from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "artifacts" / "results"
CLASSES = ["negative", "neutral", "positive"]
SEED = 42

# Старые модели HW1: имя в HW2 -> файл, как он назывался в HW1 (artifacts/ в папке HW1).
OLD_MODEL_FILES = {
    "hw1_most_frequent": "most_frequent_1.joblib",
    "hw1_stratified": "stratified_1.joblib",
    "hw1_bow_logreg": "bow_logreg_1.joblib",
    "hw1_tfidf_logreg": "tfidf_logreg_1.joblib",
    "hw1_kirp_model": "kirp_model_1.joblib",
}


# ---------------------------------------------------------------------------
#  Поиск файлов
# ---------------------------------------------------------------------------

def find_file(candidates: list[str]) -> Path | None:
    """Первый существующий путь из списка (пустые строки пропускаются)."""
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def train_path() -> Path:
    path = find_file([os.environ.get("HW2_TRAIN", ""), "hw1_train.jsonl",
                      "/home/pavel-ubunt/Рабочий стол/ML_mipt/MIPT_ML_homework/solution_hw1_BKZ/dataset/hw1_train.jsonl",
                      "hw1_data/hw1_train.jsonl", "../solution_hw1_BKZ/dataset/hw1_train.jsonl"])
    assert path is not None, "не нашли hw1_train.jsonl: положите рядом или задайте HW2_TRAIN"
    return path


def flags_path() -> Path:
    path = find_file(["artifacts/dialog_flags.csv", "dialog_flags.csv", "../solution_hw1_BKZ/artifacts/dialog_flags.csv"])
    assert path is not None, "не нашли dialog_flags.csv (он лежит в artifacts/ папки HW1)"
    return path


# ---------------------------------------------------------------------------
#  Данные и разбиение 
# ---------------------------------------------------------------------------

def read_dialogs(path) -> list[dict]:
    """jsonl -> список диалогов {dialog_id, turns}. Метки не читаем: для обучения они в flags."""
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                dialogs.append({"dialog_id": rec["dialog_id"], "turns": rec["turns"]})
    return dialogs


def load_task1(train_file, flags_file) -> pd.DataFrame:
    """Диалоги задачи 1: только use_reaction == 1 (согласно EDA).
    Столбцы: dialog_id, turns, reaction, group_id.
    Порядок строк — как в файле,
    поэтому разбиение ниже даёт ровно те же train/test, что в HW1.
    """
    texts = pd.DataFrame(read_dialogs(train_file))
    flags = pd.read_csv(flags_file)[["dialog_id", "reaction", "use_reaction", "group_id"]]
    df = texts.merge(flags, on="dialog_id", how="left", validate="one_to_one")
    task1 = df[df["use_reaction"].astype(bool)].drop(columns="use_reaction").reset_index(drop=True)
    return task1


def split_task1(task1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test как в HW1: GroupShuffleSplit по group_id, 80/20, random_state=42."""
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    train_idx, test_idx = next(gss.split(task1, task1["reaction"], groups=task1["group_id"]))
    return task1.iloc[train_idx].reset_index(drop=True), task1.iloc[test_idx].reset_index(drop=True)


def get_folds(y, groups) -> list:
    """Одни и те же 5 блоков для всех моделей: группы не разрываются, доли классов сохраняются.
    Возвращает список пар (индексы_обучения, индексы_проверки)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y, groups))


# ---------------------------------------------------------------------------
#  Вход старых моделей HW1 и их токенайзер
# ---------------------------------------------------------------------------

def old_dialog_text(dialog: dict) -> str:
    """Как в HW1 достаются все реплики (и пользователя, и ассистента) и склеиваются в одну общую строку через пробел"""
    return " ".join(t.get("text", "") for t in dialog["turns"] if isinstance(t, dict))

stemmer_en = SnowballStemmer("english")
stemmer_ru = SnowballStemmer("russian")

stop_words_en = set(stopwords.words("english"))
stop_words_ru = set(stopwords.words("russian"))
all_stop_words = stop_words_en.union(stop_words_ru) # объединяем стоп-слова
WORD_REG = re.compile(r'[^\W\d_]+')

def smart_tokenizer(text: str) -> list[str]:
    """Токенизатор модели kirp_model (копия из HW1)"""

    text_lower = text.lower()
    raw_words = WORD_REG.findall(text_lower)
    
    result = []
    for word in raw_words:
        if (
            (len(word) < 2 or word in all_stop_words) and 
            (word != 'no' or 'ok' or 'yes' or 'not')
            ): continue
            
        # Проверяем, на каком языке слово (по первой букве)
        if 'а' <= word[0] <= 'я' or word[0] == 'ё':
            stemmed = stemmer_ru.stem(word) # пилим русские окончания
        else:
            stemmed = stemmer_en.stem(word) # пилим английские окончания
            
        result.append(stemmed)
    return result


def migrate_old_models(hw1_artifacts: str | Path) -> dict:
    import __main__
    __main__.smart_tokenizer = smart_tokenizer  # нужно только для загрузки старых pickle

    hw1_artifacts = Path(hw1_artifacts)
    ARTIFACTS.mkdir(exist_ok=True)
    done = {}
    for new_name, filename in OLD_MODEL_FILES.items():
        model = joblib.load(hw1_artifacts / filename)
        first_step = model.steps[0][1] if hasattr(model, "steps") else None
        if first_step is not None and getattr(first_step, "tokenizer", None) is not None:
            first_step.tokenizer = smart_tokenizer
        dst = ARTIFACTS / f"{new_name}.joblib"
        joblib.dump(model, dst, compress=3)
        done[new_name] = dst
    ...
    return done


# ---------------------------------------------------------------------------
#  Признаки для деревьев (новые модели hw2_)
# ---------------------------------------------------------------------------

NUMERIC_COLUMNS = [
    "n_user_turns_declared",   # заявлено реплик пользователя в задании
    "n_user_actual",           # сколько реплик пользователя реально
    "n_messages",              # всего сообщений в диалоге
    "n_label_turns",           # сколько реплик размечено
    "n_chars_total",           # суммарная длина всех сообщений
    "max_msg_chars",           # длина самой длинной реплики
    "first_turn_freq",         # частота первой реплики в корпусе
    "nn_sim",                  # похожесть на ближайшего соседа
    "bad_label_count",         # число плохих меток
    "bad_roles",               # неправильная последовательность ролей
    "has_code",                # есть код в диалоге
    "has_user_code",           # код у пользователя
    "has_assistant_code",      # код у ассистента
    "has_assistant_loop",      # ассистент зациклился
    "user_has_cyrillic",       # кириллица у пользователя
    "has_empty_message",       # пустое сообщение
    "has_short_turn",          # короткая реплика
    "is_template",             # шаблонный диалог
    "is_exact_dup",            # точный дубликат
    "in_near_dup_group",       # группа почти-дубликатов
]


def load_flags() -> pd.DataFrame:
    """EDA-разметка, индекс — dialog_id."""
    return pd.read_csv(flags_path()).set_index("dialog_id")


def make_frame(dialogs: list[dict], flags: pd.DataFrame) -> pd.DataFrame:
    """Диалог -> строка таблицы: два текста + числа из разметки.

    Текст (user_text, last_user) собирается из реплик — он нужен для TF-IDF.
    Все числовые признаки берутся из dialog_flags.csv по dialog_id;
    отсутствие dialog_id в разметке — ошибка вызывающего кода.
    """
    rows = []
    for d in dialogs:
        user = [t["text"] or "" for t in d["turns"] if t["role"] == "user"]
        if d["dialog_id"] not in flags.index:
            raise KeyError(f"нет в разметке: {d['dialog_id']}")

        row = {
            "dialog_id": d["dialog_id"],
            "user_text": " ".join(user),
            "last_user": user[-1] if user else "",
            "dialog_text": " ".join(
                (t.get("text") or "") for t in d["turns"] if isinstance(t, dict)
            ),
        }
        row.update(flags.loc[d["dialog_id"], NUMERIC_COLUMNS].to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def uses_frame(model) -> bool:
    """Новые модели (hw2_) принимают таблицу make_frame (первый шаг пайплайна "features"),
    старые (hw1_) — список строк."""
    return hasattr(model, "named_steps") and "features" in model.named_steps


def make_input(model, dialogs: list[dict]):
    return make_frame(dialogs) if uses_frame(model) else [old_dialog_text(d) for d in dialogs]


# ---------------------------------------------------------------------------
#  Модели: сохранение и загрузка
# ---------------------------------------------------------------------------

def save_model(name: str, model) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / f"{name}.joblib"
    joblib.dump(model, path, compress=3)   # compress=3, как в HW1: файлы меньше
    return path


def load_models(prefix: str = "") -> dict:
    import __main__
    # старые pickle из HW1 ссылаются на __main__.smart_tokenizer
    if not hasattr(__main__, "smart_tokenizer"):
        __main__.smart_tokenizer = smart_tokenizer

    return {p.stem: joblib.load(p) for p in sorted(ARTIFACTS.glob(f"{prefix}*.joblib"))}


def init_reaction_models(train_path_: str | None = None) -> dict:
    """Для solution.ipynb: все модели (старые и новые) из artifacts/."""
    models = load_models("hw1_") | load_models("hw2_")
    if not models:
        raise FileNotFoundError(f"В {ARTIFACTS} нет моделей *.joblib")
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    return [str(a) for a in model.predict(make_input(model, dialogs))]


# ---------------------------------------------------------------------------
#  Метрики (набор как в таблице HW1) и общие таблицы результатов
# ---------------------------------------------------------------------------

def all_metrics(y_true, proba, classes=CLASSES) -> dict[str, float]:
    """Все метрики HW1 по вероятностям proba (n x 3, столбцы в порядке classes).

    Предсказание = класс с наибольшей вероятностью. Основная метрика — macro_f1.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    row = {
        "macro_f1": f1_score(y_true, pred, labels=classes, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes),
        "pr_auc_macro": average_precision_score(label_binarize(y_true, classes=classes), proba, average="macro"),
    }
    prec = precision_score(y_true, pred, labels=classes, average=None, zero_division=0)
    rec = recall_score(y_true, pred, labels=classes, average=None, zero_division=0)
    for cls, p, r in zip(classes, prec, rec):
        row[f"precision_{cls}"], row[f"recall_{cls}"] = p, r
    return row


def cv_metrics(model, X, y, folds, n_jobs=-1) -> dict[str, float]:
    """Метрики модели на кросс-валидации с заданными блоками folds."""

    proba = cross_val_predict(clone(model), X, y, cv=folds, method="predict_proba", n_jobs=n_jobs)
    return all_metrics(y, proba)


def save_result(table_file: str, model_name: str, row: dict) -> pd.DataFrame:
    """Добавить/обновить строку модели в results/<table_file> (cv_table.csv или test_table.csv).
    Так вся команда собирает одну общую таблицу."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / table_file
    table = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()
    table.loc[model_name, list(row)] = list(row.values())
    table = table.sort_values("macro_f1", ascending=False)
    table.to_csv(path)
    return table


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, list[str]]) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for name, answers in predictions.items():
            for d, a in zip(dialogs, answers):
                writer.writerow([d["dialog_id"], "reaction", name, a])
