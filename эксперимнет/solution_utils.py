from __future__ import annotations

import csv
import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

import os

import joblib
from sklearn.dummy import DummyClassifier  # noqa: F401  (нужен при загрузке сохраненных моделей)

WORD = re.compile(r"[^\W\d_]+")
CLASSES = ["negative", "neutral", "positive"]  # порядок классов везде одинаковый
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# -------------------------------------------------------------------------------------
#   Блок 1. Чтение датасета
# -------------------------------------------------------------------------------------

def read_dialogs(path: str) -> list[dict]:
    """Читает json: одна строка файла = один диалог (словарь)."""
    dialogs = []
    with open(path, encoding="utf-8") as file: # откр на чтение файл
        for line in file:
            line = line.strip() # убираем пробельные символы и переносы строк
            if line:  # пустые строки пропускаем
                dialogs.append(json.loads(line))
    return dialogs


def read_labels(path: str, task: str) -> dict[str, str]:
    """Ответы по задаче task ("reaction" или "topic"): dialog_id -> метка.

    Диалоги, у которых такой метки нет, пропускаются.
    """
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or {}).get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


def build_frames(dialogs_raw: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Превращает список диалогов в две таблицы.
    dialogs  — одна строка на диалог (признаки уровня диалога);
    messages — одна строка на реплику пользователя/ассистента (уровень labels).
    """
    dialog_rows, message_rows = [], []
    for dialog in dialogs_raw:
        labels = dialog.get("labels") or {}
        turn_labels = labels.get("user_turns") or []
        turns = dialog.get("turns") or []

        dialog_rows.append({
            "dialog_id": dialog.get("dialog_id"),
            "language": dialog.get("language"),
            "n_user_turns_declared": dialog.get("n_user_turns"),
            "n_messages": len(turns),
            "n_label_turns": len(turn_labels),
            "task_type": labels.get("task_type"),
            "reaction": labels.get("reaction"),
        })

        user_idx = 0  # номер реплики пользователя внутри диалога
        for pos, turn in enumerate(turns):
            text = turn.get("text")
            row = {
                "dialog_id": dialog.get("dialog_id"),
                "pos": pos,
                "role": turn.get("role"),
                "text_is_none": text is None,
                "text": text if isinstance(text, str) else "",
                "user_idx": np.nan,
                "turn_task_type": None,
                "turn_state": None,
                "turn_reaction": None,
                "turn_reaction_reason": None,
            }
            if turn.get("role") == "user":
                row["user_idx"] = user_idx
                if user_idx < len(turn_labels):
                    lab = turn_labels[user_idx] or {}
                    row["turn_task_type"] = lab.get("task_type")
                    row["turn_state"] = lab.get("state")
                    row["turn_reaction"] = lab.get("reaction")
                    row["turn_reaction_reason"] = lab.get("reaction_reason")
                user_idx += 1
            message_rows.append(row)

    return pd.DataFrame(dialog_rows), pd.DataFrame(message_rows)

# -------------------------------------------------------------------------------------
#   Блок 2. Облагораживаем текст
# -------------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Приводим текст к однотипному состоянию (нижний регистр, сиволы на пробел)"""
    return re.sub(r"\W+", " ", text.lower()).strip()

# -------------------------------------------------------------------------------------
#   Блок 3. Почти-дубликаты и сводки (из EDA)
# -------------------------------------------------------------------------------------

def nearest_neighbor(matrix, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой строки L2-нормированной разреженной матрицы ищет самую
    похожую другую строку по косинусной близости.

    Считаем кусками по `chunk` строк: полная матрица 10000 x 10000 плотной
    заняла бы 800 МБ, а кусок 1000 x 10000 — 80 МБ.
    Возвращает (индекс соседа, косинусная близость к нему).
    """
    n = matrix.shape[0]
    best_idx = np.zeros(n, dtype=int)
    best_sim = np.zeros(n)
    matrix_t = matrix.T.tocsr()
    for start in range(0, n, chunk):
        block = (matrix[start:start + chunk] @ matrix_t).toarray()
        rows = np.arange(block.shape[0])
        block[rows, start + rows] = -1.0  # сам себя соседом не считаем
        best_idx[start:start + chunk] = block.argmax(axis=1)
        best_sim[start:start + chunk] = block.max(axis=1)
    return best_idx, best_sim


def make_groups(nn_idx: np.ndarray, nn_sim: np.ndarray, threshold: float) -> np.ndarray:
    """Номер группы для каждого диалога: диалоги, связанные цепочкой
    «ближайший сосед с близостью >= threshold», получают один номер.

    Эти номера дальше подаются в GroupKFold, чтобы почти-копии
    не оказались по разные стороны разбиения.
    """
    n = len(nn_idx)
    linked = np.flatnonzero(nn_sim >= threshold)
    graph = coo_matrix((np.ones(len(linked)), (linked, nn_idx[linked])), shape=(n, n))
    _, group = connected_components(graph, directed=False)
    return group


def column_report(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по столбцам из семинара: тип, пропуски, уникальные, самое частое."""
    rows = []
    for column in df.columns:
        values = df[column]
        counts = values.value_counts(dropna=True)
        rows.append({
            "column": column,
            "dtype": str(values.dtype),
            "n_missing": int(values.isna().sum()),
            "missing_share": float(values.isna().mean()),
            "n_unique": int(values.nunique(dropna=True)),
            "top": counts.index[0] if len(counts) else None,
            "top_share": float(counts.iloc[0] / len(values)) if len(counts) else 0.0,
        })
    return pd.DataFrame(rows).set_index("column")


def iqr_sides(values: pd.Series, k: float = 1.5) -> tuple[int, int]:
    """Сколько значений ниже нижней и выше верхней границы IQR."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return int((values < low).sum()), int((values > high).sum())

# -------------------------------------------------------------------------------------
#   Блок 4. Задача 1 (реакция): подготовка текстов
# -------------------------------------------------------------------------------------

MAX_MSG_CHARS = 600   # длинные реплики (код, огромные вставки) обрезаем: сигнал реакции в начале реплики
MAX_NEXT_CHARS = 400  # ответ модели после реплики пользователя: извинения стоят в самом начале


def _text(turn: dict) -> str:
    """Текст реплики; None вместо текста (если вдруг встретится) считаем пустой строкой."""
    return turn.get("text") or ""


def dialog_text_full(dialog: dict, max_chars: int = MAX_MSG_CHARS) -> str:
    """Весь диалог одной строкой (реплики пользователя и модели), каждая обрезана до max_chars."""
    return " \n ".join(_text(t)[:max_chars] for t in dialog.get("turns") or [])


def dialog_text_part(dialog: dict, part: str, max_chars: int = MAX_MSG_CHARS) -> str:
    """Часть диалога для проверки гипотезы «что лучше всего предсказывает реакцию».

    part: all_user, user_wo_first, first_user, last_user, last2_user, assistant, full.
    """
    turns = dialog.get("turns") or []
    users = [_text(t)[:max_chars] for t in turns if t.get("role") == "user"]
    assistant = [_text(t)[:max_chars] for t in turns if t.get("role") == "assistant"]
    if part == "all_user":
        chosen = users
    elif part == "user_wo_first":
        chosen = users[1:]
    elif part == "first_user":
        chosen = users[:1]
    elif part == "last_user":
        chosen = users[-1:]
    elif part == "last2_user":
        chosen = users[-2:]
    elif part == "assistant":
        chosen = assistant
    elif part == "full":
        chosen = [_text(t)[:max_chars] for t in turns]
    else:
        raise ValueError(f"неизвестная часть диалога: {part}")
    return " \n ".join(chosen)


def user_turn_table(dialogs: list[dict], with_labels: bool = False) -> pd.DataFrame:
    """Таблица «одна строка = одна реплика пользователя, кроме первой».

    Первая реплика пользователя ни на что не реагирует (в разметке у нее reaction = null),
    поэтому в таблицу она не попадает. Для каждой реплики берем:
      u       — текст реплики пользователя;
      a_next  — ответ модели сразу после нее (если реплика была жалобой, модель обычно извиняется);
      rel     — относительная позиция реплики в диалоге (0..1), is_last — последняя ли она;
      ulen    — log(1 + длина реплики).
    Если with_labels=True, добавляется y — реакция на уровне реплики из разметки.
    """
    records = []
    for di, dialog in enumerate(dialogs):
        turns = dialog.get("turns") or []
        user_positions = [i for i, t in enumerate(turns) if t.get("role") == "user"]
        n_users = len(user_positions)
        turn_labels = ((dialog.get("labels") or {}).get("user_turns")) or []
        for k, pos in enumerate(user_positions):
            if k == 0:
                continue
            u = _text(turns[pos])[:MAX_MSG_CHARS]
            a_next = _text(turns[pos + 1])[:MAX_NEXT_CHARS] if pos + 1 < len(turns) else ""
            row = {
                "di": di, "k": k, "u": u, "a_next": a_next,
                "rel": k / max(n_users - 1, 1),
                "is_last": float(k == n_users - 1),
                "ulen": float(np.log1p(len(u))),
            }
            if with_labels:
                row["y"] = turn_labels[k]["reaction"] if k < len(turn_labels) and turn_labels[k] else None
            records.append(row)
    columns = ["di", "k", "u", "a_next", "rel", "is_last", "ulen"] + (["y"] if with_labels else [])
    return pd.DataFrame(records, columns=columns)


def aggregate_turns(turns: pd.DataFrame, probs: np.ndarray, n_dialogs: int,
                    ta: float, tb: float) -> list[str]:
    """Из вероятностей по репликам получаем ответ по диалогу.

    В разметке реакция диалога = реакция ПОСЛЕДНЕЙ не нейтральной реплики пользователя
    (все нейтральные -> neutral), поэтому и мы идем по репликам по порядку и запоминаем последнюю,
    у которой P(negative) > ta или P(positive) > tb. Если сработали оба порога, берем тот,
    который превышен сильнее (в долях от порога). probs — столбцы в порядке CLASSES.
    """
    answers = ["neutral"] * n_dialogs
    di = turns["di"].to_numpy()
    for row_index in range(len(turns)):  # turns уже упорядочены по (диалог, номер реплики)
        p_neg, p_pos = probs[row_index, 0], probs[row_index, 2]
        neg, pos = p_neg > ta, p_pos > tb
        if neg and pos:
            answers[di[row_index]] = "negative" if p_neg / ta >= p_pos / tb else "positive"
        elif neg:
            answers[di[row_index]] = "negative"
        elif pos:
            answers[di[row_index]] = "positive"
    return answers

# -------------------------------------------------------------------------------------
#   Блок 5. Задача 1 (реакция): модели
#   Все модели имеют метод predict(dialogs) -> список ответов positive/neutral/negative
# -------------------------------------------------------------------------------------

class DialogTextModel:
    """Базовая модель: TF-IDF по тексту всего диалога + линейный классификатор или наивный Байес."""

    def __init__(self, classifier, max_features: int = 60000):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                          max_features=max_features)
        self.classifier = classifier

    def fit(self, dialogs: list[dict], labels: list[str]):
        texts = [dialog_text_full(d) for d in dialogs]
        self.classifier.fit(self.vectorizer.fit_transform(texts), labels)
        return self

    def predict(self, dialogs: list[dict]) -> list[str]:
        texts = [dialog_text_full(d) for d in dialogs]
        return [str(a) for a in self.classifier.predict(self.vectorizer.transform(texts))]


class TurnReactionModel:
    """Своя модель. Классифицируем каждую реплику пользователя (кроме первой) на
    negative / neutral / positive по тексту реплики и по ответу модели на нее,
    а потом собираем ответ по диалогу так, как собрана разметка: реакция последней
    не нейтральной реплики. Порогам ta и tb нужны для macro-F1 (см. aggregate_turns).
    """

    def __init__(self, C: float = 1.0, ta: float = 0.6, tb: float = 0.4, max_features: int = 60000,
                 use_next: bool = True):
        # use_next=False выключает признаки ответа модели (нужно только для проверки их вклада)
        self.C, self.ta, self.tb, self.max_features, self.use_next = C, ta, tb, max_features, use_next

    def _vectorizers(self):
        kwargs = dict(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=self.max_features)
        return TfidfVectorizer(**kwargs), TfidfVectorizer(**kwargs)

    def _matrix(self, turns: pd.DataFrame):
        numeric = csr_matrix(turns[["rel", "is_last", "ulen"]].to_numpy() * np.array([1.0, 1.0, 0.2]))
        blocks = [self.vec_user.transform(turns["u"])]
        if self.use_next:
            blocks.append(self.vec_next.transform(turns["a_next"]))
        return hstack(blocks + [numeric]).tocsr()

    def fit(self, dialogs: list[dict]):
        turns = user_turn_table(dialogs, with_labels=True)
        turns = turns[turns["y"].notna()].reset_index(drop=True)
        self.vec_user, self.vec_next = self._vectorizers()
        self.vec_user.fit(turns["u"])
        if self.use_next:
            self.vec_next.fit(turns["a_next"])
        self.classifier = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000)
        self.classifier.fit(self._matrix(turns), turns["y"])
        assert list(self.classifier.classes_) == CLASSES
        return self

    def turn_probabilities(self, dialogs: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
        """Таблица реплик и матрица вероятностей (столбцы: negative, neutral, positive)."""
        turns = user_turn_table(dialogs)
        if len(turns) == 0:
            return turns, np.zeros((0, 3))
        return turns, self.classifier.predict_proba(self._matrix(turns))

    def predict(self, dialogs: list[dict]) -> list[str]:
        turns, probs = self.turn_probabilities(dialogs)
        return aggregate_turns(turns, probs, len(dialogs), self.ta, self.tb)

    @property
    def classes_(self):
        return np.array(CLASSES)

    def predict_proba(self, dialogs: list[dict]) -> np.ndarray:
        """Оценки классов для диалога (столбцы как в CLASSES), нужны для ROC-AUC и PR-AUC.
        Это не настоящие вероятности: для negative и positive берем максимум по репликам,
        neutral = 1 - max(...), строки нормируем на сумму 1."""
        turns, probs = self.turn_probabilities(dialogs)
        scores = np.zeros((len(dialogs), 3))
        scores[:, 1] = 1.0
        for di in np.unique(turns["di"].to_numpy()) if len(turns) else []:
            mask = turns["di"].to_numpy() == di
            neg, pos = probs[mask, 0].max(), probs[mask, 2].max()
            scores[di] = [neg, max(1.0 - max(neg, pos), 0.0), pos]
        return scores / scores.sum(axis=1, keepdims=True)

# -------------------------------------------------------------------------------------
#   Блок 5а. Текстовые модели: предобработка как на семинаре 4 (токены, стоп-слова, основы слов)
#   Вся предобработка живет здесь, а не в ноутбуке: сохраненная модель загружается в другом
#   процессе (solution.ipynb) и должна найти эти функции через import solution_utils.
# -------------------------------------------------------------------------------------

_NLP: dict = {}
_STEM_CACHE: dict = {}


def _nlp() -> dict:
    """Стоп-слова и стеммеры nltk. Загружаются при первом использовании: без стоп-слов и основ
    слов (stop=False, stem=False) nltk для работы моделей не нужен."""
    if not _NLP:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        _NLP["stop"] = set(stopwords.words("english")) | set(stopwords.words("russian"))
        _NLP["en"], _NLP["ru"] = SnowballStemmer("english"), SnowballStemmer("russian")
    return _NLP


def tokenize(text: str, stop: bool = True, stem: bool = True) -> list[str]:
    """Слова из букв, нижний регистр, длина >= 2; по желанию без стоп-слов и/или с основами слов.
    При stop=True, stem=True это ровно smart_tokenizer из ноутбука."""
    words = [w for w in WORD.findall(text.lower()) if len(w) >= 2]
    if not (stop or stem):
        return words
    nlp = _nlp()
    result = []
    for word in words:
        if stop and word in nlp["stop"]:
            continue
        if stem:
            stemmed = _STEM_CACHE.get(word)
            if stemmed is None:  # кэш: у слов повторы, стеммер вызываем один раз на слово
                stemmer = nlp["ru"] if ("а" <= word[0] <= "я" or word[0] == "ё") else nlp["en"]
                stemmed = _STEM_CACHE[word] = stemmer.stem(word)
            word = stemmed
        result.append(word)
    return result


def smart_tokenizer(text: str) -> list[str]:
    """Токенизатор kirp_model из исходного ноутбука (стоп-слова + основы слов)."""
    return tokenize(text, stop=True, stem=True)


def dialog_text_cut(dialog: dict, max_chars: int | None = MAX_MSG_CHARS) -> str:
    """Все реплики диалога (и пользователя, и модели) одной строкой; каждая обрезана до max_chars (None = не обрезать)."""
    return " ".join(_text(t)[:max_chars] if max_chars else _text(t) for t in dialog.get("turns") or [])


def prepare_dialog(dialog: dict, max_chars: int | None = MAX_MSG_CHARS, stop: bool = False, stem: bool = False) -> str:
    """Диалог -> строка из токенов через пробел (дальше ее векторизует TF-IDF с token_pattern=r"\\S+")."""
    return " ".join(tokenize(dialog_text_cut(dialog, max_chars), stop=stop, stem=stem))


class TextReactionModel:
    """Модель на тексте диалога: диалог -> prepare_dialog -> sklearn-модель (векторизатор + классификатор).

    Предобработка (обрезка реплик, стоп-слова, основы слов) задается параметрами и хранится в самой модели,
    поэтому predict принимает список диалогов (словарей), как все остальные модели задачи 1.
    """

    def __init__(self, estimator, max_chars: int | None = MAX_MSG_CHARS, stop: bool = False, stem: bool = False):
        self.estimator, self.max_chars, self.stop, self.stem = estimator, max_chars, stop, stem

    def _texts(self, dialogs: list[dict]) -> list[str]:
        return [prepare_dialog(d, self.max_chars, self.stop, self.stem) for d in dialogs]

    def fit(self, dialogs: list[dict], labels):
        self.estimator.fit(self._texts(dialogs), labels)
        return self

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict(self, dialogs: list[dict]) -> list[str]:
        return [str(a) for a in self.estimator.predict(self._texts(dialogs))]

    def predict_proba(self, dialogs: list[dict]) -> np.ndarray:
        return self.estimator.predict_proba(self._texts(dialogs))


# -------------------------------------------------------------------------------------
#   Блок 6. Задача 1 (реакция): интерфейс для solution.ipynb
# -------------------------------------------------------------------------------------

REACTION_ARTIFACTS = {
    "most_frequent": "reaction_most_frequent.joblib",
    "stratified": "reaction_stratified.joblib",
    "bow_logreg": "reaction_bow_logreg.joblib",
    "tfidf_logreg": "reaction_tfidf_logreg.joblib",
    "tfidf_cnb": "reaction_tfidf_cnb.joblib",
    "kirp_model": "reaction_kirp_model.joblib",
    "turn_level_logreg": "reaction_turn_level_logreg.joblib",
}


def init_reaction_models(train_path: str | None = None, artifacts_dir: str = ARTIFACTS_DIR) -> dict:
    """Все модели задачи 1: имя -> готовая к предсказанию модель.

    Модели загружаются из папки artifacts (их туда кладет ноутбук с обучением).
    train_path не нужен: оставлен в сигнатуре, чтобы совпадать с примером преподавателя.
    """
    models = {}
    for name, filename in REACTION_ARTIFACTS.items():
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"нет файла модели {path}: запустите ноутбук с обучением, он сохранит модели в artifacts")
        models[name] = joblib.load(path)
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: "positive", "neutral" или "negative"."""
    return [str(label) for label in model.predict(dialogs)]

# -------------------------------------------------------------------------------------
#   Блок 7. Метрики и сохранение ответов
# -------------------------------------------------------------------------------------

def reaction_report(y_true, y_pred) -> dict[str, float]:
    """Все метрики реакции: macro-F1 (основная), accuracy, balanced accuracy,
    precision и recall каждого класса."""
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    report = {
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    for cls, p, r in zip(CLASSES, precision, recall):
        report[f"precision_{cls}"] = float(p)
        report[f"recall_{cls}"] = float(r)
    return {name: float(value) for name, value in report.items()}


def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """Метрики по тем диалогам, для которых известен ответ."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    return reaction_report(y_true, y_pred)


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, dict[str, list[str]]]) -> None:
    """Сохранить ответы всех моделей: dialog_id, task, model, prediction."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])
