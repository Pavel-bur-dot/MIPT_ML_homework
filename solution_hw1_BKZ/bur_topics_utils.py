"""HW1, задача 2 (темы). Каркас ваш (bur_topics_utils.py); исправления и продолжение отмечены в ноутбуке."""
from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score, homogeneity_completeness_v_measure,
                             normalized_mutual_info_score, precision_recall_fscore_support)
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, normalize

import solution_utils as su

# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
SEED = 42
TEST_SIZE = 0.2
MAX_CHARS_PER_TURN = 1000     # каждая реплика пользователя обрезается до 1000 символов (огромные вставки мешают TF-IDF)

# ---------------------------------------------------------------------------
# Чтение данных
# ---------------------------------------------------------------------------
def load_dialogs_and_flags(data_path: str | None = None):
    """Возвращает (dialogs, messages, flags). flags — результат EDA (dialog_flags.csv), он обязателен."""
    candidates = [data_path or "", os.environ.get("HW1_TRAIN", ""), "hw1_train.jsonl", "dataset/hw1_train.jsonl"]
    data_path = next((p for p in candidates if p and os.path.exists(p)), None)
    assert data_path is not None, f"не нашли hw1_train.jsonl, искали: {candidates}"
    flags_path = next((p for p in ["artifacts/dialog_flags.csv"] if os.path.exists(p)), None)
    assert flags_path is not None, "не нашли dialog_flags.csv (его делает ноутбук EDA)"

    dialogs, messages = su.build_frames(su.read_dialogs(data_path))
    return dialogs, messages, pd.read_csv(flags_path)


# ---------------------------------------------------------------------------
# Фильтры и маска (одно место правды: воронка и маска используют одни и те же правила)
# ---------------------------------------------------------------------------
def topic_filters(flags: pd.DataFrame) -> dict:
    """Правила «оставить диалог» для задачи про темы. Метки reaction/task_type для тем не нужны.
    Диалоги в группах почти-копий НЕ выкидываем: они остаются, а при разбиении группа идёт целиком в одну часть."""
    return {
        "~is_template": ~flags["is_template"],
        "~has_empty_message": ~flags["has_empty_message"],
        "use_topics": flags["use_topics"],
    }


def filter_funnel(flags: pd.DataFrame) -> pd.DataFrame:
    """Сколько диалогов остаётся после каждого фильтра по очереди."""
    current = np.ones(len(flags), dtype=bool)
    rows = [{"шаг": "всего", "осталось": len(flags)}]
    for name, rule in topic_filters(flags).items():
        current = current & rule.to_numpy()
        rows.append({"шаг": name, "осталось": int(current.sum())})
    return pd.DataFrame(rows)


def build_train_mask(flags: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(flags), dtype=bool)
    for rule in topic_filters(flags).values():
        mask = mask & rule.to_numpy()
    return mask


# ---------------------------------------------------------------------------
# Текст диалога: все реплики пользователя -> без стоп-слов -> стемминг
# ---------------------------------------------------------------------------
RU_STOPWORDS = (
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня "
    "еще нет о из ему теперь когда даже ну ли если уже или ни быть был него до вас вам там потом себя ничего ей может "
    "они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет тогда кто этот "
    "того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех можно при "
    "два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя свою этой перед "
    "лучше чуть том такой им более всегда конечно между"
).split()
# «Слова-просьбы»: есть в запросах на любую тему и тянут кластеры к стилю просьбы («write…», «make…»), а не к теме.
# Список составлен по вашей таблице топ-слов (ноутбук, раздел «Топ слов»): тех из них, что не входят в стоп-слова sklearn.
REQUEST_WORDS = (
    "write using use make want like need create help list follow following work provide new different example "
    "question tell explain thing way know let try look say hello hi word words"
).split()
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | frozenset(RU_STOPWORDS) | frozenset(REQUEST_WORDS)

WORD = re.compile(r"[^\W\d_]{3,}")                      # слово: 3+ буквы любого алфавита, без цифр
CYRILLIC = re.compile(r"[а-яё]")
STEMMERS = {"english": SnowballStemmer("english"), "russian": SnowballStemmer("russian")}
STEM_CACHE: dict = {}


def stem(word: str) -> str:
    """Основа слова. Язык определяем ПО СЛОВУ (а не по всему тексту): в английском диалоге бывают русские вставки."""
    if word not in STEM_CACHE:
        STEM_CACHE[word] = STEMMERS["russian" if CYRILLIC.search(word) else "english"].stem(word)
    return STEM_CACHE[word]


def preprocess_for_topics(text: str) -> str:
    """Нижний регистр -> слова -> без стоп-слов (до стемминга) -> стемминг -> одна строка."""
    if not isinstance(text, str):
        return ""
    return " ".join(stem(w) for w in WORD.findall(text.lower()) if w not in STOP_WORDS)


def extract_user_text(messages: pd.DataFrame, max_chars: int = MAX_CHARS_PER_TURN) -> pd.Series:
    """Все реплики пользователя диалога через пробел (каждая обрезана до max_chars). Индекс — dialog_id."""
    user = messages[messages["role"] == "user"]
    return user["text"].fillna("").str.slice(0, max_chars).groupby(user["dialog_id"]).agg(" ".join)


def prepare_topic_texts(messages: pd.DataFrame) -> pd.Series:
    """Из таблицы реплик в готовые для моделей тексты. Это единственная подготовка, нужная на скрытом тесте."""
    return extract_user_text(messages).map(preprocess_for_topics)


def extract_preview(messages: pd.DataFrame, n_turns: int = 3, max_chars: int = 110) -> pd.Series:
    """Первые реплики пользователя в исходном виде (для чтения глазами)."""
    user = messages[messages["role"] == "user"]
    first = user["text"].fillna("").str.slice(0, max_chars).str.replace("\n", " ", regex=False)
    return first.groupby(user["dialog_id"]).apply(lambda s: " | ".join(s.iloc[:n_turns]))


# ---------------------------------------------------------------------------
# Разбиение train/test по группам почти-копий
# ---------------------------------------------------------------------------
def group_split(flags: pd.DataFrame, mask: np.ndarray, test_size: float = TEST_SIZE, seed: int = SEED):
    """GroupShuffleSplit по group_id. Возвращает (train_ids, test_ids)."""
    subset = flags.loc[mask]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(subset, groups=subset["group_id"].to_numpy()))
    return subset.iloc[train_idx]["dialog_id"].to_numpy(), subset.iloc[test_idx]["dialog_id"].to_numpy()


# ---------------------------------------------------------------------------
# TF-IDF, слова тем, типичные диалоги
# ---------------------------------------------------------------------------
def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(min_df=3, max_df=0.5, max_features=30000, sublinear_tf=True, ngram_range=(1, 1))


def distinctive_terms(texts: pd.Series, labels: np.ndarray, n_terms: int = 10) -> dict:
    """Для каждой темы слова, у которых средний TF-IDF внутри темы больше всего превышает средний вне темы."""
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    result = {}
    for label in np.unique(labels):
        mask = labels == label
        diff = np.asarray(matrix[mask].mean(axis=0)).ravel() - np.asarray(matrix[~mask].mean(axis=0)).ravel()
        result[label] = list(vocab[np.argsort(diff)[::-1][:n_terms]])
    return result


def representatives(X: np.ndarray, labels: np.ndarray, n: int = 5) -> dict:
    """Позиции n диалогов, ближайших к центру каждой темы (X должна быть L2-нормирована)."""
    result = {}
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        similarity = X[idx] @ X[idx].mean(axis=0)
        result[label] = list(idx[np.argsort(similarity)[::-1][:n]])
    return result


# ---------------------------------------------------------------------------
# Метрики (те же, что считает проверка на скрытом тесте)
# ---------------------------------------------------------------------------
def purity_score(ref, pred) -> float:
    """Чистота: для каждой НАШЕЙ темы берём самую частую эталонную; доля диалогов, попавших в «свою» эталонную тему."""
    table = contingency_matrix(ref, pred)             # строки — эталон, столбцы — наши темы
    return float(table.max(axis=0).sum() / table.sum())


def topic_metrics(ref, pred) -> dict:
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(ref, pred)
    return {"ARI": adjusted_rand_score(ref, pred), "NMI": normalized_mutual_info_score(ref, pred),
            "homogeneity": homogeneity, "completeness": completeness, "V": v_measure, "purity": purity_score(ref, pred)}


def supervised_metrics(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_F1": f1_score(y_true, y_pred, average="macro"),
            **topic_metrics(y_true, y_pred)}


def per_topic_report(y_true, y_pred, labels) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    table = pd.DataFrame({"precision": precision, "recall": recall, "F1": f1, "support": support}, index=labels)
    table["доля"] = table["support"] / table["support"].sum()
    return table


# ---------------------------------------------------------------------------
# Модели: на входе одна текстовая колонка (результат prepare_topic_texts)
# ---------------------------------------------------------------------------
class CosineCentroidClassifier(BaseEstimator, ClassifierMixin):
    """«Ближайший центр» по КОСИНУСНОЙ близости (классический метод Rocchio).

    fit:     для каждой темы считаем центр (средний вектор диалогов темы) и нормируем его до длины 1.
    predict: диалог относим к теме, чей центр на него больше всего «похож»: сходство = скалярное произведение
             (если у диалога длина тоже 1, это и есть косинус угла между векторами).
    """

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        centers = np.vstack([np.asarray(X[y == c].mean(axis=0)).ravel() for c in self.classes_])
        self.centers_ = normalize(centers)               # длина каждого центра = 1
        return self

    def predict(self, X):
        scores = np.asarray(X @ self.centers_.T)         # сходство с каждым центром, форма (диалогов, тем)
        return self.classes_[scores.argmax(axis=1)]


def make_topic_models(seed: int = SEED, own_percentile: int = 50) -> dict:
    """
    dummy_most_frequent    — нижняя планка: всегда самая частая тема.
    tfidf_nearest_centroid — базовая 1: ближайший центр темы (евклидово расстояние).
    tfidf_naive_bayes      — базовая 2: наивный Байес (какие слова чаще встречаются в какой теме).
    tfidf_logreg           — базовая 3: логистическая регрессия (вес каждого слова «за» и «против» каждой темы).
    own_cosine_centroid    — СВОЯ: базовая 1, улучшенная: косинус + отбор слов по chi² (own_percentile — сколько
                             процентов слов оставить; 100 = без отбора).
    """
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "tfidf_nearest_centroid": make_pipeline(make_tfidf(), NearestCentroid()),
        "tfidf_naive_bayes": make_pipeline(make_tfidf(), MultinomialNB()),
        "tfidf_logreg": make_pipeline(make_tfidf(), LogisticRegression(max_iter=300, random_state=seed)),
        "own_cosine_centroid": make_pipeline(make_tfidf(), SelectPercentile(chi2, percentile=own_percentile),
                                             Normalizer(), CosineCentroidClassifier()),
    }


# ---------------------------------------------------------------------------
# Сохранение и инференс
# ---------------------------------------------------------------------------
def save_topic_models(models: dict, topic_table: pd.DataFrame, directory=ARTIFACTS_DIR / "topics") -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, directory / f"{name}.joblib", compress=3)
    topic_table.to_csv(directory / "topics.csv", index=False)


def init_topic_models(directory=ARTIFACTS_DIR / "topics") -> dict:
    """Загружает все модели задачи 2: {имя модели: обученная модель}."""
    return {path.stem: joblib.load(path) for path in sorted(Path(directory).glob("*.joblib"))}


def topic_predictions_frame(models: dict, texts: pd.Series) -> pd.DataFrame:
    """Ответы в формате проверки: dialog_id, task, model, prediction (строка на пару «диалог × модель»)."""
    parts = [pd.DataFrame({"dialog_id": texts.index.to_numpy(), "task": "topic", "model": name,
                           "prediction": np.asarray(model.predict(texts), dtype=object)})
             for name, model in models.items()]
    return pd.concat(parts, ignore_index=True)
