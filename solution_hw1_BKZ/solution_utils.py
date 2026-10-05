from __future__ import annotations

import json, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}

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
#   Блок 3. Чтение датасета
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