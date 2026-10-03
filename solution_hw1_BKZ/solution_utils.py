from __future__ import annotations

import csv
import json
import re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}


def read_raw(path: str) -> list[dict]:
    """Читает jsonl со всеми метками
        возвращает список словарей - диалогов"""
    records = []
    with open(path, encoding="utf-8") as fh: # открываем на чтение файл
        for line in fh: # бежим по строкам-диалогам
            if line.strip():# if - проверка на непустую строку и
                            # line.strip() - убирает все пробелы, переносы и табы
                records.append(json.loads(line)) # парсим строку на словарь со всеми метками диалога
    return records


def to_dialogs_frame(records: list[dict]) -> pd.DataFrame:
    """Создает DataFrame: одна строка на диалог->
    Столбцы:
        dialog_id, language, n_user_turns, n_turns,
        n_user_replies, n_assistant_replies,
        user_chars, asst_chars, total_chars,
        task_type, reaction.
    """
    rows = []
    for r in records:
        turns = r["turns"]  # просто переобозначаем
        user_chars = sum(len(t["text"]) for t in turns if t["role"] == "user") # сумма символов для пользователя
        asst_chars = sum(len(t["text"]) for t in turns if t["role"] == "assistant") # сумма символов для ассистента
        labels = r.get("labels") or {}  # получаем более "глубокий" подуровень - метки по репликам
        rows.append({
            "dialog_id": r["dialog_id"],
            "language": r.get("language"),
            "n_user_turns": r.get("n_user_turns"),
            "n_turns": len(turns),
            "n_user_replies": sum(1 for t in turns if t["role"] == "user"),
            "n_asst_replies": sum(1 for t in turns if t["role"] == "assistant"),
            "user_chars": user_chars,
            "asst_chars": asst_chars,
            "total_chars": user_chars + asst_chars,
            "task_type": labels.get("task_type"),
            "reaction": labels.get("reaction"),
        })
    return pd.DataFrame(rows)


def to_turns_frame(records: list[dict]) -> pd.DataFrame:
    """Одна строка на i реплику пользователя с её метками для такого то id"""
    rows = []
    for r in records:
        user_indices = [i for i, t in enumerate(r["turns"]) if t["role"] == "user"] # мщем индексы реплик пользователя в turns
        user_labels = (r.get("labels") or {}).get("user_turns") or [] # сопоставляем реплики c labels
        for pos, (i, lab) in enumerate(zip(user_indices, user_labels)): # дружим между собой turns и labels
            turn = r["turns"][i]
            rows.append({
                "dialog_id": r["dialog_id"],
                "language": r.get("language"),
                "position": pos,
                "is_last": pos == len(user_indices) - 1, # True для последней реплики
                "is_first": pos == 0,                    # True для первой реплики
                "text_len": len(turn["text"]),
                "task_type": lab.get("task_type"),
                "state": lab.get("state"),
                "reaction": lab.get("reaction"),
                "reaction_reason": lab.get("reaction_reason"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Блок A. Базовые распределения
# ---------------------------------------------------------------------------

def describe_lengths(dialogs_df: pd.DataFrame) -> pd.DataFrame:
    """Квартили и выбросы по n_turns, user_chars, asst_chars, total_chars."""
    cols = ["n_turns", "n_user_turns", "user_chars", "asst_chars", "total_chars"]
    return dialogs_df[cols].describe(percentiles=[.5, .9, .99]).T


def language_counts(dialogs_df: pd.DataFrame) -> pd.Series:
    return dialogs_df["language"].value_counts(dropna=False)


def reaction_distribution(dialogs_df: pd.DataFrame) -> pd.Series:
    return dialogs_df["reaction"].value_counts(dropna=False)


def task_type_distribution(dialogs_df: pd.DataFrame) -> pd.Series:
    return dialogs_df["task_type"].value_counts(dropna=False)


# ---------------------------------------------------------------------------
#  Блок B. Метки и их согласованность
# ---------------------------------------------------------------------------

def turn_label_distributions(turns_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "task_type": turns_df["task_type"].value_counts(dropna=False),
        "state": turns_df["state"].value_counts(dropna=False),
        "reaction": turns_df["reaction"].value_counts(dropna=False),
        "reaction_reason": turns_df["reaction_reason"].value_counts(dropna=False),
    }


def dialog_vs_turn_reaction(records: list[dict]) -> pd.DataFrame:
    """Для каждого диалога — реакция верхнего уровня и набор реакций в user_turns.
    Показывает, есть ли расхождения neutral<->negative/positive."""
    rows = []
    for r in records:
        top = (r.get("labels") or {}).get("reaction")
        inner = [(t.get("reaction")) for t in ((r.get("labels") or {}).get("user_turns") or [])]
        rows.append({
            "dialog_id": r["dialog_id"],
            "top_reaction": top,
            "n_negative": inner.count("negative"),
            "n_positive": inner.count("positive"),
            "n_neutral": inner.count("neutral"),
            "n_null": sum(1 for x in inner if x is None),
        })
    return pd.DataFrame(rows)


def dialog_vs_turn_task_type(records: list[dict]) -> pd.DataFrame:
    """Совпадает ли task_type диалога с task_type хотя бы одной user-реплики."""
    rows = []
    for r in records:
        top = (r.get("labels") or {}).get("task_type")
        inner = [(t.get("task_type")) for t in ((r.get("labels") or {}).get("user_turns") or [])]
        rows.append({
            "dialog_id": r["dialog_id"],
            "top_task_type": top,
            "n_matching_turns": sum(1 for x in inner if x == top),
            "n_user_turns": len(inner),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Блок C. Мусор, шаблоны, дубликаты
# ---------------------------------------------------------------------------

TEMPLATE_PATTERNS = {
    "ai_language_model": re.compile(r"as an ai language model", re.I),
    "apologize": re.compile(r"\bi apologi[sz]e\b", re.I),
    "sorry": re.compile(r"\bi'?m sorry\b", re.I),
    "as_of_my_last": re.compile(r"as of my last knowledge", re.I),
    "cannot_browse": re.compile(r"i (do not|don't) have real-?time access", re.I),
}


def template_phrase_counts(records: list[dict]) -> pd.DataFrame:
    """Сколько реплик ассистента содержат шаблонные фразы, и в скольких диалогах."""
    rows = []
    for name, pattern in TEMPLATE_PATTERNS.items():
        turn_hits, dialog_hits = 0, set()
        for r in records:
            hit_here = False
            for t in r["turns"]:
                if t["role"] == "assistant" and pattern.search(t["text"]):
                    turn_hits += 1
                    hit_here = True
            if hit_here:
                dialog_hits.add(r["dialog_id"])
        rows.append({"pattern": name, "n_turns": turn_hits, "n_dialogs": len(dialog_hits)})
    return pd.DataFrame(rows).set_index("pattern")


def empty_or_short_turns(records: list[dict], min_len: int = 3) -> pd.DataFrame:
    """Реплики короче min_len символов (потенциальный мусор)."""
    rows = []
    for r in records:
        for i, t in enumerate(r["turns"]):
            text = t["text"]
            if len(text.strip()) < min_len:
                rows.append({"dialog_id": r["dialog_id"], "turn_idx": i,
                             "role": t["role"], "text": text})
    return pd.DataFrame(rows)


def duplicate_first_user_messages(records: list[dict]) -> pd.DataFrame:
    """Частота нормализованных первых реплик пользователя.
    Нужна, чтобы понять риск утечки между train и test."""
    counter = Counter()
    for r in records:
        first = next((t["text"] for t in r["turns"] if t["role"] == "user"), None)
        if first:
            counter[first.strip().lower()] += 1
    dupes = [(text, n) for text, n in counter.most_common() if n > 1]
    return pd.DataFrame(dupes, columns=["first_user_message", "count"])


# ---------------------------------------------------------------------------
#  Блок D. Связь с reaction
# ---------------------------------------------------------------------------

def reaction_by_length(dialogs_df: pd.DataFrame) -> pd.DataFrame:
    """Медиана/среднее длины диалога по классам реакции."""
    return dialogs_df.groupby("reaction")[["n_turns", "user_chars", "asst_chars", "total_chars"]].agg(["mean", "median", "count"])


def reaction_by_template(records: list[dict]) -> pd.DataFrame:
    """Доля диалогов с шаблонными фразами ассистента по классам реакции."""
    rows = []
    for r in records:
        top = (r.get("labels") or {}).get("reaction")
        if top is None:
            continue
        for name, pattern in TEMPLATE_PATTERNS.items():
            hit = any(t["role"] == "assistant" and pattern.search(t["text"]) for t in r["turns"])
            rows.append({"dialog_id": r["dialog_id"], "reaction": top, "pattern": name, "hit": hit})
    df = pd.DataFrame(rows)
    return df.groupby(["reaction", "pattern"])["hit"].mean().unstack()


def reaction_reason_by_position(turns_df: pd.DataFrame) -> pd.DataFrame:
    """Как причины недовольства распределены по позиции реплики."""
    sub = turns_df[turns_df["reaction_reason"].notna()]
    return sub.groupby(["position", "reaction_reason"]).size().unstack(fill_value=0)