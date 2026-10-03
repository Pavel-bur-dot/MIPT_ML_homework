from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}

# ---------------------------------------------------------------------------
#  Блок 1. Чтение данных и сохранение основных данных
# ---------------------------------------------------------------------------

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
#  Блок 2. Базовые исследования на длины, доли и вывод распределений
# ---------------------------------------------------------------------------

def language_counts(dialogs_df: pd.DataFrame) -> pd.Series:
    return dialogs_df["language"].value_counts(dropna=False)

def analyze_dialog_lengths_by_turns(dialogs_df: pd.DataFrame) -> dict:
    """
    Анализ длины диалогов по количеству реплик.
    Использует колонки: n_user_replies, n_asst_replies, n_turns.
    """
    stats = {}
    
    # --- Пользователь ---
    user = dialogs_df["n_user_replies"]
    stats["user"] = {
        "min": user.min(),
        "max": user.max(),
        "mean": user.mean(),
        "median": user.median(),
        "std": user.std(),
    }
    
    # --- Ассистент ---
    asst = dialogs_df["n_asst_replies"]
    stats["assistant"] = {
        "min": asst.min(),
        "max": asst.max(),
        "mean": asst.mean(),
        "median": asst.median(),
        "std": asst.std(),
    }
    
    # --- Общее ---
    total = dialogs_df["n_turns"]
    stats["total"] = {
        "min": total.min(),
        "max": total.max(),
        "mean": total.mean(),
        "median": total.median(),
        "std": total.std(),
    }
    
    return stats


def plot_dialog_lengths_by_turns(dialogs_df: pd.DataFrame):
    """
    Три гистограммы: распределение количества реплик
    пользователя, ассистента и общего.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Пользователь
    axes[0].hist(dialogs_df["n_user_replies"], bins=30,
                 color="steelblue", edgecolor="black")
    axes[0].set_title("Реплики пользователя")
    axes[0].set_xlabel("Количество реплик")
    axes[0].set_ylabel("Число диалогов")
    
    # Ассистент
    axes[1].hist(dialogs_df["n_asst_replies"], bins=30,
                 color="coral", edgecolor="black")
    axes[1].set_title("Реплики ассистента")
    axes[1].set_xlabel("Количество реплик")
    axes[1].set_ylabel("Число диалогов")
    
    # Общее
    axes[2].hist(dialogs_df["n_turns"], bins=30,
                 color="seagreen", edgecolor="black")
    axes[2].set_title("Общее число реплик")
    axes[2].set_xlabel("Количество реплик")
    axes[2].set_ylabel("Число диалогов")
    
    plt.tight_layout()
    plt.show()

def analyze_dialog_lengths_by_chars(dialogs_df: pd.DataFrame) -> dict:
    """
    Анализ длины диалогов по количеству символов.
    Использует колонки: user_chars, asst_chars, total_chars.
    """
    stats = {}
    
    for name, col in [("user", "user_chars"),
                      ("assistant", "asst_chars"),
                      ("total", "total_chars")]:
        s = dialogs_df[col]
        stats[name] = {
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "q25": s.quantile(0.25),
            "q50": s.quantile(0.50),
            "q75": s.quantile(0.75),
            "q90": s.quantile(0.90),
        }
    return stats


def plot_dialog_lengths_by_chars(dialogs_df: pd.DataFrame):
    """
    Гистограммы и boxplot'ы для распределения количества символов.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    cols = [
        ("user_chars", "Символы пользователя", "steelblue"),
        ("asst_chars", "Символы ассистента", "coral"),
        ("total_chars", "Общее число символов", "seagreen"),
    ]
    
    # Верхний ряд: гистограммы
    for i, (col, title, color) in enumerate(cols):
        axes[0, i].hist(dialogs_df[col], bins=50,
                        color=color, edgecolor="black")
        axes[0, i].set_title(title)
        axes[0, i].set_xlabel("Количество символов")
        axes[0, i].set_ylabel("Число диалогов")
    
    # Нижний ряд: boxplot'ы
    for i, (col, title, color) in enumerate(cols):
        axes[1, i].boxplot(dialogs_df[col].dropna(), vert=True)
        axes[1, i].set_title(f"Boxplot: {title.lower()}")
        axes[1, i].set_ylabel("Количество символов")
    
    plt.tight_layout()
    plt.show()