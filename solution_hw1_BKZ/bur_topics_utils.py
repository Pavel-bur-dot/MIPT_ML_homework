from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
import solution_utils as su


# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

DIALOG_FLAGS_PATH = Path("dialog_flags.csv")

FIRST_TURN_MAX_CHARS = 4000
NEAR_DUP_THRESHOLD = 0.85
TEMPLATE_FREQ_THRESHOLD = 1

SEED = 42
TEST_SIZE = 0.2


_STOPWORDS = {
    "english": set(stopwords.words("english")),
    "russian": set(stopwords.words("russian")),
}
_STEMMERS = {
    "english": su.STEMMERS["english"],
    "russian": su.STEMMERS["russian"],
}
_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")

# ---------------------------------------------------------------------------
# Чтение данных
# ---------------------------------------------------------------------------

def load_dialogs_and_flags(
    data_path: str | None = None,
    flags_path: Path = DIALOG_FLAGS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Возвращает (dialogs, messages, flags).
    Если dialog_flags.csv отсутствует, признаки считаются на месте.
    """
    if data_path is None:
        candidates = [os.environ.get("HW1_TRAIN", ""), "hw1_train.jsonl", "dataset/hw1_train.jsonl"]
        data_path = next((p for p in candidates if p and os.path.exists(p)), None)
        assert data_path is not None, f"не нашли hw1_train.jsonl, искали: {candidates}"

    dialogs_raw = su.read_dialogs(data_path)
    dialogs, messages = su.build_frames(dialogs_raw)

    if flags_path.exists():
        flags = pd.read_csv(flags_path)
    else:
        flags = build_flags(dialogs, messages)
        flags.to_csv(flags_path, index=False)

    return dialogs, messages, flags


# ---------------------------------------------------------------------------
# Fallback: пересборка флагов
# ---------------------------------------------------------------------------

def _has_code(text: str) -> bool:
    if not isinstance(text, str):
        return False
    markers = ("```", "def ", "class ", "import ", "function ", "console.log",
               "<html", "SELECT ", "SELECT*", "```python")
    return any(m in text for m in markers)


def _has_cyrillic(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[а-яА-ЯёЁ]", text))


def _has_loop(text: str) -> bool:
    if not isinstance(text, str) or len(text) < 100:
        return False
    head = text[:80]
    return text.count(head) >= 3


def build_flags(dialogs: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    """Минимальная пересборка флагов, если dialog_flags.csv недоступен."""
    messages = messages.copy()
    messages["n_chars"] = messages["text"].str.len()
    messages["is_user"] = messages["role"] == "user"
    messages["is_assistant"] = messages["role"] == "assistant"

    user_msgs = messages[messages["is_user"]]
    asst_msgs = messages[messages["is_assistant"]]
    first_user = user_msgs.groupby("dialog_id")["text"].first()

    agg = messages.groupby("dialog_id").agg(
        n_user_actual=("is_user", "sum"),
        n_chars_total=("n_chars", "sum"),
    )

    first_user_freq = first_user.map(first_user.value_counts()).fillna(1).astype(int)
    empty_user = user_msgs.assign(empty=user_msgs["text"].str.strip().eq("")).groupby("dialog_id")["empty"].any()
    code_user = user_msgs.assign(code=user_msgs["text"].map(_has_code)).groupby("dialog_id")["code"].any()
    code_asst = asst_msgs.assign(code=asst_msgs["text"].map(_has_code)).groupby("dialog_id")["code"].any()
    loop_asst = asst_msgs.assign(loop=asst_msgs["text"].map(_has_loop)).groupby("dialog_id")["loop"].any()
    cyrillic_user = user_msgs.assign(cyr=user_msgs["text"].map(_has_cyrillic)).groupby("dialog_id")["cyr"].any()

    template = (first_user_freq > TEMPLATE_FREQ_THRESHOLD) | (
        first_user.map(lambda t: bool(t) and len(t.split()) <= 2)
    )

    flags = dialogs[["dialog_id", "language", "n_label_turns", "reaction", "task_type"]].copy()
    flags["n_user_actual"] = flags["dialog_id"].map(agg["n_user_actual"])
    flags["n_chars_total"] = flags["dialog_id"].map(agg["n_chars_total"])
    flags["first_turn_freq"] = flags["dialog_id"].map(first_user_freq).fillna(1).astype(int)
    flags["unlabeled"] = flags["reaction"].isna() | flags["task_type"].isna()
    flags["has_empty_message"] = flags["dialog_id"].map(empty_user).fillna(False)
    flags["has_user_code"] = flags["dialog_id"].map(code_user).fillna(False)
    flags["has_assistant_code"] = flags["dialog_id"].map(code_asst).fillna(False)
    flags["has_code"] = flags["has_user_code"] | flags["has_assistant_code"]
    flags["has_assistant_loop"] = flags["dialog_id"].map(loop_asst).fillna(False)
    flags["user_has_cyrillic"] = flags["dialog_id"].map(cyrillic_user).fillna(False)
    flags["is_template"] = flags["dialog_id"].map(template).fillna(False)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=2, max_df=0.5)
    first_texts = flags["dialog_id"].map(first_user).fillna("").tolist()
    X = vec.fit_transform(first_texts)
    nn_idx, nn_sim = su.nearest_neighbor(X, chunk=2000)
    group = su.make_groups(nn_idx, nn_sim, threshold=NEAR_DUP_THRESHOLD)
    flags["group_id"] = group
    flags["in_near_dup_group"] = pd.Series(group).duplicated(keep=False).to_numpy()

    return flags


# ---------------------------------------------------------------------------
# Воронка фильтров и маска
# ---------------------------------------------------------------------------

def filter_funnel(flags: pd.DataFrame) -> pd.DataFrame:
    """
    Воронка фильтров для тем.
    ВАЖНО: `unlabeled` не применяется — тема от разметки не зависит.
    Шаг `~unlabeled` включён как справка и помечен.
    """
    total = len(flags)
    current = np.ones(total, dtype=bool)

    steps = [
        ("всего", np.ones(total, dtype=bool), False),
        ("~is_template", (~flags["is_template"]).to_numpy(), False),
        ("~has_empty_message", (~flags["has_empty_message"]).to_numpy(), False),
        ("~in_near_dup_group", (~flags["in_near_dup_group"]).to_numpy(), False),
        ("use_topics", (flags["use_topics"]).to_numpy(),False)
    ]

    rows = []
    for name, mask, is_reference in steps:
        if is_reference:
            rows.append({
                "шаг": name,
                "осталось": int((current & mask).sum()),
                "доля": float((current & mask).mean()),
                "применяется": False,
            })
        else:
            current = current & mask
            rows.append({
                "шаг": name,
                "осталось": int(current.sum()),
                "доля": float(current.mean()),
                "применяется": True,
            })
    return pd.DataFrame(rows)


def build_train_mask(flags: pd.DataFrame) -> np.ndarray:
    """
    Маска для обучения тем.
    
    Отсекаем только явный мусор:
    - шаблонные первые реплики,
    - пустые сообщения,
    - диалоги в near-dup группах (чтобы почти-копии не размазывались по train/test).
    """
    mask = (
        (~flags["is_template"]).to_numpy()
        & (~flags["has_empty_message"]).to_numpy()
        & (~flags["in_near_dup_group"]).to_numpy()
        & (flags["use_topics"]).to_numpy()
    )
    return mask


# ---------------------------------------------------------------------------
# Первая реплика пользователя и чистка
# ---------------------------------------------------------------------------

def extract_first_user_text(
    messages: pd.DataFrame,
    max_chars: int = FIRST_TURN_MAX_CHARS,
) -> pd.Series:
    """Первая реплика пользователя на диалог, обрезанная до max_chars символов."""
    user_msgs = messages[messages["role"] == "user"]
    first = user_msgs.groupby("dialog_id")["text"].first().fillna("")
    return first.str.slice(0, max_chars)

def detect_language(text: str) -> str:
    return "russian" if _CYRILLIC.search(text) else "english"

def preprocess_for_topics(text: str) -> str:
    """Автоопределение языка + стоп-слова + стемминг."""
    if not isinstance(text, str):
        return ""
    language = detect_language(text)
    stop = _STOPWORDS[language]
    stemmer = _STEMMERS[language]
    tokens = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop and len(t) > 1]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Разбиение train/test по near-dup группам
# ---------------------------------------------------------------------------

def group_split(
    flags: pd.DataFrame,
    mask: np.ndarray,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupShuffleSplit по group_id. Возвращает (train_ids, test_ids)."""
    from sklearn.model_selection import GroupShuffleSplit

    subset = flags.loc[mask].copy()
    groups = subset["group_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(subset, groups=groups))
    train_ids = subset.iloc[train_idx]["dialog_id"].to_numpy()
    test_ids = subset.iloc[test_idx]["dialog_id"].to_numpy()
    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Сохранение артефактов
# ---------------------------------------------------------------------------

def save_stage1_artifacts(
    flags: pd.DataFrame,
    first_user_text: pd.Series,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> None:
    flags.to_csv(ARTIFACTS_DIR / "topics_flags.csv", index=False)
    first_user_text.to_frame("first_user_text").to_csv(
        ARTIFACTS_DIR / "first_user_text.csv", index=False
    )
    pd.Series(train_ids, name="dialog_id").to_csv(
        ARTIFACTS_DIR / "topics_train_ids.csv", index=False
    )
    pd.Series(test_ids, name="dialog_id").to_csv(
        ARTIFACTS_DIR / "topics_test_ids.csv", index=False
    )