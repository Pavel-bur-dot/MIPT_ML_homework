"""Проверка, что окружение курса собрано правильно.

    python tools/check_env.py

Скрипт никогда не падает сам: любая беда с пакетом превращается в понятную
строчку отчета, а не в трассировку. Если он упал — это ошибка в самом
скрипте, напишите мне.
"""

from __future__ import annotations

import platform
import sys

CORE = [
    ("numpy", "1.26"),
    ("pandas", "2.1"),
    ("scipy", "1.11"),
    ("sklearn", "1.9"),
    ("matplotlib", "3.8"),
]
LATER = ["seaborn", "data_profiling", "nltk", "pymorphy3", "catboost", "lightgbm", "optuna", "datasketch"]

#: Подсказки для известных поломок, которые не лечатся переустановкой пакета.
HINTS = {
    "libomp": (
        "нужна библиотека OpenMP. На macOS: brew install libomp, "
        "либо conda install -c conda-forge libomp"
    ),
    "GLIBCXX": "не хватает системных библиотек C++: conda install -c conda-forge libstdcxx-ng",
}


def probe(name: str) -> tuple[str | None, str | None]:
    """Вернуть версию пакета и текст проблемы, если она есть.

    Пакет может не установиться, а может установиться и не импортироваться:
    у lightgbm и catboost на macOS так бывает из-за отсутствия OpenMP.
    Второй случай важно отличать от первого, потому что и лечится он иначе.
    """
    try:
        module = __import__(name)
    except ImportError:
        return None, None
    except Exception as exc:                     # noqa: BLE001
        text = str(exc)
        for marker, hint in HINTS.items():
            if marker in text:
                return None, hint
        return None, f"{type(exc).__name__}: {text.splitlines()[0][:110]}"
    return getattr(module, "__version__", "?"), None


def as_tuple(v: str) -> tuple:
    parts = []
    for chunk in v.split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def main() -> int:
    print(f"python {platform.python_version()}  ({sys.executable})")
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        print("  ! нужен python от 3.11 до 3.13")

    problems = 0
    print("\nобязательные пакеты:")
    for name, minimum in CORE:
        got, trouble = probe(name)
        if trouble:
            print(f"  СЛОМАН   {name}: {trouble}")
            problems += 1
        elif got is None:
            print(f"  НЕТ      {name}  (нужен >= {minimum})")
            problems += 1
        elif got != "?" and as_tuple(got) < as_tuple(minimum):
            print(f"  СТАРЫЙ   {name} {got}  (нужен >= {minimum})")
            problems += 1
        else:
            print(f"  ok       {name} {got}")

    broken_later = 0
    print("\nпонадобятся позже по курсу:")
    for name in LATER:
        got, trouble = probe(name)
        if trouble:
            print(f"  сломан   {name}: {trouble}")
            broken_later += 1
        elif got is None:
            print(f"  нет пока {name}")
        else:
            print(f"  ok       {name} {got}")

    print()
    if problems:
        print(f"Проблем с обязательными пакетами: {problems}.")
        print("Установка: pip install -r requirements.txt")
        return 1

    print("Обязательное окружение готово, можно работать.")
    if broken_later:
        print(f"Отложенных пакетов сломано: {broken_later}. "
              "Это не мешает первым занятиям, почините до нужной недели.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
