# HW2 — командная заготовка (деревья и ансамбли)
# Переделаем под отчет
Всё устроено как в HW1: флаги `dialog_flags.csv`, диалоги с `use_reaction == 1` (2930),
разбиение `GroupShuffleSplit` по `group_id`, кросс-валидация `StratifiedGroupKFold`, метрики из таблицы HW1.

## Запуск
1. Положите папку `hw2` рядом с `hw1` (или поправьте `HW1_ARTIFACTS` в `00_setup_old_models.ipynb`).
   `hw1_train.jsonl` ищется в `hw1_train.jsonl`, `dataset/`, `hw1_data/`, `../hw1/dataset/` или в `HW2_TRAIN`.
2. `00_setup_old_models.ipynb`: переносит модели HW1 в `artifacts/` как `hw1_*.joblib`, строит общие таблицы
   `results/cv_table.csv` и `results/test_table.csv`. Проверьте сверку: train=2344, test=586, macro-F1 `tfidf_logreg`≈0.593.
3. Новая модель: копия `01_random_forest.ipynb` с первой ячейкой подготовки. Имя `hw2_<имя>`,
   сохранение `su.save_model`, строки в таблицы `su.save_result`.
   Модель на деревьях принимает таблицу `su.make_frame(...)` (первый шаг Pipeline должен называться `features`).
   Нужен другой вход — правьте `uses_frame` и `make_input` в `solution_utils.py`.
4. `solution.ipynb` сам подхватит все `hw1_*` и `hw2_*` из `artifacts/`.
