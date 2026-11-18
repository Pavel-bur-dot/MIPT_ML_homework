"""Вспомогательный код для 02_gradient_boosting.ipynb.

Лежат в отдельном модуле, а не в ноутбуке: модели сохраняются через joblib, и чтобы их можно было загрузить
в solution.ipynb, классы должны импортироваться из файла. Положите этот файл рядом с solution_utils.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ORDER = ["negative", "neutral", "positive"]                    # порядок реакций от плохой к хорошей


class BalancedFit(BaseEstimator, ClassifierMixin):
    """Обучает estimator с весами объектов, уравнивающими классы (аналог class_weight="balanced").

    У GradientBoostingClassifier нет параметра class_weight, а редкий класс positive (~10%) иначе тонет.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


# ---------------------------------------------------------------------------
#  Качество в зависимости от числа деревьев: одно обучение на блок, оценки на каждом этапе
# ---------------------------------------------------------------------------

def _stage_labels(model, X_train, y_train, X_valid, sizes):
    """Метки на проверочном блоке после каждого числа деревьев из sizes (модель учится один раз)."""
    top = max(sizes)
    if isinstance(model, HistGradientBoostingClassifier):
        model.fit(X_train, y_train)
        stages = list(model.staged_predict_proba(X_valid))
        return [model.classes_[stages[min(n, len(stages)) - 1].argmax(axis=1)] for n in sizes]
    if isinstance(model, BalancedFit) and isinstance(model.estimator, GradientBoostingClassifier):
        model.fit(X_train, y_train)
        stages = list(model.estimator_.staged_predict_proba(X_valid))
        return [model.classes_[stages[min(n, len(stages)) - 1].argmax(axis=1)] for n in sizes]
    model.fit(X_train, y_train)                                  # CatBoost: ntree_end — сколько деревьев использовать
    return [model.classes_[model.predict_proba(X_valid, ntree_end=min(n, top)).argmax(axis=1)] for n in sizes]


def staged_macro_f1(pipeline, X, y, folds, sizes):
    """macro-F1 на проверочных блоках для каждого числа деревьев из sizes: массив (блоки, len(sizes)).

    pipeline — Pipeline из шагов features и gb, число деревьев в gb уже равно max(sizes). Признаки учатся
    только на обучающей части блока, как при обычной кросс-валидации.
    """
    scores = np.zeros((len(folds), len(sizes)))
    for k, (train_part, valid_part) in enumerate(folds):
        features = clone(pipeline[:-1])
        model = clone(pipeline[-1])
        X_train = features.fit_transform(X.iloc[train_part], y[train_part])
        X_valid = features.transform(X.iloc[valid_part])
        labels = _stage_labels(model, X_train, y[train_part], X_valid, sizes)
        scores[k] = [f1_score(y[valid_part], lab, labels=ORDER, average="macro", zero_division=0) for lab in labels]
    return scores


