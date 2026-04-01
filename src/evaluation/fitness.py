"""Fitness and model evaluation functions."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


def penalized_accuracy(y_true, y_pred, n_features: int, penalty: float = 0.01) -> float:
    """Return accuracy penalized by selected feature count."""
    return float(accuracy_score(y_true, y_pred) / (1 + penalty * n_features))


def penalized_f1_micro(
    y_true,
    y_pred,
    n_features: int,
    penalty: float = 1.0,
) -> float:
    """Return micro F1 penalized by selected feature count."""
    return float(f1_score(y_true, y_pred, average="micro") / (1 + penalty * n_features))


def score_with_model(
    x_train,
    x_test,
    y_train,
    y_test,
    model_name: str = "svm",
    use_f1: bool = False,
    penalty: float = 0.01,
) -> float:
    """Train a classifier and return penalized score."""
    model = SVC() if model_name == "svm" else RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    n_features = int(x_train.shape[1])

    if use_f1:
        return penalized_f1_micro(y_test, predictions, n_features=n_features, penalty=1.0)
    return penalized_accuracy(y_test, predictions, n_features=n_features, penalty=penalty)


def pure_accuracy(x_train, x_test, y_train, y_test, model_name: str = "svm") -> float:
    """Train a classifier and return non-penalized accuracy."""
    model = SVC() if model_name == "svm" else RandomForestClassifier(n_estimators=200)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return float(accuracy_score(y_test, predictions))
