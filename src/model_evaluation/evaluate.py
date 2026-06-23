"""Model evaluation metrics for biosensor classification and regression."""

from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


OVERFIT_THRESHOLD = 0.10


def evaluate_classification(y_true: Iterable[Any], y_pred: Iterable[Any]) -> dict[str, Any]:
    """Calculate classification metrics used by the V&V plan."""

    true_values = list(y_true)
    pred_values = list(y_pred)
    return {
        "accuracy": accuracy_score(true_values, pred_values),
        "macro_precision": precision_score(true_values, pred_values, average="macro", zero_division=0),
        "macro_recall": recall_score(true_values, pred_values, average="macro", zero_division=0),
        "macro_f1": f1_score(true_values, pred_values, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(true_values, pred_values).tolist(),
    }


def evaluate_regression(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Calculate regression metrics used by the V&V plan."""

    true_values = list(y_true)
    pred_values = list(y_pred)
    return {
        "r2": r2_score(true_values, pred_values),
        "rmse": mean_squared_error(true_values, pred_values) ** 0.5,
        "mae": mean_absolute_error(true_values, pred_values),
    }


def compare_train_test_performance(train_metric: float, test_metric: float) -> dict[str, bool]:
    """Flag overfitting when train performance exceeds test performance by more than 0.10."""

    return {"overfit_warning": train_metric - test_metric > OVERFIT_THRESHOLD}
