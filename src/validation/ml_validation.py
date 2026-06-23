"""ML output validation and leakage checks for V&V."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable


CLASSIFIER_ACCURACY_THRESHOLD = 0.80
CLASSIFIER_MACRO_F1_THRESHOLD = 0.75
REGRESSOR_R2_THRESHOLD = 0.75


@dataclass(frozen=True)
class MLValidationResult:
    task_type: str
    passed: bool
    metrics: dict[str, float]
    threshold_results: dict[str, bool]
    messages: list[str]


def validate_classifier(y_true: Iterable[Any], y_pred: Iterable[Any]) -> MLValidationResult:
    """Validate classifier predictions against V&V performance thresholds."""

    true_values, pred_values = _paired_values(y_true, y_pred)
    labels = sorted(set(true_values) | set(pred_values), key=str)

    accuracy = sum(true == pred for true, pred in zip(true_values, pred_values)) / len(true_values)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for label in labels:
        true_positive = sum(
            true == label and pred == label for true, pred in zip(true_values, pred_values)
        )
        false_positive = sum(
            true != label and pred == label for true, pred in zip(true_values, pred_values)
        )
        false_negative = sum(
            true == label and pred != label for true, pred in zip(true_values, pred_values)
        )

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    metrics = {
        "accuracy": accuracy,
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
    }
    threshold_results = {
        "accuracy": metrics["accuracy"] >= CLASSIFIER_ACCURACY_THRESHOLD,
        "macro_f1": metrics["macro_f1"] >= CLASSIFIER_MACRO_F1_THRESHOLD,
    }

    return MLValidationResult(
        task_type="classifier",
        passed=all(threshold_results.values()),
        metrics=metrics,
        threshold_results=threshold_results,
        messages=_threshold_messages(threshold_results),
    )


def validate_regressor(y_true: Iterable[float], y_pred: Iterable[float]) -> MLValidationResult:
    """Validate regressor predictions against V&V performance thresholds."""

    true_values, pred_values = _paired_float_values(y_true, y_pred)
    mean_true = sum(true_values) / len(true_values)
    residual_sum_squares = sum((true - pred) ** 2 for true, pred in zip(true_values, pred_values))
    total_sum_squares = sum((true - mean_true) ** 2 for true in true_values)
    r2 = 1.0 if total_sum_squares == 0 and residual_sum_squares == 0 else 0.0
    if total_sum_squares != 0:
        r2 = 1 - residual_sum_squares / total_sum_squares

    metrics = {
        "r2": r2,
        "rmse": sqrt(residual_sum_squares / len(true_values)),
        "mae": sum(abs(true - pred) for true, pred in zip(true_values, pred_values))
        / len(true_values),
    }
    threshold_results = {"r2": metrics["r2"] >= REGRESSOR_R2_THRESHOLD}

    return MLValidationResult(
        task_type="regressor",
        passed=all(threshold_results.values()),
        metrics=metrics,
        threshold_results=threshold_results,
        messages=_threshold_messages(threshold_results),
    )


def verify_experiment_level_split(
    train_experiments: Iterable[Any],
    test_experiments: Iterable[Any],
) -> MLValidationResult:
    """Verify train and test sets do not share experiment identifiers."""

    overlap = set(train_experiments) & set(test_experiments)
    threshold_results = {"no_overlap": not overlap}
    messages = ["No experiment ID overlap detected."]
    if overlap:
        overlap_ids = ", ".join(str(experiment_id) for experiment_id in sorted(overlap, key=str))
        messages = [f"Experiment ID overlap detected: {overlap_ids}"]

    return MLValidationResult(
        task_type="experiment_split",
        passed=threshold_results["no_overlap"],
        metrics={},
        threshold_results=threshold_results,
        messages=messages,
    )


def _paired_values(y_true: Iterable[Any], y_pred: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length.")
    if not true_values:
        raise ValueError("y_true and y_pred must not be empty.")
    return true_values, pred_values


def _paired_float_values(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> tuple[list[float], list[float]]:
    true_values, pred_values = _paired_values(y_true, y_pred)
    return [float(value) for value in true_values], [float(value) for value in pred_values]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _threshold_messages(threshold_results: dict[str, bool]) -> list[str]:
    return [
        f"{metric_name} threshold {'passed' if passed else 'failed'}"
        for metric_name, passed in threshold_results.items()
    ]
