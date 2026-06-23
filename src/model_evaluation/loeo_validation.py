"""Leave-one-experiment-out validation for biosensor feature tables."""

from __future__ import annotations

from math import isfinite
from typing import Any, Sequence

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.model_evaluation.evaluate import evaluate_classification, evaluate_regression
from src.model_training.models import (
    RANDOM_SEED,
    predict_classifier,
    predict_regressor,
    train_classifier,
    train_regressor,
)


CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

REGRESSION_METRICS: tuple[str, ...] = (
    "r2",
    "rmse",
    "mae",
)


def run_loeo_classification(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run leave-one-experiment-out validation for chemical classification."""

    _validate_loeo_input(feature_df, "chemical", feature_columns)
    fold_results = []

    for held_out_experiment in _experiment_values(feature_df):
        train_df, test_df = _split_by_experiment(feature_df, held_out_experiment)
        model, resolved_feature_columns = _train_classifier(train_df, feature_columns)
        predictions = predict_classifier(model, test_df, resolved_feature_columns)
        metrics = evaluate_classification(test_df["chemical"], predictions)
        filtered_metrics = _select_metrics(metrics, CLASSIFICATION_METRICS)
        fold_results.append(
            _fold_result(
                held_out_experiment=held_out_experiment,
                train_df=train_df,
                test_df=test_df,
                metrics=filtered_metrics,
            )
        )

    return {
        "task_type": "classification",
        "per_experiment": fold_results,
        "mean_metrics": _mean_metrics(fold_results, CLASSIFICATION_METRICS),
    }


def run_loeo_regression(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run leave-one-experiment-out validation for concentration regression."""

    _validate_loeo_input(feature_df, "concentration", feature_columns)
    fold_results = []

    for held_out_experiment in _experiment_values(feature_df):
        train_df, test_df = _split_by_experiment(feature_df, held_out_experiment)
        model, resolved_feature_columns = _train_regressor(train_df, feature_columns)
        predictions = predict_regressor(model, test_df, resolved_feature_columns)
        metrics = evaluate_regression(test_df["concentration"], predictions)
        filtered_metrics = _select_metrics(metrics, REGRESSION_METRICS)
        fold_results.append(
            _fold_result(
                held_out_experiment=held_out_experiment,
                train_df=train_df,
                test_df=test_df,
                metrics=filtered_metrics,
            )
        )

    return {
        "task_type": "regression",
        "per_experiment": fold_results,
        "mean_metrics": _mean_metrics(fold_results, REGRESSION_METRICS),
    }


def _validate_loeo_input(
    feature_df: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
) -> None:
    if "experiment" not in feature_df.columns:
        raise ValueError("Feature dataframe must include an experiment column.")
    if target_column not in feature_df.columns:
        raise ValueError(f"Feature dataframe must include a {target_column} column.")
    if feature_df["experiment"].nunique() < 2:
        raise ValueError("LOEO validation requires at least two experiments.")
    if feature_columns is not None:
        missing_columns = [column for column in feature_columns if column not in feature_df.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")


def _train_classifier(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str] | None,
):
    if feature_columns is None:
        return train_classifier(train_df)

    resolved_feature_columns = list(feature_columns)
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(train_df[resolved_feature_columns], train_df["chemical"])
    return model, resolved_feature_columns


def _train_regressor(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str] | None,
):
    if feature_columns is None:
        return train_regressor(train_df)

    resolved_feature_columns = list(feature_columns)
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(train_df[resolved_feature_columns], pd.to_numeric(train_df["concentration"]))
    return model, resolved_feature_columns


def _experiment_values(feature_df: pd.DataFrame) -> list[Any]:
    return sorted(feature_df["experiment"].dropna().unique().tolist(), key=str)


def _split_by_experiment(
    feature_df: pd.DataFrame,
    held_out_experiment: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = feature_df["experiment"] == held_out_experiment
    train_df = feature_df.loc[~test_mask].reset_index(drop=True)
    test_df = feature_df.loc[test_mask].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("Each LOEO fold must contain training and test rows.")
    return train_df, test_df


def _select_metrics(metrics: dict[str, Any], metric_names: tuple[str, ...]) -> dict[str, float]:
    return {metric_name: float(metrics[metric_name]) for metric_name in metric_names}


def _fold_result(
    held_out_experiment: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict[str, float],
) -> dict[str, Any]:
    train_experiments = sorted(train_df["experiment"].dropna().unique().tolist(), key=str)
    return {
        "held_out_experiment": _json_safe_scalar(held_out_experiment),
        "train_experiments": [_json_safe_scalar(experiment) for experiment in train_experiments],
        "test_experiments": [_json_safe_scalar(held_out_experiment)],
        "test_rows": int(len(test_df)),
        "metrics": metrics,
    }


def _mean_metrics(
    fold_results: list[dict[str, Any]],
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    mean_metrics = {}
    for metric_name in metric_names:
        values = [
            fold_result["metrics"][metric_name]
            for fold_result in fold_results
            if isfinite(fold_result["metrics"][metric_name])
        ]
        mean_metrics[metric_name] = sum(values) / len(values) if values else float("nan")
    return mean_metrics


def _json_safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
