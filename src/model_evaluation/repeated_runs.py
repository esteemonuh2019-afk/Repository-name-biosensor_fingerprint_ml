"""Repeated random-seed robustness analysis for classification metrics."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.model_evaluation.evaluate import evaluate_classification
from src.model_training.models import NUMERIC_FEATURE_COLUMNS


DEFAULT_SEEDS: tuple[int, ...] = (1, 7, 11, 21, 42, 101, 123, 202, 555, 999)
REPEATED_RUN_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
RUN_COLUMNS: tuple[str, ...] = ("seed", *REPEATED_RUN_METRICS)
SUMMARY_COLUMNS: tuple[str, ...] = ("metric", "mean", "std", "min", "max")


def run_repeated_seed_evaluation(
    feature_df: pd.DataFrame,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_size: float = 0.2,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Train and evaluate a classifier across multiple random seeds."""

    resolved_feature_columns = _validate_repeated_run_input(
        feature_df,
        seeds,
        test_size,
        feature_columns,
    )

    rows = []
    for seed in seeds:
        train_df, test_df = _split_train_test(feature_df, int(seed), test_size)
        model = RandomForestClassifier(n_estimators=100, random_state=int(seed))
        model.fit(train_df[resolved_feature_columns], train_df["chemical"])

        predictions = model.predict(test_df[resolved_feature_columns])
        metrics = evaluate_classification(test_df["chemical"], predictions)
        rows.append(
            {
                "seed": int(seed),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["macro_precision"]),
                "recall": float(metrics["macro_recall"]),
                "f1": float(metrics["macro_f1"]),
            }
        )

    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def summarize_repeated_run_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated-run metric stability with mean, std, min, and max."""

    _validate_metrics_dataframe(metrics_df)
    rows = []
    for metric in REPEATED_RUN_METRICS:
        values = pd.to_numeric(metrics_df[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def create_repeated_run_boxplot(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create a boxplot showing repeated-run metric variability."""

    _validate_metrics_dataframe(metrics_df)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = [metrics_df[metric].to_numpy(dtype=float) for metric in REPEATED_RUN_METRICS]
    ax.boxplot(values, showmeans=True)
    ax.set_xticks(range(1, len(REPEATED_RUN_METRICS) + 1), REPEATED_RUN_METRICS)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Repeated-Run Classification Robustness")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)

    return destination


def _validate_repeated_run_input(
    feature_df: pd.DataFrame,
    seeds: Sequence[int],
    test_size: float,
    feature_columns: Sequence[str] | None,
) -> list[str]:
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if "chemical" not in feature_df.columns:
        raise ValueError("Missing required label column: chemical")
    if feature_df["chemical"].nunique() < 2:
        raise ValueError("Classification requires at least two chemicals.")

    resolved_feature_columns = (
        list(NUMERIC_FEATURE_COLUMNS) if feature_columns is None else list(feature_columns)
    )
    missing_columns = [
        column for column in resolved_feature_columns if column not in feature_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")

    return resolved_feature_columns


def _split_train_test(
    feature_df: pd.DataFrame,
    seed: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = _stratification_labels(feature_df, test_size)
    train_df, test_df = train_test_split(
        feature_df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _stratification_labels(feature_df: pd.DataFrame, test_size: float):
    class_counts = feature_df["chemical"].value_counts()
    if class_counts.min() < 2:
        return None

    number_of_classes = len(class_counts)
    test_rows = ceil(len(feature_df) * test_size)
    train_rows = len(feature_df) - test_rows
    if test_rows < number_of_classes or train_rows < number_of_classes:
        return None

    return feature_df["chemical"]


def _validate_metrics_dataframe(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        raise ValueError("Repeated-run metrics dataframe must not be empty.")
    missing_columns = [
        column for column in REPEATED_RUN_METRICS if column not in metrics_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing repeated-run metric columns: {', '.join(missing_columns)}")
