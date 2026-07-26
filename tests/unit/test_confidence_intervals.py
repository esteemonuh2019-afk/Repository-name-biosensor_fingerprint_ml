import pandas as pd

from src.model_evaluation.confidence_intervals import (
    bootstrap_confidence_interval,
    summarize_metric_confidence_intervals,
)


REQUIRED_INTERVAL_KEYS = {
    "mean",
    "ci_lower",
    "ci_upper",
    "confidence",
    "n_bootstrap",
}
REQUIRED_SUMMARY_COLUMNS = {
    "metric",
    "mean",
    "ci_lower",
    "ci_upper",
    "confidence",
    "n_bootstrap",
}


def test_bootstrap_confidence_interval_returns_required_keys() -> None:
    interval = bootstrap_confidence_interval([0.4, 0.6, 0.8], n_bootstrap=200)

    assert REQUIRED_INTERVAL_KEYS <= set(interval)


def test_bootstrap_confidence_interval_bounds_contain_mean() -> None:
    interval = bootstrap_confidence_interval([0.4, 0.6, 0.8], n_bootstrap=200)

    assert interval["ci_lower"] <= interval["mean"]
    assert interval["ci_upper"] >= interval["mean"]


def test_summarize_metric_confidence_intervals_contains_required_columns() -> None:
    metrics_df = pd.DataFrame(
        {
            "chemical": ["Boric Acid", "DEET", "Diazinon"],
            "precision": [0.97, 0.46, 0.28],
            "recall": [1.0, 0.53, 0.25],
            "f1": [0.99, 0.49, 0.26],
        }
    )

    summary = summarize_metric_confidence_intervals(metrics_df)

    assert REQUIRED_SUMMARY_COLUMNS <= set(summary.columns)
    assert set(summary["metric"]) == {"precision", "recall", "f1"}


def test_bootstrap_confidence_interval_handles_small_numeric_arrays() -> None:
    interval = bootstrap_confidence_interval([0.75], n_bootstrap=50)

    assert interval["mean"] == 0.75
    assert interval["ci_lower"] <= interval["mean"]
    assert interval["ci_upper"] >= interval["mean"]
