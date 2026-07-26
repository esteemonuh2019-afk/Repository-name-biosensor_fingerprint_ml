"""Bootstrap confidence intervals for model performance metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_STATE = 42
SUMMARY_COLUMNS: tuple[str, ...] = (
    "metric",
    "mean",
    "ci_lower",
    "ci_upper",
    "confidence",
    "n_bootstrap",
)


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray | pd.Series,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, float | int]:
    """Estimate a bootstrap confidence interval for the sample mean."""

    numeric_values = _coerce_numeric_values(values)
    _validate_bootstrap_parameters(confidence, n_bootstrap)

    rng = np.random.default_rng(random_state)
    bootstrap_samples = rng.choice(
        numeric_values,
        size=(n_bootstrap, numeric_values.size),
        replace=True,
    )
    bootstrap_means = bootstrap_samples.mean(axis=1)

    mean = float(numeric_values.mean())
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": mean,
        "ci_lower": min(ci_lower, mean),
        "ci_upper": max(ci_upper, mean),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
    }


def summarize_metric_confidence_intervals(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute bootstrap confidence intervals for numeric metric columns."""

    rows = []
    for column in metrics_df.columns:
        if not is_numeric_dtype(metrics_df[column]):
            continue

        interval = bootstrap_confidence_interval(metrics_df[column].to_numpy())
        rows.append({"metric": column, **interval})

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _coerce_numeric_values(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    try:
        numeric_values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be numeric.") from exc

    if numeric_values.ndim == 0:
        numeric_values = numeric_values.reshape(1)

    numeric_values = numeric_values.ravel()
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        raise ValueError("values must contain at least one finite numeric value.")

    return numeric_values


def _validate_bootstrap_parameters(confidence: float, n_bootstrap: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
