"""Experiment-normalized feature engineering utilities."""

from __future__ import annotations

import pandas as pd


BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "auc",
    "max_signal",
    "min_signal",
    "time_to_peak",
    "initial_slope",
    "final_signal",
)

EXPERIMENT_ZSCORE_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_zexp" for column in BASE_FEATURE_COLUMNS
)

STRAIN_EXPERIMENT_ZSCORE_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_zstrain_exp" for column in BASE_FEATURE_COLUMNS
)


def add_experiment_zscore_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-experiment z-score normalized feature columns."""

    _validate_columns(feature_df, ("experiment", *BASE_FEATURE_COLUMNS))
    normalized_df = feature_df.copy()
    for source_column, normalized_column in zip(
        BASE_FEATURE_COLUMNS,
        EXPERIMENT_ZSCORE_COLUMNS,
        strict=True,
    ):
        normalized_df[normalized_column] = normalized_df.groupby("experiment")[
            source_column
        ].transform(_zscore_series)
    return normalized_df


def add_strain_experiment_zscore_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-strain, per-experiment z-score normalized feature columns."""

    _validate_columns(feature_df, ("strain", "experiment", *BASE_FEATURE_COLUMNS))
    normalized_df = feature_df.copy()
    for source_column, normalized_column in zip(
        BASE_FEATURE_COLUMNS,
        STRAIN_EXPERIMENT_ZSCORE_COLUMNS,
        strict=True,
    ):
        normalized_df[normalized_column] = normalized_df.groupby(["strain", "experiment"])[
            source_column
        ].transform(_zscore_series)
    return normalized_df


def get_normalized_feature_columns() -> list[str]:
    """Return all experiment-normalized feature columns."""

    return list(EXPERIMENT_ZSCORE_COLUMNS + STRAIN_EXPERIMENT_ZSCORE_COLUMNS)


def _zscore_series(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)
    if standard_deviation == 0 or pd.isna(standard_deviation):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / standard_deviation


def _validate_columns(dataframe: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
