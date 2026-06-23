"""Baseline model training helpers for biosensor feature tables."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "auc",
    "max_signal",
    "min_signal",
    "time_to_peak",
    "initial_slope",
    "final_signal",
)

RANDOM_SEED = 42


def train_classifier(feature_df: pd.DataFrame) -> tuple[RandomForestClassifier, list[str]]:
    """Train a Random Forest classifier to predict chemical identity."""

    feature_columns = _validate_feature_columns(feature_df)
    if "chemical" not in feature_df.columns:
        raise ValueError("Missing required label column: chemical")

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(feature_df[feature_columns], feature_df["chemical"])
    return model, feature_columns


def train_regressor(feature_df: pd.DataFrame) -> tuple[RandomForestRegressor, list[str]]:
    """Train a Random Forest regressor to predict numeric concentration."""

    feature_columns = _validate_feature_columns(feature_df)
    if "concentration" not in feature_df.columns:
        raise ValueError("Missing required target column: concentration")

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(feature_df[feature_columns], pd.to_numeric(feature_df["concentration"]))
    return model, feature_columns


def predict_classifier(
    model: RandomForestClassifier,
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
):
    """Predict chemical identity labels from a trained classifier."""

    _validate_prediction_columns(feature_df, feature_columns)
    return model.predict(feature_df[list(feature_columns)])


def predict_regressor(
    model: RandomForestRegressor,
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
):
    """Predict numeric concentration values from a trained regressor."""

    _validate_prediction_columns(feature_df, feature_columns)
    return model.predict(feature_df[list(feature_columns)])


def _validate_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    feature_columns = list(NUMERIC_FEATURE_COLUMNS)
    _validate_prediction_columns(feature_df, feature_columns)
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")
    return feature_columns


def _validate_prediction_columns(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    missing_columns = [column for column in feature_columns if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")
