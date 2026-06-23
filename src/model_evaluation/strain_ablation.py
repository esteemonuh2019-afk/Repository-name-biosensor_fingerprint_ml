"""Strain ablation analysis using LOEO classification validation."""

from __future__ import annotations

from math import nan
from typing import Any

import pandas as pd

from src.model_evaluation.loeo_validation import run_loeo_classification


METRIC_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

SINGLE_STRAIN_COLUMNS: tuple[str, ...] = (
    "strain",
    *METRIC_COLUMNS,
    "number_of_samples",
    "status",
    "error",
)

LEAVE_ONE_STRAIN_OUT_COLUMNS: tuple[str, ...] = (
    "removed_strain",
    *METRIC_COLUMNS,
    "number_of_samples",
    "status",
    "error",
)


def run_single_strain_loeo(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Run LOEO classification independently for each strain."""

    _validate_ablation_input(feature_df)
    rows = []
    for strain in _strain_values(feature_df):
        subset = feature_df.loc[feature_df["strain"] == strain].reset_index(drop=True)
        rows.append(_run_ablation_fold("strain", strain, subset))

    return _sort_results(pd.DataFrame(rows, columns=SINGLE_STRAIN_COLUMNS))


def run_leave_one_strain_out_loeo(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Run LOEO classification after removing each strain."""

    _validate_ablation_input(feature_df)
    rows = []
    for strain in _strain_values(feature_df):
        subset = feature_df.loc[feature_df["strain"] != strain].reset_index(drop=True)
        rows.append(_run_ablation_fold("removed_strain", strain, subset))

    return _sort_results(pd.DataFrame(rows, columns=LEAVE_ONE_STRAIN_OUT_COLUMNS))


def _run_ablation_fold(
    strain_column: str,
    strain: Any,
    subset: pd.DataFrame,
) -> dict[str, Any]:
    row = {
        strain_column: strain,
        "number_of_samples": int(len(subset)),
        "status": "success",
        "error": "",
    }

    if subset["experiment"].nunique() < 2:
        return _failed_row(row, "LOEO requires at least two experiments.")
    if subset["chemical"].nunique() < 2:
        return _failed_row(row, "Classification requires at least two chemicals.")

    try:
        loeo_result = run_loeo_classification(subset)
    except ValueError as error:
        return _failed_row(row, str(error))

    row.update(loeo_result["mean_metrics"])
    return row


def _failed_row(row: dict[str, Any], error: str) -> dict[str, Any]:
    row.update({metric: nan for metric in METRIC_COLUMNS})
    row["status"] = "failed"
    row["error"] = error
    return row


def _sort_results(result_df: pd.DataFrame) -> pd.DataFrame:
    return result_df.sort_values(
        by="macro_f1",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def _validate_ablation_input(feature_df: pd.DataFrame) -> None:
    required_columns = {"strain", "chemical", "experiment"}
    missing_columns = sorted(required_columns - set(feature_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")


def _strain_values(feature_df: pd.DataFrame) -> list[Any]:
    return sorted(feature_df["strain"].dropna().unique().tolist(), key=str)
