"""Strain panel optimization using LOEO classification validation."""

from __future__ import annotations

from math import nan
from typing import Any, Sequence

import pandas as pd

from src.model_evaluation.loeo_validation import run_loeo_classification


CANDIDATE_PANELS: dict[str, tuple[str, ...]] = {
    "Panel_A": ("BL027",),
    "Panel_B": ("BL027", "BL011"),
    "Panel_C": ("BL027", "BL011", "BL030"),
    "Panel_D": ("BL027", "BL011", "BL030", "BL029"),
    "Panel_E": ("BL027", "BL011", "BL030", "BL029", "BL032"),
}

METRIC_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "panel_name",
    "strains",
    *METRIC_COLUMNS,
    "sample_count",
    "status",
    "error",
)


def evaluate_strain_panel(
    feature_df: pd.DataFrame,
    strains: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a selected strain panel with LOEO classification."""

    _validate_panel_input(feature_df)
    selected_strains = tuple(strains)
    panel_df = feature_df.loc[feature_df["strain"].isin(selected_strains)].reset_index(drop=True)
    result = {
        "panel_name": _panel_name(selected_strains),
        "strains": _format_strains(selected_strains),
        "sample_count": int(len(panel_df)),
        "status": "success",
        "error": "",
    }

    if panel_df.empty:
        return _failed_result(result, "No rows found for selected strains.")
    if panel_df["experiment"].nunique() < 2:
        return _failed_result(result, "LOEO requires at least two experiments.")
    if panel_df["chemical"].nunique() < 2:
        return _failed_result(result, "Classification requires at least two chemicals.")

    try:
        loeo_result = run_loeo_classification(panel_df)
    except ValueError as error:
        return _failed_result(result, str(error))

    result.update(loeo_result["mean_metrics"])
    return result


def run_candidate_panels(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate predefined strain panels and return results sorted by macro F1."""

    _validate_panel_input(feature_df)
    rows = []
    for panel_name, strains in CANDIDATE_PANELS.items():
        row = evaluate_strain_panel(feature_df, strains)
        row["panel_name"] = panel_name
        rows.append(row)

    all_strains = tuple(sorted(feature_df["strain"].dropna().unique().tolist(), key=str))
    all_strain_row = evaluate_strain_panel(feature_df, all_strains)
    all_strain_row["panel_name"] = "Panel_F"
    rows.append(all_strain_row)

    return (
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        .sort_values("macro_f1", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _validate_panel_input(feature_df: pd.DataFrame) -> None:
    required_columns = {"strain", "chemical", "experiment"}
    missing_columns = sorted(required_columns - set(feature_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")


def _failed_result(result: dict[str, Any], error: str) -> dict[str, Any]:
    result.update({metric: nan for metric in METRIC_COLUMNS})
    result["status"] = "failed"
    result["error"] = error
    return result


def _panel_name(strains: Sequence[str]) -> str:
    return " + ".join(str(strain) for strain in strains)


def _format_strains(strains: Sequence[str]) -> str:
    return ", ".join(str(strain) for strain in strains)
