"""LOEO panel comparison using original and advanced biosensor features."""

from __future__ import annotations

from math import nan
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.feature_engineering.advanced_features import ADVANCED_FEATURE_COLUMNS
from src.feature_engineering.features import GROUP_COLUMNS
from src.model_evaluation.loeo_validation import run_loeo_classification
from src.model_training.models import NUMERIC_FEATURE_COLUMNS


ADVANCED_PANEL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Panel_A": ("BL027",),
    "Panel_B": ("BL027", "BL011"),
    "Panel_C": ("BL027", "BL011", "BL030"),
}

ADVANCED_LOEO_FEATURE_COLUMNS: tuple[str, ...] = (
    *NUMERIC_FEATURE_COLUMNS,
    *ADVANCED_FEATURE_COLUMNS,
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "panel_name",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "sample_count",
)


def combine_original_and_advanced_features(
    original_feature_df: pd.DataFrame,
    advanced_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge original feature columns into the advanced feature table."""

    _validate_columns(original_feature_df, (*GROUP_COLUMNS, *NUMERIC_FEATURE_COLUMNS), "original")
    _validate_columns(advanced_feature_df, (*GROUP_COLUMNS, *ADVANCED_FEATURE_COLUMNS), "advanced")

    if all(column in advanced_feature_df.columns for column in NUMERIC_FEATURE_COLUMNS):
        return advanced_feature_df.copy()

    merged = pd.merge(
        original_feature_df[list((*GROUP_COLUMNS, *NUMERIC_FEATURE_COLUMNS))],
        advanced_feature_df,
        on=list(GROUP_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Original and advanced feature tables did not share any matching rows.")
    return merged


def get_advanced_loeo_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return all original and advanced numeric feature columns required for comparison."""

    _validate_columns(feature_df, ADVANCED_LOEO_FEATURE_COLUMNS, "combined")
    return list(ADVANCED_LOEO_FEATURE_COLUMNS)


def run_advanced_panel_comparison(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the requested strain panels with advanced LOEO classification."""

    _validate_columns(feature_df, ("strain", "chemical", "experiment"), "feature")
    feature_columns = get_advanced_loeo_feature_columns(feature_df)
    rows = []

    for panel_name, strains in ADVANCED_PANEL_CANDIDATES.items():
        rows.append(_evaluate_panel(feature_df, panel_name, strains, feature_columns))

    return (
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        .sort_values("macro_f1", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def plot_advanced_panel_macro_f1(panel_results: pd.DataFrame, output_path: str | Path) -> None:
    """Generate a macro-F1 bar plot for advanced panel comparison."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = panel_results.dropna(subset=["macro_f1"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(plot_df["panel_name"], plot_df["macro_f1"])
    ax.set_xlabel("Panel")
    ax.set_ylabel("Mean LOEO Macro F1")
    ax.set_title("Advanced Feature Panel Macro F1")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _evaluate_panel(
    feature_df: pd.DataFrame,
    panel_name: str,
    strains: Sequence[str],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    panel_df = feature_df.loc[feature_df["strain"].isin(strains)].reset_index(drop=True)
    panel_df = _coerce_model_input(panel_df, feature_columns)
    result: dict[str, Any] = {
        "panel_name": panel_name,
        "sample_count": int(len(panel_df)),
    }

    if panel_df.empty or panel_df["experiment"].nunique() < 2 or panel_df["chemical"].nunique() < 2:
        result.update({metric: nan for metric in OUTPUT_COLUMNS if metric not in {"panel_name", "sample_count"}})
        return result

    try:
        loeo_result = run_loeo_classification(panel_df, feature_columns=feature_columns)
    except ValueError:
        result.update({metric: nan for metric in OUTPUT_COLUMNS if metric not in {"panel_name", "sample_count"}})
        return result

    result.update(loeo_result["mean_metrics"])
    return result


def _coerce_model_input(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    model_df = feature_df.copy()
    model_df[list(feature_columns)] = model_df[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    model_df = model_df.replace([float("inf"), float("-inf")], pd.NA)
    return model_df.dropna(subset=[*feature_columns, "chemical", "experiment"]).reset_index(drop=True)


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing {table_name} feature columns: {', '.join(missing_columns)}")
