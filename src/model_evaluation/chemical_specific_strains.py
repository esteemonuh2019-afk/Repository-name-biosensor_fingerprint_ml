"""Chemical-specific strain ranking with LOEO binary classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.feature_engineering.advanced_features import ADVANCED_FEATURE_COLUMNS
from src.model_evaluation.loeo_validation import (
    _experiment_values,
    _split_by_experiment,
    _train_classifier,
)
from src.model_training.models import NUMERIC_FEATURE_COLUMNS, predict_classifier


CHEMICALS: tuple[str, ...] = (
    "Boric Acid",
    "DEET",
    "Diazinon",
    "Metaldehyde",
    "Propoxur",
    "Trimethoprim",
)

STRAINS: tuple[str, ...] = (
    "BL011",
    "BL027",
    "BL029",
    "BL030",
    "BL031",
    "BL032",
)

RANKING_COLUMNS: tuple[str, ...] = (
    "chemical",
    "strain",
    "precision",
    "recall",
    "f1",
    "support",
)

FEATURE_CANDIDATES: tuple[str, ...] = (
    *NUMERIC_FEATURE_COLUMNS,
    *ADVANCED_FEATURE_COLUMNS,
)


def evaluate_strain_for_chemical(
    feature_df: pd.DataFrame,
    strain: str,
    target_chemical: str,
) -> dict[str, Any]:
    """Evaluate one strain as target-chemical-vs-other using LOEO validation."""

    _validate_input(feature_df)
    feature_columns = _feature_columns(feature_df)
    strain_df = feature_df.loc[feature_df["strain"] == strain].reset_index(drop=True)
    strain_df = _prepare_binary_input(strain_df, feature_columns, target_chemical)

    result: dict[str, Any] = {
        "chemical": target_chemical,
        "strain": strain,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": int((strain_df["chemical"] == target_chemical).sum()) if not strain_df.empty else 0,
    }

    if (
        strain_df.empty
        or strain_df["experiment"].nunique() < 2
        or strain_df["chemical"].nunique() < 2
        or result["support"] == 0
    ):
        return result

    predictions_df = _aggregate_binary_loeo_predictions(strain_df, feature_columns)
    if predictions_df.empty:
        result["support"] = 0
        return result

    precision, recall, f1, support = precision_recall_fscore_support(
        predictions_df["y_true"],
        predictions_df["y_pred"],
        labels=[target_chemical],
        zero_division=0,
    )
    result.update(
        {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        }
    )
    return result


def rank_strains_per_chemical(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Rank fixed biosensor strains for each contaminant by binary LOEO F1."""

    rows = [
        evaluate_strain_for_chemical(feature_df, strain, chemical)
        for chemical in CHEMICALS
        for strain in STRAINS
    ]
    ranking = pd.DataFrame(rows, columns=RANKING_COLUMNS)
    ranking["chemical"] = pd.Categorical(ranking["chemical"], categories=CHEMICALS, ordered=True)
    ranking["strain"] = pd.Categorical(ranking["strain"], categories=STRAINS, ordered=True)
    ranking = ranking.sort_values(
        ["chemical", "f1", "recall", "precision", "strain"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    ranking["chemical"] = ranking["chemical"].astype(str)
    ranking["strain"] = ranking["strain"].astype(str)
    return ranking


def plot_chemical_specific_strain_heatmap(
    ranking_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a chemical-by-strain F1 heatmap."""

    missing_columns = [column for column in ("chemical", "strain", "f1") if column not in ranking_df.columns]
    if missing_columns:
        raise ValueError(f"Missing ranking columns: {', '.join(missing_columns)}")

    heatmap_data = (
        ranking_df.pivot_table(
            index="chemical",
            columns="strain",
            values="f1",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(index=CHEMICALS, columns=STRAINS)
        .fillna(0.0)
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap_data.to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(STRAINS)), STRAINS, rotation=45, ha="right")
    ax.set_yticks(range(len(CHEMICALS)), CHEMICALS)
    ax.set_xlabel("Strain")
    ax.set_ylabel("Chemical")
    ax.set_title("Chemical-Specific Strain F1")
    fig.colorbar(image, ax=ax, label="LOEO F1")

    for row_index, chemical in enumerate(CHEMICALS):
        for column_index, strain in enumerate(STRAINS):
            value = heatmap_data.loc[chemical, strain]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _aggregate_binary_loeo_predictions(
    binary_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    prediction_rows = []
    for held_out_experiment in _experiment_values(binary_df):
        train_df, test_df = _split_by_experiment(binary_df, held_out_experiment)
        if train_df["chemical"].nunique() < 2:
            continue

        model, resolved_feature_columns = _train_classifier(train_df, feature_columns=feature_columns)
        predictions = predict_classifier(model, test_df, resolved_feature_columns)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "experiment": held_out_experiment,
                    "y_true": test_df["chemical"].to_numpy(),
                    "y_pred": predictions,
                }
            )
        )

    if not prediction_rows:
        return pd.DataFrame(columns=["experiment", "y_true", "y_pred"])
    return pd.concat(prediction_rows, ignore_index=True)


def _prepare_binary_input(
    strain_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_chemical: str,
) -> pd.DataFrame:
    model_df = strain_df.copy()
    model_df[list(feature_columns)] = model_df[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    model_df = model_df.replace([float("inf"), float("-inf")], pd.NA)
    model_df = model_df.dropna(subset=[*feature_columns, "chemical", "experiment"]).reset_index(drop=True)
    if model_df.empty:
        return model_df

    model_df["chemical"] = model_df["chemical"].where(
        model_df["chemical"] == target_chemical,
        "Other",
    )
    return model_df


def _feature_columns(feature_df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in FEATURE_CANDIDATES if column in feature_df.columns]
    if not feature_columns:
        raise ValueError("No supported original or advanced feature columns were found.")
    return feature_columns


def _validate_input(feature_df: pd.DataFrame) -> None:
    required_columns = {"strain", "chemical", "experiment"}
    missing_columns = sorted(required_columns - set(feature_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")
