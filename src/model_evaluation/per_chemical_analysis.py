"""Per-chemical LOEO performance analysis for selected strain panels."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.model_evaluation.loeo_validation import (
    _experiment_values,
    _split_by_experiment,
    _train_classifier,
)
from src.model_training.models import predict_classifier


PER_CHEMICAL_COLUMNS: tuple[str, ...] = (
    "chemical",
    "precision",
    "recall",
    "f1",
    "support",
)


def run_per_chemical_loeo(
    feature_df: pd.DataFrame,
    strains: Sequence[str],
) -> pd.DataFrame:
    """Run LOEO classification and return per-chemical held-out metrics."""

    predictions_df = _aggregate_loeo_predictions(feature_df, strains)
    labels = sorted(predictions_df["y_true"].unique().tolist(), key=str)
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions_df["y_true"],
        predictions_df["y_pred"],
        labels=labels,
        zero_division=0,
    )
    return (
        pd.DataFrame(
            {
                "chemical": labels,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            },
            columns=PER_CHEMICAL_COLUMNS,
        )
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )


def generate_normalized_confusion_matrix(
    feature_df: pd.DataFrame,
    strains: Sequence[str],
    output_path: str | Path = Path("outputs") / "figures" / "normalized_confusion_matrix.png",
) -> Path:
    """Generate a row-normalized LOEO confusion-matrix figure."""

    predictions_df = _aggregate_loeo_predictions(feature_df, strains)
    labels = sorted(predictions_df["y_true"].unique().tolist(), key=str)
    matrix = confusion_matrix(
        predictions_df["y_true"],
        predictions_df["y_pred"],
        labels=labels,
        normalize="true",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted chemical")
    ax.set_ylabel("True chemical")
    ax.set_title("LOEO Normalized Confusion Matrix")
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _aggregate_loeo_predictions(
    feature_df: pd.DataFrame,
    strains: Sequence[str],
) -> pd.DataFrame:
    _validate_input(feature_df)
    panel_df = feature_df.loc[feature_df["strain"].isin(strains)].reset_index(drop=True)
    if panel_df.empty:
        raise ValueError("No rows found for selected strains.")
    if panel_df["experiment"].nunique() < 2:
        raise ValueError("LOEO requires at least two experiments.")
    if panel_df["chemical"].nunique() < 2:
        raise ValueError("Classification requires at least two chemicals.")

    prediction_rows = []
    for held_out_experiment in _experiment_values(panel_df):
        train_df, test_df = _split_by_experiment(panel_df, held_out_experiment)
        model, feature_columns = _train_classifier(train_df, feature_columns=None)
        predictions = predict_classifier(model, test_df, feature_columns)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "experiment": held_out_experiment,
                    "y_true": test_df["chemical"].to_numpy(),
                    "y_pred": predictions,
                }
            )
        )

    return pd.concat(prediction_rows, ignore_index=True)


def _validate_input(feature_df: pd.DataFrame) -> None:
    required_columns = {"strain", "chemical", "experiment"}
    missing_columns = sorted(required_columns - set(feature_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")
