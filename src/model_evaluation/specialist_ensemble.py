"""Specialist-strain ensemble classification with LOEO validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.feature_engineering.advanced_features import ADVANCED_FEATURE_COLUMNS
from src.model_evaluation.evaluate import evaluate_classification
from src.model_evaluation.loeo_validation import (
    CLASSIFICATION_METRICS,
    _experiment_values,
    _select_metrics,
    _split_by_experiment,
    _train_classifier,
)
from src.model_training.models import NUMERIC_FEATURE_COLUMNS


DEFAULT_CONFUSION_MATRIX_PATH = (
    Path("outputs") / "figures" / "specialist_ensemble_confusion_matrix.png"
)

CASE_COLUMNS: tuple[str, ...] = (
    "experiment",
    "chemical",
    "concentration",
    "replicate",
)

FEATURE_CANDIDATES: tuple[str, ...] = (
    *NUMERIC_FEATURE_COLUMNS,
    *ADVANCED_FEATURE_COLUMNS,
)


def get_specialist_mapping() -> dict[str, str]:
    """Return the contaminant-to-specialist-strain mapping."""

    return {
        "Boric Acid": "BL027",
        "DEET": "BL030",
        "Diazinon": "BL029",
        "Metaldehyde": "BL027",
        "Propoxur": "BL027",
        "Trimethoprim": "BL032",
    }


def run_specialist_ensemble_loeo(feature_df: pd.DataFrame) -> dict[str, Any]:
    """Run LOEO specialist one-vs-rest classification and aggregate multiclass predictions."""

    _validate_input(feature_df)
    mapping = get_specialist_mapping()
    feature_columns = _feature_columns(feature_df)
    model_df = _coerce_model_input(feature_df, feature_columns)

    score_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for target_chemical, specialist_strain in mapping.items():
        strain_df = model_df.loc[model_df["strain"] == specialist_strain].reset_index(drop=True)
        if strain_df.empty or strain_df["experiment"].nunique() < 2:
            continue

        _score_target_chemical(
            score_rows=score_rows,
            strain_df=strain_df,
            target_chemical=target_chemical,
            feature_columns=feature_columns,
        )

    predictions = _predictions_from_scores(score_rows)
    if not predictions:
        raise ValueError("Specialist ensemble did not generate any held-out predictions.")

    y_true = [row["y_true"] for row in predictions]
    y_pred = [row["y_pred"] for row in predictions]
    metrics = _select_metrics(evaluate_classification(y_true, y_pred), CLASSIFICATION_METRICS)

    return {
        "task_type": "specialist_ensemble_classification",
        "specialist_mapping": mapping,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "prediction_count": len(predictions),
        "predictions": predictions,
    }


def generate_specialist_confusion_matrix(
    loeo_result: dict[str, Any] | None = None,
    output_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
) -> Path:
    """Generate a normalized confusion-matrix figure for specialist ensemble predictions."""

    if loeo_result is None:
        feature_path = Path("outputs") / "tables" / "features_advanced.csv"
        if not feature_path.exists():
            raise FileNotFoundError(f"Advanced feature table not found: {feature_path}")
        loeo_result = run_specialist_ensemble_loeo(pd.read_csv(feature_path))

    predictions = loeo_result.get("predictions", [])
    if not predictions:
        raise ValueError("LOEO result does not contain predictions for confusion matrix generation.")

    labels = list(get_specialist_mapping().keys())
    y_true = [row["y_true"] for row in predictions]
    y_pred = [row["y_pred"] for row in predictions]
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted chemical")
    ax.set_ylabel("True chemical")
    ax.set_title("Specialist Ensemble LOEO Confusion Matrix")
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


def _score_target_chemical(
    score_rows: dict[tuple[Any, ...], dict[str, Any]],
    strain_df: pd.DataFrame,
    target_chemical: str,
    feature_columns: Sequence[str],
) -> None:
    for held_out_experiment in _experiment_values(strain_df):
        train_df, test_df = _split_by_experiment(strain_df, held_out_experiment)
        binary_train_df = _binary_label_dataframe(train_df, target_chemical)
        if binary_train_df["chemical"].nunique() < 2:
            continue

        model, resolved_feature_columns = _train_classifier(
            binary_train_df,
            feature_columns=feature_columns,
        )
        target_probability = _target_probabilities(model, test_df, resolved_feature_columns, target_chemical)

        for row_index, (_, row) in enumerate(test_df.iterrows()):
            key = _case_key(row)
            score_row = score_rows.setdefault(
                key,
                {
                    "experiment": _json_safe_scalar(row["experiment"]),
                    "chemical": _json_safe_scalar(row["chemical"]),
                    "concentration": _json_safe_scalar(row["concentration"]),
                    "replicate": _json_safe_scalar(row["replicate"]),
                    "y_true": _json_safe_scalar(row["chemical"]),
                    "scores": {},
                },
            )
            score_row["scores"][target_chemical] = float(target_probability[row_index])


def _target_probabilities(model: Any, test_df: pd.DataFrame, feature_columns: Sequence[str], target_chemical: str):
    probabilities = model.predict_proba(test_df[list(feature_columns)])
    class_labels = list(model.classes_)
    if target_chemical not in class_labels:
        return [0.0] * len(test_df)
    target_index = class_labels.index(target_chemical)
    return probabilities[:, target_index]


def _predictions_from_scores(score_rows: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for score_row in score_rows.values():
        scores = score_row["scores"]
        if not scores:
            continue

        predicted_chemical = max(scores.items(), key=lambda item: item[1])[0]
        predictions.append(
            {
                "experiment": score_row["experiment"],
                "chemical": score_row["chemical"],
                "concentration": score_row["concentration"],
                "replicate": score_row["replicate"],
                "y_true": score_row["y_true"],
                "y_pred": predicted_chemical,
                "scores": dict(sorted(scores.items())),
            }
        )
    return predictions


def _binary_label_dataframe(dataframe: pd.DataFrame, target_chemical: str) -> pd.DataFrame:
    binary_df = dataframe.copy()
    binary_df["chemical"] = binary_df["chemical"].where(
        binary_df["chemical"] == target_chemical,
        "Other",
    )
    return binary_df


def _coerce_model_input(feature_df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    model_df = feature_df.copy()
    model_df[list(feature_columns)] = model_df[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    model_df = model_df.replace([float("inf"), float("-inf")], pd.NA)
    return model_df.dropna(subset=[*feature_columns, *CASE_COLUMNS, "strain"]).reset_index(drop=True)


def _feature_columns(feature_df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in FEATURE_CANDIDATES if column in feature_df.columns]
    if not feature_columns:
        raise ValueError("No supported original or advanced feature columns were found.")
    return feature_columns


def _case_key(row: pd.Series) -> tuple[Any, ...]:
    return tuple(_json_safe_scalar(row[column]) for column in CASE_COLUMNS)


def _validate_input(feature_df: pd.DataFrame) -> None:
    required_columns = {"strain", *CASE_COLUMNS}
    missing_columns = sorted(required_columns - set(feature_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if feature_df.empty:
        raise ValueError("Feature dataframe must not be empty.")


def _json_safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
