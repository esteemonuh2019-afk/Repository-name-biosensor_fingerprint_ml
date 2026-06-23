"""Feature-importance and experiment-effect analysis utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.model_training.models import NUMERIC_FEATURE_COLUMNS, RANDOM_SEED


def calculate_random_forest_feature_importance(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Train a Random Forest classifier and return sorted feature importances."""

    _validate_columns(feature_df, (*NUMERIC_FEATURE_COLUMNS, "chemical"))
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(feature_df[list(NUMERIC_FEATURE_COLUMNS)], feature_df["chemical"])

    importance_df = pd.DataFrame(
        {
            "feature": list(NUMERIC_FEATURE_COLUMNS),
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Generate and save a feature-importance bar plot."""

    _validate_columns(feature_importance_df, ("feature", "importance"))
    destination = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(feature_importance_df["feature"], feature_importance_df["importance"])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def generate_pca_by_chemical(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Generate a PCA plot colored by chemical identity."""

    _validate_columns(feature_df, (*NUMERIC_FEATURE_COLUMNS, "chemical"))
    return _generate_pca_plot(feature_df, "chemical", output_path, "PCA by Chemical")


def generate_pca_by_experiment(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Generate a PCA plot colored by experiment ID."""

    _validate_columns(feature_df, (*NUMERIC_FEATURE_COLUMNS, "experiment"))
    return _generate_pca_plot(feature_df, "experiment", output_path, "PCA by Experiment")


def _generate_pca_plot(
    feature_df: pd.DataFrame,
    color_column: str,
    output_path: str | Path,
    title: str,
) -> Path:
    destination = _prepare_output_path(output_path)
    projection = _pca_projection(feature_df[list(NUMERIC_FEATURE_COLUMNS)])
    labels = feature_df[color_column].astype(str)
    categories = list(pd.unique(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    for category in categories:
        mask = labels == category
        ax.scatter(
            projection.loc[mask, "PC1"],
            projection.loc[mask, "PC2"],
            s=18,
            alpha=0.75,
            label=category,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize="small", markerscale=1.3)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _pca_projection(feature_values: pd.DataFrame) -> pd.DataFrame:
    if len(feature_values) < 2:
        raise ValueError("At least two rows are required for PCA.")

    scaled_values = StandardScaler().fit_transform(feature_values)
    projection = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(scaled_values)
    return pd.DataFrame(projection, columns=["PC1", "PC2"], index=feature_values.index)


def _prepare_output_path(output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _validate_columns(dataframe: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if dataframe.empty:
        raise ValueError("Dataframe must not be empty.")
