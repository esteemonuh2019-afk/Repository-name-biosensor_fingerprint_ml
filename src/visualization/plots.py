"""Visualization helpers for biosensor feature and raw data outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "auc",
    "max_signal",
    "min_signal",
    "time_to_peak",
    "initial_slope",
    "final_signal",
)


def plot_heatmap(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot mean AUC values as a chemical-by-strain heatmap."""

    _validate_columns(feature_df, ("chemical", "strain", "auc"))
    destination = _prepare_output_path(output_path)

    heatmap_data = feature_df.pivot_table(
        index="chemical",
        columns="strain",
        values="auc",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(heatmap_data.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)), heatmap_data.index)
    ax.set_xlabel("Strain")
    ax.set_ylabel("Chemical")
    ax.set_title("AUC Heatmap")
    fig.colorbar(image, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_pca(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot a two-dimensional PCA projection from numeric feature columns."""

    destination = _prepare_output_path(output_path)
    numeric_columns = [
        column
        for column in NUMERIC_FEATURE_COLUMNS
        if column in feature_df.columns and pd.api.types.is_numeric_dtype(feature_df[column])
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns are available for PCA plotting.")

    projection = _pca_projection(feature_df[numeric_columns].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(projection[:, 0], projection[:, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_dose_response(feature_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot concentration against AUC values."""

    _validate_columns(feature_df, ("concentration", "auc"))
    destination = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    if "chemical" in feature_df.columns:
        for chemical, group in feature_df.groupby("chemical", sort=False):
            ax.plot(group["concentration"], group["auc"], marker="o", label=str(chemical))
        ax.legend()
    else:
        ax.plot(feature_df["concentration"], feature_df["auc"], marker="o")

    ax.set_xlabel("Concentration")
    ax.set_ylabel("AUC")
    ax.set_title("Dose Response")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_time_course(raw_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot luminescence over time."""

    _validate_columns(raw_df, ("time", "luminescence"))
    destination = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    group_columns = [
        column for column in ("strain", "chemical", "concentration", "replicate") if column in raw_df.columns
    ]
    if group_columns:
        for group_values, group in raw_df.groupby(group_columns, sort=False):
            label = " / ".join(str(value) for value in _as_tuple(group_values))
            sorted_group = group.sort_values("time")
            ax.plot(sorted_group["time"], sorted_group["luminescence"], marker="o", label=label)
        ax.legend(fontsize="small")
    else:
        sorted_data = raw_df.sort_values("time")
        ax.plot(sorted_data["time"], sorted_data["luminescence"], marker="o")

    ax.set_xlabel("Time")
    ax.set_ylabel("Luminescence")
    ax.set_title("Time Course")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


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


def _pca_projection(values: np.ndarray) -> np.ndarray:
    centered_values = values - values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation == 0] = 1
    scaled_values = centered_values / standard_deviation

    if scaled_values.shape[0] == 1:
        return np.zeros((1, 2))

    _, _, vt = np.linalg.svd(scaled_values, full_matrices=False)
    projection = scaled_values @ vt[: min(2, vt.shape[0])].T
    if projection.shape[1] == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(projection.shape[0])])
    return projection[:, :2]


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)
