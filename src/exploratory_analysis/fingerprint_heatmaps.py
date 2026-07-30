"""Heatmap tables and publication-oriented figures for Stage 7B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src.exploratory_analysis.pca_analysis import FEATURE_COLUMNS, scale_feature_frame


MAX_HEATMAP_LABELS = 40


def create_heatmap_tables(
    consensus: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
    scaling: str = "zscore",
    distance: str = "euclidean",
) -> dict[str, pd.DataFrame]:
    """Create deterministic heatmap source tables."""

    feature_columns = [feature for feature in feature_columns if feature in consensus.columns]
    if consensus.empty or not feature_columns:
        return {
            "consensus_fingerprint_heatmap_table": pd.DataFrame(),
            "strain_chemical_heatmap_table": pd.DataFrame(),
            "chemical_similarity_heatmap_table": pd.DataFrame(),
            "concentration_response_heatmap_table": pd.DataFrame(),
            "feature_loading_heatmap_table": pd.DataFrame(),
        }
    scaled = consensus.copy(deep=True)
    scaled_features, _, _ = scale_feature_frame(scaled.loc[:, feature_columns], method=scaling)
    for feature in feature_columns:
        scaled[feature] = scaled_features[feature].astype(float).to_numpy()

    chemical_feature = scaled.pivot_table(
        index="Chemical",
        values=feature_columns,
        aggfunc="median",
    ).sort_index()

    work = scaled.copy()
    work["Fingerprint_Magnitude"] = np.linalg.norm(work.loc[:, feature_columns].to_numpy(dtype=float), axis=1)
    strain_chemical = work.pivot_table(
        index="Strain",
        columns="Chemical",
        values="Fingerprint_Magnitude",
        aggfunc="median",
    ).sort_index(axis=0).sort_index(axis=1)

    chemical_vectors = scaled.pivot_table(
        index="Chemical",
        values=feature_columns,
        aggfunc="median",
    ).sort_index()
    chemical_similarity = _distance_table(chemical_vectors, distance=distance)

    concentration = scaled.copy()
    concentration["Concentration_Numeric"] = concentration["Concentration"].map(_parse_numeric)
    concentration["Trajectory_Label"] = (
        concentration["Strain"].astype(str)
        + " | "
        + concentration["Chemical"].astype(str)
        + " | "
        + concentration["Concentration"].astype(str)
    )
    concentration = concentration.sort_values(
        ["Strain", "Chemical", "Concentration_Numeric", "Concentration"],
        na_position="last",
    )
    concentration_response = concentration.set_index("Trajectory_Label").loc[:, feature_columns]

    loading_table = (
        pca_loadings.set_index("feature")
        if not pca_loadings.empty and "feature" in pca_loadings.columns
        else pd.DataFrame()
    )
    return {
        "consensus_fingerprint_heatmap_table": chemical_feature,
        "strain_chemical_heatmap_table": strain_chemical,
        "chemical_similarity_heatmap_table": chemical_similarity,
        "concentration_response_heatmap_table": concentration_response,
        "feature_loading_heatmap_table": loading_table,
    }


def write_exploratory_figures(result: Any, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write all Stage 7B figures as deterministic PNG/PDF artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    created.extend(_write_pca_figures(result, target, overwrite=overwrite))
    figure_specs = [
        ("consensus_fingerprint_heatmap", "consensus_fingerprint_heatmap_table", "Chemical x Feature Consensus Fingerprint", "Feature", "Chemical"),
        ("strain_chemical_heatmap", "strain_chemical_heatmap_table", "Strain x Chemical Fingerprint Magnitude", "Chemical", "Strain"),
        ("chemical_similarity_heatmap", "chemical_similarity_heatmap_table", "Chemical Similarity Distance", "Chemical", "Chemical"),
        ("concentration_trajectory_plot", "concentration_response_heatmap_table", "Concentration-Response Fingerprint Heatmap", "Feature", "Strain | Chemical | Concentration"),
        ("feature_loading_heatmap", "feature_loading_heatmap_table", "PCA Feature Loading Heatmap", "Principal Component", "Feature"),
    ]
    for filename, table_key, title, xlabel, ylabel in figure_specs:
        table = result.heatmap_tables.get(table_key, pd.DataFrame())
        if table.empty:
            continue
        created.extend(
            _write_heatmap(
                table,
                target / filename,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                overwrite=overwrite,
            )
        )

    linkage = result.clustering_results.get("linkage_information", pd.DataFrame())
    assignments = result.clustering_results.get("cluster_assignments", pd.DataFrame())
    if not linkage.empty:
        created.extend(_write_dendrogram(linkage, assignments, target, overwrite=overwrite))
    return created


def _write_pca_figures(result: Any, target: Path, *, overwrite: bool) -> list[Path]:
    scores = result.pca_scores
    if scores.empty or not {"PC1", "PC2"} <= set(scores.columns):
        return []
    created: list[Path] = []
    created.extend(
        _write_scatter(
            scores,
            target / "pca_pc1_pc2",
            x="PC1",
            y="PC2",
            title="Consensus Fingerprint PCA: PC1 vs PC2",
            overwrite=overwrite,
        )
    )
    if "PC3" in scores.columns:
        created.extend(
            _write_scatter(
                scores,
                target / "pca_pc1_pc3",
                x="PC1",
                y="PC3",
                title="Consensus Fingerprint PCA: PC1 vs PC3",
                overwrite=overwrite,
                formats=("png",),
            )
        )
    return created


def _write_scatter(
    scores: pd.DataFrame,
    stem: Path,
    *,
    x: str,
    y: str,
    title: str,
    overwrite: bool,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[Path]:
    paths = [stem.with_suffix(f".{suffix}") for suffix in formats]
    for path in paths:
        _ensure_can_write(path, overwrite=overwrite)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    groups = scores["Chemical"].astype(str) if "Chemical" in scores.columns else pd.Series("fingerprint", index=scores.index)
    unique_groups = groups.unique().tolist()
    for group_value in unique_groups[:20]:
        mask = groups.eq(group_value)
        ax.scatter(scores.loc[mask, x], scores.loc[mask, y], s=20, alpha=0.75, label=group_value)
    if len(unique_groups) <= 12:
        ax.legend(loc="best", fontsize=7, frameon=True)
    ax.set_xlabel(f"{x} score")
    ax.set_ylabel(f"{y} score")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def _write_heatmap(
    table: pd.DataFrame,
    stem: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    overwrite: bool,
) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        _ensure_can_write(path, overwrite=overwrite)
    display = _limited_table(table)
    height = min(max(4.5, 0.24 * max(len(display), 1)), 14)
    width = min(max(7.0, 0.24 * max(len(display.columns), 1)), 16)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    values = display.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(display.columns)), display.columns.astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(display.index)), display.index.astype(str), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Scaled value / distance")
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def _write_dendrogram(
    linkage_table: pd.DataFrame,
    assignments: pd.DataFrame,
    target: Path,
    *,
    overwrite: bool,
) -> list[Path]:
    from scipy.cluster.hierarchy import dendrogram

    path = target / "hierarchical_dendrogram.png"
    _ensure_can_write(path, overwrite=overwrite)
    linkage_matrix = linkage_table.loc[:, ["cluster_a", "cluster_b", "distance", "sample_count"]].to_numpy(dtype=float)
    labels = assignments["Consensus_ID"].astype(str).tolist() if "Consensus_ID" in assignments.columns else None
    if labels and len(labels) > MAX_HEATMAP_LABELS:
        labels = None
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    dendrogram(linkage_matrix, labels=labels, no_labels=labels is None, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Consensus fingerprint")
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _distance_table(vectors: pd.DataFrame, *, distance: str) -> pd.DataFrame:
    if len(vectors) < 2:
        return pd.DataFrame(index=vectors.index, columns=vectors.index, data=0.0)
    metric = str(distance).strip().casefold()
    if metric not in {"euclidean", "cosine", "correlation"}:
        metric = "euclidean"
    square = squareform(pdist(vectors.to_numpy(dtype=float), metric=metric))
    return pd.DataFrame(square, index=vectors.index, columns=vectors.index)


def _limited_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if len(result) > MAX_HEATMAP_LABELS:
        result = result.head(MAX_HEATMAP_LABELS)
    if len(result.columns) > MAX_HEATMAP_LABELS:
        result = result.iloc[:, :MAX_HEATMAP_LABELS]
    return result


def _parse_numeric(value: object) -> float:
    try:
        import re

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
        return float(match.group(0)) if match else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Figure already exists: {path}")
