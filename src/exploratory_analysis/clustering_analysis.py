"""Hierarchical clustering utilities for Stage 7B."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from src.exploratory_analysis.pca_analysis import FEATURE_COLUMNS, scale_feature_frame


SUPPORTED_DISTANCES = {"euclidean", "cosine", "correlation"}
SUPPORTED_LINKAGES = {"ward", "average", "complete"}


def run_hierarchical_clustering(
    dataframe: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
    distance: str = "euclidean",
    linkage_method: str = "ward",
    scaling: str = "zscore",
    max_clusters: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Cluster consensus fingerprints without assigning biological meaning."""

    distance = _canonical_distance(distance)
    linkage_method = _canonical_linkage(linkage_method)
    if linkage_method == "ward" and distance != "euclidean":
        raise ValueError("Ward linkage is mathematically valid only with Euclidean distance.")

    warnings: list[str] = []
    feature_columns = [feature for feature in feature_columns if feature in dataframe.columns]
    if dataframe.empty or len(dataframe) < 2:
        raise ValueError("Clustering requires at least two consensus fingerprints.")
    if not feature_columns:
        raise ValueError("No fingerprint features are available for clustering.")

    numeric = dataframe.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    excluded_count = int((~finite_mask).sum())
    if excluded_count:
        warnings.append(f"Excluded {excluded_count} non-finite rows from clustering.")
    working = dataframe.loc[finite_mask].reset_index(drop=True)
    numeric = numeric.loc[finite_mask].reset_index(drop=True)
    if len(working) < 2:
        raise ValueError("Clustering requires at least two finite consensus fingerprints.")

    scaled, _, scaling_warnings = scale_feature_frame(numeric, method=scaling)
    warnings.extend(scaling_warnings)
    values = scaled.to_numpy(dtype=float)

    if linkage_method == "ward":
        linkage_matrix = linkage(values, method="ward", metric="euclidean", optimal_ordering=True)
    else:
        condensed = pdist(values, metric=distance)
        linkage_matrix = linkage(condensed, method=linkage_method, optimal_ordering=True)

    cluster_count = max_clusters or min(6, max(2, int(round(np.sqrt(len(working))))))
    cluster_count = min(cluster_count, len(working))
    assignments = fcluster(linkage_matrix, t=cluster_count, criterion="maxclust")

    assignment_table = _assignment_table(working, assignments)
    linkage_table = _linkage_table(linkage_matrix)
    composition = _cluster_composition(assignment_table)
    distance_matrix = _distance_matrix(working, values, distance)
    distance_summary = _distance_summary(distance_matrix)

    return (
        {
            "cluster_assignments": assignment_table,
            "linkage_information": linkage_table,
            "dendrogram_table": linkage_table.copy(),
            "cluster_composition": composition,
        },
        {
            "consensus_distance_matrix": distance_matrix,
            "distance_summary": distance_summary,
        },
        warnings,
    )


def _assignment_table(dataframe: pd.DataFrame, assignments: np.ndarray) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in (
            "Consensus_ID",
            "Strain",
            "Chemical",
            "Concentration",
            "Replicate_Count",
            "QC_Status",
        )
        if column in dataframe.columns
    ]
    result = dataframe.loc[:, metadata_columns].copy()
    result["cluster_id"] = assignments.astype(int)
    return result.sort_values(["cluster_id", *metadata_columns]).reset_index(drop=True)


def _linkage_table(linkage_matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "merge_step": range(1, len(linkage_matrix) + 1),
            "cluster_a": linkage_matrix[:, 0].astype(int),
            "cluster_b": linkage_matrix[:, 1].astype(int),
            "distance": linkage_matrix[:, 2].astype(float),
            "sample_count": linkage_matrix[:, 3].astype(int),
        }
    )


def _cluster_composition(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cluster_id, group in assignments.groupby("cluster_id", sort=True):
        chemical_counts = group["Chemical"].astype(str).value_counts().to_dict() if "Chemical" in group.columns else {}
        for chemical, count in chemical_counts.items():
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "chemical": chemical,
                    "count": int(count),
                    "cluster_size": int(len(group)),
                    "strains": _joined_unique(group, "Strain"),
                    "concentrations": _joined_unique(group, "Concentration"),
                }
            )
    return pd.DataFrame(rows)


def _distance_matrix(dataframe: pd.DataFrame, values: np.ndarray, distance: str) -> pd.DataFrame:
    labels = dataframe["Consensus_ID"].astype(str).tolist() if "Consensus_ID" in dataframe.columns else [
        str(index) for index in dataframe.index.tolist()
    ]
    condensed = pdist(values, metric=distance)
    square = squareform(condensed)
    return pd.DataFrame(square, index=labels, columns=labels)


def _distance_summary(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    if distance_matrix.empty:
        return pd.DataFrame()
    values = distance_matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices_from(values, k=1)]
    if len(upper) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "pair_count": int(len(upper)),
                "minimum_distance": float(np.min(upper)),
                "median_distance": float(np.median(upper)),
                "mean_distance": float(np.mean(upper)),
                "maximum_distance": float(np.max(upper)),
            }
        ]
    )


def _joined_unique(dataframe: pd.DataFrame, column: str) -> str:
    if column not in dataframe.columns:
        return ""
    return ";".join(sorted(dataframe[column].dropna().astype(str).unique().tolist()))


def _canonical_distance(distance: str) -> str:
    value = str(distance).strip().casefold()
    if value not in SUPPORTED_DISTANCES:
        raise ValueError("Distance must be one of: euclidean, cosine, correlation.")
    return value


def _canonical_linkage(linkage_method: str) -> str:
    value = str(linkage_method).strip().casefold()
    if value not in SUPPORTED_LINKAGES:
        raise ValueError("Linkage must be one of: ward, average, complete.")
    return value
