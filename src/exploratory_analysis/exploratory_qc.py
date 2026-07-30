"""Input validation and non-plot exploratory summaries for Stage 7B."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src.exploratory_analysis.pca_analysis import FEATURE_COLUMNS, scale_feature_frame
from src.fingerprint.fingerprint_similarity import (
    correlation_distance,
    cosine_distance,
    euclidean_distance,
)


def validate_exploratory_inputs(
    individual: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate fingerprint inputs for exploratory analysis."""

    warnings: list[str] = []
    errors: list[str] = []
    feature_columns = [feature for feature in feature_columns if feature in consensus.columns]
    if consensus.empty:
        errors.append("Consensus fingerprint dataset is empty.")
    if individual.empty:
        warnings.append("Individual fingerprint dataset is empty; replicate summaries are unavailable.")
    if not feature_columns:
        errors.append("No fingerprint feature columns are available.")

    nonfinite_consensus = _nonfinite_row_count(consensus, feature_columns)
    nonfinite_individual = _nonfinite_row_count(individual, [f for f in feature_columns if f in individual.columns])
    if nonfinite_consensus:
        warnings.append(f"Consensus fingerprints contain {nonfinite_consensus} non-finite feature rows.")
    if nonfinite_individual:
        warnings.append(f"Individual fingerprints contain {nonfinite_individual} non-finite feature rows.")

    required_metadata = ["Strain", "Chemical", "Concentration"]
    missing_metadata = [column for column in required_metadata if column not in consensus.columns]
    if missing_metadata:
        errors.append("Missing consensus metadata columns: " + ", ".join(missing_metadata) + ".")

    metadata = {
        "individual_fingerprint_count": int(len(individual)),
        "consensus_fingerprint_count": int(len(consensus)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
        "excluded_for_analysis_count": int(nonfinite_consensus),
    }
    return warnings, errors, metadata


def enrich_consensus_metadata(consensus: pd.DataFrame, individual: pd.DataFrame) -> pd.DataFrame:
    """Add duration, experiment, and source-type summaries to consensus fingerprints."""

    result = consensus.copy(deep=True)
    group_columns = [column for column in ("Strain", "Chemical", "Concentration") if column in result.columns]
    if individual.empty or not group_columns:
        return result

    work = individual.copy(deep=True)
    if "Source_Type" not in work.columns:
        work["Source_Type"] = work.get("Source_File", pd.Series(dtype=object)).map(_source_type_from_file)
    aggregations: dict[str, tuple[str, Any]] = {}
    if "Duration" in work.columns:
        aggregations["Median_Duration"] = ("Duration", lambda values: _median_numeric(values))
    if "Experiment_ID" in work.columns:
        aggregations["Experiment_Count"] = ("Experiment_ID", lambda values: int(values.dropna().astype(str).nunique()))
        aggregations["Experiment_IDs"] = ("Experiment_ID", lambda values: _joined_unique(values))
    if "Source_File" in work.columns:
        aggregations["Source_File_Count"] = ("Source_File", lambda values: int(values.dropna().astype(str).nunique()))
    if "Source_Type" in work.columns:
        aggregations["Source_Type"] = ("Source_Type", lambda values: _joined_unique(values))

    if not aggregations:
        return result
    summary = work.groupby(group_columns, dropna=False, sort=True).agg(**aggregations).reset_index()
    return result.merge(summary, on=group_columns, how="left")


def calculate_replicate_to_consensus_distances(
    individual: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
    distance: str = "euclidean",
    scaling: str = "zscore",
) -> pd.DataFrame:
    """Measure each individual fingerprint's distance to its consensus fingerprint."""

    group_columns = [column for column in ("Strain", "Chemical", "Concentration") if column in individual.columns and column in consensus.columns]
    feature_columns = [feature for feature in feature_columns if feature in individual.columns and feature in consensus.columns]
    if individual.empty or consensus.empty or not group_columns or not feature_columns:
        return pd.DataFrame()

    combined = pd.concat(
        [
            individual.loc[:, [*group_columns, *feature_columns]].assign(_source="individual"),
            consensus.loc[:, [*group_columns, *feature_columns]].assign(_source="consensus"),
        ],
        ignore_index=True,
    )
    finite = combined.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(finite.to_numpy(dtype=float)).all(axis=1)
    combined = combined.loc[finite_mask].reset_index(drop=True)
    scaled, _, _ = scale_feature_frame(combined.loc[:, feature_columns], method=scaling)
    _assign_scaled_features(combined, scaled, feature_columns)
    scaled_individual = combined.loc[combined["_source"].eq("individual")].drop(columns=["_source"]).reset_index(drop=True)
    scaled_consensus = combined.loc[combined["_source"].eq("consensus")].drop(columns=["_source"]).reset_index(drop=True)

    consensus_lookup = {
        _group_key(row, group_columns): row
        for _, row in scaled_consensus.iterrows()
    }
    original_individual = individual.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, row in scaled_individual.iterrows():
        key = _group_key(row, group_columns)
        consensus_row = consensus_lookup.get(key)
        if consensus_row is None:
            continue
        distance_value = _distance(
            row.loc[feature_columns].to_numpy(dtype=float),
            consensus_row.loc[feature_columns].to_numpy(dtype=float),
            distance,
        )
        original = original_individual.iloc[index]
        rows.append(
            {
                "Fingerprint_ID": original.get("Fingerprint_ID", pd.NA),
                "Measurement_Unit_ID": original.get("Measurement_Unit_ID", pd.NA),
                "Strain": original.get("Strain", pd.NA),
                "Chemical": original.get("Chemical", pd.NA),
                "Concentration": original.get("Concentration", pd.NA),
                "Replicate_ID": original.get("Replicate_ID", pd.NA),
                "distance_to_consensus": float(distance_value),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["group_replicate_count"] = result.groupby(
        ["Strain", "Chemical", "Concentration"],
        dropna=False,
    )["distance_to_consensus"].transform("size")
    result["insufficient_replicates"] = result["group_replicate_count"] < 2
    result["unusually_distant_replicate"] = result.groupby(
        ["Strain", "Chemical", "Concentration"],
        dropna=False,
    )["distance_to_consensus"].transform(_unusually_distant_mask).astype(bool)
    return result.sort_values(["Strain", "Chemical", "Concentration", "distance_to_consensus"]).reset_index(drop=True)


def calculate_concentration_trajectories(
    consensus: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
    scaling: str = "zscore",
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate adjacent concentration distances within chemical/strain groups."""

    warnings: list[str] = []
    feature_columns = [feature for feature in feature_columns if feature in consensus.columns]
    required = {"Strain", "Chemical", "Concentration"}
    if consensus.empty or not required <= set(consensus.columns) or not feature_columns:
        return pd.DataFrame(), warnings

    work = consensus.copy(deep=True)
    work["Concentration_Numeric"] = work["Concentration"].map(parse_numeric_concentration)
    missing_count = int(work["Concentration_Numeric"].isna().sum())
    if missing_count:
        warnings.append(f"Missing numeric concentration values for {missing_count} consensus fingerprints.")
    work = work.dropna(subset=["Concentration_Numeric"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(), warnings

    scaled, _, scaling_warnings = scale_feature_frame(work.loc[:, feature_columns], method=scaling)
    warnings.extend(scaling_warnings)
    _assign_scaled_features(work, scaled, feature_columns)

    rows: list[dict[str, Any]] = []
    for (strain, chemical), group in work.groupby(["Strain", "Chemical"], dropna=False, sort=True):
        ordered = group.sort_values(["Concentration_Numeric", "Concentration"]).reset_index(drop=True)
        if len(ordered) < 2:
            continue
        norms = np.linalg.norm(ordered.loc[:, feature_columns].to_numpy(dtype=float), axis=1)
        trajectory_label = _monotonic_label(norms)
        for index in range(len(ordered) - 1):
            left = ordered.iloc[index]
            right = ordered.iloc[index + 1]
            rows.append(
                {
                    "Strain": strain,
                    "Chemical": chemical,
                    "from_concentration": left["Concentration"],
                    "to_concentration": right["Concentration"],
                    "from_concentration_numeric": float(left["Concentration_Numeric"]),
                    "to_concentration_numeric": float(right["Concentration_Numeric"]),
                    "adjacent_distance": float(
                        euclidean_distance(
                            left.loc[feature_columns].to_numpy(dtype=float),
                            right.loc[feature_columns].to_numpy(dtype=float),
                        )
                    ),
                    "trajectory_label": trajectory_label,
                }
            )
    return pd.DataFrame(rows), warnings


def calculate_strain_dispersion(
    consensus: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
    scaling: str = "zscore",
) -> pd.DataFrame:
    """Summarize exploratory fingerprint variation by strain."""

    feature_columns = [feature for feature in feature_columns if feature in consensus.columns]
    if consensus.empty or "Strain" not in consensus.columns or not feature_columns:
        return pd.DataFrame()
    work = consensus.copy(deep=True)
    scaled, _, _ = scale_feature_frame(work.loc[:, feature_columns], method=scaling)
    _assign_scaled_features(work, scaled, feature_columns)
    rows: list[dict[str, Any]] = []
    for strain, group in work.groupby("Strain", dropna=False, sort=True):
        values = group.loc[:, feature_columns].to_numpy(dtype=float)
        centroid = values.mean(axis=0)
        distances = np.linalg.norm(values - centroid, axis=1)
        chemical_count = int(group["Chemical"].dropna().astype(str).nunique()) if "Chemical" in group.columns else 0
        rows.append(
            {
                "Strain": strain,
                "consensus_count": int(len(group)),
                "chemical_count": chemical_count,
                "mean_distance_to_strain_centroid": float(np.mean(distances)) if len(distances) else 0.0,
                "median_distance_to_strain_centroid": float(np.median(distances)) if len(distances) else 0.0,
                "mean_feature_standard_deviation": float(group.loc[:, feature_columns].std(ddof=0).mean()),
                "candidate_informative_strain": bool(chemical_count >= 2 and np.mean(distances) > 0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_distance_to_strain_centroid",
        ascending=False,
    ).reset_index(drop=True)


def parse_numeric_concentration(value: Any) -> float | None:
    """Extract a numeric concentration for ordering without changing labels."""

    if pd.isna(value):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def _assign_scaled_features(
    dataframe: pd.DataFrame,
    scaled: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    for feature in feature_columns:
        dataframe[feature] = scaled[feature].astype(float).to_numpy()


def _nonfinite_row_count(dataframe: pd.DataFrame, feature_columns: list[str]) -> int:
    if dataframe.empty or not feature_columns:
        return 0
    values = dataframe.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return int((~np.isfinite(values).all(axis=1)).sum())


def _distance(vector_a: np.ndarray, vector_b: np.ndarray, distance: str) -> float:
    distance = str(distance).strip().casefold()
    if distance == "euclidean":
        return euclidean_distance(vector_a, vector_b)
    if distance == "cosine":
        return cosine_distance(vector_a, vector_b)
    if distance == "correlation":
        return correlation_distance(vector_a, vector_b)
    raise ValueError("Distance must be one of: euclidean, cosine, correlation.")


def _source_type_from_file(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    suffix = str(value).lower().rsplit(".", 1)[-1] if "." in str(value) else ""
    if suffix == "csv":
        return "csv"
    if suffix in {"xlsx", "xls"}:
        return "excel"
    return "unknown"


def _median_numeric(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def _joined_unique(values: pd.Series) -> str:
    return ";".join(sorted(values.dropna().astype(str).unique().tolist()))


def _group_key(row: pd.Series, columns: list[str]) -> tuple[Any, ...]:
    return tuple(row[column] for column in columns)


def _unusually_distant_mask(values: pd.Series) -> pd.Series:
    if len(values) < 3:
        return pd.Series(False, index=values.index)
    median = float(values.median())
    mad = float((values - median).abs().median())
    threshold = median + 3.0 * (mad if mad > 0 else values.std(ddof=0))
    if not math.isfinite(threshold) or threshold == median:
        return pd.Series(False, index=values.index)
    return values > threshold


def _monotonic_label(values: np.ndarray) -> str:
    differences = np.diff(values)
    if np.all(differences >= -1e-12):
        return "monotonic_increasing_fingerprint_norm"
    if np.all(differences <= 1e-12):
        return "monotonic_decreasing_fingerprint_norm"
    return "non_monotonic_fingerprint_norm"
