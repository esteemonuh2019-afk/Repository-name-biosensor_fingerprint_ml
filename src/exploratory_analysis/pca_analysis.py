"""PCA utilities for Stage 7B exploratory fingerprint analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.fingerprint.fingerprint_builder import FINGERPRINT_FEATURE_COLUMNS


FEATURE_COLUMNS: tuple[str, ...] = FINGERPRINT_FEATURE_COLUMNS
PCA_METADATA_COLUMNS: tuple[str, ...] = (
    "Consensus_ID",
    "Fingerprint_ID",
    "Experiment_ID",
    "Measurement_Unit_ID",
    "Source_File",
    "Source_Type",
    "Strain",
    "Chemical",
    "Concentration",
    "Replicate_ID",
    "Duration",
    "Median_Duration",
    "Experiment_Count",
    "Source_File_Count",
    "QC_Status",
)


def run_pca_analysis(
    dataframe: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
    scaling: str = "zscore",
    max_components: int = 3,
    top_n_features: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Run deterministic PCA on finite fingerprint feature rows."""

    feature_columns = [feature for feature in feature_columns if feature in dataframe.columns]
    warnings: list[str] = []
    if dataframe.empty:
        raise ValueError("PCA requires at least two finite fingerprint rows.")
    if not feature_columns:
        raise ValueError("No fingerprint feature columns are available for PCA.")

    original = dataframe.copy(deep=True)
    numeric = original.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    excluded_count = int((~finite_mask).sum())
    if excluded_count:
        warnings.append(f"Excluded {excluded_count} non-finite rows from PCA.")
    working = original.loc[finite_mask].reset_index(drop=True)
    numeric = numeric.loc[finite_mask].reset_index(drop=True)
    if len(working) < 2:
        raise ValueError("PCA requires at least two finite fingerprint rows.")

    scaled, scaling_parameters, scaling_warnings = scale_feature_frame(numeric, method=scaling)
    warnings.extend(scaling_warnings)
    variable_columns = [
        column
        for column in scaled.columns
        if float(pd.to_numeric(scaled[column], errors="coerce").std(ddof=0)) > 0.0
    ]
    constant_columns = [column for column in scaled.columns if column not in variable_columns]
    if constant_columns:
        warnings.append(
            "Constant features excluded from PCA: " + ", ".join(constant_columns) + "."
        )
    if not variable_columns:
        raise ValueError("PCA requires at least one non-constant finite feature.")

    values = scaled.loc[:, variable_columns].to_numpy(dtype=float)
    component_count = min(int(max_components), values.shape[0], values.shape[1])
    if component_count < 1:
        raise ValueError("PCA requires at least one valid component.")

    centered = values - values.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:component_count].copy()
    scores = centered @ components.T
    components, scores = _orient_components(components, scores)

    total_variance = float(np.sum(singular_values**2))
    explained = singular_values[:component_count] ** 2
    explained_ratio = explained / total_variance if total_variance > 0 else np.zeros_like(explained)

    score_columns = [f"PC{index}" for index in range(1, component_count + 1)]
    scores_df = _metadata_frame(working)
    for index, column in enumerate(score_columns):
        scores_df[column] = scores[:, index]

    loadings_df = pd.DataFrame({"feature": variable_columns})
    for index, column in enumerate(score_columns):
        loadings_df[column] = components[index, :]

    explained_df = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance": explained.astype(float),
            "explained_variance_ratio": explained_ratio.astype(float),
            "cumulative_explained_variance_ratio": np.cumsum(explained_ratio).astype(float),
        }
    )
    top_features = _top_component_features(loadings_df, top_n_features=top_n_features)
    metadata = {
        "scaling_method": _canonical_scaling_method(scaling),
        "scaling_parameters": scaling_parameters,
        "feature_columns_used": variable_columns,
        "constant_features_excluded": constant_columns,
        "excluded_nonfinite_rows": excluded_count,
        "component_count": component_count,
    }
    return scores_df, loadings_df, explained_df, top_features, warnings, metadata


def scale_feature_frame(
    dataframe: pd.DataFrame,
    *,
    method: str = "zscore",
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Scale finite features using an explicit deterministic method."""

    method = _canonical_scaling_method(method)
    values = dataframe.copy(deep=True).apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Feature scaling requires finite numeric values.")
    warnings: list[str] = []
    if method == "none":
        return values.astype(float), {"method": method, "zero_scale_features": []}, warnings

    zero_scale_features: list[str] = []
    parameters: dict[str, Any] = {"method": method, "zero_scale_features": zero_scale_features}
    if method == "zscore":
        centers = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scaled = _scale(values, centers, scales, zero_scale_features)
        parameters["centers"] = _series_to_float_dict(centers)
        parameters["scales"] = _series_to_float_dict(scales)
    elif method == "minmax":
        minimums = values.min(axis=0)
        maximums = values.max(axis=0)
        scales = maximums - minimums
        scaled = _scale(values, minimums, scales, zero_scale_features)
        parameters["minimums"] = _series_to_float_dict(minimums)
        parameters["maximums"] = _series_to_float_dict(maximums)
    elif method == "robust":
        medians = values.median(axis=0)
        lower = values.quantile(0.25, axis=0)
        upper = values.quantile(0.75, axis=0)
        scales = upper - lower
        scaled = _scale(values, medians, scales, zero_scale_features)
        parameters["medians"] = _series_to_float_dict(medians)
        parameters["iqr"] = _series_to_float_dict(scales)
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    if zero_scale_features:
        warnings.append(
            "Zero-scale features set to 0 during scaling: "
            + ", ".join(zero_scale_features)
            + "."
        )
    return scaled.astype(float), parameters, warnings


def _metadata_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in PCA_METADATA_COLUMNS if column in dataframe.columns]
    return dataframe.loc[:, columns].copy().reset_index(drop=True)


def _top_component_features(loadings: pd.DataFrame, *, top_n_features: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    component_columns = [column for column in loadings.columns if column.startswith("PC")]
    for component in component_columns:
        ranked = loadings.assign(abs_loading=loadings[component].abs()).sort_values(
            ["abs_loading", "feature"],
            ascending=[False, True],
        )
        for rank, row in enumerate(ranked.head(top_n_features).to_dict("records"), start=1):
            rows.append(
                {
                    "component": component,
                    "rank": rank,
                    "feature": row["feature"],
                    "loading": float(row[component]),
                    "absolute_loading": float(row["abs_loading"]),
                }
            )
    return pd.DataFrame(rows)


def _orient_components(components: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oriented_components = components.copy()
    oriented_scores = scores.copy()
    for index in range(oriented_components.shape[0]):
        loading_row = oriented_components[index]
        max_index = int(np.argmax(np.abs(loading_row)))
        if loading_row[max_index] < 0:
            oriented_components[index, :] *= -1.0
            oriented_scores[:, index] *= -1.0
    return oriented_components, oriented_scores


def _scale(
    values: pd.DataFrame,
    centers: pd.Series,
    scales: pd.Series,
    zero_scale_features: list[str],
) -> pd.DataFrame:
    adjusted = scales.copy()
    for feature, scale in scales.items():
        if pd.isna(scale) or float(scale) == 0.0:
            zero_scale_features.append(str(feature))
            adjusted.loc[feature] = 1.0
    scaled = (values - centers) / adjusted
    for feature in zero_scale_features:
        scaled.loc[:, feature] = 0.0
    return scaled


def _series_to_float_dict(series: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in series.items()}


def _canonical_scaling_method(method: str) -> str:
    normalized = str(method).strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "zscore": "zscore",
        "z": "zscore",
        "robust": "robust",
        "robustscaling": "robust",
        "minmax": "minmax",
        "none": "none",
    }
    if normalized not in aliases:
        raise ValueError("Scaling must be one of: zscore, robust, minmax, none.")
    return aliases[normalized]
