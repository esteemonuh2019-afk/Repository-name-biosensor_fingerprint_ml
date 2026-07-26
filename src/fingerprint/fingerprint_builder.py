"""Build Stage 7A fingerprint datasets from validated feature results."""

from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd

from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS, FEATURE_ENGINE_VERSION
from src.feature_validation.feature_validator import FeatureValidationResult
from src.fingerprint.fingerprint_dataset import FingerprintDataset
from src.fingerprint.fingerprint_qc import evaluate_fingerprint_qc


FINGERPRINT_VERSION = "0.1.0"
FEATURE_VERSION = f"6B-core-{FEATURE_ENGINE_VERSION}"
DEFAULT_NORMALIZATION = "zscore"

FINGERPRINT_METADATA_COLUMNS: tuple[str, ...] = (
    "Fingerprint_ID",
    "Experiment_ID",
    "Measurement_Unit_ID",
    "Source_File",
    "Strain",
    "Chemical",
    "Concentration",
    "Replicate_ID",
    "Duration",
    "QC_Status",
    "Feature_QC_Flags",
)

FINGERPRINT_FEATURE_COLUMNS: tuple[str, ...] = CORE_FEATURE_COLUMNS
FINGERPRINT_DATASET_COLUMNS: tuple[str, ...] = (
    *FINGERPRINT_METADATA_COLUMNS,
    *FINGERPRINT_FEATURE_COLUMNS,
)

NormalizationMethod = Literal["none", "zscore", "minmax", "robust"]


def build_fingerprint_dataset(
    validation_result: FeatureValidationResult,
    *,
    normalization: NormalizationMethod | str = DEFAULT_NORMALIZATION,
) -> FingerprintDataset:
    """Build a fingerprint matrix from a Stage 6C validation result."""

    if not isinstance(validation_result, FeatureValidationResult):
        raise TypeError("Fingerprint builder requires a FeatureValidationResult input.")

    source_dataframe = _source_fingerprint_dataframe(validation_result.validated_dataframe)
    source_feature_order = list(validation_result.metadata.get("feature_columns", []))
    eligible_dataframe, excluded_dataframe = _eligible_fingerprint_rows(source_dataframe)
    normalized_dataframe, normalization_parameters, normalization_warnings = normalize_fingerprint_dataframe(
        eligible_dataframe,
        feature_names=FINGERPRINT_FEATURE_COLUMNS,
        method=normalization,
    )

    qc = evaluate_fingerprint_qc(
        eligible_dataframe,
        feature_names=FINGERPRINT_FEATURE_COLUMNS,
        expected_feature_names=FINGERPRINT_FEATURE_COLUMNS,
        source_dataframe=source_dataframe,
        excluded_dataframe=excluded_dataframe,
        source_feature_order=source_feature_order,
    )
    summary = _summary(
        validation_result=validation_result,
        fingerprint_dataframe=eligible_dataframe,
        excluded_dataframe=excluded_dataframe,
        qc_summary=qc.summary,
        qc_passed=qc.passed,
        normalization_method=normalization_parameters["method"],
        normalization_parameters=normalization_parameters,
    )
    warnings = [
        *qc.warnings,
        *normalization_warnings,
        *[f"Input feature validation warning: {warning}" for warning in validation_result.warnings],
        *[
            "Input feature validation error retained as fingerprint exclusion context: "
            f"{error}"
            for error in validation_result.errors
        ],
    ]
    errors = list(qc.errors)
    summary["fingerprint_qc_warning_count"] = len(qc.warnings)
    summary["fingerprint_warning_count"] = len(warnings)
    summary["fingerprint_error_count"] = len(errors)

    return FingerprintDataset(
        dataframe=eligible_dataframe.copy(deep=True),
        normalized_dataframe=normalized_dataframe.copy(deep=True),
        metadata=_metadata(validation_result, normalization_parameters),
        feature_names=list(FINGERPRINT_FEATURE_COLUMNS),
        feature_version=FEATURE_VERSION,
        fingerprint_version=FINGERPRINT_VERSION,
        qc=qc,
        summary=summary,
        warnings=warnings,
        errors=errors,
        normalization_method=normalization_parameters["method"],
        normalization_parameters=normalization_parameters,
        excluded_dataframe=excluded_dataframe.copy(deep=True),
    )


def normalize_fingerprint_dataframe(
    dataframe: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str],
    method: NormalizationMethod | str = DEFAULT_NORMALIZATION,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Return a normalised copy of a fingerprint dataframe."""

    method = _canonical_normalization_method(method)
    feature_names = list(feature_names)
    normalized = dataframe.copy(deep=True)
    warnings: list[str] = []

    if dataframe.empty:
        return normalized, {"method": method, "zero_scale_features": []}, warnings

    values = dataframe.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    finite_mask = values.apply(lambda column: column.map(_is_finite))
    if not finite_mask.all().all():
        raise ValueError("Fingerprint normalisation requires finite numeric feature values.")

    if method == "none":
        return normalized, {"method": method, "zero_scale_features": []}, warnings

    zero_scale_features: list[str] = []
    parameters: dict[str, Any] = {"method": method, "zero_scale_features": zero_scale_features}
    if method == "zscore":
        centers = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        transformed = _scale_values(values, centers, scales, zero_scale_features)
        parameters["centers"] = _float_mapping(centers)
        parameters["scales"] = _float_mapping(scales)
    elif method == "minmax":
        minimums = values.min(axis=0)
        maximums = values.max(axis=0)
        scales = maximums - minimums
        transformed = _scale_values(values, minimums, scales, zero_scale_features)
        parameters["minimums"] = _float_mapping(minimums)
        parameters["maximums"] = _float_mapping(maximums)
        parameters["scales"] = _float_mapping(scales)
    elif method == "robust":
        medians = values.median(axis=0)
        lower = values.quantile(0.25, axis=0)
        upper = values.quantile(0.75, axis=0)
        scales = upper - lower
        transformed = _scale_values(values, medians, scales, zero_scale_features)
        parameters["medians"] = _float_mapping(medians)
        parameters["iqr"] = _float_mapping(scales)
    else:
        raise ValueError(f"Unsupported normalisation method: {method}")

    if zero_scale_features:
        warnings.append(
            "Zero-scale fingerprint features set to 0 during normalisation: "
            + ", ".join(zero_scale_features)
            + "."
        )
    normalized.loc[:, feature_names] = transformed
    return normalized, parameters, warnings


def _source_fingerprint_dataframe(validated_dataframe: pd.DataFrame) -> pd.DataFrame:
    source = validated_dataframe.copy(deep=True)
    if "Fingerprint_ID" not in source.columns:
        source.insert(0, "Fingerprint_ID", source.apply(_fingerprint_id, axis=1))
    for column in FINGERPRINT_METADATA_COLUMNS:
        if column not in source.columns:
            source[column] = pd.NA
    for feature in FINGERPRINT_FEATURE_COLUMNS:
        if feature not in source.columns:
            source[feature] = pd.NA
    return source.loc[:, list(FINGERPRINT_DATASET_COLUMNS)].copy()


def _eligible_fingerprint_rows(source_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_dataframe.empty:
        return source_dataframe.copy(deep=True), source_dataframe.copy(deep=True)

    feature_values = source_dataframe.loc[:, list(FINGERPRINT_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_feature_mask = feature_values.apply(lambda column: column.map(_is_finite)).all(axis=1)
    qc_status = source_dataframe["QC_Status"].astype("string").fillna("")
    feature_qc_pass_mask = ~qc_status.eq("fail")
    eligible_mask = finite_feature_mask & feature_qc_pass_mask

    eligible = source_dataframe.loc[eligible_mask].copy()
    excluded = source_dataframe.loc[~eligible_mask].copy()
    if not excluded.empty:
        excluded["Fingerprint_Exclusion_Reason"] = [
            _exclusion_reason(qc_failed=not feature_qc_pass, finite_features=finite_features)
            for feature_qc_pass, finite_features in zip(
                feature_qc_pass_mask.loc[~eligible_mask].tolist(),
                finite_feature_mask.loc[~eligible_mask].tolist(),
                strict=True,
            )
        ]

    eligible.loc[:, list(FINGERPRINT_FEATURE_COLUMNS)] = feature_values.loc[eligible_mask].astype(float)
    return (
        eligible.loc[:, list(FINGERPRINT_DATASET_COLUMNS)].reset_index(drop=True),
        excluded.reset_index(drop=True),
    )


def _summary(
    *,
    validation_result: FeatureValidationResult,
    fingerprint_dataframe: pd.DataFrame,
    excluded_dataframe: pd.DataFrame,
    qc_summary: dict[str, Any],
    qc_passed: bool,
    normalization_method: str,
    normalization_parameters: dict[str, Any],
) -> dict[str, Any]:
    feature_rows = int(validation_result.metadata.get("feature_rows", len(validation_result.validated_dataframe)))
    fingerprint_rows = int(len(fingerprint_dataframe))
    excluded_rows = int(len(excluded_dataframe))
    status_counts = (
        fingerprint_dataframe["QC_Status"].astype("string").value_counts(dropna=False).to_dict()
        if "QC_Status" in fingerprint_dataframe.columns and not fingerprint_dataframe.empty
        else {}
    )
    excluded_reason_counts = (
        excluded_dataframe["Fingerprint_Exclusion_Reason"].value_counts(dropna=False).to_dict()
        if "Fingerprint_Exclusion_Reason" in excluded_dataframe.columns and not excluded_dataframe.empty
        else {}
    )
    return {
        "feature_rows": feature_rows,
        "fingerprint_rows": fingerprint_rows,
        "excluded_rows": excluded_rows,
        "feature_columns": list(FINGERPRINT_FEATURE_COLUMNS),
        "feature_count": len(FINGERPRINT_FEATURE_COLUMNS),
        "fingerprint_version": FINGERPRINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "normalization_method": normalization_method,
        "normalization_zero_scale_feature_count": len(
            normalization_parameters.get("zero_scale_features", [])
        ),
        "normalization_zero_scale_features": list(
            normalization_parameters.get("zero_scale_features", [])
        ),
        "distance_matrix_rows": fingerprint_rows,
        "distance_matrix_columns": fingerprint_rows,
        "feature_validation_passed": bool(validation_result.validation_passed),
        "feature_validation_errors": list(validation_result.errors),
        "feature_validation_warnings": list(validation_result.warnings),
        "fingerprint_qc_passed": bool(qc_passed),
        "fingerprint_qc_warning_count": 0,
        "fingerprint_warning_count": 0,
        "fingerprint_error_count": 0,
        "qc_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "excluded_reason_counts": {
            str(key): int(value) for key, value in excluded_reason_counts.items()
        },
        "duplicate_fingerprint_row_count": int(qc_summary.get("duplicate_fingerprint_row_count", 0)),
        "duplicated_measurement_unit_row_count": int(
            qc_summary.get("duplicated_measurement_unit_row_count", 0)
        ),
    }


def _metadata(
    validation_result: FeatureValidationResult,
    normalization_parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "7A",
        "fingerprint_version": FINGERPRINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "input_contract": "FeatureValidationResult",
        "feature_validation_stage": validation_result.metadata.get("stage", "6C"),
        "raw_readers_used_by_builder": False,
        "feature_validation_bypassed": False,
        "normalization_method": normalization_parameters["method"],
        "original_features_preserved": True,
        "normalised_features_written_separately": True,
        "distance_metrics": ["euclidean", "cosine", "manhattan", "correlation"],
        "pca_performed": False,
        "clustering_performed": False,
        "machine_learning_performed": False,
    }


def _canonical_normalization_method(method: NormalizationMethod | str) -> str:
    normalized = str(method).strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "none": "none",
        "no": "none",
        "zscore": "zscore",
        "z": "zscore",
        "minmax": "minmax",
        "robust": "robust",
        "robustscaling": "robust",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported normalisation method. Expected one of: none, zscore, minmax, robust."
        )
    return aliases[normalized]


def _scale_values(
    values: pd.DataFrame,
    centers: pd.Series,
    scales: pd.Series,
    zero_scale_features: list[str],
) -> pd.DataFrame:
    adjusted_scales = scales.copy()
    for feature, scale in scales.items():
        if pd.isna(scale) or float(scale) == 0.0:
            zero_scale_features.append(str(feature))
            adjusted_scales.loc[feature] = 1.0
    transformed = (values - centers) / adjusted_scales
    for feature in zero_scale_features:
        transformed.loc[:, feature] = 0.0
    return transformed


def _float_mapping(series: pd.Series) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in series.items():
        result[str(key)] = None if pd.isna(value) else float(value)
    return result


def _fingerprint_id(row: pd.Series) -> str:
    parts = [
        row.get("Experiment_ID", ""),
        row.get("Source_File", ""),
        row.get("Measurement_Unit_ID", ""),
    ]
    return "::".join("" if pd.isna(part) else str(part) for part in parts)


def _is_finite(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _exclusion_reason(*, qc_failed: bool, finite_features: bool) -> str:
    reasons: list[str] = []
    if qc_failed:
        reasons.append("feature_qc_fail")
    if not finite_features:
        reasons.append("missing_or_nonfinite_core_feature")
    return ";".join(reasons) if reasons else "excluded_feature_row"
