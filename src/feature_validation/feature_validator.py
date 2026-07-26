"""Stage 6C feature validation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.feature_engine.feature_dataset import FeatureDataset
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.feature_validation.feature_selection_report import recommend_features
from src.feature_validation.feature_statistics import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_DOMINANT_PROPORTION_THRESHOLD,
    DEFAULT_LOW_VARIANCE_THRESHOLD,
    calculate_feature_statistics,
    correlation_table,
    feature_columns_for_validation,
    highly_correlated_pairs,
    identify_constant_features,
    identify_low_variance_features,
    summarize_missing_values,
    summarize_nonfinite_values,
    validate_feature_ranges,
)
from src.feature_validation.replicate_reproducibility import (
    DEFAULT_ACCEPTABLE_CV_THRESHOLD,
    DEFAULT_STABLE_CV_THRESHOLD,
    calculate_replicate_consistency,
    summarize_replicate_consistency,
)


@dataclass(frozen=True)
class FeatureValidationResult:
    """Structured result from Stage 6C feature validation."""

    validated_dataframe: pd.DataFrame
    feature_statistics: pd.DataFrame
    missing_value_summary: pd.DataFrame
    infinite_value_summary: pd.DataFrame
    constant_feature_summary: pd.DataFrame
    low_variance_feature_summary: pd.DataFrame
    correlation_summary: dict[str, pd.DataFrame]
    replicate_reproducibility_summary: pd.DataFrame
    excluded_feature_candidates: list[str]
    retained_feature_candidates: list[str]
    warnings: list[str]
    errors: list[str]
    validation_passed: bool
    metadata: dict[str, Any]
    range_validation_summary: pd.DataFrame
    feature_recommendations: pd.DataFrame
    replicate_consistency_detail: pd.DataFrame

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable validation summary."""

        recommendation_counts = (
            self.feature_recommendations["recommendation"].value_counts().to_dict()
            if not self.feature_recommendations.empty
            else {}
        )
        return {
            "metadata": self.metadata,
            "validation_passed": self.validation_passed,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "missing_value_total": _safe_sum(self.missing_value_summary, "missing_count"),
            "nonfinite_value_total": _safe_sum(self.infinite_value_summary, "nonfinite_count"),
            "constant_features": list(self.constant_feature_summary.get("feature", pd.Series(dtype=str)).astype(str)),
            "low_variance_features": list(self.low_variance_feature_summary.get("feature", pd.Series(dtype=str)).astype(str)),
            "highly_correlated_pair_count": len(
                self.correlation_summary.get("highly_correlated_pairs", pd.DataFrame())
            ),
            "recommendation_counts": {
                str(key): int(value) for key, value in recommendation_counts.items()
            },
            "excluded_feature_candidates": list(self.excluded_feature_candidates),
            "retained_feature_candidates": list(self.retained_feature_candidates),
        }


def validate_features(
    feature_dataset: FeatureDataset | pd.DataFrame,
    *,
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
    dominant_proportion_threshold: float = DEFAULT_DOMINANT_PROPORTION_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    stable_cv_threshold: float = DEFAULT_STABLE_CV_THRESHOLD,
    acceptable_cv_threshold: float = DEFAULT_ACCEPTABLE_CV_THRESHOLD,
) -> FeatureValidationResult:
    """Validate extracted core features without modifying feature values."""

    dataframe = _feature_dataframe(feature_dataset)
    feature_columns = feature_columns_for_validation(dataframe, CORE_FEATURE_COLUMNS)
    valid_dataframe = _numerically_assessable_rows(dataframe)

    missing_summary = summarize_missing_values(dataframe, feature_columns)
    nonfinite_summary = summarize_nonfinite_values(dataframe, feature_columns)
    feature_statistics = calculate_feature_statistics(valid_dataframe, feature_columns)
    constant_features = identify_constant_features(feature_statistics)
    low_variance_features = identify_low_variance_features(
        feature_statistics,
        variance_threshold=low_variance_threshold,
        dominant_proportion_threshold=dominant_proportion_threshold,
    )
    range_summary = validate_feature_ranges(dataframe)
    pearson = correlation_table(valid_dataframe, feature_columns, method="pearson")
    spearman = correlation_table(valid_dataframe, feature_columns, method="spearman")
    correlated_pairs = highly_correlated_pairs(
        pearson,
        spearman,
        threshold=correlation_threshold,
    )
    replicate_detail = calculate_replicate_consistency(
        valid_dataframe,
        feature_columns,
        stable_cv_threshold=stable_cv_threshold,
        acceptable_cv_threshold=acceptable_cv_threshold,
    )
    replicate_summary = summarize_replicate_consistency(replicate_detail)
    recommendations = recommend_features(
        feature_columns,
        missing_value_summary=missing_summary,
        nonfinite_value_summary=nonfinite_summary,
        feature_statistics=feature_statistics,
        constant_feature_summary=constant_features,
        low_variance_feature_summary=low_variance_features,
        range_validation_summary=range_summary,
        highly_correlated_pairs=correlated_pairs,
        replicate_reproducibility_summary=replicate_summary,
    )

    warnings, errors = _validation_messages(
        dataframe=dataframe,
        feature_columns=feature_columns,
        missing_summary=missing_summary,
        nonfinite_summary=nonfinite_summary,
        constant_features=constant_features,
        low_variance_features=low_variance_features,
        range_summary=range_summary,
        correlated_pairs=correlated_pairs,
    )

    metadata = {
        "stage": "6C",
        "feature_rows": int(len(dataframe)),
        "valid_feature_rows": int(len(valid_dataframe)),
        "feature_columns": list(feature_columns),
        "feature_columns_assessed": int(len(feature_columns)),
        "metadata_columns_excluded": True,
        "low_variance_threshold": low_variance_threshold,
        "dominant_proportion_threshold": dominant_proportion_threshold,
        "correlation_threshold": correlation_threshold,
        "stable_cv_threshold": stable_cv_threshold,
        "acceptable_cv_threshold": acceptable_cv_threshold,
        "selection_is_supervised": False,
        "biological_reproducibility_claimed": False,
        "replicate_assessment_label": "replicate consistency",
    }

    excluded = _features_by_recommendation(recommendations, "Exclude")
    retained = _features_by_recommendation(recommendations, "Retain")
    return FeatureValidationResult(
        validated_dataframe=dataframe.copy(deep=True),
        feature_statistics=feature_statistics,
        missing_value_summary=missing_summary,
        infinite_value_summary=nonfinite_summary,
        constant_feature_summary=constant_features,
        low_variance_feature_summary=low_variance_features,
        correlation_summary={
            "pearson": pearson,
            "spearman": spearman,
            "highly_correlated_pairs": correlated_pairs,
        },
        replicate_reproducibility_summary=replicate_summary,
        excluded_feature_candidates=excluded,
        retained_feature_candidates=retained,
        warnings=warnings,
        errors=errors,
        validation_passed=not errors,
        metadata=metadata,
        range_validation_summary=range_summary,
        feature_recommendations=recommendations,
        replicate_consistency_detail=replicate_detail,
    )


def _feature_dataframe(feature_dataset: FeatureDataset | pd.DataFrame) -> pd.DataFrame:
    if isinstance(feature_dataset, FeatureDataset):
        return feature_dataset.dataframe.copy(deep=True)
    return feature_dataset.copy(deep=True)


def _numerically_assessable_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or "QC_Status" not in dataframe.columns:
        return dataframe.copy(deep=True)
    return dataframe.loc[~dataframe["QC_Status"].astype("string").eq("fail")].copy(deep=True)


def _validation_messages(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    missing_summary: pd.DataFrame,
    nonfinite_summary: pd.DataFrame,
    constant_features: pd.DataFrame,
    low_variance_features: pd.DataFrame,
    range_summary: pd.DataFrame,
    correlated_pairs: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if dataframe.empty:
        errors.append("Feature dataset is empty.")
    if not feature_columns:
        errors.append("No core feature columns are available for validation.")

    failed_rows = _failed_row_count(dataframe)
    if failed_rows:
        errors.append(f"Feature dataset contains {failed_rows} failed feature rows.")

    missing_total = _safe_sum(missing_summary, "missing_count")
    if missing_total:
        warnings.append(f"Missing feature values detected: {missing_total}.")

    serious_nonfinite = _serious_nonfinite_count(nonfinite_summary)
    if serious_nonfinite:
        errors.append(f"Infinite or non-numeric feature values detected: {serious_nonfinite}.")

    if not constant_features.empty:
        warnings.append(f"Constant features detected: {len(constant_features)}.")
    if not low_variance_features.empty:
        warnings.append(f"Low-variance features detected: {len(low_variance_features)}.")
    if not range_summary.empty:
        errors.append(f"Feature range violations detected: {len(range_summary)}.")
    if not correlated_pairs.empty:
        warnings.append(f"Highly correlated feature pairs detected: {len(correlated_pairs)}.")
    return warnings, errors


def _failed_row_count(dataframe: pd.DataFrame) -> int:
    if "QC_Status" not in dataframe.columns or dataframe.empty:
        return 0
    return int(dataframe["QC_Status"].astype("string").eq("fail").sum())


def _serious_nonfinite_count(nonfinite_summary: pd.DataFrame) -> int:
    if nonfinite_summary.empty:
        return 0
    columns = ["positive_infinity_count", "negative_infinity_count", "non_numeric_count"]
    return int(sum(nonfinite_summary[column].sum() for column in columns if column in nonfinite_summary.columns))


def _features_by_recommendation(recommendations: pd.DataFrame, recommendation: str) -> list[str]:
    if recommendations.empty:
        return []
    return sorted(
        recommendations.loc[
            recommendations["recommendation"].astype(str).eq(recommendation),
            "feature",
        ].astype(str).tolist()
    )


def _safe_sum(dataframe: pd.DataFrame, column: str) -> int:
    if dataframe.empty or column not in dataframe.columns:
        return 0
    return int(pd.to_numeric(dataframe[column], errors="coerce").fillna(0).sum())

