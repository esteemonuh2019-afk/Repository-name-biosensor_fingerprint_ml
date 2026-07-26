"""Descriptive statistics and numeric validation for feature datasets."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Iterable

import pandas as pd


DEFAULT_LOW_VARIANCE_THRESHOLD = 1e-12
DEFAULT_DOMINANT_PROPORTION_THRESHOLD = 0.95
DEFAULT_CORRELATION_THRESHOLD = 0.95

METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "Experiment_ID",
        "Measurement_Unit_ID",
        "Source_File",
        "Strain",
        "Chemical",
        "Concentration",
        "Replicate_ID",
        "QC_Status",
        "Source_QC_Statuses",
        "Source_QC_Flags",
        "Feature_QC_Flags",
        "Input_Row_Count",
        "Valid_Observation_Count",
        "Missing_Observation_Count",
        "Duplicate_Timestamp_Count",
        "Duplicate_Timestamp_Group_Count",
    }
)


def feature_columns_for_validation(
    dataframe: pd.DataFrame,
    candidate_features: Iterable[str],
) -> list[str]:
    """Return candidate feature columns present in the DataFrame, excluding metadata."""

    return [
        column
        for column in candidate_features
        if column in dataframe.columns and column not in METADATA_COLUMNS
    ]


def summarize_missing_values(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize missing values and affected metadata labels for each feature."""

    rows: list[dict[str, Any]] = []
    row_count = len(dataframe)
    for feature in feature_columns:
        missing_mask = dataframe[feature].isna()
        affected = dataframe.loc[missing_mask]
        missing_count = int(missing_mask.sum())
        rows.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_percentage": _percentage(missing_count, row_count),
                "affected_strains": _joined_unique(affected, "Strain"),
                "affected_chemicals": _joined_unique(affected, "Chemical"),
                "affected_concentrations": _joined_unique(affected, "Concentration"),
                "affected_source_files": _joined_unique(affected, "Source_File"),
            }
        )
    return pd.DataFrame(rows)


def summarize_nonfinite_values(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize infinity, NaN, and non-numeric values for each feature."""

    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        series = dataframe[feature] if feature in dataframe.columns else pd.Series(dtype=object)
        numeric = pd.to_numeric(series, errors="coerce")
        missing_mask = series.isna()
        non_numeric_mask = _non_numeric_mask(series, numeric)
        positive_inf_mask = numeric.map(lambda value: _is_positive_infinity(value))
        negative_inf_mask = numeric.map(lambda value: _is_negative_infinity(value))
        nan_count = int(missing_mask.sum())
        non_numeric_count = int(non_numeric_mask.sum())
        positive_inf_count = int(positive_inf_mask.sum())
        negative_inf_count = int(negative_inf_mask.sum())
        rows.append(
            {
                "feature": feature,
                "positive_infinity_count": positive_inf_count,
                "negative_infinity_count": negative_inf_count,
                "nan_count": nan_count,
                "non_numeric_count": non_numeric_count,
                "nonfinite_count": (
                    positive_inf_count
                    + negative_inf_count
                    + nan_count
                    + non_numeric_count
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_feature_statistics(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Calculate finite numeric descriptive statistics per feature."""

    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        finite_values = finite_numeric_series(dataframe, feature)
        unique_count = int(finite_values.nunique(dropna=True))
        dominant_proportion = _dominant_value_proportion(finite_values)
        rows.append(
            {
                "feature": feature,
                "finite_count": int(len(finite_values)),
                "mean": _safe_stat(finite_values.mean() if not finite_values.empty else pd.NA),
                "variance": _safe_stat(finite_values.var(ddof=0) if len(finite_values) else pd.NA),
                "standard_deviation": _safe_stat(
                    finite_values.std(ddof=0) if len(finite_values) else pd.NA
                ),
                "minimum": _safe_stat(finite_values.min() if not finite_values.empty else pd.NA),
                "maximum": _safe_stat(finite_values.max() if not finite_values.empty else pd.NA),
                "unique_value_count": unique_count,
                "dominant_value_proportion": dominant_proportion,
            }
        )
    return pd.DataFrame(rows)


def identify_constant_features(feature_statistics: pd.DataFrame) -> pd.DataFrame:
    """Return features with one unique finite value across valid feature rows."""

    if feature_statistics.empty:
        return pd.DataFrame(columns=list(feature_statistics.columns) + ["reason"])
    result = feature_statistics.loc[
        (feature_statistics["finite_count"] > 0)
        & (feature_statistics["unique_value_count"] == 1)
    ].copy()
    result["reason"] = "one_unique_finite_value"
    return result.reset_index(drop=True)


def identify_low_variance_features(
    feature_statistics: pd.DataFrame,
    *,
    variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
    dominant_proportion_threshold: float = DEFAULT_DOMINANT_PROPORTION_THRESHOLD,
) -> pd.DataFrame:
    """Return non-constant near-constant or low-variance features."""

    if feature_statistics.empty:
        return pd.DataFrame(columns=list(feature_statistics.columns) + ["reason"])

    rows = []
    for row in feature_statistics.to_dict("records"):
        if row["finite_count"] <= 0 or row["unique_value_count"] <= 1:
            continue
        reasons: list[str] = []
        variance = row.get("variance")
        dominant = row.get("dominant_value_proportion")
        if pd.notna(variance) and float(variance) <= variance_threshold:
            reasons.append(f"variance<={variance_threshold}")
        if pd.notna(dominant) and float(dominant) >= dominant_proportion_threshold:
            reasons.append(f"dominant_proportion>={dominant_proportion_threshold}")
        if reasons:
            rows.append({**row, "reason": ";".join(reasons)})
    return pd.DataFrame(rows)


def validate_feature_ranges(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Identify impossible or suspicious values based on Stage 6A/6B definitions."""

    rows: list[dict[str, Any]] = []
    for index, row in dataframe.iterrows():
        _append_range_violation(rows, row, index, "time_to_peak", _time_to_peak_violation(row))
        _append_range_violation(rows, row, index, "dynamic_range", _dynamic_range_violation(row))
        _append_range_violation(rows, row, index, "auc", _auc_violation(row))
        _append_range_violation(rows, row, index, "fold_change", _fold_change_violation(row))
        _append_range_violation(rows, row, index, "peak_minimum", _peak_minimum_violation(row))
        _append_range_violation(rows, row, index, "endpoint", _endpoint_violation(row))
    return pd.DataFrame(rows)


def correlation_table(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    method: str,
) -> pd.DataFrame:
    """Calculate pairwise Pearson or Spearman correlations among numeric features."""

    numeric = _numeric_feature_frame(dataframe, feature_columns)
    rows: list[dict[str, Any]] = []
    for feature_a, feature_b in combinations(numeric.columns, 2):
        pair = numeric[[feature_a, feature_b]].replace([math.inf, -math.inf], pd.NA).dropna()
        if len(pair) < 2:
            correlation = pd.NA
        else:
            correlation = pair[feature_a].corr(pair[feature_b], method=method)
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "method": method,
                "correlation": _safe_stat(correlation),
                "pair_count": int(len(pair)),
            }
        )
    return pd.DataFrame(rows)


def highly_correlated_pairs(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    *,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Return pairs whose absolute Pearson or Spearman correlation exceeds threshold."""

    combined = pd.concat([pearson, spearman], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "method", "correlation", "pair_count", "threshold"])
    numeric_corr = pd.to_numeric(combined["correlation"], errors="coerce")
    result = combined.loc[numeric_corr.abs() >= threshold].copy()
    result["threshold"] = threshold
    return result.sort_values(["feature_a", "feature_b", "method"]).reset_index(drop=True)


def finite_numeric_series(dataframe: pd.DataFrame, feature: str) -> pd.Series:
    """Return finite numeric values for one feature."""

    if feature not in dataframe.columns:
        return pd.Series(dtype="float64")
    numeric = pd.to_numeric(dataframe[feature], errors="coerce")
    finite_mask = numeric.map(lambda value: _is_finite(value))
    return numeric.loc[finite_mask].astype(float)


def _numeric_feature_frame(dataframe: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=dataframe.index)
    for feature in feature_columns:
        if feature in dataframe.columns and feature not in METADATA_COLUMNS:
            numeric[feature] = pd.to_numeric(dataframe[feature], errors="coerce")
    return numeric


def _append_range_violation(
    rows: list[dict[str, Any]],
    row: pd.Series,
    row_index: Any,
    feature: str,
    violation: str | None,
) -> None:
    if violation is None:
        return
    rows.append(
        {
            "row_index": row_index,
            "Experiment_ID": row.get("Experiment_ID", pd.NA),
            "Source_File": row.get("Source_File", pd.NA),
            "Measurement_Unit_ID": row.get("Measurement_Unit_ID", pd.NA),
            "feature": feature,
            "violation": violation,
        }
    )


def _time_to_peak_violation(row: pd.Series) -> str | None:
    time_to_peak = _optional_float(row.get("time_to_peak"))
    if time_to_peak is None:
        return None
    if time_to_peak < 0:
        return "negative_time_to_peak"
    duration = _optional_float(row.get("Duration"))
    start_time = _optional_float(row.get("Start_Time"))
    end_time = _optional_float(row.get("End_Time"))
    if end_time is not None and time_to_peak > end_time:
        return "time_to_peak_greater_than_end_time"
    if start_time is not None and time_to_peak < start_time:
        return "time_to_peak_less_than_start_time"
    if duration is not None and (start_time is None or start_time <= 0) and time_to_peak > duration:
        return "time_to_peak_greater_than_duration"
    return None


def _dynamic_range_violation(row: pd.Series) -> str | None:
    dynamic_range = _optional_float(row.get("dynamic_range"))
    if dynamic_range is not None and dynamic_range < 0:
        return "negative_dynamic_range"
    return None


def _auc_violation(row: pd.Series) -> str | None:
    auc = _optional_float(row.get("auc"))
    minimum = _optional_float(row.get("minimum"))
    if auc is not None and minimum is not None and minimum >= 0 and auc < 0:
        return "negative_auc_with_nonnegative_signal_range"
    return None


def _fold_change_violation(row: pd.Series) -> str | None:
    baseline = _optional_float(row.get("baseline"))
    fold_change = _optional_float(row.get("fold_change"))
    if baseline == 0 and fold_change is None:
        return "undefined_fold_change_from_zero_baseline"
    return None


def _peak_minimum_violation(row: pd.Series) -> str | None:
    peak = _optional_float(row.get("peak"))
    minimum = _optional_float(row.get("minimum"))
    dynamic_range = _optional_float(row.get("dynamic_range"))
    if peak is not None and minimum is not None and peak < minimum:
        return "peak_less_than_minimum"
    if peak is not None and minimum is not None and dynamic_range is not None:
        if not math.isclose(dynamic_range, peak - minimum, rel_tol=1e-9, abs_tol=1e-9):
            return "dynamic_range_inconsistent_with_peak_and_minimum"
    return None


def _endpoint_violation(row: pd.Series) -> str | None:
    endpoint = _optional_float(row.get("endpoint"))
    peak = _optional_float(row.get("peak"))
    minimum = _optional_float(row.get("minimum"))
    if endpoint is None or peak is None or minimum is None:
        return None
    if endpoint > peak:
        return "endpoint_greater_than_peak"
    if endpoint < minimum:
        return "endpoint_less_than_minimum"
    return None


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def _non_numeric_mask(series: pd.Series, numeric: pd.Series) -> pd.Series:
    missing = series.isna()
    normalized = series.astype("string").str.strip().str.casefold()
    explicit_nan = normalized.eq("nan")
    return (~missing) & numeric.isna() & (~explicit_nan.fillna(False))


def _dominant_value_proportion(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.value_counts(dropna=True).iloc[0] / len(values))


def _percentage(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(count / denominator * 100.0)


def _joined_unique(dataframe: pd.DataFrame, column: str) -> str:
    if column not in dataframe.columns or dataframe.empty:
        return ""
    values = [
        str(value)
        for value in dataframe[column].dropna().astype(str).unique().tolist()
        if str(value).strip()
    ]
    return ";".join(sorted(values))


def _safe_stat(value: Any) -> float | None:
    if pd.isna(value):
        return None
    converted = float(value)
    if math.isnan(converted):
        return None
    return converted


def _is_finite(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_positive_infinity(value: Any) -> bool:
    return not pd.isna(value) and float(value) == math.inf


def _is_negative_infinity(value: Any) -> bool:
    return not pd.isna(value) and float(value) == -math.inf

