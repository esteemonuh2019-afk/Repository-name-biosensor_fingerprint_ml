"""Core canonical feature extraction for biosensor luminescence series."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import pandas as pd

from src.data_schema.canonical_schema import (
    CANONICAL_COLUMNS,
    SERIES_GROUPING_KEY_COLUMNS,
    coerce_canonical_dtypes,
)
from src.feature_engine.feature_dataset import FeatureDataset
from src.feature_engine.feature_qc import evaluate_feature_qc


FEATURE_ENGINE_VERSION = "0.1.0"

SERIES_KEY_COLUMNS: tuple[str, ...] = SERIES_GROUPING_KEY_COLUMNS

METADATA_COLUMNS: tuple[str, ...] = (
    "Experiment_ID",
    "Measurement_Unit_ID",
    "Source_File",
    "Strain",
    "Chemical",
    "Concentration",
    "Replicate_ID",
    "Duration",
    "QC_Status",
)

CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "baseline",
    "peak",
    "minimum",
    "endpoint",
    "dynamic_range",
    "time_to_peak",
    "auc",
    "initial_slope",
    "maximum_slope",
    "fold_change",
    "log2_fold_change",
)

DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "Start_Time",
    "End_Time",
    "Input_Row_Count",
    "Valid_Observation_Count",
    "Missing_Observation_Count",
    "Duplicate_Timestamp_Count",
    "Duplicate_Timestamp_Group_Count",
    "Source_QC_Statuses",
    "Source_QC_Flags",
    "Feature_QC_Flags",
)

FEATURE_DATASET_COLUMNS: tuple[str, ...] = (
    *METADATA_COLUMNS,
    *CORE_FEATURE_COLUMNS,
    *DIAGNOSTIC_COLUMNS,
)


def extract_features(canonical_dataframe: pd.DataFrame) -> FeatureDataset:
    """Extract Stage 6B core features from a canonical biosensor DataFrame.

    The engine consumes canonical rows only. It does not read raw CSV or Excel
    files and does not average duplicate timestamps.
    """

    _validate_canonical_columns(canonical_dataframe)
    canonical = coerce_canonical_dtypes(canonical_dataframe)

    if canonical.empty:
        empty = pd.DataFrame(columns=list(FEATURE_DATASET_COLUMNS))
        qc = evaluate_feature_qc(empty, feature_columns=CORE_FEATURE_COLUMNS)
        metadata = _metadata(input_rows=0)
        summary = _summary(empty, canonical, qc)
        return FeatureDataset(dataframe=empty, metadata=metadata, summary=summary, qc=qc)

    rows: list[dict[str, Any]] = []
    grouped = canonical.groupby(list(SERIES_KEY_COLUMNS), dropna=False, sort=True)
    for _, series in grouped:
        rows.append(_extract_series_features(series))

    feature_dataframe = pd.DataFrame(rows, columns=list(FEATURE_DATASET_COLUMNS))
    qc = evaluate_feature_qc(feature_dataframe, feature_columns=CORE_FEATURE_COLUMNS)
    metadata = _metadata(input_rows=len(canonical))
    summary = _summary(feature_dataframe, canonical, qc)
    return FeatureDataset(
        dataframe=feature_dataframe,
        metadata=metadata,
        summary=summary,
        qc=qc,
    )


def _extract_series_features(series: pd.DataFrame) -> dict[str, Any]:
    flags: list[str] = []
    input_row_count = int(len(series))
    source_statuses = _joined_unique(series, "QC_Status")
    source_flags = _joined_unique_flags(series, "QC_Flags")

    if input_row_count == 0:
        flags.append("empty_series")

    if "fail" in _string_values(series, "QC_Status"):
        flags.append("source_qc_fail")
    elif "warning" in _string_values(series, "QC_Status"):
        flags.append("source_qc_warning")

    if _has_invalid_record_rows(series):
        flags.append("record_invalid_rows_excluded")

    numeric = _numeric_measurements(series)
    missing_observation_count = int(input_row_count - len(numeric))
    if missing_observation_count:
        flags.append("missing_time_or_signal_rows")

    if numeric.empty:
        flags.append("no_valid_observations")
        return _feature_row(
            series=series,
            flags=flags,
            source_statuses=source_statuses,
            source_flags=source_flags,
            input_row_count=input_row_count,
            valid_observation_count=0,
            missing_observation_count=missing_observation_count,
        )

    negative_time_count = int((numeric["Time_Minutes"] < 0).sum())
    if negative_time_count:
        flags.append("negative_time_values")

    duplicate_info = _duplicate_timestamp_info(numeric)
    if duplicate_info["duplicate_timestamp_count"]:
        flags.append("duplicate_timestamps")
    if duplicate_info["conflicting_duplicate_timestamp_count"]:
        flags.append("conflicting_duplicate_timestamps")

    ordered = numeric.sort_values(
        ["Time_Minutes", "Source_Row_ID"],
        kind="mergesort",
        na_position="last",
    )
    time_values = ordered["Time_Minutes"].astype(float).tolist()
    signal_values = ordered["Luminescence_Raw"].astype(float).tolist()

    start_time = float(min(time_values))
    end_time = float(max(time_values))
    duration = end_time - start_time

    baseline = signal_values[0]
    peak = max(signal_values)
    minimum = min(signal_values)
    endpoint = _endpoint_value(ordered, flags)
    dynamic_range = peak - minimum
    time_to_peak = _time_to_peak(time_values, signal_values, peak)
    auc = _auc(time_values, signal_values, duplicate_info, flags)
    initial_slope = _initial_slope(time_values, signal_values, duplicate_info, flags)
    maximum_slope = _maximum_slope(time_values, signal_values, duplicate_info, flags)
    fold_change = _fold_change(peak, baseline, flags)
    log2_fold_change = _log2_fold_change(endpoint, baseline, flags)

    return _feature_row(
        series=series,
        flags=flags,
        source_statuses=source_statuses,
        source_flags=source_flags,
        input_row_count=input_row_count,
        valid_observation_count=len(numeric),
        missing_observation_count=missing_observation_count,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        duplicate_timestamp_count=duplicate_info["duplicate_timestamp_count"],
        duplicate_timestamp_group_count=duplicate_info["duplicate_timestamp_group_count"],
        baseline=baseline,
        peak=peak,
        minimum=minimum,
        endpoint=endpoint,
        dynamic_range=dynamic_range,
        time_to_peak=time_to_peak,
        auc=auc,
        initial_slope=initial_slope,
        maximum_slope=maximum_slope,
        fold_change=fold_change,
        log2_fold_change=log2_fold_change,
    )


def _feature_row(
    *,
    series: pd.DataFrame,
    flags: list[str],
    source_statuses: str,
    source_flags: str,
    input_row_count: int,
    valid_observation_count: int,
    missing_observation_count: int,
    start_time: float | None = None,
    end_time: float | None = None,
    duration: float | None = None,
    duplicate_timestamp_count: int = 0,
    duplicate_timestamp_group_count: int = 0,
    baseline: float | None = None,
    peak: float | None = None,
    minimum: float | None = None,
    endpoint: float | None = None,
    dynamic_range: float | None = None,
    time_to_peak: float | None = None,
    auc: float | None = None,
    initial_slope: float | None = None,
    maximum_slope: float | None = None,
    fold_change: float | None = None,
    log2_fold_change: float | None = None,
) -> dict[str, Any]:
    row = {
        "Experiment_ID": _single_value(series, "Experiment_ID"),
        "Measurement_Unit_ID": _single_value(series, "Measurement_Unit_ID"),
        "Source_File": _single_value(series, "Source_File"),
        "Strain": _single_value(series, "Strain_Original"),
        "Chemical": _single_value(series, "Chemical_Name_Original"),
        "Concentration": _single_value(series, "Concentration_Label"),
        "Replicate_ID": _single_value(series, "Replicate_ID"),
        "Duration": duration,
        "QC_Status": _feature_qc_status(flags),
        "baseline": baseline,
        "peak": peak,
        "minimum": minimum,
        "endpoint": endpoint,
        "dynamic_range": dynamic_range,
        "time_to_peak": time_to_peak,
        "auc": auc,
        "initial_slope": initial_slope,
        "maximum_slope": maximum_slope,
        "fold_change": fold_change,
        "log2_fold_change": log2_fold_change,
        "Start_Time": start_time,
        "End_Time": end_time,
        "Input_Row_Count": input_row_count,
        "Valid_Observation_Count": valid_observation_count,
        "Missing_Observation_Count": missing_observation_count,
        "Duplicate_Timestamp_Count": duplicate_timestamp_count,
        "Duplicate_Timestamp_Group_Count": duplicate_timestamp_group_count,
        "Source_QC_Statuses": source_statuses,
        "Source_QC_Flags": source_flags,
        "Feature_QC_Flags": _join_flags(flags),
    }
    return row


def _validate_canonical_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {', '.join(missing)}")


def _numeric_measurements(series: pd.DataFrame) -> pd.DataFrame:
    valid_scope = pd.Series(True, index=series.index)
    if "Record_Valid" in series.columns:
        valid_scope = series["Record_Valid"].astype("boolean").ne(False).fillna(True)

    numeric = series.loc[valid_scope, ["Time_Minutes", "Luminescence_Raw", "Source_Row_ID"]].copy()
    numeric["Time_Minutes"] = pd.to_numeric(numeric["Time_Minutes"], errors="coerce")
    numeric["Luminescence_Raw"] = pd.to_numeric(numeric["Luminescence_Raw"], errors="coerce")
    numeric["Source_Row_ID"] = pd.to_numeric(numeric["Source_Row_ID"], errors="coerce")
    finite_mask = (
        numeric["Time_Minutes"].map(_is_finite_number)
        & numeric["Luminescence_Raw"].map(_is_finite_number)
    )
    return numeric.loc[finite_mask].copy()


def _duplicate_timestamp_info(numeric: pd.DataFrame) -> dict[str, int]:
    if numeric.empty:
        return {
            "duplicate_timestamp_count": 0,
            "duplicate_timestamp_group_count": 0,
            "conflicting_duplicate_timestamp_count": 0,
        }

    counts = (
        numeric.groupby("Time_Minutes", dropna=False, sort=True)
        .agg(
            row_count=("Luminescence_Raw", "size"),
            distinct_values=("Luminescence_Raw", lambda values: int(values.nunique(dropna=False))),
        )
        .reset_index()
    )
    duplicate_groups = counts.loc[counts["row_count"] > 1]
    conflicting = duplicate_groups.loc[duplicate_groups["distinct_values"] > 1]
    return {
        "duplicate_timestamp_count": int(duplicate_groups["row_count"].sum()) if not duplicate_groups.empty else 0,
        "duplicate_timestamp_group_count": int(len(duplicate_groups)),
        "conflicting_duplicate_timestamp_count": (
            int(conflicting["row_count"].sum()) if not conflicting.empty else 0
        ),
    }


def _endpoint_value(ordered: pd.DataFrame, flags: list[str]) -> float | None:
    last_time = ordered["Time_Minutes"].max()
    endpoint_rows = ordered.loc[ordered["Time_Minutes"].eq(last_time), "Luminescence_Raw"]
    if endpoint_rows.empty:
        flags.append("missing_endpoint")
        return None
    if endpoint_rows.nunique(dropna=False) > 1:
        flags.append("conflicting_endpoint_timestamp")
        return None
    return float(endpoint_rows.iloc[-1])


def _time_to_peak(time_values: list[float], signal_values: list[float], peak: float) -> float | None:
    for time_value, signal_value in zip(time_values, signal_values, strict=True):
        if signal_value == peak:
            return float(time_value)
    return None


def _auc(
    time_values: list[float],
    signal_values: list[float],
    duplicate_info: dict[str, int],
    flags: list[str],
) -> float | None:
    if len(time_values) < 2:
        flags.append("insufficient_distinct_timepoints_for_auc")
        return None
    if duplicate_info["duplicate_timestamp_group_count"]:
        flags.append("duplicate_timestamps_prevent_auc")
        return None

    total = 0.0
    for index in range(len(time_values) - 1):
        time_delta = time_values[index + 1] - time_values[index]
        if time_delta <= 0:
            flags.append("nonpositive_time_delta_prevents_auc")
            return None
        total += time_delta * (signal_values[index] + signal_values[index + 1]) / 2.0
    return float(total)


def _initial_slope(
    time_values: list[float],
    signal_values: list[float],
    duplicate_info: dict[str, int],
    flags: list[str],
) -> float | None:
    if duplicate_info["duplicate_timestamp_group_count"]:
        flags.append("duplicate_timestamps_prevent_initial_slope")
        return None

    if len(time_values) < 2:
        flags.append("insufficient_distinct_timepoints_for_initial_slope")
        return None

    time_delta = time_values[1] - time_values[0]
    if time_delta <= 0:
        flags.append("nonpositive_time_delta_prevents_initial_slope")
        return None
    return float((signal_values[1] - signal_values[0]) / time_delta)


def _maximum_slope(
    time_values: list[float],
    signal_values: list[float],
    duplicate_info: dict[str, int],
    flags: list[str],
) -> float | None:
    if duplicate_info["duplicate_timestamp_group_count"]:
        flags.append("duplicate_timestamps_prevent_maximum_slope")
        return None

    if len(time_values) < 2:
        flags.append("insufficient_distinct_timepoints_for_maximum_slope")
        return None

    slopes: list[float] = []
    for index in range(len(time_values) - 1):
        time_delta = time_values[index + 1] - time_values[index]
        if time_delta <= 0:
            flags.append("nonpositive_time_delta_prevents_maximum_slope")
            return None
        slopes.append((signal_values[index + 1] - signal_values[index]) / time_delta)
    return float(max(slopes))


def _fold_change(peak: float, baseline: float, flags: list[str]) -> float | None:
    if baseline == 0:
        flags.append("zero_baseline_for_fold_change")
        return None
    if baseline < 0:
        flags.append("negative_baseline_for_fold_change")
        return None
    return float((peak - baseline) / baseline)


def _log2_fold_change(endpoint: float | None, baseline: float, flags: list[str]) -> float | None:
    if endpoint is None:
        flags.append("missing_endpoint_for_log2_fold_change")
        return None
    if baseline == 0:
        flags.append("zero_baseline_for_log2_fold_change")
        return None
    if baseline < 0:
        flags.append("negative_baseline_for_log2_fold_change")
        return None
    if endpoint <= 0:
        flags.append("nonpositive_endpoint_for_log2_fold_change")
        return None
    return float(math.log2(endpoint / baseline))


def _feature_qc_status(flags: list[str]) -> str:
    fatal_flags = {
        "empty_series",
        "no_valid_observations",
        "negative_time_values",
        "conflicting_duplicate_timestamps",
        "conflicting_endpoint_timestamp",
        "source_qc_fail",
    }
    if any(flag in fatal_flags for flag in flags):
        return "fail"
    if flags:
        return "warning"
    return "pass"


def _summary(
    feature_dataframe: pd.DataFrame,
    canonical_dataframe: pd.DataFrame,
    qc_result: Any,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    if "QC_Status" in feature_dataframe.columns and not feature_dataframe.empty:
        status_counts = {
            str(key): int(value)
            for key, value in feature_dataframe["QC_Status"].value_counts(dropna=False).items()
        }

    return {
        "feature_engine_version": FEATURE_ENGINE_VERSION,
        "input_canonical_rows": int(len(canonical_dataframe)),
        "feature_rows": int(len(feature_dataframe)),
        "core_feature_count": len(CORE_FEATURE_COLUMNS),
        "core_features": list(CORE_FEATURE_COLUMNS),
        "series_key_columns": list(SERIES_KEY_COLUMNS),
        "feature_qc_passed": bool(qc_result.passed),
        "feature_qc_status_counts": status_counts,
        "missing_feature_value_count": int(qc_result.summary["missing_feature_value_count"]),
        "infinite_feature_value_count": int(qc_result.summary["infinite_feature_value_count"]),
        "zero_baseline_count": int(qc_result.summary["zero_baseline_count"]),
        "duplicate_measurement_unit_row_count": int(
            qc_result.summary["duplicate_measurement_unit_row_count"]
        ),
    }


def _metadata(input_rows: int) -> dict[str, Any]:
    return {
        "stage": "6B",
        "feature_engine_version": FEATURE_ENGINE_VERSION,
        "input_contract": "canonical_schema_v1.1.0",
        "input_rows": int(input_rows),
        "implemented_priority": "Core features requested for Stage 6B only",
        "normalisation_used": False,
        "duplicate_timestamps_averaged": False,
        "raw_readers_used": False,
    }


def _single_value(series: pd.DataFrame, column: str) -> Any:
    if column not in series.columns or series.empty:
        return pd.NA
    values = [value for value in series[column].dropna().unique().tolist()]
    if len(values) > 1:
        return ";".join(str(value) for value in values)
    if not values:
        return pd.NA
    value = values[0]
    if hasattr(value, "item"):
        return value.item()
    return value


def _joined_unique(series: pd.DataFrame, column: str) -> str:
    values = _string_values(series, column)
    return ";".join(sorted(values))


def _joined_unique_flags(series: pd.DataFrame, column: str) -> str:
    flags: set[str] = set()
    if column not in series.columns:
        return ""
    for value in series[column].dropna().astype(str):
        flags.update(part.strip() for part in value.split(";") if part.strip())
    return ";".join(sorted(flags))


def _string_values(series: pd.DataFrame, column: str) -> set[str]:
    if column not in series.columns:
        return set()
    return {
        str(value)
        for value in series[column].dropna().astype(str).tolist()
        if str(value).strip()
    }


def _has_invalid_record_rows(series: pd.DataFrame) -> bool:
    if "Record_Valid" not in series.columns:
        return False
    return bool(series["Record_Valid"].astype("boolean").eq(False).fillna(False).any())


def _is_finite_number(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _join_flags(flags: list[str]) -> str:
    counts = Counter(flags)
    ordered = sorted(counts)
    return ";".join(ordered)

