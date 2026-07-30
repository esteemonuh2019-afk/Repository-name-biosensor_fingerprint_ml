"""Stage 8C Feature Engine V2 for canonical luminescence time series."""

from __future__ import annotations

from collections import Counter
import math
import time as timer
from typing import Any

import numpy as np
import pandas as pd

from src.data_schema.canonical_schema import (
    CANONICAL_COLUMNS,
    SERIES_GROUPING_KEY_COLUMNS,
    coerce_canonical_dtypes,
    validate_canonical_schema,
)
from src.feature_engine_v2.feature_dataset_v2 import AdvancedFeatureDataset
from src.feature_engine_v2.feature_definitions import (
    BASELINE_FEATURES,
    FEATURE_ENGINE_V2_VERSION,
    FEATURE_FAMILIES,
    FREQUENCY_FEATURES,
    NORMALIZED_FEATURES,
    RESPONSE_DYNAMICS,
    SHAPE_DESCRIPTORS,
    STRAIN_INTERACTION,
    TEMPORAL_KINETICS,
    WINDOW_FEATURES,
    feature_columns_by_family,
    feature_dictionary,
)


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
    "Advanced_Feature_QC_Flags",
)

WINDOW_BOUNDS_HOURS: tuple[tuple[str, float, float], ...] = (
    ("0_2h", 0.0, 2.0),
    ("2_6h", 2.0, 6.0),
    ("6_12h", 6.0, 12.0),
    ("12_24h", 12.0, 24.0),
)


def extract_advanced_features(canonical_dataframe: pd.DataFrame) -> AdvancedFeatureDataset:
    """Extract Stage 8C advanced features from validated canonical rows."""

    started = timer.perf_counter()
    _validate_canonical_columns(canonical_dataframe)
    schema_result = validate_canonical_schema(canonical_dataframe)
    canonical = coerce_canonical_dtypes(canonical_dataframe)
    warnings = [f"Canonical schema warning: {warning}" for warning in schema_result.warnings]
    errors: list[str] = []
    if schema_result.errors:
        warnings.extend(f"Canonical schema error retained as QC context: {error}" for error in schema_result.errors)

    if canonical.empty:
        dataframe = _empty_dataframe()
        return _dataset(dataframe, warnings, errors, started, input_rows=0, usable_rows=0)

    usable = _usable_canonical_rows(canonical)
    excluded_rows = int(len(canonical) - len(usable))
    if excluded_rows:
        warnings.append(f"Canonical rows excluded by Stage 8C QC scope: {excluded_rows}.")

    rows: list[dict[str, Any]] = []
    for _, series in usable.groupby(list(SERIES_GROUPING_KEY_COLUMNS), dropna=False, sort=True):
        rows.append(_extract_series(series))
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe = _empty_dataframe()
    else:
        dataframe = _add_strain_interaction_features(dataframe)
        dataframe = _order_columns(dataframe)
    return _dataset(dataframe, warnings, errors, started, input_rows=len(canonical), usable_rows=len(usable))


def _extract_series(series: pd.DataFrame) -> dict[str, Any]:
    flags: list[str] = []
    numeric = _numeric_series(series)
    if numeric.empty:
        flags.append("no_valid_observations")
        return _metadata_row(series, flags)

    ordered = numeric.sort_values(["Time_Minutes", "Source_Row_ID"], kind="mergesort").drop_duplicates(
        subset=["Time_Minutes"],
        keep="first",
    )
    if len(ordered) < len(numeric):
        flags.append("duplicate_timestamps_first_value_used_for_v2_features")

    time_values = ordered["Time_Minutes"].to_numpy(dtype=float)
    signal_values = ordered["Luminescence_Raw"].to_numpy(dtype=float)
    if len(time_values) < 2:
        flags.append("insufficient_timepoints")

    baseline = float(signal_values[0])
    peak = float(np.max(signal_values))
    minimum = float(np.min(signal_values))
    endpoint = float(signal_values[-1])
    start_time = float(time_values[0])
    end_time = float(time_values[-1])
    duration = end_time - start_time
    derivatives = _derivatives(time_values, signal_values, flags)
    row = {
        **_metadata_row(series, flags),
        **_temporal_features(time_values, signal_values, derivatives, baseline, peak, endpoint, start_time, end_time),
        **_window_features(time_values, signal_values),
        **_shape_features(time_values, signal_values, baseline),
        **_frequency_features(time_values, signal_values),
        **_response_dynamics(time_values, signal_values, baseline, peak, minimum, endpoint, duration),
        **_baseline_features(time_values, signal_values),
        **_normalized_features(time_values, signal_values, baseline, peak, minimum, endpoint, duration),
        "_response_magnitude_for_strain_interaction": _auc(time_values, signal_values),
    }
    row["Advanced_Feature_QC_Flags"] = _join_flags(flags)
    return row


def _temporal_features(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    derivatives: np.ndarray,
    baseline: float,
    peak: float,
    endpoint: float,
    start_time: float,
    end_time: float,
) -> dict[str, float | None]:
    threshold = baseline + 0.5 * (peak - baseline)
    peak_index = int(np.argmax(signal_values))
    time_to_peak = float(time_values[peak_index])
    above = signal_values >= threshold
    return {
        "temporal_time_to_peak": time_to_peak,
        "temporal_time_to_half_peak": _first_threshold_time(time_values, signal_values, threshold, direction="above"),
        "temporal_rise_time": time_to_peak - start_time,
        "temporal_decay_time": end_time - time_to_peak,
        "temporal_recovery_time": _recovery_time(time_values, signal_values, baseline, peak_index),
        "temporal_peak_width": _duration_masked(time_values, above),
        "temporal_peak_prominence": peak - max(baseline, endpoint),
        "temporal_maximum_derivative": _finite_or_none(np.max(derivatives)) if len(derivatives) else None,
        "temporal_minimum_derivative": _finite_or_none(np.min(derivatives)) if len(derivatives) else None,
        "temporal_derivative_variance": _finite_or_none(np.var(derivatives)) if len(derivatives) else None,
        "temporal_derivative_entropy": _entropy(np.abs(derivatives)) if len(derivatives) else None,
    }


def _window_features(time_values: np.ndarray, signal_values: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    hours = time_values / 60.0
    for label, start_hour, end_hour in WINDOW_BOUNDS_HOURS:
        mask = (hours >= start_hour) & (hours <= end_hour)
        window_time = time_values[mask]
        window_signal = signal_values[mask]
        prefix = f"window_{label}"
        if len(window_signal) == 0:
            for stat in ("mean", "median", "maximum", "minimum", "variance", "slope", "auc", "standard_deviation"):
                result[f"{prefix}_{stat}"] = None
            continue
        result[f"{prefix}_mean"] = float(np.mean(window_signal))
        result[f"{prefix}_median"] = float(np.median(window_signal))
        result[f"{prefix}_maximum"] = float(np.max(window_signal))
        result[f"{prefix}_minimum"] = float(np.min(window_signal))
        result[f"{prefix}_variance"] = float(np.var(window_signal))
        result[f"{prefix}_standard_deviation"] = float(np.std(window_signal))
        result[f"{prefix}_slope"] = _slope(window_time, window_signal)
        result[f"{prefix}_auc"] = _auc(window_time, window_signal)
    return result


def _shape_features(time_values: np.ndarray, signal_values: np.ndarray, baseline: float) -> dict[str, float | int | None]:
    centered = signal_values - baseline
    mean = float(np.mean(signal_values))
    std = float(np.std(signal_values))
    half = len(signal_values) // 2
    early_auc = _auc(time_values[: max(2, half)], signal_values[: max(2, half)]) if len(signal_values) >= 2 else None
    late_auc = _auc(time_values[-max(2, len(signal_values) - half):], signal_values[-max(2, len(signal_values) - half):]) if len(signal_values) >= 2 else None
    total_auc = _auc(time_values, signal_values)
    return {
        "shape_skewness": _skewness(signal_values),
        "shape_kurtosis": _kurtosis(signal_values),
        "shape_entropy": _entropy(signal_values),
        "shape_signal_energy": float(np.sum(signal_values ** 2)),
        "shape_roughness": float(np.sum(np.abs(np.diff(signal_values)))) if len(signal_values) >= 2 else None,
        "shape_symmetry": _symmetry(early_auc, late_auc, total_auc),
        "shape_peak_count": _peak_count(signal_values),
        "shape_zero_crossings": _zero_crossings(centered),
        "shape_coefficient_of_variation": (std / abs(mean)) if mean != 0 else None,
    }


def _frequency_features(time_values: np.ndarray, signal_values: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "frequency_dominant_frequency": None,
        "frequency_spectral_entropy": None,
        "frequency_spectral_energy": None,
    }
    for index in range(1, 6):
        result[f"frequency_fft_coefficient_{index}"] = 0.0
    if len(signal_values) < 2:
        return result
    centered = signal_values - np.mean(signal_values)
    deltas = np.diff(time_values)
    sample_spacing = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 1.0
    fft_values = np.fft.rfft(centered)
    magnitudes = np.abs(fft_values)
    power = magnitudes ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=sample_spacing)
    if len(magnitudes) > 1:
        dominant_index = int(np.argmax(magnitudes[1:]) + 1)
        result["frequency_dominant_frequency"] = float(frequencies[dominant_index])
    result["frequency_spectral_entropy"] = _probability_entropy(power)
    result["frequency_spectral_energy"] = float(np.sum(power))
    for index in range(1, 6):
        if index < len(magnitudes):
            result[f"frequency_fft_coefficient_{index}"] = float(magnitudes[index])
    return result


def _response_dynamics(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    baseline: float,
    peak: float,
    minimum: float,
    endpoint: float,
    duration: float,
) -> dict[str, float | None]:
    dynamic_range = peak - minimum
    positive_threshold = baseline + 0.1 * dynamic_range
    negative_threshold = baseline - 0.1 * dynamic_range
    outside = (signal_values > positive_threshold) | (signal_values < negative_threshold)
    late_start = time_values[0] + 0.75 * duration if duration > 0 else time_values[-1]
    late_signal = signal_values[time_values >= late_start]
    return {
        "response_induction_delay": _first_threshold_time(
            time_values,
            signal_values,
            positive_threshold,
            direction="above",
            fallback=float(duration),
        ),
        "response_inhibition_delay": _first_threshold_time(
            time_values,
            signal_values,
            negative_threshold,
            direction="below",
            fallback=float(duration),
        ),
        "response_duration": _duration_masked(time_values, outside),
        "response_recovery_fraction": ((peak - endpoint) / (peak - baseline)) if peak != baseline else None,
        "response_sustained_response_score": (
            float(np.mean(np.abs(late_signal - baseline)) / dynamic_range)
            if len(late_signal) and dynamic_range != 0
            else None
        ),
    }


def _baseline_features(time_values: np.ndarray, signal_values: np.ndarray) -> dict[str, float | None]:
    count = max(2, int(math.ceil(len(signal_values) * 0.1)))
    baseline_time = time_values[:count]
    baseline_signal = signal_values[:count]
    mean = float(np.mean(baseline_signal)) if len(baseline_signal) else 0.0
    std = float(np.std(baseline_signal)) if len(baseline_signal) else 0.0
    cv = std / abs(mean) if mean != 0 else None
    return {
        "baseline_stability": (1.0 / (1.0 + cv)) if cv is not None else None,
        "baseline_noise": std,
        "baseline_drift": _slope(baseline_time, baseline_signal),
    }


def _normalized_features(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    baseline: float,
    peak: float,
    minimum: float,
    endpoint: float,
    duration: float,
) -> dict[str, float | None]:
    auc = _auc(time_values, signal_values)
    centered = signal_values - baseline
    positive_area = _auc(time_values, np.maximum(centered, 0.0))
    total_abs_area = _auc(time_values, np.abs(centered))
    std = float(np.std(signal_values))
    z_signal = (signal_values - np.mean(signal_values)) / std if std > 0 else np.zeros_like(signal_values)
    return {
        "normalized_peak_over_baseline": (peak / baseline) if baseline != 0 else None,
        "normalized_endpoint_over_baseline": (endpoint / baseline) if baseline != 0 else None,
        "normalized_auc_over_baseline_duration": (auc / (baseline * duration)) if baseline != 0 and duration > 0 and auc is not None else None,
        "normalized_dynamic_range_over_baseline": ((peak - minimum) / baseline) if baseline != 0 else None,
        "normalized_positive_area_over_total_area": (positive_area / total_abs_area) if total_abs_area not in (None, 0) else None,
        "normalized_signal_zscore_auc": _auc(time_values, z_signal),
    }


def _add_strain_interaction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy(deep=True)
    group_columns = [
        column
        for column in ("Experiment_ID", "Source_File", "Chemical", "Concentration", "Replicate_ID", "Duration")
        if column in result.columns
    ]
    values = pd.to_numeric(result["_response_magnitude_for_strain_interaction"], errors="coerce")
    result["_strain_response_value"] = values
    if not group_columns:
        result["strain_interaction_difference"] = pd.NA
        result["strain_interaction_ratio"] = pd.NA
        result["strain_interaction_mean"] = pd.NA
        result["strain_interaction_variance"] = pd.NA
    else:
        grouped = result.groupby(group_columns, dropna=False)["_strain_response_value"]
        mean = grouped.transform("mean")
        variance = grouped.transform(lambda series: float(np.nanvar(series.to_numpy(dtype=float))))
        result["strain_interaction_difference"] = result["_strain_response_value"] - mean
        result["strain_interaction_ratio"] = result["_strain_response_value"] / mean.replace(0, np.nan)
        result["strain_interaction_mean"] = mean
        result["strain_interaction_variance"] = variance
    return result.drop(columns=["_response_magnitude_for_strain_interaction", "_strain_response_value"], errors="ignore")


def _metadata_row(series: pd.DataFrame, flags: list[str]) -> dict[str, Any]:
    return {
        "Experiment_ID": _single_value(series, "Experiment_ID"),
        "Measurement_Unit_ID": _single_value(series, "Measurement_Unit_ID"),
        "Source_File": _single_value(series, "Source_File"),
        "Strain": _single_value(series, "Strain_Original"),
        "Chemical": _single_value(series, "Chemical_Name_Original"),
        "Concentration": _single_value(series, "Concentration_Label"),
        "Replicate_ID": _single_value(series, "Replicate_ID"),
        "Duration": _series_duration(series),
        "QC_Status": "fail" if "no_valid_observations" in flags else "warning" if flags else "pass",
        "Advanced_Feature_QC_Flags": _join_flags(flags),
    }


def _dataset(
    dataframe: pd.DataFrame,
    warnings: list[str],
    errors: list[str],
    started: float,
    *,
    input_rows: int,
    usable_rows: int,
) -> AdvancedFeatureDataset:
    definitions = feature_dictionary()
    dictionary = pd.DataFrame([definition.__dict__ for definition in definitions])
    grouped = feature_columns_by_family()
    feature_columns = [definition.feature_name for definition in definitions]
    missing_count = int(dataframe.loc[:, [c for c in feature_columns if c in dataframe.columns]].isna().sum().sum()) if not dataframe.empty else 0
    summary = {
        "feature_engine_v2_version": FEATURE_ENGINE_V2_VERSION,
        "input_canonical_rows": int(input_rows),
        "usable_canonical_rows": int(usable_rows),
        "advanced_feature_rows": int(len(dataframe)),
        "advanced_feature_count": int(len(feature_columns)),
        "feature_family_count": int(len(FEATURE_FAMILIES)),
        "missing_advanced_feature_values": missing_count,
        "runtime_seconds": float(timer.perf_counter() - started),
        "existing_feature_engine_replaced": False,
    }
    metadata = {
        "stage": "8C",
        "input_contract": "validated canonical dataset",
        "raw_readers_used": False,
        "qc_bypassed": False,
        "complete_time_series_required": True,
        "feature_engine_v2_version": FEATURE_ENGINE_V2_VERSION,
    }
    return AdvancedFeatureDataset(
        dataframe=dataframe.copy(deep=True),
        feature_dictionary=dictionary,
        feature_columns_by_family=grouped,
        metadata=metadata,
        summary=summary,
        warnings=warnings,
        errors=errors,
    )


def _validate_canonical_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {', '.join(missing)}")


def _usable_canonical_rows(canonical: pd.DataFrame) -> pd.DataFrame:
    record_valid = canonical["Record_Valid"].astype("boolean").ne(False).fillna(True)
    status = canonical["QC_Status"].astype("string").fillna("")
    return canonical.loc[record_valid & ~status.eq("fail")].copy(deep=True)


def _numeric_series(series: pd.DataFrame) -> pd.DataFrame:
    numeric = series.loc[:, ["Time_Minutes", "Luminescence_Raw", "Source_Row_ID"]].copy()
    numeric["Time_Minutes"] = pd.to_numeric(numeric["Time_Minutes"], errors="coerce")
    numeric["Luminescence_Raw"] = pd.to_numeric(numeric["Luminescence_Raw"], errors="coerce")
    numeric["Source_Row_ID"] = pd.to_numeric(numeric["Source_Row_ID"], errors="coerce")
    finite = np.isfinite(numeric["Time_Minutes"].to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
        numeric["Luminescence_Raw"].to_numpy(dtype=float, na_value=np.nan)
    )
    return numeric.loc[finite].copy()


def _order_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [definition.feature_name for definition in feature_dictionary()]
    columns = [*METADATA_COLUMNS, *feature_columns]
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    return dataframe.loc[:, columns].reset_index(drop=True)


def _empty_dataframe() -> pd.DataFrame:
    feature_columns = [definition.feature_name for definition in feature_dictionary()]
    return pd.DataFrame(columns=[*METADATA_COLUMNS, *feature_columns])


def _derivatives(time_values: np.ndarray, signal_values: np.ndarray, flags: list[str]) -> np.ndarray:
    if len(time_values) < 2:
        return np.asarray([], dtype=float)
    deltas = np.diff(time_values)
    if np.any(deltas <= 0):
        flags.append("nonpositive_time_delta")
        valid = deltas > 0
        return np.diff(signal_values)[valid] / deltas[valid] if np.any(valid) else np.asarray([], dtype=float)
    return np.diff(signal_values) / deltas


def _auc(time_values: np.ndarray, signal_values: np.ndarray) -> float | None:
    if len(time_values) < 2:
        return None
    return float(np.trapezoid(signal_values, time_values))


def _slope(time_values: np.ndarray, signal_values: np.ndarray) -> float | None:
    if len(time_values) < 2:
        return None
    delta = float(time_values[-1] - time_values[0])
    if delta == 0:
        return None
    return float((signal_values[-1] - signal_values[0]) / delta)


def _first_threshold_time(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    threshold: float,
    *,
    direction: str,
    fallback: float | None = None,
) -> float | None:
    if direction == "above":
        mask = signal_values >= threshold
    else:
        mask = signal_values <= threshold
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return fallback
    return float(time_values[int(indices[0])])


def _recovery_time(time_values: np.ndarray, signal_values: np.ndarray, baseline: float, peak_index: int) -> float | None:
    tolerance = abs(float(np.max(signal_values)) - baseline) * 0.1
    if tolerance == 0:
        return None
    post_peak = np.arange(len(signal_values)) > peak_index
    recovered = post_peak & (np.abs(signal_values - baseline) <= tolerance)
    indices = np.flatnonzero(recovered)
    if len(indices) == 0:
        return float(time_values[-1] - time_values[peak_index])
    return float(time_values[int(indices[0])] - time_values[peak_index])


def _duration_masked(time_values: np.ndarray, mask: np.ndarray) -> float | None:
    if len(time_values) < 2 or not np.any(mask):
        return 0.0
    total = 0.0
    for index in range(len(time_values) - 1):
        if mask[index] or mask[index + 1]:
            total += float(time_values[index + 1] - time_values[index])
    return total


def _entropy(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    if np.all(finite == finite[0]):
        return 0.0
    counts, _ = np.histogram(finite, bins=min(10, max(2, len(finite))))
    return _probability_entropy(counts.astype(float))


def _probability_entropy(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _skewness(values: np.ndarray) -> float | None:
    std = float(np.std(values))
    if std == 0:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 3))


def _kurtosis(values: np.ndarray) -> float | None:
    std = float(np.std(values))
    if std == 0:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 4) - 3.0)


def _symmetry(early_auc: float | None, late_auc: float | None, total_auc: float | None) -> float | None:
    if early_auc is None or late_auc is None or total_auc in (None, 0):
        return None
    return float(1.0 - abs(early_auc - late_auc) / abs(total_auc))


def _peak_count(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    return int(sum(1 for i in range(1, len(values) - 1) if values[i] > values[i - 1] and values[i] >= values[i + 1]))


def _zero_crossings(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    signs = np.sign(values)
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _single_value(series: pd.DataFrame, column: str) -> Any:
    values = series[column].dropna().unique().tolist() if column in series.columns else []
    if len(values) > 1:
        return ";".join(str(value) for value in values)
    if not values:
        return pd.NA
    value = values[0]
    return value.item() if hasattr(value, "item") else value


def _series_duration(series: pd.DataFrame) -> float | None:
    if "Time_Minutes" not in series.columns:
        return None
    values = pd.to_numeric(series["Time_Minutes"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max() - values.min())


def _join_flags(flags: list[str]) -> str:
    counts = Counter(flags)
    return ";".join(sorted(counts))
