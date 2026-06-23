"""Advanced kinetic features for biosensor luminescence time series."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.feature_engineering.features import GROUP_COLUMNS


ADVANCED_FEATURE_COLUMNS: tuple[str, ...] = (
    "peak_to_baseline_ratio",
    "fold_change",
    "max_derivative",
    "min_derivative",
    "signal_decay_rate",
    "auc_early",
    "auc_mid",
    "auc_late",
)

OUTPUT_COLUMNS: tuple[str, ...] = (*GROUP_COLUMNS, *ADVANCED_FEATURE_COLUMNS)


def calculate_peak_to_baseline_ratio(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate max(signal) / signal[0]."""

    _, signal_values = _paired_numeric_values(time, signal)
    baseline = signal_values[0]
    if baseline == 0:
        return float("nan")
    return max(signal_values) / baseline


def calculate_fold_change(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate (signal[-1] - signal[0]) / signal[0]."""

    _, signal_values = _paired_numeric_values(time, signal)
    baseline = signal_values[0]
    if baseline == 0:
        return float("nan")
    return (signal_values[-1] - baseline) / baseline


def calculate_max_derivative(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate the maximum first derivative across adjacent time points."""

    return max(_calculate_derivatives(time, signal))


def calculate_min_derivative(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate the minimum first derivative across adjacent time points."""

    return min(_calculate_derivatives(time, signal))


def calculate_signal_decay_rate(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate the slope from peak signal to final signal."""

    time_values, signal_values = _paired_numeric_values(time, signal)
    peak_index = signal_values.index(max(signal_values))
    final_index = len(signal_values) - 1
    time_delta = time_values[final_index] - time_values[peak_index]
    if peak_index == final_index or time_delta == 0:
        return 0.0
    return (signal_values[final_index] - signal_values[peak_index]) / time_delta


def calculate_auc_early(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate AUC over 0-6 hours."""

    return _calculate_segment_auc(time, signal, "early")


def calculate_auc_mid(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate AUC over 6-12 hours."""

    return _calculate_segment_auc(time, signal, "mid")


def calculate_auc_late(time: Iterable[float], signal: Iterable[float]) -> float:
    """Calculate AUC over 12-24 hours."""

    return _calculate_segment_auc(time, signal, "late")


def extract_advanced_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract advanced kinetic features for each biosensor condition."""

    missing_columns = [
        column for column in (*GROUP_COLUMNS, "time", "luminescence") if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    feature_rows = []
    for group_values, group in dataframe.groupby(list(GROUP_COLUMNS), sort=False):
        sorted_group = group.sort_values("time")
        time_values = sorted_group["time"].tolist()
        signal_values = sorted_group["luminescence"].tolist()

        feature_rows.append(
            {
                **dict(zip(GROUP_COLUMNS, group_values, strict=True)),
                "peak_to_baseline_ratio": calculate_peak_to_baseline_ratio(time_values, signal_values),
                "fold_change": calculate_fold_change(time_values, signal_values),
                "max_derivative": calculate_max_derivative(time_values, signal_values),
                "min_derivative": calculate_min_derivative(time_values, signal_values),
                "signal_decay_rate": calculate_signal_decay_rate(time_values, signal_values),
                "auc_early": calculate_auc_early(time_values, signal_values),
                "auc_mid": calculate_auc_mid(time_values, signal_values),
                "auc_late": calculate_auc_late(time_values, signal_values),
            }
        )

    return pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS)


def _calculate_derivatives(time: Iterable[float], signal: Iterable[float]) -> list[float]:
    time_values, signal_values = _paired_numeric_values(time, signal)
    if len(time_values) < 2:
        raise ValueError("At least two time points are required to calculate derivatives.")

    derivatives = []
    for index in range(len(time_values) - 1):
        time_delta = time_values[index + 1] - time_values[index]
        if time_delta == 0:
            raise ValueError("Derivative calculation requires distinct adjacent time points.")
        derivatives.append((signal_values[index + 1] - signal_values[index]) / time_delta)
    return derivatives


def _calculate_segment_auc(
    time: Iterable[float],
    signal: Iterable[float],
    segment: str,
) -> float:
    time_values, signal_values = _paired_numeric_values(time, signal)
    pairs = sorted(zip(time_values, signal_values, strict=True))
    time_values = [pair[0] for pair in pairs]
    signal_values = [pair[1] for pair in pairs]

    start, end = _segment_bounds(time_values)[segment]
    overlap_start = max(start, time_values[0])
    overlap_end = min(end, time_values[-1])
    if overlap_start >= overlap_end:
        return 0.0

    points = [(overlap_start, _interpolate_signal(time_values, signal_values, overlap_start))]
    points.extend(
        (time_value, signal_value)
        for time_value, signal_value in zip(time_values, signal_values, strict=True)
        if overlap_start < time_value < overlap_end
    )
    points.append((overlap_end, _interpolate_signal(time_values, signal_values, overlap_end)))
    points = sorted(_deduplicate_points(points))

    if len(points) < 2:
        return 0.0

    return sum(
        (points[index + 1][0] - points[index][0])
        * (points[index][1] + points[index + 1][1])
        / 2
        for index in range(len(points) - 1)
    )


def _segment_bounds(time_values: list[float]) -> dict[str, tuple[float, float]]:
    if max(time_values) <= 24:
        return {
            "early": (0.0, 6.0),
            "mid": (6.0, 12.0),
            "late": (12.0, 24.0),
        }
    return {
        "early": (0.0, 360.0),
        "mid": (360.0, 720.0),
        "late": (720.0, 1440.0),
    }


def _interpolate_signal(
    time_values: list[float],
    signal_values: list[float],
    target_time: float,
) -> float:
    if target_time <= time_values[0]:
        return signal_values[0]
    if target_time >= time_values[-1]:
        return signal_values[-1]

    for index in range(len(time_values) - 1):
        left_time = time_values[index]
        right_time = time_values[index + 1]
        if left_time <= target_time <= right_time:
            time_delta = right_time - left_time
            if time_delta == 0:
                return signal_values[index]
            fraction = (target_time - left_time) / time_delta
            return signal_values[index] + fraction * (signal_values[index + 1] - signal_values[index])

    return signal_values[-1]


def _deduplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduplicated = {}
    for time_value, signal_value in points:
        deduplicated[time_value] = signal_value
    return list(deduplicated.items())


def _paired_numeric_values(
    time: Iterable[float],
    signal: Iterable[float],
) -> tuple[list[float], list[float]]:
    time_values = _numeric_values(time)
    signal_values = _numeric_values(signal)
    if len(time_values) != len(signal_values):
        raise ValueError("time and signal must have the same length.")
    return time_values, signal_values


def _numeric_values(values: Iterable[float]) -> list[float]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("values must not be empty.")
    return numeric_values
