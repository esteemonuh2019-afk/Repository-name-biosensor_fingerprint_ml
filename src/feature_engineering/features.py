"""Kinetic feature extraction for biosensor time-series data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


GROUP_COLUMNS: tuple[str, ...] = (
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "auc",
    "max_signal",
    "min_signal",
    "time_to_peak",
    "initial_slope",
    "final_signal",
)


def calculate_auc(time: Iterable[float], values: Iterable[float]) -> float:
    """Calculate area under the curve using trapezoidal integration."""

    time_values, signal_values = _paired_numeric_values(time, values)
    return sum(
        (time_values[index + 1] - time_values[index])
        * (signal_values[index] + signal_values[index + 1])
        / 2
        for index in range(len(time_values) - 1)
    )


def calculate_max_signal(values: Iterable[float]) -> float:
    """Return the maximum signal value."""

    signal_values = _numeric_values(values)
    return max(signal_values)


def calculate_min_signal(values: Iterable[float]) -> float:
    """Return the minimum signal value."""

    signal_values = _numeric_values(values)
    return min(signal_values)


def calculate_time_to_peak(time: Iterable[float], values: Iterable[float]) -> float:
    """Return the time point where the maximum signal occurs."""

    time_values, signal_values = _paired_numeric_values(time, values)
    peak_index = signal_values.index(max(signal_values))
    return time_values[peak_index]


def calculate_initial_slope(time: Iterable[float], values: Iterable[float]) -> float:
    """Calculate the slope between the first two time points."""

    time_values, signal_values = _paired_numeric_values(time, values)
    if len(time_values) < 2:
        raise ValueError("At least two time points are required to calculate initial slope.")

    time_delta = time_values[1] - time_values[0]
    if time_delta == 0:
        raise ValueError("Initial slope requires distinct first two time points.")

    return (signal_values[1] - signal_values[0]) / time_delta


def calculate_final_signal(values: Iterable[float]) -> float:
    """Return the final signal value."""

    signal_values = _numeric_values(values)
    return signal_values[-1]


def extract_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract kinetic features grouped by strain, chemical, concentration, experiment, and replicate."""

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
                "auc": calculate_auc(time_values, signal_values),
                "max_signal": calculate_max_signal(signal_values),
                "min_signal": calculate_min_signal(signal_values),
                "time_to_peak": calculate_time_to_peak(time_values, signal_values),
                "initial_slope": calculate_initial_slope(time_values, signal_values),
                "final_signal": calculate_final_signal(signal_values),
            }
        )

    return pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)


def _paired_numeric_values(
    time: Iterable[float],
    values: Iterable[float],
) -> tuple[list[float], list[float]]:
    time_values = _numeric_values(time)
    signal_values = _numeric_values(values)
    if len(time_values) != len(signal_values):
        raise ValueError("time and values must have the same length.")
    return time_values, signal_values


def _numeric_values(values: Iterable[float]) -> list[float]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("values must not be empty.")
    return numeric_values
