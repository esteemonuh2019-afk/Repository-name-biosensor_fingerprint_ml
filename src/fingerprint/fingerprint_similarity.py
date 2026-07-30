"""Reusable distance metrics for fingerprint vectors."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


DistanceMetric = Literal["euclidean", "manhattan", "cosine", "correlation"]
DEFAULT_CSV_BYTES_PER_CELL = 24
DEFAULT_LABEL_BYTES_PER_ROW = 128


def euclidean_distance(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Return Euclidean distance between two finite vectors."""

    a, b = _paired_vectors(vector_a, vector_b)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Return Manhattan distance between two finite vectors."""

    a, b = _paired_vectors(vector_a, vector_b)
    return float(np.sum(np.abs(a - b)))


def cosine_distance(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Return cosine distance, defined as 1 minus cosine similarity."""

    a, b = _paired_vectors(vector_a, vector_b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return math.nan
    return float(1.0 - np.dot(a, b) / denominator)


def correlation_distance(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Return correlation distance, defined as 1 minus Pearson correlation."""

    a, b = _paired_vectors(vector_a, vector_b)
    centered_a = a - np.mean(a)
    centered_b = b - np.mean(b)
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0:
        return math.nan
    return float(1.0 - np.dot(centered_a, centered_b) / denominator)


def calculate_distance_matrix(
    dataframe: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    metric: DistanceMetric,
    label_column: str = "Fingerprint_ID",
) -> pd.DataFrame:
    """Calculate a full square distance matrix for a fingerprint dataframe."""

    feature_names = list(feature_names)
    values = _finite_matrix(dataframe, feature_names)
    labels = _labels(dataframe, label_column)
    distances = _distance_chunk(values, values, metric)
    return pd.DataFrame(distances, index=labels, columns=labels)


def write_distance_matrix_csv(
    dataframe: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    metric: DistanceMetric,
    output_path: str | Path,
    label_column: str = "Fingerprint_ID",
    chunk_size: int = 128,
) -> tuple[int, int]:
    """Write a square distance matrix to CSV using row chunks."""

    feature_names = list(feature_names)
    values = _finite_matrix(dataframe, feature_names)
    labels = _labels(dataframe, label_column)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([label_column, *labels])
        for start in range(0, len(values), chunk_size):
            end = min(start + chunk_size, len(values))
            chunk = _distance_chunk(values[start:end], values, metric)
            for row_label, row_values in zip(labels[start:end], chunk, strict=True):
                writer.writerow([row_label, *(_format_distance(value) for value in row_values)])
    return int(len(values)), int(len(values))


def estimate_distance_matrix_size(
    row_count: int,
    *,
    csv_bytes_per_cell: int = DEFAULT_CSV_BYTES_PER_CELL,
    label_bytes_per_row: int = DEFAULT_LABEL_BYTES_PER_ROW,
) -> dict[str, int]:
    """Estimate square distance-matrix size before calculation."""

    rows = int(row_count)
    if rows < 0:
        raise ValueError("row_count must be non-negative.")
    cells = rows * rows
    estimated_memory_bytes = cells * 8
    estimated_csv_bytes = (
        cells * int(csv_bytes_per_cell)
        + rows * int(label_bytes_per_row) * 2
        + max(rows, 1) * 2
    )
    return {
        "rows": rows,
        "columns": rows,
        "cells": cells,
        "estimated_memory_bytes": int(estimated_memory_bytes),
        "estimated_csv_bytes": int(estimated_csv_bytes),
    }


def _distance_chunk(
    left: np.ndarray,
    right: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    if metric == "euclidean":
        left_norm = np.sum(left**2, axis=1)[:, np.newaxis]
        right_norm = np.sum(right**2, axis=1)[np.newaxis, :]
        squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
        return np.sqrt(squared)
    if metric == "manhattan":
        distances = np.zeros((left.shape[0], right.shape[0]), dtype=float)
        for feature_index in range(left.shape[1]):
            distances += np.abs(
                left[:, feature_index][:, np.newaxis]
                - right[:, feature_index][np.newaxis, :]
            )
        return distances
    if metric == "cosine":
        return _cosine_chunk(left, right)
    if metric == "correlation":
        centered_left = left - np.mean(left, axis=1)[:, np.newaxis]
        centered_right = right - np.mean(right, axis=1)[:, np.newaxis]
        return _cosine_chunk(centered_left, centered_right)
    raise ValueError(f"Unsupported distance metric: {metric}")


def _cosine_chunk(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, np.newaxis] * np.linalg.norm(
        right,
        axis=1,
    )[np.newaxis, :]
    similarity = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=similarity, where=denominator != 0)
    return 1.0 - similarity


def _paired_vectors(
    vector_a: Iterable[float],
    vector_b: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(list(vector_a), dtype=float)
    b = np.asarray(list(vector_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("Distance vectors must have the same shape.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Distance vectors must contain only finite numeric values.")
    return a, b


def _finite_matrix(dataframe: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    missing = [feature for feature in feature_names if feature not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing fingerprint feature columns: {', '.join(missing)}")
    numeric = dataframe.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Distance matrix requires finite fingerprint values.")
    return values


def _labels(dataframe: pd.DataFrame, label_column: str) -> list[str]:
    if label_column in dataframe.columns:
        return dataframe[label_column].astype(str).tolist()
    return [str(index) for index in dataframe.index.tolist()]


def _format_distance(value: float) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.12g}"
