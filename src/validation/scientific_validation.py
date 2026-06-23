"""Scientific validation checks for generated biosensor outputs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Any, Iterable, Sequence


CLUSTER_MINIMUM_COUNT = 2
SPEARMAN_ABSOLUTE_RHO_THRESHOLD = 0.5
REPRODUCIBILITY_CORRELATION_THRESHOLD = 0.7
FINGERPRINT_MINIMUM_MEAN_DISTANCE = 0.0


@dataclass(frozen=True)
class ScientificValidationResult:
    validation_type: str
    passed: bool
    metrics: dict[str, float]
    messages: list[str]


def validate_cluster_separation(cluster_labels: Iterable[Any]) -> ScientificValidationResult:
    """Validate that scientific outputs contain at least two clusters."""

    cluster_count = len(set(cluster_labels))
    passed = cluster_count >= CLUSTER_MINIMUM_COUNT
    return ScientificValidationResult(
        validation_type="cluster_separation",
        passed=passed,
        metrics={"cluster_count": float(cluster_count)},
        messages=[f"Detected {cluster_count} unique cluster(s)."],
    )


def validate_concentration_dependence(
    concentrations: Iterable[float],
    response_values: Iterable[float],
) -> ScientificValidationResult:
    """Validate concentration dependence using Spearman rank correlation."""

    concentration_values, response_value_list = _paired_float_values(
        concentrations,
        response_values,
    )
    rho = _spearman_correlation(concentration_values, response_value_list)
    passed = abs(rho) >= SPEARMAN_ABSOLUTE_RHO_THRESHOLD
    return ScientificValidationResult(
        validation_type="concentration_dependence",
        passed=passed,
        metrics={"rho": rho},
        messages=[f"Spearman rho: {rho:.3f}."],
    )


def validate_reproducibility(
    replicate_matrix: Iterable[Iterable[float]],
) -> ScientificValidationResult:
    """Validate replicate reproducibility using average pairwise Pearson correlation."""

    rows = _matrix_rows(replicate_matrix)
    if len(rows) < 2:
        return ScientificValidationResult(
            validation_type="reproducibility",
            passed=False,
            metrics={"average_correlation": 0.0},
            messages=["At least two replicates are required."],
        )

    correlations = [
        _pearson_correlation(first_row, second_row)
        for first_row, second_row in combinations(rows, 2)
    ]
    average_correlation = sum(correlations) / len(correlations)
    passed = average_correlation >= REPRODUCIBILITY_CORRELATION_THRESHOLD
    return ScientificValidationResult(
        validation_type="reproducibility",
        passed=passed,
        metrics={"average_correlation": average_correlation},
        messages=[f"Average pairwise Pearson correlation: {average_correlation:.3f}."],
    )


def validate_fingerprint_distinctiveness(
    fingerprint_matrix: Iterable[Iterable[float]],
) -> ScientificValidationResult:
    """Validate that fingerprint vectors are separated by nonzero Euclidean distance."""

    rows = _matrix_rows(fingerprint_matrix)
    if len(rows) < 2:
        return ScientificValidationResult(
            validation_type="fingerprint_distinctiveness",
            passed=False,
            metrics={"mean_distance": 0.0},
            messages=["At least two fingerprints are required."],
        )

    distances = [
        _euclidean_distance(first_row, second_row)
        for first_row, second_row in combinations(rows, 2)
    ]
    mean_distance = sum(distances) / len(distances)
    passed = mean_distance > FINGERPRINT_MINIMUM_MEAN_DISTANCE
    return ScientificValidationResult(
        validation_type="fingerprint_distinctiveness",
        passed=passed,
        metrics={"mean_distance": mean_distance},
        messages=[f"Mean pairwise Euclidean distance: {mean_distance:.3f}."],
    )


def _paired_float_values(
    first: Iterable[float],
    second: Iterable[float],
) -> tuple[list[float], list[float]]:
    first_values = [float(value) for value in first]
    second_values = [float(value) for value in second]
    if len(first_values) != len(second_values):
        raise ValueError("Input sequences must have the same length.")
    if not first_values:
        raise ValueError("Input sequences must not be empty.")
    return first_values, second_values


def _matrix_rows(matrix: Iterable[Iterable[float]]) -> list[list[float]]:
    rows = [[float(value) for value in row] for row in matrix]
    if not rows:
        raise ValueError("Matrix must not be empty.")

    row_length = len(rows[0])
    if row_length == 0:
        raise ValueError("Matrix rows must not be empty.")
    if any(len(row) != row_length for row in rows):
        raise ValueError("Matrix rows must have the same length.")
    return rows


def _spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    return _pearson_correlation(_ranks(first), _ranks(second))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0

    while index < len(indexed_values):
        tie_end = index
        while (
            tie_end + 1 < len(indexed_values)
            and indexed_values[tie_end + 1][1] == indexed_values[index][1]
        ):
            tie_end += 1

        average_rank = (index + 1 + tie_end + 1) / 2
        for ranked_index in range(index, tie_end + 1):
            original_index = indexed_values[ranked_index][0]
            ranks[original_index] = average_rank

        index = tie_end + 1

    return ranks


def _pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Input sequences must have the same length.")
    if not first:
        raise ValueError("Input sequences must not be empty.")

    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (first_value - first_mean) * (second_value - second_mean)
        for first_value, second_value in zip(first, second)
    )
    first_denominator = sum((value - first_mean) ** 2 for value in first)
    second_denominator = sum((value - second_mean) ** 2 for value in second)
    denominator = sqrt(first_denominator * second_denominator)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _euclidean_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Input sequences must have the same length.")
    return sqrt(sum((first_value - second_value) ** 2 for first_value, second_value in zip(first, second)))
