from src.validation.scientific_validation import (
    validate_cluster_separation,
    validate_concentration_dependence,
    validate_fingerprint_distinctiveness,
    validate_reproducibility,
)


def test_cluster_separation_passes_with_multiple_clusters() -> None:
    result = validate_cluster_separation(["A", "A", "B", "C"])

    assert result.validation_type == "cluster_separation"
    assert result.passed is True
    assert result.metrics["cluster_count"] == 3


def test_cluster_separation_fails_with_one_cluster() -> None:
    result = validate_cluster_separation(["A", "A", "A"])

    assert result.validation_type == "cluster_separation"
    assert result.passed is False
    assert result.metrics["cluster_count"] == 1


def test_concentration_dependence_passes_with_monotonic_response() -> None:
    result = validate_concentration_dependence(
        concentrations=[0.05, 0.5, 5.0, 50.0, 500.0],
        response_values=[1.0, 2.0, 3.0, 4.0, 5.0],
    )

    assert result.validation_type == "concentration_dependence"
    assert result.passed is True
    assert result.metrics["rho"] == 1.0


def test_concentration_dependence_fails_with_flat_response() -> None:
    result = validate_concentration_dependence(
        concentrations=[0.05, 0.5, 5.0, 50.0, 500.0],
        response_values=[2.0, 2.0, 2.0, 2.0, 2.0],
    )

    assert result.validation_type == "concentration_dependence"
    assert result.passed is False
    assert result.metrics["rho"] == 0.0


def test_reproducibility_passes_with_correlated_replicates() -> None:
    result = validate_reproducibility(
        replicate_matrix=[
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            [0.9, 1.9, 2.9, 3.9],
        ],
    )

    assert result.validation_type == "reproducibility"
    assert result.passed is True
    assert result.metrics["average_correlation"] >= 0.7


def test_reproducibility_fails_with_uncorrelated_replicates() -> None:
    result = validate_reproducibility(
        replicate_matrix=[
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 4.0, 1.0, 3.0],
        ],
    )

    assert result.validation_type == "reproducibility"
    assert result.passed is False
    assert result.metrics["average_correlation"] < 0.7


def test_fingerprint_distinctiveness_passes_with_nonzero_distances() -> None:
    result = validate_fingerprint_distinctiveness(
        fingerprint_matrix=[
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 5.0],
        ],
    )

    assert result.validation_type == "fingerprint_distinctiveness"
    assert result.passed is True
    assert result.metrics["mean_distance"] > 0.0
