from src.validation.ml_validation import (
    validate_classifier,
    validate_regressor,
    verify_experiment_level_split,
)


def test_classifier_passes_when_metrics_exceed_thresholds() -> None:
    result = validate_classifier(
        y_true=["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        y_pred=["A", "A", "B", "B", "C", "C", "D", "D", "E", "A"],
    )

    assert result.task_type == "classifier"
    assert result.passed is True
    assert result.threshold_results == {"accuracy": True, "macro_f1": True}
    assert result.metrics["accuracy"] == 0.9
    assert result.metrics["macro_f1"] >= 0.75


def test_classifier_fails_when_metrics_are_below_thresholds() -> None:
    result = validate_classifier(
        y_true=["A", "A", "B", "B", "C", "C"],
        y_pred=["B", "B", "C", "C", "A", "A"],
    )

    assert result.task_type == "classifier"
    assert result.passed is False
    assert result.threshold_results == {"accuracy": False, "macro_f1": False}
    assert result.metrics["accuracy"] == 0.0


def test_regressor_passes_when_r2_exceeds_threshold() -> None:
    result = validate_regressor(
        y_true=[1.0, 2.0, 3.0, 4.0, 5.0],
        y_pred=[1.0, 2.1, 2.9, 4.1, 5.0],
    )

    assert result.task_type == "regressor"
    assert result.passed is True
    assert result.threshold_results == {"r2": True}
    assert result.metrics["r2"] >= 0.75
    assert result.metrics["rmse"] >= 0.0
    assert result.metrics["mae"] >= 0.0


def test_regressor_fails_when_r2_is_below_threshold() -> None:
    result = validate_regressor(
        y_true=[1.0, 2.0, 3.0, 4.0, 5.0],
        y_pred=[5.0, 4.0, 3.0, 2.0, 1.0],
    )

    assert result.task_type == "regressor"
    assert result.passed is False
    assert result.threshold_results == {"r2": False}
    assert result.metrics["r2"] < 0.75


def test_experiment_split_passes_with_no_overlap() -> None:
    result = verify_experiment_level_split(
        train_experiments=["EXP-001", "EXP-002"],
        test_experiments=["EXP-003"],
    )

    assert result.task_type == "experiment_split"
    assert result.passed is True
    assert result.threshold_results == {"no_overlap": True}
    assert result.messages == ["No experiment ID overlap detected."]


def test_experiment_split_fails_with_overlap() -> None:
    result = verify_experiment_level_split(
        train_experiments=["EXP-001", "EXP-002"],
        test_experiments=["EXP-002", "EXP-003"],
    )

    assert result.task_type == "experiment_split"
    assert result.passed is False
    assert result.threshold_results == {"no_overlap": False}
    assert "EXP-002" in result.messages[0]
