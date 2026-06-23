from src.model_evaluation.evaluate import (
    compare_train_test_performance,
    evaluate_classification,
    evaluate_regression,
)


def test_classification_metrics_are_calculated() -> None:
    metrics = evaluate_classification(
        y_true=["A", "A", "B", "B"],
        y_pred=["A", "B", "B", "B"],
    )

    assert set(metrics) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "confusion_matrix",
    }
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] > 0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_regression_metrics_are_calculated() -> None:
    metrics = evaluate_regression(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.0, 2.5],
    )

    assert set(metrics) == {"r2", "rmse", "mae"}
    assert metrics["r2"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_overfitting_warning_triggers_correctly() -> None:
    result = compare_train_test_performance(train_metric=0.95, test_metric=0.80)

    assert result == {"overfit_warning": True}


def test_overfitting_warning_does_not_trigger_when_difference_is_acceptable() -> None:
    result = compare_train_test_performance(train_metric=0.85, test_metric=0.78)

    assert result == {"overfit_warning": False}
