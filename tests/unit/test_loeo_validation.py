import pandas as pd

from src.model_evaluation.loeo_validation import (
    run_loeo_classification,
    run_loeo_regression,
)


CLASSIFICATION_METRICS = {"accuracy", "macro_precision", "macro_recall", "macro_f1"}
REGRESSION_METRICS = {"r2", "rmse", "mae"}


def test_loeo_classification_runs() -> None:
    result = run_loeo_classification(_feature_dataframe())

    assert result["task_type"] == "classification"
    assert len(result["per_experiment"]) == 3
    assert set(result["mean_metrics"]) == CLASSIFICATION_METRICS


def test_loeo_regression_runs() -> None:
    result = run_loeo_regression(_feature_dataframe())

    assert result["task_type"] == "regression"
    assert len(result["per_experiment"]) == 3
    assert set(result["mean_metrics"]) == REGRESSION_METRICS


def test_no_experiment_appears_in_both_train_and_test() -> None:
    result = run_loeo_classification(_feature_dataframe())

    for fold_result in result["per_experiment"]:
        assert set(fold_result["train_experiments"]).isdisjoint(
            fold_result["test_experiments"]
        )


def test_output_structure_contains_required_metrics() -> None:
    classification_result = run_loeo_classification(_feature_dataframe())
    regression_result = run_loeo_regression(_feature_dataframe())

    for fold_result in classification_result["per_experiment"]:
        assert set(fold_result["metrics"]) == CLASSIFICATION_METRICS
        assert "held_out_experiment" in fold_result
        assert "test_rows" in fold_result

    for fold_result in regression_result["per_experiment"]:
        assert set(fold_result["metrics"]) == REGRESSION_METRICS
        assert "held_out_experiment" in fold_result
        assert "test_rows" in fold_result


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for experiment in ("EXP-001", "EXP-002", "EXP-003"):
        rows.extend(
            [
                _feature_row(experiment, "Diazinon", 5.0, 5000.0, 1000.0),
                _feature_row(experiment, "Diazinon", 50.0, 7000.0, 1300.0),
                _feature_row(experiment, "DEET", 5.0, 3500.0, 800.0),
                _feature_row(experiment, "DEET", 50.0, 5200.0, 1100.0),
            ]
        )
    return pd.DataFrame(rows)


def _feature_row(
    experiment: str,
    chemical: str,
    concentration: float,
    auc: float,
    max_signal: float,
) -> dict[str, float | str]:
    experiment_offset = {"EXP-001": 0.0, "EXP-002": 25.0, "EXP-003": -25.0}[experiment]
    return {
        "strain": "BL011",
        "chemical": chemical,
        "concentration": concentration,
        "experiment": experiment,
        "replicate": 1,
        "auc": auc + experiment_offset,
        "max_signal": max_signal + experiment_offset,
        "min_signal": 500.0 + experiment_offset,
        "time_to_peak": 5.0 if chemical == "Diazinon" else 10.0,
        "initial_slope": 40.0 if chemical == "Diazinon" else 20.0,
        "final_signal": max_signal - 100.0 + experiment_offset,
    }
