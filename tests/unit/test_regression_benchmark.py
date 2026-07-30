import math

import pandas as pd
import pytest

from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.regression_benchmark import (
    RegressionBenchmarkConfig,
    RegressionBenchmarkResult,
    available_regression_model_specs,
    make_validation_splits,
    prepare_regression_data,
    rank_regression_models,
    required_regression_model_ids,
    run_regression_benchmark,
)


def test_required_regressors_are_available() -> None:
    available, skipped = available_regression_model_specs(model_ids=required_regression_model_ids())

    assert {spec.model_id for spec in available} == set(required_regression_model_ids())
    assert skipped == []


def test_benchmark_evaluates_required_regressors_and_selects_best_deterministically() -> None:
    dataframe = _fingerprint_dataframe()
    config = RegressionBenchmarkConfig(
        validation_strategy="repeated_kfold",
        n_splits=3,
        n_repeats=1,
        model_ids=required_regression_model_ids(),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    first = run_regression_benchmark(dataframe, config=config)
    second = run_regression_benchmark(dataframe, config=config)

    assert isinstance(first, RegressionBenchmarkResult)
    assert set(first.summary["model_id"]) == set(required_regression_model_ids())
    assert first.rankings.iloc[0]["rank"] == 1
    assert first.best_model_metrics["model_id"] == first.rankings.iloc[0]["model_id"]
    assert "r2_ci95_low" in first.summary.columns
    assert first.errors == []
    pd.testing.assert_series_equal(
        first.rankings["model_id"],
        second.rankings["model_id"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        first.summary.sort_values("model_id")["r2_mean"].reset_index(drop=True),
        second.summary.sort_values("model_id")["r2_mean"].reset_index(drop=True),
        check_names=False,
    )


def test_preprocessing_is_pipeline_based_and_input_is_not_mutated() -> None:
    dataframe = _fingerprint_dataframe()
    before = dataframe.copy(deep=True)
    config = RegressionBenchmarkConfig(
        validation_strategy="repeated_kfold",
        preprocessing="robust",
        n_splits=2,
        n_repeats=1,
        model_ids=("ridge",),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    result = run_regression_benchmark(dataframe, config=config)

    pd.testing.assert_frame_equal(dataframe, before)
    assert result.metadata["uses_sklearn_pipelines"] is True
    assert result.metadata["full_dataset_scaled_before_splitting"] is False
    assert result.metadata["preprocessing"] == "robust"


def test_validation_split_strategies_are_supported() -> None:
    prepared = prepare_regression_data(
        _fingerprint_dataframe(),
        feature_names=list(CORE_FEATURE_COLUMNS),
    )

    repeated, repeated_meta, _ = make_validation_splits(
        prepared,
        validation_strategy="repeated_kfold",
        n_splits=3,
        n_repeats=2,
    )
    loso, loso_meta, _ = make_validation_splits(
        prepared,
        validation_strategy="leave_one_strain_out",
    )
    loco, loco_meta, loco_warnings = make_validation_splits(
        prepared,
        validation_strategy="leave_one_chemical_out",
    )

    assert len(repeated) == 6
    assert repeated_meta["fold_count"] == 6
    assert repeated_meta["effective_n_splits"] == 3
    assert len(loso) == 2
    assert loso_meta["held_out_group_column"] == "Strain"
    assert len(loco) == 3
    assert loco_meta["research_mode"] == "leave_one_chemical_out"
    assert any("research mode" in warning for warning in loco_warnings)


def test_leave_one_chemical_out_research_mode_runs() -> None:
    config = RegressionBenchmarkConfig(
        validation_strategy="leave_one_chemical_out",
        model_ids=("ridge",),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    result = run_regression_benchmark(_fingerprint_dataframe(), config=config)

    assert result.metadata["research_mode"] == "leave_one_chemical_out"
    assert result.metadata["fold_count"] == 3
    assert result.summary.iloc[0]["model_id"] == "ridge"


def test_numeric_concentration_labels_are_parsed_and_unusable_rows_are_excluded() -> None:
    dataframe = _fingerprint_dataframe()
    dataframe.loc[0, "Concentration"] = "1 mg/L"
    dataframe.loc[1, "Concentration"] = "1000 ng/mL"
    dataframe.loc[2, "Concentration"] = "0.5 ug/mL"
    dataframe.loc[3, "Concentration"] = "Control"
    dataframe.loc[4, "Concentration"] = "3 mM"
    dataframe.loc[5, "baseline"] = math.nan

    prepared = prepare_regression_data(
        dataframe,
        feature_names=list(CORE_FEATURE_COLUMNS),
    )

    assert prepared.metadata["excluded_row_count"] == 3
    assert prepared.metadata["sample_count"] == len(dataframe) - 3
    assert prepared.dataframe.loc[prepared.dataframe["Measurement_Unit_ID"].eq("Chem-A-BL011-1"), "Concentration_Target_ug_mL"].iloc[0] == pytest.approx(1.0)
    assert prepared.dataframe.loc[prepared.dataframe["Measurement_Unit_ID"].eq("Chem-A-BL011-2"), "Concentration_Target_ug_mL"].iloc[0] == pytest.approx(1.0)
    assert prepared.dataframe.loc[prepared.dataframe["Measurement_Unit_ID"].eq("Chem-A-BL011-3"), "Concentration_Target_ug_mL"].iloc[0] == pytest.approx(0.5)
    assert any("unsupported concentration" in warning for warning in prepared.warnings)


def test_zero_concentration_is_retained_and_mape_is_defined_on_nonzero_actuals() -> None:
    dataframe = _fingerprint_dataframe(concentrations=(0.0, 1.0, 10.0))
    config = RegressionBenchmarkConfig(
        validation_strategy="repeated_kfold",
        n_splits=2,
        n_repeats=1,
        model_ids=("ridge",),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    result = run_regression_benchmark(dataframe, config=config)

    assert result.metadata["zero_concentration_count"] > 0
    assert result.summary.iloc[0]["mape_valid_count"] > 0
    assert any("MAPE excludes zero-valued actuals" in warning for warning in result.warnings)


def test_rank_regression_models_uses_r2_then_rmse_then_mae() -> None:
    summary = pd.DataFrame(
        [
            {"model_id": "a", "model_name": "A", "r2_mean": 0.8, "rmse_mean": 2.0, "mae_mean": 1.0},
            {"model_id": "b", "model_name": "B", "r2_mean": 0.8, "rmse_mean": 1.0, "mae_mean": 1.5},
            {"model_id": "c", "model_name": "C", "r2_mean": 0.7, "rmse_mean": 0.5, "mae_mean": 0.5},
        ]
    )

    ranked = rank_regression_models(summary)

    assert ranked.iloc[0]["model_id"] == "b"
    assert ranked.iloc[1]["model_id"] == "a"
    assert ranked.iloc[2]["model_id"] == "c"


def test_output_files_and_figures_are_written(tmp_path) -> None:
    config = RegressionBenchmarkConfig(
        validation_strategy="repeated_kfold",
        n_splits=2,
        n_repeats=1,
        model_ids=("random_forest",),
        permutation_repeats=2,
        run_leave_one_strain_importance=True,
    )
    result = run_regression_benchmark(_fingerprint_dataframe(), config=config)

    paths = result.write_outputs(tmp_path)
    names = {path.name for path in paths}

    assert "regression_summary.csv" in names
    assert "best_regression_model.json" in names
    assert "per_model_metrics.csv" in names
    assert "fold_metrics.csv" in names
    assert "prediction_vs_actual.csv" in names
    assert "residuals.csv" in names
    assert "model_rankings.csv" in names
    assert "regression_report.md" in names
    assert "feature_importance.csv" in names
    assert "permutation_importance.csv" in names
    assert "prediction_vs_actual.png" in names
    assert "prediction_vs_actual.pdf" in names
    assert "residual_plot.png" in names
    assert "residual_histogram.pdf" in names
    assert "fold_performance.png" in names
    assert not result.feature_importance.empty
    assert not result.permutation_importance.empty


def _fingerprint_dataframe(concentrations: tuple[float, ...] = (0.5, 5.0, 50.0, 500.0)) -> pd.DataFrame:
    rows = []
    chemicals = ["Chem-A", "Chem-B", "Chem-C"]
    for chemical_index, chemical in enumerate(chemicals):
        for strain_index, strain in enumerate(["BL011", "BL032"]):
            for concentration in concentrations:
                for replicate in range(1, 4):
                    rows.append(
                        _fingerprint_row(
                            chemical_index=chemical_index,
                            chemical=chemical,
                            strain_index=strain_index,
                            strain=strain,
                            concentration=concentration,
                            replicate=replicate,
                        )
                    )
    return pd.DataFrame(rows)


def _fingerprint_row(
    *,
    chemical_index: int,
    chemical: str,
    strain_index: int,
    strain: str,
    concentration: float,
    replicate: int,
) -> dict[str, object]:
    signal = math.log10(concentration + 1.0) * 20.0 + chemical_index * 4.0 + strain_index * 1.5
    baseline = 10.0 + signal + replicate * 0.05
    peak = baseline + signal * 0.8 + 3.0
    minimum = baseline - 1.0
    endpoint = baseline + signal * 0.4 + 1.0
    dynamic_range = peak - minimum
    return {
        "Fingerprint_ID": f"EXP-1::synthetic.csv::{chemical}-{strain}-{concentration}-{replicate}",
        "Experiment_ID": "EXP-1",
        "Measurement_Unit_ID": f"{chemical}-{strain}-{replicate}",
        "Source_File": "synthetic.csv",
        "Strain": strain,
        "Chemical": chemical,
        "Concentration": f"{concentration:g}",
        "Replicate_ID": str(replicate),
        "Duration": 10.0,
        "QC_Status": "pass",
        "Feature_QC_Flags": "",
        "baseline": baseline,
        "peak": peak,
        "minimum": minimum,
        "endpoint": endpoint,
        "dynamic_range": dynamic_range,
        "time_to_peak": 5.0 + min(concentration, 10.0) * 0.01,
        "auc": 100.0 + signal * 12.0 + replicate,
        "initial_slope": 1.0 + signal * 0.08,
        "maximum_slope": 2.0 + signal * 0.1,
        "fold_change": (peak - baseline) / baseline,
        "log2_fold_change": math.log2(endpoint / baseline),
    }
