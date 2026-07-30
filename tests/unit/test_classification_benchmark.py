import math

import pandas as pd
import pytest

from src.classification_benchmark import (
    BenchmarkConfig,
    ClassificationBenchmarkResult,
    available_model_specs,
    make_validation_splits,
    prepare_classification_data,
    rank_models,
    required_model_ids,
    run_classification_benchmark,
)
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS


def test_required_classifiers_are_available() -> None:
    available, skipped = available_model_specs(model_ids=required_model_ids())

    assert {spec.model_id for spec in available} == set(required_model_ids())
    assert skipped == []


def test_benchmark_evaluates_required_models_and_selects_best_deterministically() -> None:
    dataframe = _fingerprint_dataframe()
    config = BenchmarkConfig(
        validation_strategy="stratified_kfold",
        n_splits=3,
        model_ids=required_model_ids(),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    first = run_classification_benchmark(dataframe, config=config)
    second = run_classification_benchmark(dataframe, config=config)

    assert isinstance(first, ClassificationBenchmarkResult)
    assert set(first.summary["model_id"]) == set(required_model_ids())
    assert first.rankings.iloc[0]["rank"] == 1
    assert first.best_model_metrics["model_id"] == first.rankings.iloc[0]["model_id"]
    assert first.summary["f1_macro_mean"].between(0, 1).all()
    assert "f1_macro_ci95_low" in first.summary.columns
    assert first.errors == []
    pd.testing.assert_series_equal(
        first.rankings["model_id"],
        second.rankings["model_id"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        first.summary.sort_values("model_id")["f1_macro_mean"].reset_index(drop=True),
        second.summary.sort_values("model_id")["f1_macro_mean"].reset_index(drop=True),
        check_names=False,
    )


def test_preprocessing_is_pipeline_based_and_input_is_not_mutated() -> None:
    dataframe = _fingerprint_dataframe()
    before = dataframe.copy(deep=True)
    config = BenchmarkConfig(
        validation_strategy="stratified_kfold",
        preprocessing="robust",
        n_splits=2,
        model_ids=("logistic_regression",),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    result = run_classification_benchmark(dataframe, config=config)

    pd.testing.assert_frame_equal(dataframe, before)
    assert result.metadata["uses_sklearn_pipelines"] is True
    assert result.metadata["full_dataset_scaled_before_splitting"] is False
    assert result.metadata["preprocessing"] == "robust"


def test_validation_split_strategies_are_supported() -> None:
    prepared = prepare_classification_data(
        _fingerprint_dataframe(),
        feature_names=list(CORE_FEATURE_COLUMNS),
    )

    train_test, train_test_meta, _ = make_validation_splits(
        prepared,
        validation_strategy="train_test",
        test_size=0.25,
    )
    kfold, kfold_meta, _ = make_validation_splits(
        prepared,
        validation_strategy="stratified_kfold",
        n_splits=3,
    )
    repeated, repeated_meta, _ = make_validation_splits(
        prepared,
        validation_strategy="repeated_stratified_kfold",
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

    assert len(train_test) == 1
    assert train_test_meta["effective_n_splits"] == 1
    assert len(kfold) == 3
    assert kfold_meta["effective_n_splits"] == 3
    assert len(repeated) == 6
    assert repeated_meta["fold_count"] == 6
    assert len(loso) == 2
    assert loso_meta["held_out_group_column"] == "Strain"
    assert len(loco) == 3
    assert loco_meta["research_mode"] == "leave_one_chemical_out"
    assert any("research mode" in warning for warning in loco_warnings)


def test_leave_one_chemical_out_research_mode_runs() -> None:
    config = BenchmarkConfig(
        validation_strategy="leave_one_chemical_out",
        model_ids=("logistic_regression",),
        run_permutation_importance=False,
        run_leave_one_strain_importance=False,
    )

    result = run_classification_benchmark(_fingerprint_dataframe(), config=config)

    assert result.metadata["research_mode"] == "leave_one_chemical_out"
    assert result.metadata["fold_count"] == 3
    assert result.summary.iloc[0]["model_id"] == "logistic_regression"


def test_unusable_rows_are_excluded_with_warning() -> None:
    dataframe = _fingerprint_dataframe()
    dataframe.loc[0, "baseline"] = math.nan
    dataframe.loc[1, "Chemical"] = pd.NA

    prepared = prepare_classification_data(
        dataframe,
        feature_names=list(CORE_FEATURE_COLUMNS),
    )

    assert prepared.metadata["excluded_row_count"] == 2
    assert prepared.metadata["sample_count"] == len(dataframe) - 2
    assert any("Rows excluded" in warning for warning in prepared.warnings)


def test_rank_models_uses_macro_f1_then_balanced_accuracy_then_accuracy() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_id": "a",
                "model_name": "A",
                "f1_macro_mean": 0.8,
                "balanced_accuracy_mean": 0.7,
                "accuracy_mean": 0.95,
            },
            {
                "model_id": "b",
                "model_name": "B",
                "f1_macro_mean": 0.8,
                "balanced_accuracy_mean": 0.9,
                "accuracy_mean": 0.80,
            },
            {
                "model_id": "c",
                "model_name": "C",
                "f1_macro_mean": 0.7,
                "balanced_accuracy_mean": 1.0,
                "accuracy_mean": 1.0,
            },
        ]
    )

    ranked = rank_models(summary)

    assert ranked.iloc[0]["model_id"] == "b"
    assert ranked.iloc[1]["model_id"] == "a"
    assert ranked.iloc[2]["model_id"] == "c"


def test_output_files_are_written(tmp_path) -> None:
    config = BenchmarkConfig(
        validation_strategy="stratified_kfold",
        n_splits=2,
        model_ids=("random_forest",),
        permutation_repeats=2,
        run_leave_one_strain_importance=True,
    )
    result = run_classification_benchmark(_fingerprint_dataframe(), config=config)

    paths = result.write_outputs(tmp_path)
    names = {path.name for path in paths}

    assert "classification_summary.csv" in names
    assert "best_model_metrics.json" in names
    assert "confusion_matrix.csv" in names
    assert "per_class_metrics.csv" in names
    assert "feature_importance.csv" in names
    assert "permutation_importance.csv" in names
    assert "model_rankings.csv" in names
    assert "classification_report.md" in names
    assert "leave_one_strain_importance.csv" in names
    assert not result.feature_importance.empty
    assert not result.permutation_importance.empty


def _fingerprint_dataframe() -> pd.DataFrame:
    rows = []
    for chemical_index, chemical in enumerate(["Chem-A", "Chem-B", "Chem-C"]):
        for strain_index, strain in enumerate(["BL011", "BL032"]):
            for replicate in range(1, 7):
                rows.append(_fingerprint_row(chemical_index, chemical, strain_index, strain, replicate))
    return pd.DataFrame(rows)


def _fingerprint_row(
    chemical_index: int,
    chemical: str,
    strain_index: int,
    strain: str,
    replicate: int,
) -> dict[str, object]:
    baseline = 10.0 + chemical_index * 20.0 + strain_index + replicate * 0.1
    peak = baseline + 5.0 + chemical_index * 2.0
    minimum = baseline - 1.0 - chemical_index
    endpoint = baseline + 2.5 + chemical_index
    dynamic_range = peak - minimum
    return {
        "Fingerprint_ID": f"EXP-1::synthetic.csv::{chemical}-{strain}-{replicate}",
        "Experiment_ID": "EXP-1",
        "Measurement_Unit_ID": f"{chemical}-{strain}-{replicate}",
        "Source_File": "synthetic.csv",
        "Strain": strain,
        "Chemical": chemical,
        "Concentration": "10 ug/mL",
        "Replicate_ID": str(replicate),
        "Duration": 10.0,
        "QC_Status": "pass",
        "Feature_QC_Flags": "",
        "baseline": baseline,
        "peak": peak,
        "minimum": minimum,
        "endpoint": endpoint,
        "dynamic_range": dynamic_range,
        "time_to_peak": 5.0 + chemical_index,
        "auc": 120.0 + chemical_index * 100.0 + strain_index * 5.0 + replicate,
        "initial_slope": 1.0 + chemical_index * 0.5 + strain_index * 0.1,
        "maximum_slope": 2.0 + chemical_index * 0.6 + strain_index * 0.1,
        "fold_change": (peak - baseline) / baseline,
        "log2_fold_change": math.log2(endpoint / baseline),
    }
