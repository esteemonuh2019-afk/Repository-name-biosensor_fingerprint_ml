import csv
import json
from pathlib import Path

import pytest

from src.scientific_reasoning.observation_engine import (
    build_observation_database,
    render_markdown,
    write_observation_outputs,
)
from src.scientific_reasoning.observation_registry import CATEGORIES


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def touch_output(root: Path, relative_path: str) -> None:
    path = root / "outputs" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("source", encoding="utf-8")


def create_validated_project(tmp_path: Path, *, remove_metric: bool = False) -> Path:
    root = tmp_path / "project"
    source_files = [
        "qc/qc_summary.json",
        "features/feature_summary.json",
        "fingerprints/fingerprint_summary.json",
        "exploratory/pca_explained_variance.csv",
        "exploratory/cluster_assignments.csv",
        "classification/best_model_metrics.json",
        "regression/best_regression_model.json",
        "feature_engineering/stage_8c_summary.json",
        "feature_engineering/feature_family_ablation_summary.csv",
        "feature_selection/feature_selection_summary.csv",
        "feature_selection/selected_features.csv",
        "tables/leave_one_strain_out_loeo.csv",
        "tables/chemical_specific_strain_rankings.csv",
        "tables/single_strain_loeo.csv",
        "blind/blind_prediction_summary.json",
    ]
    for source_file in source_files:
        touch_output(root, source_file)

    classification_metrics = [
        metric("f1_macro_mean", 0.71, source="classification/best_model_metrics.json"),
        metric("balanced_accuracy_mean", 0.72, source="classification/best_model_metrics.json"),
        metric("accuracy_mean", 0.73, source="classification/best_model_metrics.json"),
        metric("precision_macro_mean", 0.74, source="classification/best_model_metrics.json"),
        metric("recall_macro_mean", 0.75, source="classification/best_model_metrics.json"),
        metric("roc_auc_ovr_weighted_mean", 0.76, source="classification/best_model_metrics.json"),
        metric("sample_count", 90, source="classification/best_model_metrics.json"),
        metric("fold_count", 3, source="classification/best_model_metrics.json"),
    ]
    if remove_metric:
        classification_metrics = [row for row in classification_metrics if row["metric_name"] != "accuracy_mean"]

    summary = {
        "package_passed": True,
        "project_summary": {
            "blind_prediction_context": {
                "prediction_passed": False,
                "true_labels_included": False,
                "predicted_chemical": "Boric Acid",
                "chemical_confidence": 0.21,
                "predicted_concentration": 346.4,
                "concentration_units": "ug/mL",
                "novelty_status": "Out of Distribution",
                "source_file": "blind/blind_prediction_summary.json",
            }
        },
        "dataset_summary": {
            "input_canonical_rows": 100,
            "feature_rows": 10,
            "core_feature_count": 2,
            "chemical_count": 3,
            "strain_count": 2,
            "feature_summary_source": "features/feature_summary.json",
            "canonical_qc_source": "qc/qc_summary.json",
        },
        "quality_control_summary": {
            "canonical_qc_passed": True,
            "canonical_error_count": 0,
            "canonical_warning_count": 1,
            "feature_qc_passed": True,
            "feature_failed_rows": 0,
            "fingerprint_qc_passed": True,
            "canonical_qc_source": "qc/qc_summary.json",
            "feature_summary_source": "features/feature_summary.json",
            "fingerprint_summary_source": "fingerprints/fingerprint_summary.json",
        },
        "fingerprint_summary": {
            "fingerprint_rows": 10,
            "consensus_fingerprint_rows": 4,
            "feature_count": 2,
            "normalization_method": "zscore",
            "distance_matrix_rows": 4,
            "distance_matrix_columns": 4,
            "source_file": "fingerprints/fingerprint_summary.json",
        },
        "exploratory_results": {
            "cumulative_variance_pc3": 0.85,
            "cluster_count": 2,
            "cluster_assignment_rows": 4,
            "pca_source": "exploratory/pca_explained_variance.csv",
            "cluster_source": "exploratory/cluster_assignments.csv",
        },
        "classification_results": {
            "selected_model": {
                "model_name": "Extra Trees",
                "rank": 1,
                "selection_metric": "f1_macro_mean",
                "source_file": "classification/best_model_metrics.json",
            },
            "selected_metrics": classification_metrics,
        },
        "regression_results": {
            "selected_model": {
                "model_name": "Extra Trees Regressor",
                "rank": 1,
                "selection_metric": "r2_mean",
                "source_file": "regression/best_regression_model.json",
            },
            "selected_metrics": [
                metric("r2_mean", 0.31, source="regression/best_regression_model.json"),
                metric("rmse_mean", 11.0, "ug/mL", "regression/best_regression_model.json"),
                metric("mae_mean", 5.0, "ug/mL", "regression/best_regression_model.json"),
                metric("median_absolute_error_mean", 4.0, "ug/mL", "regression/best_regression_model.json"),
                metric("explained_variance_mean", 0.32, source="regression/best_regression_model.json"),
                metric("sample_count", 80, source="regression/best_regression_model.json"),
                metric("fold_count", 3, source="regression/best_regression_model.json"),
                metric("concentration_min", 0.5, "ug/mL", "regression/best_regression_model.json"),
                metric("concentration_max", 50, "ug/mL", "regression/best_regression_model.json"),
            ],
        },
        "feature_engineering_results": {
            "best_feature_family": "window_features",
            "classification_improvement": 0.1,
            "regression_improvement": 0.2,
            "runtime_increase_seconds": 3.0,
            "feature_family_count": 2,
            "summary_source": "feature_engineering/stage_8c_summary.json",
            "benchmark_source": "feature_engineering/feature_family_ablation_summary.csv",
        },
        "feature_selection_results": {
            "summary_rows": [{"task": "classification"}],
            "selected_feature_count": 4,
            "recommended_defaults": [
                {"task": "classification", "macro_f1_mean": 0.8, "feature_count": 4, "selector_method": "rfe"},
                {"task": "regression", "r2_mean": 0.3, "feature_count": 4, "selector_method": "rfe"},
            ],
            "summary_source": "feature_selection/feature_selection_summary.csv",
            "selected_features_source": "feature_selection/selected_features.csv",
        },
        "strain_results": {
            "leave_one_strain_count": 2,
            "chemical_specific_count": 3,
            "single_strain_count": 2,
            "loeo_source": "tables/leave_one_strain_out_loeo.csv",
            "chemical_source": "tables/chemical_specific_strain_rankings.csv",
            "single_source": "tables/single_strain_loeo.csv",
        },
    }
    write_json(root / "outputs" / "supervisor_results" / "supervisor_results_summary.json", summary)
    write_json(root / "outputs" / "supervisor_results" / "report_validation.json", {"passed": True, "checks": []})
    return root


def metric(name, value, units=None, source=None):
    return {
        "metric_name": name,
        "metric_value": value,
        "metric_units": units,
        "source_file": source,
        "status": "SUPPORTED",
    }


def test_builds_observations_for_all_supported_categories(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    database = build_observation_database(root)
    assert [observation.category for observation in database.observations] == list(CATEGORIES)
    assert database.complete_count == len(CATEGORIES)


def test_observation_ids_are_stable(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    observations = build_observation_database(root).observations
    assert observations[0].id == "OBS-0001"
    assert observations[-1].id == "OBS-0010"


def test_classification_observation_uses_authoritative_source_and_metrics(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    observation = next(item for item in build_observation_database(root).observations if item.category == "Classification")
    metrics = {row["metric_name"]: row["metric_value"] for row in observation.supporting_metrics}
    assert "classification/best_model_metrics.json" in observation.supporting_files
    assert "Extra Trees" in observation.statement
    assert metrics["accuracy_mean"] == 0.73


def test_regression_observation_preserves_metric_units(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    observation = next(item for item in build_observation_database(root).observations if item.category == "Regression")
    rmse = next(row for row in observation.supporting_metrics if row["metric_name"] == "rmse_mean")
    assert rmse["metric_value"] == 11.0
    assert rmse["metric_units"] == "ug/mL"


def test_missing_validation_returns_incomplete_observations(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    (root / "outputs" / "supervisor_results" / "report_validation.json").unlink()
    database = build_observation_database(root)
    assert database.incomplete_count == len(CATEGORIES)
    assert database.errors


def test_failed_validation_skips_extraction(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    write_json(root / "outputs" / "supervisor_results" / "report_validation.json", {"passed": False})
    database = build_observation_database(root)
    assert database.incomplete_count == len(CATEGORIES)
    assert database.metadata["validated_input_used"] is False


def test_missing_supporting_file_marks_observation_incomplete(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    (root / "outputs" / "classification" / "best_model_metrics.json").unlink()
    database = build_observation_database(root)
    observation = next(item for item in database.observations if item.category == "Classification")
    assert observation.status == "INCOMPLETE"
    assert database.warnings


def test_missing_metric_marks_observation_incomplete(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path, remove_metric=True)
    observation = next(item for item in build_observation_database(root).observations if item.category == "Classification")
    assert observation.status == "INCOMPLETE"
    assert "accuracy_mean" in observation.notes


def test_write_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    database = build_observation_database(root)
    output_dir = root / "outputs" / "scientific_observations"
    paths = write_observation_outputs(database, output_dir, overwrite=True)
    assert [path.name for path in paths] == ["observations.json", "observations.csv", "observations.md"]
    assert all(path.exists() for path in paths)
    with (output_dir / "observations.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "OBS-0001"


def test_existing_outputs_are_protected_without_overwrite(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    output_dir = root / "outputs" / "scientific_observations"
    database = build_observation_database(root)
    write_observation_outputs(database, output_dir, overwrite=True)
    with pytest.raises(FileExistsError):
        write_observation_outputs(database, output_dir, overwrite=False)


def test_markdown_is_grouped_by_category(tmp_path: Path) -> None:
    root = create_validated_project(tmp_path)
    markdown = render_markdown(build_observation_database(root))
    assert "## QC" in markdown
    assert "## Classification" in markdown
    assert "Observation:" in markdown
    assert "Status: COMPLETE" in markdown
