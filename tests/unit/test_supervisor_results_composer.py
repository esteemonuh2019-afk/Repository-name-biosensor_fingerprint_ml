import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.supervisor_report.authoritative_source_loader import load_selected_sources
from src.supervisor_report.results_composer import (
    build_supervisor_results_package,
    render_markdown,
    validate_package,
    write_supervisor_results_package,
)


PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def create_project(tmp_path, selected_has_median=True):
    root = tmp_path / "project"
    output = root / "outputs"
    (output / "results_inventory_2").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "LIMITATIONS_AND_RISKS.md").write_text("limited dataset", encoding="utf-8")

    selected_rows = [
        {
            "report_section": "Dataset summary",
            "analysis_type": "feature extraction",
            "selected_file": "features/feature_dataset.csv",
            "selected_run": "features",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "features/feature_summary.json",
            "scientific_role": "dataset summary",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "QC",
            "analysis_type": "canonical QC; feature validation",
            "selected_file": "qc/stage/qc_summary.json",
            "selected_run": "qc/stage",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "",
            "scientific_role": "canonical QC",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Fingerprint",
            "analysis_type": "fingerprint generation",
            "selected_file": "fingerprints/fingerprint_dataset.csv",
            "selected_run": "fingerprints",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "fingerprints/fingerprint_summary.json",
            "scientific_role": "fingerprint QC",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Exploratory",
            "analysis_type": "exploratory analysis",
            "selected_file": "exploratory/pca_explained_variance.csv",
            "selected_run": "exploratory",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "exploratory/cluster_assignments.csv; exploratory/consensus_fingerprint_heatmap.png; exploratory/chemical_similarity_heatmap.png",
            "scientific_role": "PCA",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Classification",
            "analysis_type": "classification",
            "selected_file": "classification/best_model_metrics.json",
            "selected_run": "classification",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "classification/model_rankings.csv; classification/per_class_metrics.csv; classification/confusion_matrix.csv",
            "scientific_role": "classification performance",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Regression",
            "analysis_type": "regression",
            "selected_file": "regression/best_regression_model.json",
            "selected_run": "regression",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "regression/model_rankings.csv; regression/prediction_vs_actual.csv; regression/residuals.csv",
            "scientific_role": "regression performance",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Feature engineering",
            "analysis_type": "advanced feature engineering",
            "selected_file": "feature_engineering/advanced_feature_dataset.csv",
            "selected_run": "feature_engineering",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "feature_engineering/feature_family_ablation_summary.csv; feature_engineering/stage_8c_summary.json",
            "scientific_role": "feature-family benchmark",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Feature selection",
            "analysis_type": "feature selection",
            "selected_file": "feature_selection/classification_after_selection.csv",
            "selected_run": "feature_selection",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "feature_selection/feature_selection_summary.csv; feature_selection/selected_features.csv; feature_selection/performance_vs_feature_count.csv",
            "scientific_role": "feature selection",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Strains",
            "analysis_type": "strain ablation",
            "selected_file": "figures/chemical_specific_strain_heatmap.png",
            "selected_run": "strain",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "tables/leave_one_strain_out_loeo.csv; tables/chemical_specific_strain_rankings.csv; tables/single_strain_loeo.csv",
            "scientific_role": "strain ablation",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Docs",
            "analysis_type": "documentation",
            "selected_file": "docs/LIMITATIONS_AND_RISKS.md",
            "selected_run": "docs",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "",
            "scientific_role": "documentation",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
        {
            "report_section": "Blind",
            "analysis_type": "real blind validation",
            "selected_file": "blind/input.csv",
            "selected_run": "blind",
            "status": "FOUND",
            "selection_reason": "",
            "companion_files": "blind/blind_prediction_summary.json",
            "scientific_role": "blind prediction",
            "include_in_supervisor_report": "True",
            "notes": "",
        },
    ]
    selected_path = output / "results_inventory_2" / "selected_results.csv"
    write_csv(selected_path, selected_rows)
    write_csv(output / "features" / "feature_dataset.csv", [{"id": "1", "feature": "2"}])
    write_json(
        output / "features" / "feature_summary.json",
        {
            "summary": {
                "feature_rows": 10,
                "core_feature_count": 2,
                "core_features": ["baseline", "peak"],
                "input_canonical_rows": 100,
            },
            "qc": {
                "passed": False,
                "failed_feature_rows": 1,
                "warning_feature_rows": 2,
                "missing_feature_value_count": 3,
            },
        },
    )
    write_json(
        output / "qc" / "stage" / "qc_summary.json",
        {
            "qc_passed": True,
            "row_count": 100,
            "errors": [],
            "warnings": ["warning"],
            "chemicals_detected": ["Boric Acid", "DEET"],
            "strains_detected": ["BL011", "BL032"],
            "concentrations_detected": ["0.5", "Control"],
            "source_files": ["input.csv"],
        },
    )
    write_csv(output / "fingerprints" / "fingerprint_dataset.csv", [{"id": "1"}])
    write_json(
        output / "fingerprints" / "fingerprint_summary.json",
        {
            "summary": {
                "fingerprint_rows": 10,
                "consensus_fingerprint_rows": 4,
                "feature_count": 2,
                "normalization_method": "zscore",
                "distance_matrix_rows": 4,
                "distance_matrix_columns": 4,
                "duplicate_fingerprint_row_count": 1,
                "excluded_rows": 1,
            },
            "qc": {"passed": True, "excluded_rows": 1, "warnings": []},
        },
    )
    write_csv(
        output / "exploratory" / "pca_explained_variance.csv",
        [
            {"component": "PC1", "explained_variance_ratio": "0.50", "cumulative_explained_variance_ratio": "0.50"},
            {"component": "PC2", "explained_variance_ratio": "0.25", "cumulative_explained_variance_ratio": "0.75"},
            {"component": "PC3", "explained_variance_ratio": "0.10", "cumulative_explained_variance_ratio": "0.85"},
        ],
    )
    write_csv(output / "exploratory" / "cluster_assignments.csv", [{"id": "a", "cluster_id": "1"}, {"id": "b", "cluster_id": "2"}])
    (output / "exploratory" / "consensus_fingerprint_heatmap.png").write_bytes(PNG_BYTES)
    (output / "exploratory" / "chemical_similarity_heatmap.png").write_bytes(PNG_BYTES)
    write_json(
        output / "classification" / "best_model_metrics.json",
        {
            "model_id": "extra_trees",
            "model_name": "Extra Trees",
            "rank": 1,
            "selection_metric": "f1_macro_mean",
            "f1_macro_mean": 0.71,
            "balanced_accuracy_mean": 0.72,
            "accuracy_mean": 0.73,
            "sample_count": 90,
        },
    )
    write_csv(
        output / "classification" / "model_rankings.csv",
        [
            {"rank": "1", "model_name": "Extra Trees", "f1_macro_mean": "0.71", "balanced_accuracy_mean": "0.72", "fold_count": "3"},
            {"rank": "2", "model_name": "Random Forest", "f1_macro_mean": "0.69", "balanced_accuracy_mean": "0.70", "fold_count": "3"},
        ],
    )
    write_csv(output / "classification" / "per_class_metrics.csv", [{"chemical": "Boric Acid", "precision": "0.8", "recall": "0.7", "f1": "0.75", "support": "5"}])
    write_csv(output / "classification" / "confusion_matrix.csv", [{"label": "Boric Acid", "Boric Acid": "5"}])
    best_regression = {
        "model_id": "extra_trees",
        "model_name": "Extra Trees Regressor",
        "rank": 1,
        "selection_metric": "r2_mean",
        "r2_mean": 0.31,
        "rmse_mean": 11.0,
        "mae_mean": 5.0,
        "explained_variance_mean": 0.32,
        "sample_count": 80,
        "concentration_min": 0.5,
        "concentration_max": 50,
        "target_units": "ug/mL",
    }
    if selected_has_median:
        best_regression["median_absolute_error_mean"] = 4.0
    write_json(output / "regression" / "best_regression_model.json", best_regression)
    selected_regression_row = {
        "rank": "1",
        "model_name": "Extra Trees Regressor",
        "r2_mean": "0.31",
        "rmse_mean": "11.0",
        "mae_mean": "5.0",
        "fold_count": "3",
    }
    if selected_has_median:
        selected_regression_row["median_absolute_error_mean"] = "4.0"
    write_csv(
        output / "regression" / "model_rankings.csv",
        [
            selected_regression_row,
            {"rank": "2", "model_name": "XGBoost Regressor", "r2_mean": "0.22", "rmse_mean": "12.0", "mae_mean": "6.0", "median_absolute_error_mean": "9.9", "fold_count": "3"},
        ],
    )
    write_csv(output / "regression" / "prediction_vs_actual.csv", [{"actual": "1", "predicted": "1.1"}])
    write_csv(output / "regression" / "residuals.csv", [{"residual": "0.1"}])
    write_csv(output / "feature_engineering" / "advanced_feature_dataset.csv", [{"id": "1"}])
    write_json(
        output / "feature_engineering" / "stage_8c_summary.json",
        {"best_feature_family": "window_features", "classification_improvement": 0.1, "regression_improvement": 0.2, "runtime_increase_seconds": 3, "metadata": {"feature_family_count": 2}},
    )
    write_csv(output / "feature_engineering" / "feature_family_ablation_summary.csv", [{"feature_family": "window_features", "classification_macro_f1": "0.8", "regression_r2": "0.3"}])
    write_csv(output / "feature_selection" / "classification_after_selection.csv", [{"id": "1"}])
    write_csv(output / "feature_selection" / "feature_selection_summary.csv", [{"task": "classification", "model_name": "Extra Trees", "macro_f1_mean": "0.75", "r2_mean": "", "recommended_default": "True"}])
    write_csv(output / "feature_selection" / "selected_features.csv", [{"task": "classification", "feature_name": "baseline"}])
    write_csv(output / "feature_selection" / "performance_vs_feature_count.csv", [{"feature_count": "1", "score": "0.75"}])
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "chemical_specific_strain_heatmap.png").write_bytes(PNG_BYTES)
    write_csv(output / "tables" / "leave_one_strain_out_loeo.csv", [{"removed_strain": "BL011", "macro_f1": "0.5"}])
    write_csv(output / "tables" / "chemical_specific_strain_rankings.csv", [{"chemical": "Boric Acid", "strain": "BL011", "f1": "0.9"}])
    write_csv(output / "tables" / "single_strain_loeo.csv", [{"strain": "BL011", "macro_f1": "0.4"}])
    write_csv(output / "blind" / "input.csv", [{"x": "1"}])
    write_json(output / "blind" / "blind_prediction_summary.json", {"true_labels_included": False, "prediction_passed": False, "predicted_chemical": "Boric Acid"})
    return root, selected_path


@pytest.fixture
def synthetic_package(tmp_path):
    root, selected = create_project(tmp_path)
    return build_supervisor_results_package(root, selected)


def test_loader_reads_selected_and_companion_sources(tmp_path):
    root, selected = create_project(tmp_path)
    sources = load_selected_sources(root, selected)
    assert any(source.source_file == "classification/best_model_metrics.json" for source in sources)
    assert any(source.source_file == "classification/model_rankings.csv" for source in sources)


def test_loader_resolves_project_root_documentation_paths(tmp_path):
    root, selected = create_project(tmp_path)
    docs = [source for source in load_selected_sources(root, selected) if source.analysis_type == "documentation"]
    assert docs[0].exists is True
    assert Path(docs[0].resolved_path) == root / "docs" / "LIMITATIONS_AND_RISKS.md"


def test_package_schema_has_required_top_level_fields(synthetic_package):
    required = {
        "project_summary",
        "dataset_summary",
        "quality_control_summary",
        "fingerprint_summary",
        "exploratory_results",
        "classification_results",
        "regression_results",
        "selected_tables",
        "provenance",
    }
    assert required.issubset(synthetic_package.to_dict())


def test_authoritative_classifier_from_best_json(synthetic_package):
    assert synthetic_package.classification_results["selected_model"]["model_name"] == "Extra Trees"


def test_classifier_primary_macro_f1_from_best_json(synthetic_package):
    macro = next(row for row in synthetic_package.classification_results["selected_metrics"] if row["metric_name"] == "f1_macro_mean")
    assert macro["metric_value"] == 0.71
    assert macro["model_name"] == "Extra Trees"
    assert macro["source_file"] == "classification/best_model_metrics.json"


def test_classifier_comparison_keeps_other_models_separate(synthetic_package):
    comparison = synthetic_package.classification_results["model_comparison"]
    assert {row["selection_status"] for row in comparison} == {"SELECTED", "COMPARISON"}
    assert any(row["model_name"] == "Random Forest" for row in comparison)


def test_authoritative_regressor_from_best_json(synthetic_package):
    assert synthetic_package.regression_results["selected_model"]["model_name"] == "Extra Trees Regressor"


def test_regression_primary_metrics_from_selected_model(synthetic_package):
    populated = [row for row in synthetic_package.regression_results["selected_metrics"] if row["status"] == "SUPPORTED"]
    assert {row["model_name"] for row in populated} == {"Extra Trees Regressor"}


def test_regression_does_not_substitute_other_model_metrics(tmp_path):
    root, selected = create_project(tmp_path, selected_has_median=False)
    package = build_supervisor_results_package(root, selected)
    median = next(row for row in package.regression_results["selected_metrics"] if row["metric_name"] == "median_absolute_error_mean")
    assert median["status"] == "MISSING"
    assert median["metric_value"] is None


def test_missing_selected_regression_metric_is_marked_missing(tmp_path):
    root, selected = create_project(tmp_path, selected_has_median=False)
    package = build_supervisor_results_package(root, selected)
    missing = [row for row in package.regression_results["selected_metrics"] if row["status"] == "MISSING"]
    assert any(row["metric_name"] == "median_absolute_error_mean" for row in missing)


def test_blind_prediction_not_claimed_as_validation(synthetic_package):
    assert synthetic_package.project_summary["blind_prediction_context"]["true_labels_included"] is False
    assert validate_package(synthetic_package)["checks"][6]["passed"] is True


def test_figure_selector_copies_only_inventory_listed_images(tmp_path):
    root, selected = create_project(tmp_path)
    package = build_supervisor_results_package(root, selected)
    output_dir = root / "outputs" / "supervisor_results"
    write_supervisor_results_package(package, output_dir, overwrite=True)
    figures = {row["figure_id"] for row in package.selected_figures}
    assert "consensus_fingerprint_heatmap" in figures
    assert (output_dir / "figures" / "consensus_fingerprint_heatmap.png").exists()


def test_provenance_references_metric_sources(synthetic_package):
    metric_records = [row for row in synthetic_package.provenance if row["record_type"] == "metric" and row["status"] == "SUPPORTED"]
    assert metric_records
    assert all(row["source_file"] for row in metric_records)


def test_validation_fails_on_tampered_model_mixing(synthetic_package):
    metric = next(row for row in synthetic_package.regression_results["selected_metrics"] if row["metric_name"] == "rmse_mean")
    metric["model_name"] = "XGBoost Regressor"
    validation = validate_package(synthetic_package)
    assert validation["passed"] is False
    assert any(check["check"] == "regression_primary_model_coherent" and not check["passed"] for check in validation["checks"])


def test_selected_tables_include_required_titles(synthetic_package):
    table_ids = {table["table_id"] for table in synthetic_package.selected_tables}
    assert {"classifier_comparison", "selected_classifier_metrics", "regressor_comparison", "selected_regressor_metrics"}.issubset(table_ids)


def test_writer_creates_required_files(tmp_path):
    root, selected = create_project(tmp_path)
    package = build_supervisor_results_package(root, selected)
    output_dir = root / "outputs" / "supervisor_results"
    result = write_supervisor_results_package(package, output_dir, overwrite=True)
    assert result["package_passed"] is True
    for name in [
        "supervisor_results_summary.json",
        "supervisor_results_tables.xlsx",
        "supervisor_results_report.md",
        "supervisor_results_report.docx",
        "supervisor_results_report.pdf",
        "selected_figures.csv",
        "selected_tables.csv",
        "provenance_index.csv",
        "report_validation.json",
    ]:
        assert (output_dir / name).exists()


def test_writer_keeps_input_outputs_unmodified(tmp_path):
    root, selected = create_project(tmp_path)
    best_path = root / "outputs" / "regression" / "best_regression_model.json"
    before = hashlib.sha256(best_path.read_bytes()).hexdigest()
    package = build_supervisor_results_package(root, selected)
    write_supervisor_results_package(package, root / "outputs" / "supervisor_results", overwrite=True)
    after = hashlib.sha256(best_path.read_bytes()).hexdigest()
    assert before == after


def test_missing_files_produce_warnings(tmp_path):
    root, selected = create_project(tmp_path)
    (root / "docs" / "LIMITATIONS_AND_RISKS.md").unlink()
    package = build_supervisor_results_package(root, selected)
    assert any("Listed source is unavailable" in warning for warning in package.warnings)


def test_unsupported_quantitative_claims_are_rejected(synthetic_package):
    synthetic_package.provenance.append(
        {
            "record_type": "metric",
            "status": "SUPPORTED",
            "source_file": None,
            "metric_name": "unsupported_metric",
            "metric_value": 1,
        }
    )
    validation = validate_package(synthetic_package)
    assert validation["passed"] is False
    assert any(check["check"] == "no_unsupported_quantitative_claims" and not check["passed"] for check in validation["checks"])


def test_existing_output_directory_is_protected(tmp_path):
    root, selected = create_project(tmp_path)
    package = build_supervisor_results_package(root, selected)
    output_dir = root / "outputs" / "supervisor_results"
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_supervisor_results_package(package, output_dir, overwrite=False)


def test_report_rendering_is_deterministic_for_same_package(synthetic_package):
    assert render_markdown(synthetic_package) == render_markdown(synthetic_package)


def test_validation_failure_blocks_docx_and_pdf(tmp_path):
    root, selected = create_project(tmp_path)
    package = build_supervisor_results_package(root, selected)
    metric = next(row for row in package.classification_results["selected_metrics"] if row["metric_name"] == "f1_macro_mean")
    metric["model_name"] = "Other Model"
    output_dir = root / "outputs" / "supervisor_results"
    result = write_supervisor_results_package(package, output_dir, overwrite=True)
    assert result["package_passed"] is False
    assert result["report_docx"] is None
    assert result["report_pdf"] is None
    assert not (output_dir / "supervisor_results_report.docx").exists()
    assert not (output_dir / "supervisor_results_report.pdf").exists()
