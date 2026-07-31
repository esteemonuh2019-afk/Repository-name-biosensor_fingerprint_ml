import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def create_minimal_project(tmp_path):
    root = tmp_path / "project"
    output = root / "outputs"
    rows = [
        ["Dataset", "feature extraction", "features/feature_dataset.csv", "features", "features/feature_summary.json", "dataset summary"],
        ["QC", "canonical QC; feature validation", "qc/qc_summary.json", "qc", "", "canonical QC"],
        ["Fingerprint", "fingerprint generation", "fingerprints/fingerprint_dataset.csv", "fingerprints", "fingerprints/fingerprint_summary.json", "fingerprint QC"],
        ["Exploratory", "exploratory analysis", "exploratory/pca_explained_variance.csv", "exploratory", "exploratory/cluster_assignments.csv; exploratory/consensus_fingerprint_heatmap.png", "PCA"],
        ["Classification", "classification", "classification/best_model_metrics.json", "classification", "classification/model_rankings.csv; classification/per_class_metrics.csv; classification/confusion_matrix.csv", "classification performance"],
        ["Regression", "regression", "regression/best_regression_model.json", "regression", "regression/model_rankings.csv; regression/prediction_vs_actual.csv; regression/residuals.csv", "regression performance"],
        ["Feature engineering", "advanced feature engineering", "feature_engineering/advanced_feature_dataset.csv", "feature_engineering", "feature_engineering/feature_family_ablation_summary.csv; feature_engineering/stage_8c_summary.json", "feature-family benchmark"],
        ["Feature selection", "feature selection", "feature_selection/classification_after_selection.csv", "feature_selection", "feature_selection/feature_selection_summary.csv; feature_selection/selected_features.csv; feature_selection/performance_vs_feature_count.csv", "feature selection"],
        ["Strain", "strain ablation", "figures/chemical_specific_strain_heatmap.png", "strain", "tables/leave_one_strain_out_loeo.csv; tables/chemical_specific_strain_rankings.csv; tables/single_strain_loeo.csv", "strain ablation"],
        ["Blind", "real blind validation", "blind/input.csv", "blind", "blind/blind_prediction_summary.json", "blind prediction"],
    ]
    selected = output / "results_inventory_2" / "selected_results.csv"
    selected.parent.mkdir(parents=True)
    with selected.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "report_section",
                "analysis_type",
                "selected_file",
                "selected_run",
                "status",
                "selection_reason",
                "companion_files",
                "scientific_role",
                "include_in_supervisor_report",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "report_section": row[0],
                    "analysis_type": row[1],
                    "selected_file": row[2],
                    "selected_run": row[3],
                    "status": "FOUND",
                    "selection_reason": "",
                    "companion_files": row[4],
                    "scientific_role": row[5],
                    "include_in_supervisor_report": "True",
                    "notes": "",
                }
            )
    write_csv(output / "features" / "feature_dataset.csv", [{"id": "1"}])
    write_json(output / "features" / "feature_summary.json", {"summary": {"feature_rows": 8, "core_feature_count": 2, "input_canonical_rows": 80, "core_features": ["baseline", "peak"]}, "qc": {"passed": True, "failed_feature_rows": 0, "warning_feature_rows": 0, "missing_feature_value_count": 0}})
    write_json(output / "qc" / "qc_summary.json", {"qc_passed": True, "row_count": 80, "errors": [], "warnings": [], "chemicals_detected": ["Boric Acid"], "strains_detected": ["BL011"], "concentrations_detected": ["0.5"], "source_files": ["input.csv"]})
    write_csv(output / "fingerprints" / "fingerprint_dataset.csv", [{"id": "1"}])
    write_json(output / "fingerprints" / "fingerprint_summary.json", {"summary": {"fingerprint_rows": 8, "consensus_fingerprint_rows": 2, "feature_count": 2, "normalization_method": "zscore", "distance_matrix_rows": 2, "distance_matrix_columns": 2, "duplicate_fingerprint_row_count": 0, "excluded_rows": 0}, "qc": {"passed": True, "excluded_rows": 0, "warnings": []}})
    write_csv(output / "exploratory" / "pca_explained_variance.csv", [{"component": "PC1", "explained_variance_ratio": "0.6", "cumulative_explained_variance_ratio": "0.6"}, {"component": "PC2", "explained_variance_ratio": "0.2", "cumulative_explained_variance_ratio": "0.8"}, {"component": "PC3", "explained_variance_ratio": "0.1", "cumulative_explained_variance_ratio": "0.9"}])
    write_csv(output / "exploratory" / "cluster_assignments.csv", [{"id": "a", "cluster_id": "1"}])
    (output / "exploratory" / "consensus_fingerprint_heatmap.png").write_bytes(PNG_BYTES)
    write_json(output / "classification" / "best_model_metrics.json", {"model_name": "Extra Trees", "model_id": "extra_trees", "rank": 1, "selection_metric": "f1_macro_mean", "f1_macro_mean": 0.7, "balanced_accuracy_mean": 0.71, "accuracy_mean": 0.72, "sample_count": 8})
    write_csv(output / "classification" / "model_rankings.csv", [{"rank": "1", "model_name": "Extra Trees", "f1_macro_mean": "0.7", "balanced_accuracy_mean": "0.71", "fold_count": "3"}, {"rank": "2", "model_name": "Other", "f1_macro_mean": "0.5", "balanced_accuracy_mean": "0.6", "fold_count": "3"}])
    write_csv(output / "classification" / "per_class_metrics.csv", [{"chemical": "Boric Acid", "precision": "1", "recall": "1", "f1": "1", "support": "1"}])
    write_csv(output / "classification" / "confusion_matrix.csv", [{"label": "Boric Acid", "Boric Acid": "1"}])
    write_json(output / "regression" / "best_regression_model.json", {"model_name": "Extra Trees Regressor", "model_id": "extra_trees", "rank": 1, "selection_metric": "r2_mean", "r2_mean": 0.3, "rmse_mean": 10, "mae_mean": 5, "median_absolute_error_mean": 4, "explained_variance_mean": 0.31, "sample_count": 8, "concentration_min": 0.5, "concentration_max": 50, "target_units": "ug/mL"})
    write_csv(output / "regression" / "model_rankings.csv", [{"rank": "1", "model_name": "Extra Trees Regressor", "r2_mean": "0.3", "rmse_mean": "10", "mae_mean": "5", "fold_count": "3"}, {"rank": "2", "model_name": "XGBoost Regressor", "r2_mean": "0.2", "rmse_mean": "20", "mae_mean": "6", "fold_count": "3"}])
    write_csv(output / "regression" / "prediction_vs_actual.csv", [{"actual": "1", "predicted": "1"}])
    write_csv(output / "regression" / "residuals.csv", [{"residual": "0"}])
    write_csv(output / "feature_engineering" / "advanced_feature_dataset.csv", [{"id": "1"}])
    write_json(output / "feature_engineering" / "stage_8c_summary.json", {"best_feature_family": "window_features", "classification_improvement": 0.1, "regression_improvement": 0.1, "runtime_increase_seconds": 1, "metadata": {"feature_family_count": 1}})
    write_csv(output / "feature_engineering" / "feature_family_ablation_summary.csv", [{"feature_family": "window_features", "classification_macro_f1": "0.8"}])
    write_csv(output / "feature_selection" / "classification_after_selection.csv", [{"id": "1"}])
    write_csv(output / "feature_selection" / "feature_selection_summary.csv", [{"task": "classification", "model_name": "Extra Trees", "macro_f1_mean": "0.8", "r2_mean": "", "recommended_default": "True"}])
    write_csv(output / "feature_selection" / "selected_features.csv", [{"feature_name": "baseline"}])
    write_csv(output / "feature_selection" / "performance_vs_feature_count.csv", [{"feature_count": "1", "score": "0.8"}])
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "chemical_specific_strain_heatmap.png").write_bytes(PNG_BYTES)
    write_csv(output / "tables" / "leave_one_strain_out_loeo.csv", [{"removed_strain": "BL011", "macro_f1": "0.5"}])
    write_csv(output / "tables" / "chemical_specific_strain_rankings.csv", [{"chemical": "Boric Acid", "strain": "BL011", "f1": "1"}])
    write_csv(output / "tables" / "single_strain_loeo.csv", [{"strain": "BL011", "macro_f1": "1"}])
    write_csv(output / "blind" / "input.csv", [{"id": "1"}])
    write_json(output / "blind" / "blind_prediction_summary.json", {"true_labels_included": False, "prediction_passed": False})
    return root, selected


def test_supervisor_results_pipeline_cli(tmp_path):
    root, selected = create_minimal_project(tmp_path)
    best_path = root / "outputs" / "regression" / "best_regression_model.json"
    before = hashlib.sha256(best_path.read_bytes()).hexdigest()
    output_dir = root / "outputs" / "supervisor_results"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_supervisor_results_package.py"),
            "--project-root",
            str(root),
            "--selected-results",
            str(selected.relative_to(root)),
            "--output-dir",
            str(output_dir.relative_to(root)),
            "--overwrite",
            "--title",
            "Synthetic Supervisor Package",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert hashlib.sha256(best_path.read_bytes()).hexdigest() == before
    summary = json.loads((output_dir / "supervisor_results_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "report_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert summary["classification_results"]["selected_model"]["model_name"] == "Extra Trees"
    assert summary["regression_results"]["selected_model"]["model_name"] == "Extra Trees Regressor"
    assert all((output_dir / name).exists() for name in [
        "supervisor_results_report.md",
        "supervisor_results_report.docx",
        "supervisor_results_report.pdf",
        "supervisor_results_tables.xlsx",
        "provenance_index.csv",
    ])
