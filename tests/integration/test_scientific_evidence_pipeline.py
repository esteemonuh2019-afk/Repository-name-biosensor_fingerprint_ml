from __future__ import annotations

import csv
import json
from pathlib import Path

from src.scientific_narrative import build_scientific_evidence, write_evidence_outputs


def test_scientific_evidence_pipeline_with_synthetic_selected_outputs(tmp_path: Path) -> None:
    _write_selected_results(tmp_path)
    _write_json(
        tmp_path / "outputs" / "classification" / "stage_8a" / "best_model_metrics.json",
        {
            "model_name": "Extra Trees",
            "f1_macro_mean": 0.81,
            "f1_weighted_mean": 0.82,
            "balanced_accuracy_mean": 0.83,
            "precision_macro_mean": 0.84,
            "recall_macro_mean": 0.85,
            "roc_auc_ovr_weighted_mean": 0.86,
        },
    )
    _write(
        tmp_path / "outputs" / "classification" / "stage_8a" / "classification_summary.csv",
        "model_id,model_name,fold_count,f1_macro_mean\nextra_trees,Extra Trees,10,0.81\n",
    )
    _write_json(
        tmp_path / "outputs" / "regression" / "stage_8b" / "best_regression_model.json",
        {"model_name": "Extra Trees Regressor", "r2_mean": 0.3, "rmse_mean": 2.0, "mae_mean": 1.0},
    )
    _write_json(
        tmp_path / "outputs" / "feature_engineering" / "stage_8c" / "stage_8c_summary.json",
        {
            "best_feature_family": "window_features",
            "classification_improvement": 0.2,
            "regression_improvement": 0.1,
            "runtime_increase_seconds": 5.0,
        },
    )
    _write(
        tmp_path / "outputs" / "feature_selection" / "selected_features.csv",
        "feature_name,default_classification_feature_set,default_regression_feature_set,research_feature_set\n"
        "baseline,True,False,True\npeak,True,True,True\n",
    )
    _write_json(
        tmp_path / "outputs" / "fingerprints" / "stage_7a" / "fingerprint_summary.json",
        {"summary": {"fingerprint_rows": 100, "consensus_fingerprint_rows": 12, "fingerprint_qc_passed": True}},
    )
    _write(
        tmp_path / "outputs" / "exploratory" / "stage_7b" / "pca_explained_variance.csv",
        "component,explained_variance_ratio,cumulative_explained_variance_ratio\nPC1,0.6,0.6\nPC2,0.2,0.8\n",
    )
    _write(
        tmp_path / "outputs" / "exploratory" / "stage_7b" / "cluster_assignments.csv",
        "sample_id,cluster_id\nS1,1\nS2,2\n",
    )
    _write_json(
        tmp_path / "outputs" / "qc" / "stage_5c" / "qc_summary.json",
        {"qc_passed": False, "warnings": ["a"], "errors": ["b"], "row_count": 5},
    )
    _write(tmp_path / "outputs" / "exploratory" / "stage_7b" / "pca.png", "image")

    database = build_scientific_evidence(tmp_path)
    paths = write_evidence_outputs(database, tmp_path / "outputs" / "scientific_narrative")

    assert database.extraction_success
    assert len(database.parsed_files) == 9
    assert len(database.unsupported_files) == 1
    assert any(record.metric_name == "best_model" and record.metric_value == "Extra Trees" for record in database.records)
    assert any(record.metric_name == "r2_mean" and record.metric_value == 0.3 for record in database.records)
    assert any(record.metric_name == "best_feature_family" for record in database.records)
    assert any(record.metric_name == "default_classification_selected_feature_count" for record in database.records)
    assert any(record.metric_name == "fingerprint_rows" for record in database.records)
    assert any(record.metric_name == "explained_variance_ratio" for record in database.records)
    assert any(record.metric_name == "warning_count" for record in database.records)
    assert {path.name for path in paths} == {
        "scientific_evidence.json",
        "scientific_evidence.csv",
        "scientific_evidence_report.md",
    }

    csv_rows = _read_csv(tmp_path / "outputs" / "scientific_narrative" / "scientific_evidence.csv")
    assert csv_rows
    assert set(csv_rows[0]) == {
        "analysis_type",
        "source_file",
        "source_run",
        "metric_name",
        "metric_value",
        "metric_units",
        "figure_reference",
        "table_reference",
        "biological_entity",
        "model_name",
        "confidence",
        "extraction_status",
        "notes",
    }


def _write_selected_results(tmp_path: Path) -> None:
    rows = [
        ("Classification results", "classification", "classification/stage_8a/best_model_metrics.json", "stage_8a", "classification/stage_8a/classification_summary.csv"),
        ("Regression results", "regression", "regression/stage_8b/best_regression_model.json", "stage_8b", ""),
        ("Feature engineering results", "advanced feature engineering", "feature_engineering/stage_8c/stage_8c_summary.json", "stage_8c", ""),
        ("Feature-selection results", "feature selection", "feature_selection/selected_features.csv", "feature_selection", ""),
        ("Fingerprint summary", "fingerprint generation", "fingerprints/stage_7a/fingerprint_summary.json", "stage_7a", ""),
        ("PCA/exploratory analysis", "exploratory analysis", "exploratory/stage_7b/pca_explained_variance.csv", "stage_7b", "exploratory/stage_7b/cluster_assignments.csv; exploratory/stage_7b/pca.png"),
        ("Data-quality summary", "canonical QC; feature validation", "qc/stage_5c/qc_summary.json", "stage_5c", ""),
    ]
    path = tmp_path / "outputs" / "results_inventory" / "selected_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for report_section, analysis_type, selected_file, selected_run, companions in rows:
            writer.writerow(
                {
                    "report_section": report_section,
                    "analysis_type": analysis_type,
                    "selected_file": selected_file,
                    "selected_run": selected_run,
                    "status": "FOUND",
                    "selection_reason": "",
                    "companion_files": companions,
                    "scientific_role": "",
                    "include_in_supervisor_report": "True",
                    "notes": "",
                }
            )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write(path, json.dumps(payload))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))
