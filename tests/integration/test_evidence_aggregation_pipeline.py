from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from src.scientific_narrative.scientific_summary import OUTPUT_FILENAMES


EVIDENCE_FIELDS = [
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
]


def test_evidence_aggregation_cli_pipeline(tmp_path: Path) -> None:
    evidence_file = tmp_path / "scientific_evidence.csv"
    output_dir = tmp_path / "scientific_aggregation"
    _write_evidence(evidence_file)
    evidence_file.with_suffix(".json").write_text(
        json.dumps({"metadata": {"unsupported_file_count": 2, "unreadable_file_count": 0}}),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "scripts/aggregate_scientific_evidence.py",
        "--evidence-file",
        str(evidence_file),
        "--output-dir",
        str(output_dir),
        "--maximum-source-ids-per-cell",
        "3",
    ]

    result = subprocess.run(command, cwd=project_root, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "aggregation success: True" in result.stdout
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_FILENAMES)
    assert not (output_dir / "fingerprint_summary.csv").exists()

    payload = json.loads((output_dir / "aggregated_evidence.json").read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["evidence_records_received"] > 0
    assert metadata["summary_records_created"] > 0
    assert metadata["unsupported_file_count_from_json"] == 2
    assert metadata["traceability_coverage"] == 1.0
    assert payload["aggregation_passed"] is True
    assert payload["fingerprint_summary"]

    aggregated_rows = _read_csv(output_dir / "aggregated_evidence.csv")
    section_files = [
        "dataset_summary.csv",
        "qc_summary.csv",
        "exploratory_summary.csv",
        "classification_summary.csv",
        "regression_summary.csv",
        "feature_engineering_summary.csv",
        "feature_selection_summary.csv",
        "strain_summary.csv",
        "limitations_summary.csv",
        "blind_validation_status.csv",
    ]
    section_row_count = sum(len(_read_csv(output_dir / name)) for name in section_files)
    assert len(aggregated_rows) == section_row_count + len(payload["fingerprint_summary"])
    assert _read_csv(output_dir / "evidence_traceability.csv")
    assert _read_csv(output_dir / "metric_alias_registry.csv")
    assert _read_csv(output_dir / "metric_mapping_audit.csv")
    assert (output_dir / "unmapped_metrics.csv").exists()
    assert _read_csv(output_dir / "summary_population_audit.csv")
    comparison_rows = _read_csv(output_dir / "regression_model_comparison.csv")
    assert comparison_rows
    assert [row["selection_status"] for row in comparison_rows].count("SELECTED") == 1
    classification_rows = _read_csv(output_dir / "classification_summary.csv")
    regression_rows = _read_csv(output_dir / "regression_summary.csv")
    assert any(row["metric_name"] == "f1_macro_mean" and row["status"] != "MISSING" for row in classification_rows)
    assert any(row["metric_name"] == "balanced_accuracy_mean" and row["status"] != "MISSING" for row in classification_rows)
    assert any(row["metric_name"] == "r2_mean" and row["status"] != "MISSING" for row in regression_rows)
    assert any(row["metric_name"] == "rmse_mean" and row["status"] != "MISSING" for row in regression_rows)
    assert "does not write Results, Discussion, DOCX, or PDF outputs" in (output_dir / "aggregation_report.md").read_text(encoding="utf-8")
    assert "Mapped metric names" in (output_dir / "aggregation_report.md").read_text(encoding="utf-8")


def _write_evidence(path: Path) -> None:
    rows = [
        _row("canonical QC; feature validation", "qc/stage_5c_2/qc_summary.json", "row_count", 120, metric_units="count"),
        _row("canonical QC; feature validation", "qc/stage_5c_2/qc_summary.json", "measurement_unit_count", 40, metric_units="count"),
        _row("canonical QC; feature validation", "qc/stage_5c_2/qc_summary.json", "qc_passed", True),
        _row("canonical QC; feature validation", "qc/stage_5c_2/qc_summary.json", "warning_count", 1, metric_units="count"),
        _row("canonical QC; feature validation", "qc/stage_5c_2/qc_summary.json", "error_count", 0, metric_units="count"),
        _row("feature extraction", "features/feature_dataset_summary.json", "feature_row_count", 80, metric_units="count"),
        _row("fingerprint generation", "fingerprints/stage_7a/fingerprint_summary.json", "fingerprint_rows", 64, metric_units="count"),
        _row("fingerprint generation", "fingerprints/stage_7a/fingerprint_summary.json", "consensus_fingerprint_rows", 12, metric_units="count"),
        _row("exploratory analysis", "exploratory/stage_7b_3/pca_explained_variance.csv", "explained_variance_ratio", 0.55, biological_entity="PC1"),
        _matrix("exploratory analysis", "exploratory/stage_7b_3/chemical_similarity_heatmap_table.csv", "Ampicillin", "Kanamycin", 0.4),
        _row("classification", "classification/stage_8a/best_model_metrics.json", "best_model", "Extra Trees", model_name="Extra Trees"),
        _row("classification", "classification/stage_8a/best_model_metrics.json", "f1_macro_mean", 0.82, model_name="Extra Trees"),
        _row("classification", "classification/stage_8a/best_model_metrics.json", "balanced_accuracy_mean", 0.84, model_name="Extra Trees"),
        _matrix("classification", "classification/stage_8a/per_class_metrics.csv", "BL011", "f1", 0.78),
        _matrix("classification", "classification/stage_8a/confusion_matrix.csv", "BL011", "BL027", 2),
        _row("regression", "regression/stage_8b_2/best_regression_model.json", "best_model", "Extra Trees Regressor", model_name="Extra Trees Regressor"),
        _row("regression", "regression/stage_8b_2/best_regression_model.json", "r2_mean", 0.31, model_name="Extra Trees Regressor"),
        _row("regression", "regression/stage_8b_2/best_regression_model.json", "rmse_mean", 1.8, model_name="Extra Trees Regressor"),
        _row("regression", "regression/stage_8b_2/best_regression_model.json", "mae_mean", 1.1, model_name="Extra Trees Regressor"),
        _row("regression", "regression/stage_8b_2/model_rankings.csv", "r2_mean", 0.1, model_name="Other Regressor"),
        _row("regression", "regression/stage_8b_2/model_rankings.csv", "rmse_mean", 2.5, model_name="Other Regressor"),
        _row("regression", "regression/stage_8b_2/model_rankings.csv", "mae_mean", 2.0, model_name="Other Regressor"),
        _row("advanced feature engineering", "feature_engineering/stage_8c_real/stage_8c_summary.json", "best_feature_family", "advanced_windows"),
        _matrix("advanced feature engineering", "feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "advanced_windows", "regression_r2", 0.31),
        _matrix("advanced feature engineering", "feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "advanced_windows", "regression_rmse", 1.8),
        _matrix("advanced feature engineering", "feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "advanced_windows", "regression_mae", 1.1),
        _matrix("advanced feature engineering", "feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "advanced_windows", "runtime_increase_seconds", 4.0),
        _fs("classification__80pct", "classification", "mutual_info", "macro_f1_mean", 0.8),
        _fs("classification__80pct", "classification", "mutual_info", "feature_count", 10),
        _fs("regression__80pct", "regression", "mutual_info", "r2_mean", 0.28),
        _fs("regression__80pct", "regression", "mutual_info", "feature_count", 10),
        _row("feature selection", "feature_selection_3/selected_features.csv", "recommended_feature_set", "research", biological_entity="task=classification"),
        _row("strain ablation", "classification/stage_8a/leave_one_strain_importance.csv", "held_out_f1_macro", 0.73, biological_entity="BL011"),
        _row("strain ablation", "regression/stage_8b_2/leave_one_strain_importance.csv", "held_out_r2", 0.22, biological_entity="BL011"),
        _row("real blind validation", "blind_validation/real_subset_prediction.csv", "prediction_passed", True, source_run="not yet available"),
        _row("real blind validation", "blind_validation/real_subset_prediction.csv", "novelty_score", 0.12, source_run="not yet available"),
        _row("classification", "classification/stage_8a/null_metric.csv", "f1_macro_mean", "", extraction_status="NULL"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{field: "" for field in EVIDENCE_FIELDS}, **row})


def _row(
    analysis_type: str,
    source_file: str,
    metric_name: str,
    metric_value: object,
    *,
    source_run: str = "stage_test",
    metric_units: str = "unitless",
    biological_entity: str = "",
    model_name: str = "",
    extraction_status: str = "EXTRACTED",
) -> dict[str, object]:
    return {
        "analysis_type": analysis_type,
        "source_file": source_file,
        "source_run": source_run,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_units": metric_units,
        "biological_entity": biological_entity,
        "model_name": model_name,
        "confidence": "HIGH",
        "extraction_status": extraction_status,
    }


def _matrix(analysis_type: str, source_file: str, row: str, column: str, value: object) -> dict[str, object]:
    return _row(
        analysis_type,
        source_file,
        "matrix_value",
        value,
        biological_entity=f"row={row}; column={column}",
    )


def _fs(subset: str, task: str, selector: str, metric_name: str, metric_value: object) -> dict[str, object]:
    return _row(
        "feature selection",
        "feature_selection_3/feature_selection_summary.csv",
        metric_name,
        metric_value,
        biological_entity=f"feature_subset_id={subset}; task={task}; selector_method={selector}",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))
