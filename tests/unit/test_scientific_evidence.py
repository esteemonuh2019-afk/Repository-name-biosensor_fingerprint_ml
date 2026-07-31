from __future__ import annotations

import csv
import json
from pathlib import Path

from src.scientific_narrative import build_scientific_evidence, write_evidence_outputs
from src.scientific_narrative.evidence_database import OUTPUT_FILENAMES
from src.scientific_narrative.result_parser import SourceContext, parse_csv_source, parse_json_source, parse_text_source


def test_parse_classification_best_model_json(tmp_path: Path) -> None:
    path = tmp_path / "best_model_metrics.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "Extra Trees",
                "f1_macro_mean": 0.91,
                "f1_weighted_mean": 0.92,
                "balanced_accuracy_mean": 0.93,
                "roc_auc_ovr_weighted_mean": 0.94,
            }
        ),
        encoding="utf-8",
    )
    result = parse_json_source(path, _context("classification/stage_8a/best_model_metrics.json", "classification"))

    metrics = {record.metric_name: record.metric_value for record in result.records}
    assert metrics["best_model"] == "Extra Trees"
    assert metrics["f1_macro_mean"] == 0.91
    assert metrics["roc_auc_ovr_weighted_mean"] == 0.94


def test_parse_csv_metrics_and_model_names(tmp_path: Path) -> None:
    path = tmp_path / "classification_summary.csv"
    path.write_text(
        "model_id,model_name,fold_count,f1_macro_mean,precision_macro_mean,recall_macro_mean\n"
        "extra_trees,Extra Trees,10,0.8,0.7,0.75\n",
        encoding="utf-8",
    )
    result = parse_csv_source(path, _context("classification/stage_8a/classification_summary.csv", "classification"))

    by_metric = {record.metric_name: record for record in result.records}
    assert by_metric["f1_macro_mean"].metric_value == 0.8
    assert by_metric["f1_macro_mean"].model_name == "Extra Trees"
    assert by_metric["fold_count"].metric_units == "count"


def test_summary_only_dataset_extracts_counts_without_cells(tmp_path: Path) -> None:
    path = tmp_path / "feature_dataset.csv"
    path.write_text(
        "Experiment_ID,Strain,Chemical,baseline,peak\n"
        "E1,BL011,Ampicillin,1,2\n"
        "E2,BL027,Kanamycin,3,4\n",
        encoding="utf-8",
    )
    result = parse_csv_source(path, _context("features/feature_dataset.csv", "feature extraction"))

    metrics = {record.metric_name: record.metric_value for record in result.records}
    assert metrics["source_row_count"] == 2
    assert metrics["feature_column_count"] == 2
    assert "baseline" not in metrics


def test_markdown_extracts_numeric_bullets_only(tmp_path: Path) -> None:
    path = tmp_path / "classification_report.md"
    path.write_text("- Sample count: 42\n- Narrative sentence without scalar\n", encoding="utf-8")

    result = parse_text_source(path, _context("classification/stage_8a/classification_report.md", "classification"))

    assert len(result.records) == 1
    assert result.records[0].metric_name == "sample_count"
    assert result.records[0].metric_value == 42
    assert result.records[0].confidence == "MEDIUM"


def test_extractor_uses_only_selected_files(tmp_path: Path) -> None:
    _write_selected_results(
        tmp_path,
        [
            {
                "report_section": "Classification results",
                "analysis_type": "classification",
                "selected_file": "classification/stage_8a/best_model_metrics.json",
                "selected_run": "stage_8a",
                "companion_files": "",
            }
        ],
    )
    _write_json(tmp_path / "outputs" / "classification" / "stage_8a" / "best_model_metrics.json", {"model_name": "A", "f1_macro_mean": 0.5})
    _write_json(tmp_path / "outputs" / "classification" / "stage_8a" / "unlisted.json", {"f1_macro_mean": 0.99})

    database = build_scientific_evidence(tmp_path)

    assert all(record.source_file != "classification/stage_8a/unlisted.json" for record in database.records)
    assert any(record.metric_value == 0.5 for record in database.records)
    assert not any(record.metric_value == 0.99 for record in database.records)


def test_unsupported_files_are_reported(tmp_path: Path) -> None:
    _write_selected_results(
        tmp_path,
        [
            {
                "report_section": "Exploratory",
                "analysis_type": "exploratory analysis",
                "selected_file": "exploratory/stage_7b/pca.png",
                "selected_run": "stage_7b",
                "companion_files": "exploratory/stage_7b/report.pdf",
            }
        ],
    )
    _write(tmp_path / "outputs" / "exploratory" / "stage_7b" / "pca.png", "image")
    _write(tmp_path / "outputs" / "exploratory" / "stage_7b" / "report.pdf", "pdf")

    database = build_scientific_evidence(tmp_path)

    assert len(database.unsupported_files) == 2
    assert any("Image ignored" in item.notes for item in database.unsupported_files)


def test_missing_expected_metrics_are_recorded_as_null(tmp_path: Path) -> None:
    _write_selected_results(
        tmp_path,
        [
            {
                "report_section": "Classification results",
                "analysis_type": "classification",
                "selected_file": "classification/stage_8a/best_model_metrics.json",
                "selected_run": "stage_8a",
                "companion_files": "",
            }
        ],
    )
    _write_json(tmp_path / "outputs" / "classification" / "stage_8a" / "best_model_metrics.json", {"model_name": "A"})

    database = build_scientific_evidence(tmp_path)

    missing_names = {record.metric_name for record in database.missing_evidence}
    assert "macro_f1" in missing_names
    assert any(record.metric_value is None and record.extraction_status == "NULL" for record in database.missing_evidence)


def test_write_exact_evidence_output_filenames(tmp_path: Path) -> None:
    _write_selected_results(
        tmp_path,
        [
            {
                "report_section": "Regression results",
                "analysis_type": "regression",
                "selected_file": "regression/stage_8b/best_regression_model.json",
                "selected_run": "stage_8b",
                "companion_files": "",
            }
        ],
    )
    _write_json(
        tmp_path / "outputs" / "regression" / "stage_8b" / "best_regression_model.json",
        {"model_name": "Extra Trees Regressor", "r2_mean": 0.2, "rmse_mean": 1.5, "mae_mean": 1.0},
    )
    database = build_scientific_evidence(tmp_path)

    paths = write_evidence_outputs(database, tmp_path / "outputs" / "scientific_narrative")

    assert {path.name for path in paths} == set(OUTPUT_FILENAMES)
    assert (tmp_path / "outputs" / "scientific_narrative" / "scientific_evidence.csv").exists()


def _context(source_file: str, analysis_type: str) -> SourceContext:
    return SourceContext(
        analysis_type=analysis_type,
        source_file=source_file,
        source_run="stage_test",
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write(path, json.dumps(payload))


def _write_selected_results(tmp_path: Path, rows: list[dict[str, str]]) -> None:
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
        for row in rows:
            writer.writerow(
                {
                    **{field: "" for field in fields},
                    "status": "FOUND",
                    "include_in_supervisor_report": "True",
                    **row,
                }
            )
