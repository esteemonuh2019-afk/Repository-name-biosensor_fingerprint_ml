from __future__ import annotations

import csv
import os
from pathlib import Path

from src.results_inventory.inventory_report import REQUIRED_OUTPUT_FILENAMES, write_inventory_outputs
from src.results_inventory.inventory_scanner import build_results_inventory


CLASSIFICATION_REQUIRED = (
    "classification_summary.csv",
    "best_model_metrics.json",
    "confusion_matrix.csv",
    "per_class_metrics.csv",
    "model_rankings.csv",
    "classification_report.md",
)

REGRESSION_REQUIRED = (
    "regression_summary.csv",
    "best_regression_model.json",
    "prediction_vs_actual.csv",
    "residuals.csv",
    "model_rankings.csv",
    "regression_report.md",
)

EXPLORATORY_REQUIRED = (
    "pca_scores.csv",
    "pca_loadings.csv",
    "pca_explained_variance.csv",
    "cluster_assignments.csv",
    "exploratory_summary.json",
    "pca_pc1_pc2.png",
)


def test_results_inventory_pipeline_with_synthetic_outputs(tmp_path: Path) -> None:
    complete_classification = tmp_path / "outputs" / "classification" / "stage_8a"
    incomplete_classification = tmp_path / "outputs" / "classification" / "stage_8a_2"
    regression = tmp_path / "outputs" / "regression" / "stage_8b"
    exploratory = tmp_path / "outputs" / "exploratory" / "stage_7b"

    _write_complete_run(complete_classification, CLASSIFICATION_REQUIRED, mtime=100)
    _write(incomplete_classification / "classification_summary.csv", mtime=200)
    _write(incomplete_classification / "model_rankings.csv", mtime=200)
    _write_complete_run(regression, REGRESSION_REQUIRED, mtime=150)
    _write_complete_run(exploratory, EXPLORATORY_REQUIRED, mtime=160)
    _write(regression / "model_rankings.csv", mtime=150)
    _write(tmp_path / "outputs" / "misc" / "unknown_result.bin", "???", mtime=170)

    inventory = build_results_inventory(tmp_path, output_dir=tmp_path / "inventory")
    output_paths = write_inventory_outputs(inventory, tmp_path / "inventory")

    assert inventory.inventory_passed
    assert inventory.selected_runs["classification"].run_name == "stage_8a"
    assert inventory.selected_runs["regression"].run_name == "stage_8b"
    assert inventory.selected_runs["exploratory analysis"].run_name == "stage_7b"
    assert any(candidate.filename == "model_rankings.csv" for candidate in inventory.duplicate_candidates)
    assert any(record.result_role == "unknown" for record in inventory.classified_files)

    section_statuses = {row.report_section: row.status for row in inventory.missing_required_results}
    assert section_statuses["Classification results"] == "FOUND"
    assert section_statuses["Regression results"] == "FOUND"
    assert section_statuses["PCA/exploratory analysis"] == "FOUND"
    assert section_statuses["Dataset summary"] == "MISSING"
    assert inventory.project_health["report_generation_can_proceed"] is False

    assert {path.name for path in output_paths} == set(REQUIRED_OUTPUT_FILENAMES)
    selected_rows = _read_csv(tmp_path / "inventory" / "selected_results.csv")
    assert selected_rows[0].keys() >= {
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
    }
    assert (tmp_path / "inventory" / "results_inventory_report.md").exists()


def _write(path: Path, content: str = "value\n1\n", *, mtime: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json" and content == "value\n1\n":
        content = "{}"
    if path.suffix == ".md" and content == "value\n1\n":
        content = "# Report\n"
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _write_complete_run(path: Path, filenames: tuple[str, ...], *, mtime: int | None = None) -> None:
    for filename in filenames:
        _write(path / filename, mtime=mtime)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))
