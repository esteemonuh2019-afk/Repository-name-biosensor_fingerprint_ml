from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from src.results_inventory.inventory_report import REQUIRED_OUTPUT_FILENAMES, write_inventory_outputs
from src.results_inventory.inventory_scanner import build_results_inventory, scan_output_files
from src.results_inventory.result_classifier import classify_files
from src.results_inventory.run_selector import (
    detect_runs,
    identify_duplicate_candidates,
    select_preferred_runs,
)


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


def test_recursive_file_discovery(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "classification" / "stage_8a" / "classification_summary.csv")

    scan = scan_output_files(tmp_path, output_dir=None)

    assert [record.relative_path for record in scan.all_files] == [
        "classification/stage_8a/classification_summary.csv"
    ]


def test_correct_extension_detection(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "features" / "feature_dataset.CSV")

    scan = scan_output_files(tmp_path, output_dir=None)

    assert scan.all_files[0].extension == ".csv"


def test_classification_of_tables_figures_reports_and_metrics(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "classification" / "stage_8a" / "classification_summary.csv")
    _write(tmp_path / "outputs" / "classification" / "stage_8a" / "classification_report.md")
    _write(tmp_path / "outputs" / "classification" / "stage_8a" / "best_model_metrics.json", "{}")
    _write(tmp_path / "outputs" / "exploratory" / "stage_7b" / "pca_pc1_pc2.png", "png")

    classified = classify_files(scan_output_files(tmp_path, output_dir=None).all_files)
    by_name = {record.filename: record for record in classified}

    assert by_name["classification_summary.csv"].table
    assert by_name["classification_summary.csv"].model_metric
    assert by_name["classification_report.md"].report
    assert by_name["best_model_metrics.json"].machine_readable
    assert by_name["pca_pc1_pc2.png"].figure


def test_run_directory_detection(tmp_path: Path) -> None:
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED)
    classified = classify_files(scan_output_files(tmp_path, output_dir=None).all_files)

    runs = detect_runs(classified)

    assert any(run.run_name == "stage_8a" and run.analysis_type == "classification" for run in runs)


def test_newer_incomplete_run_does_not_replace_older_complete_run(tmp_path: Path) -> None:
    complete_dir = tmp_path / "outputs" / "classification" / "stage_8a"
    incomplete_dir = tmp_path / "outputs" / "classification" / "stage_8a_2"
    _write_complete_run(complete_dir, CLASSIFICATION_REQUIRED, mtime=100)
    _write(incomplete_dir / "classification_summary.csv", mtime=200)

    inventory = build_results_inventory(tmp_path, output_dir=None)

    assert inventory.selected_runs["classification"].run_name == "stage_8a"


def test_latest_complete_run_is_selected(tmp_path: Path) -> None:
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED, mtime=100)
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a_2", CLASSIFICATION_REQUIRED, mtime=200)

    inventory = build_results_inventory(tmp_path, output_dir=None)

    assert inventory.selected_runs["classification"].run_name == "stage_8a_2"


def test_duplicate_candidates_are_identified(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "classification" / "stage_8a" / "model_rankings.csv")
    _write(tmp_path / "outputs" / "regression" / "stage_8b" / "model_rankings.csv")
    classified = classify_files(scan_output_files(tmp_path, output_dir=None).all_files)

    duplicates = identify_duplicate_candidates(classified)

    assert [candidate.filename for candidate in duplicates] == ["model_rankings.csv"]


def test_empty_directories_are_handled(tmp_path: Path) -> None:
    (tmp_path / "outputs" / "classification" / "stage_8a_empty").mkdir(parents=True)
    scan = scan_output_files(tmp_path, output_dir=None)

    runs = detect_runs(classify_files(scan.all_files), empty_directories=scan.empty_directories)

    assert any(run.run_name == "stage_8a_empty" and run.likely_completion_status == "EMPTY" for run in runs)


def test_unknown_files_remain_unknown(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "misc" / "unexpected.bin", "binary-ish")

    classified = classify_files(scan_output_files(tmp_path, output_dir=None).all_files)

    assert classified[0].analysis_type == "unknown"
    assert classified[0].result_role == "unknown"


def test_large_files_are_recorded_without_unnecessary_hashing(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "tables" / "large.csv", "x" * 2048)

    scan = scan_output_files(tmp_path, output_dir=None, maximum_hash_size_mb=0.000001)

    assert scan.all_files[0].relative_path == "tables/large.csv"
    assert scan.all_files[0].content_hash == ""
    assert scan.all_files[0].hash_status == "skipped_large_file"


def test_required_section_completeness_is_calculated_correctly(tmp_path: Path) -> None:
    _write_minimal_report_ready_tree(tmp_path)

    inventory = build_results_inventory(tmp_path, output_dir=None)
    statuses = {row.report_section: row.status for row in inventory.missing_required_results}

    assert statuses["Classification results"] == "FOUND"
    assert statuses["Regression results"] == "FOUND"
    assert statuses["Blind validation status"] == "NOT YET APPLICABLE"


def test_blind_validation_is_marked_not_yet_available_when_absent(tmp_path: Path) -> None:
    _write_complete_run(
        tmp_path / "outputs" / "blind_prediction" / "sample_prediction",
        (
            "blind_prediction_summary.json",
            "chemical_probabilities.csv",
            "concentration_prediction.csv",
            "prediction_confidence.csv",
            "novelty_assessment.csv",
            "influential_features.csv",
            "influential_strains.csv",
            "blind_sample_qc_report.md",
            "blind_prediction_report.md",
        ),
    )

    inventory = build_results_inventory(tmp_path, output_dir=None)
    blind = next(row for row in inventory.missing_required_results if row.report_section == "Blind validation status")

    assert blind.status == "NOT YET APPLICABLE"


def test_file_paths_remain_deterministic(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "z" / "b.csv")
    _write(tmp_path / "outputs" / "a" / "c.csv")

    first = [record.relative_path for record in scan_output_files(tmp_path, output_dir=None).all_files]
    second = [record.relative_path for record in scan_output_files(tmp_path, output_dir=None).all_files]

    assert first == second == ["a/c.csv", "z/b.csv"]


def test_input_outputs_are_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "features" / "feature_dataset.csv"
    _write(path, "value\n1\n", mtime=123)
    before = (path.read_text(encoding="utf-8"), path.stat().st_mtime)

    scan_output_files(tmp_path, output_dir=None)

    assert (path.read_text(encoding="utf-8"), path.stat().st_mtime) == before


def test_selection_reasons_are_recorded(tmp_path: Path) -> None:
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED)

    runs = detect_runs(classify_files(scan_output_files(tmp_path, output_dir=None).all_files))
    selected = select_preferred_runs(runs)

    assert "passes completeness requirements" in selected["classification"].selection_reason


def test_exact_required_output_filenames_are_produced(tmp_path: Path) -> None:
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED)
    inventory = build_results_inventory(tmp_path, output_dir=tmp_path / "inventory")

    paths = write_inventory_outputs(inventory, tmp_path / "inventory")

    assert {path.name for path in paths} == set(REQUIRED_OUTPUT_FILENAMES)
    assert {path.name for path in (tmp_path / "inventory").iterdir()} == set(REQUIRED_OUTPUT_FILENAMES)


def test_existing_output_directories_are_not_overwritten_without_overwrite(tmp_path: Path) -> None:
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED)
    output_dir = tmp_path / "inventory"
    _write(output_dir / "stale.txt", "keep")
    inventory = build_results_inventory(tmp_path, output_dir=output_dir)

    with pytest.raises(FileExistsError):
        write_inventory_outputs(inventory, output_dir, overwrite=False)

    assert (output_dir / "stale.txt").read_text(encoding="utf-8") == "keep"


def test_windows_paths_are_handled_correctly(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "classification" / "stage_8a" / "classification_summary.csv")
    outputs_dir = Path("outputs\\classification")

    scan = scan_output_files(str(tmp_path), outputs_dir=outputs_dir, output_dir=None)

    assert scan.all_files[0].relative_path == "stage_8a/classification_summary.csv"
    assert Path(scan.all_files[0].full_path).exists()


def test_results_are_reproducible(tmp_path: Path) -> None:
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED, mtime=100)

    first = build_results_inventory(tmp_path, output_dir=None)
    second = build_results_inventory(tmp_path, output_dir=None)

    assert [record.relative_path for record in first.classified_files] == [
        record.relative_path for record in second.classified_files
    ]
    assert first.selected_runs["classification"].run_name == second.selected_runs["classification"].run_name


def test_empty_outputs_folder_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()

    inventory = build_results_inventory(tmp_path, output_dir=None)

    assert not inventory.inventory_passed
    assert "No generated result files found" in inventory.errors[0]


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


def _write_minimal_report_ready_tree(tmp_path: Path) -> None:
    _write_complete_run(
        tmp_path / "outputs" / "features",
        ("feature_dataset.csv", "feature_summary.json", "feature_qc_report.md"),
    )
    _write_complete_run(
        tmp_path / "outputs" / "qc" / "stage_5c",
        ("qc_summary.json", "missing_values.csv", "source_file_summary.csv", "canonical_qc_report.md"),
    )
    _write_complete_run(
        tmp_path / "outputs" / "feature_validation" / "stage_6c",
        (
            "feature_validation_summary.json",
            "feature_statistics.csv",
            "feature_missingness.csv",
            "feature_nonfinite_values.csv",
            "constant_features.csv",
            "low_variance_features.csv",
            "pearson_correlations.csv",
            "spearman_correlations.csv",
            "highly_correlated_pairs.csv",
            "replicate_consistency.csv",
            "feature_recommendations.csv",
            "feature_validation_report.md",
        ),
    )
    _write_complete_run(
        tmp_path / "outputs" / "fingerprints" / "stage_7a",
        (
            "fingerprint_dataset.csv",
            "fingerprint_dataset_normalized.csv",
            "fingerprint_summary.json",
            "fingerprint_qc_report.md",
        ),
    )
    _write_complete_run(tmp_path / "outputs" / "exploratory" / "stage_7b", EXPLORATORY_REQUIRED)
    _write_complete_run(tmp_path / "outputs" / "classification" / "stage_8a", CLASSIFICATION_REQUIRED)
    _write_complete_run(tmp_path / "outputs" / "regression" / "stage_8b", REGRESSION_REQUIRED)
    _write_complete_run(
        tmp_path / "outputs" / "feature_engineering" / "stage_8c",
        (
            "advanced_feature_dataset.csv",
            "advanced_feature_dictionary.csv",
            "advanced_feature_summary.json",
            "feature_family_ablation_summary.csv",
            "feature_family_vs_macro_f1.csv",
            "feature_family_vs_r2.csv",
            "feature_family_vs_rmse.csv",
            "feature_family_vs_mae.csv",
            "feature_family_runtime.csv",
            "stage_8c_summary.json",
            "stage_8c_feature_engineering_report.md",
        ),
    )
    _write_complete_run(
        tmp_path / "outputs" / "feature_selection",
        (
            "selected_features.csv",
            "feature_ranking.csv",
            "feature_selection_summary.csv",
            "classification_after_selection.csv",
            "regression_after_selection.csv",
            "performance_vs_feature_count.csv",
            "feature_selection_report.md",
        ),
    )
    _write(tmp_path / "outputs" / "tables" / "processed_data.csv")
    _write(tmp_path / "outputs" / "tables" / "single_strain_loeo.csv")
    _write(tmp_path / "docs" / "LIMITATIONS_AND_RISKS.md", "# Limitations\n")
