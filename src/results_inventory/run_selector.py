"""Run detection, successful-run evaluation, and preferred-run selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re

from src.results_inventory.inventory_models import (
    COMPLETION_COMPLETE,
    COMPLETION_DIAGNOSTIC,
    COMPLETION_EMPTY,
    COMPLETION_PARTIAL,
    COMPLETION_UNKNOWN,
    DuplicateCandidate,
    InventoryFile,
    ObsoleteCandidate,
    RunInventory,
)


@dataclass(frozen=True)
class RunRule:
    """Completeness rule for an analysis run."""

    required_files: tuple[str, ...]
    required_any: tuple[tuple[str, tuple[str, ...]], ...] = ()
    optional_files: tuple[str, ...] = ()
    required_machine_readable: tuple[str, ...] = ()


RUN_RULES: dict[str, RunRule] = {
    "canonical QC": RunRule(
        required_files=(
            "qc_summary.json",
            "missing_values.csv",
            "source_file_summary.csv",
            "canonical_qc_report.md",
        ),
        optional_files=(
            "logical_duplicate_groups.csv",
            "legacy_logical_duplicate_groups.csv",
            "source_aware_duplicate_groups.csv",
            "duplicate_value_groups.csv",
            "conflicting_duplicate_rows.csv",
            "ambiguous_measurement_identities.csv",
            "time_series_issues.csv",
        ),
        required_machine_readable=("qc_summary.json", "missing_values.csv", "source_file_summary.csv"),
    ),
    "feature extraction": RunRule(
        required_files=("feature_dataset.csv", "feature_summary.json", "feature_qc_report.md"),
        required_machine_readable=("feature_dataset.csv", "feature_summary.json"),
    ),
    "feature validation": RunRule(
        required_files=(
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
        required_machine_readable=(
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
        ),
    ),
    "fingerprint generation": RunRule(
        required_files=(
            "fingerprint_dataset.csv",
            "fingerprint_dataset_normalized.csv",
            "fingerprint_summary.json",
            "fingerprint_qc_report.md",
        ),
        optional_files=(
            "consensus_fingerprint_dataset.csv",
            "consensus_fingerprint_summary.csv",
            "consensus_distance_matrix_euclidean.csv",
            "consensus_distance_matrix_cosine.csv",
            "consensus_distance_matrix_manhattan.csv",
            "consensus_distance_matrix_correlation.csv",
            "distance_matrix_euclidean.csv",
            "distance_matrix_cosine.csv",
            "distance_matrix_manhattan.csv",
            "distance_matrix_correlation.csv",
        ),
        required_machine_readable=(
            "fingerprint_dataset.csv",
            "fingerprint_dataset_normalized.csv",
            "fingerprint_summary.json",
        ),
    ),
    "exploratory analysis": RunRule(
        required_files=(
            "pca_scores.csv",
            "pca_loadings.csv",
            "pca_explained_variance.csv",
            "cluster_assignments.csv",
        ),
        required_any=(
            ("exploratory report or summary", ("exploratory_analysis_report.md", "exploratory_summary.json")),
            (
                "major exploratory figure",
                (
                    "pca_pc1_pc2.png",
                    "pca_pc1_pc2.pdf",
                    "chemical_similarity_heatmap.png",
                    "hierarchical_dendrogram.png",
                    "consensus_fingerprint_heatmap.png",
                ),
            ),
        ),
        optional_files=(
            "cluster_composition.csv",
            "chemical_similarity_heatmap_table.csv",
            "concentration_trajectories.csv",
            "top_component_features.csv",
            "strain_dispersion.csv",
        ),
        required_machine_readable=(
            "pca_scores.csv",
            "pca_loadings.csv",
            "pca_explained_variance.csv",
            "cluster_assignments.csv",
        ),
    ),
    "classification": RunRule(
        required_files=(
            "classification_summary.csv",
            "best_model_metrics.json",
            "confusion_matrix.csv",
            "per_class_metrics.csv",
            "model_rankings.csv",
            "classification_report.md",
        ),
        optional_files=(
            "feature_importance.csv",
            "permutation_importance.csv",
            "leave_one_strain_importance.csv",
            "fold_metrics.csv",
        ),
        required_machine_readable=(
            "classification_summary.csv",
            "best_model_metrics.json",
            "confusion_matrix.csv",
            "per_class_metrics.csv",
            "model_rankings.csv",
        ),
    ),
    "regression": RunRule(
        required_files=(
            "regression_summary.csv",
            "best_regression_model.json",
            "prediction_vs_actual.csv",
            "residuals.csv",
            "model_rankings.csv",
            "regression_report.md",
        ),
        optional_files=(
            "per_model_metrics.csv",
            "fold_metrics.csv",
            "feature_importance.csv",
            "permutation_importance.csv",
            "leave_one_strain_importance.csv",
            "prediction_vs_actual.png",
            "residual_plot.png",
            "residual_histogram.png",
            "fold_performance.png",
        ),
        required_machine_readable=(
            "regression_summary.csv",
            "best_regression_model.json",
            "prediction_vs_actual.csv",
            "residuals.csv",
            "model_rankings.csv",
        ),
    ),
    "advanced feature engineering": RunRule(
        required_files=(
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
        optional_files=(
            "feature_family_importance.csv",
            "feature_family_redundancy.csv",
            "feature_family_comparison.png",
            "feature_family_ablation.png",
            "classification_improvement.png",
            "regression_improvement.png",
            "runtime_comparison.png",
        ),
        required_machine_readable=(
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
        ),
    ),
    "feature selection": RunRule(
        required_files=(
            "selected_features.csv",
            "feature_ranking.csv",
            "feature_selection_summary.csv",
            "classification_after_selection.csv",
            "regression_after_selection.csv",
            "performance_vs_feature_count.csv",
            "feature_selection_report.md",
        ),
        optional_files=(
            "performance_vs_feature_count.png",
            "feature_ranking.png",
            "feature_importance.png",
        ),
        required_machine_readable=(
            "selected_features.csv",
            "feature_ranking.csv",
            "feature_selection_summary.csv",
            "classification_after_selection.csv",
            "regression_after_selection.csv",
            "performance_vs_feature_count.csv",
        ),
    ),
    "blind prediction": RunRule(
        required_files=(
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
        required_machine_readable=(
            "blind_prediction_summary.json",
            "chemical_probabilities.csv",
            "concentration_prediction.csv",
            "prediction_confidence.csv",
            "novelty_assessment.csv",
            "influential_features.csv",
            "influential_strains.csv",
        ),
    ),
}


def detect_runs(
    files: list[InventoryFile],
    *,
    empty_directories: list[str] | None = None,
) -> list[RunInventory]:
    """Detect distinct output runs from classified file records."""

    grouped: dict[tuple[str, str, str], list[InventoryFile]] = {}
    for record in files:
        analysis_type = record.analysis_type
        run_directory, run_name = _run_identity(record)
        grouped.setdefault((analysis_type, run_directory, run_name), []).append(record)

    runs = [_build_run(key, records) for key, records in grouped.items()]
    for directory in empty_directories or []:
        analysis_type = _analysis_type_from_directory(directory)
        run_name = _run_name_from_directory(directory)
        runs.append(
            RunInventory(
                analysis_type=analysis_type,
                run_name=run_name,
                run_directory=directory,
                run_version=_run_version(run_name),
                modified_time="",
                files_present=[],
                expected_files_present=[],
                expected_files_missing=[],
                likely_completion_status=COMPLETION_EMPTY,
                warnings=["Directory contains no generated files."],
                selection_score=0.0,
                file_count=0,
                completion_ratio=0.0,
            )
        )

    return sorted(
        runs,
        key=lambda run: (
            run.analysis_type.casefold(),
            run.run_directory.casefold(),
            run.run_name.casefold(),
        ),
    )


def select_preferred_runs(runs: list[RunInventory]) -> dict[str, RunInventory]:
    """Select one preferred run for each analysis type."""

    selectable = [
        run
        for run in runs
        if run.analysis_type not in {"unknown", "results inventory", "scientific report"}
        and run.file_count > 0
    ]
    selected: dict[str, RunInventory] = {}
    for analysis_type in sorted({run.analysis_type for run in selectable}):
        candidates = [run for run in selectable if run.analysis_type == analysis_type]
        best = max(candidates, key=_selection_key)
        best.selected = True
        best.selection_reason = _selection_reason(best, candidates)
        selected[analysis_type] = best
        for run in candidates:
            if run is not best:
                run.selection_reason = f"Not selected; preferred run is {best.run_name}."
    return selected


def identify_duplicate_candidates(files: list[InventoryFile]) -> list[DuplicateCandidate]:
    """Identify exact filename duplicates across different run directories."""

    grouped: dict[str, list[InventoryFile]] = {}
    for record in files:
        grouped.setdefault(record.filename.casefold(), []).append(record)

    candidates: list[DuplicateCandidate] = []
    for records in grouped.values():
        run_dirs = {record.parent_directory for record in records}
        if len(records) < 2 or len(run_dirs) < 2:
            continue
        sorted_records = sorted(records, key=lambda item: item.relative_path.casefold())
        candidates.append(
            DuplicateCandidate(
                filename=sorted_records[0].filename,
                duplicate_count=len(sorted_records),
                analysis_types=sorted({record.analysis_type for record in sorted_records}),
                run_names=sorted({record.run_name for record in sorted_records}),
                paths=[record.relative_path for record in sorted_records],
                newest_modified_time=max(record.modified_time for record in sorted_records),
                notes="Same filename appears in multiple run directories; no files were deleted.",
            )
        )
    return sorted(candidates, key=lambda item: item.filename.casefold())


def identify_obsolete_candidates(
    runs: list[RunInventory],
    selected_runs: dict[str, RunInventory],
    files: list[InventoryFile],
    *,
    large_file_threshold_bytes: int,
) -> list[ObsoleteCandidate]:
    """Identify superseded, partial, empty, diagnostic, and large generated outputs."""

    candidates: list[ObsoleteCandidate] = []
    selected_by_type = {analysis_type: run.run_directory for analysis_type, run in selected_runs.items()}
    for run in runs:
        if run.analysis_type in {"unknown", "results inventory"}:
            continue
        if run.likely_completion_status == COMPLETION_EMPTY:
            candidates.append(
                ObsoleteCandidate(
                    candidate_type="empty_run_directory",
                    path=run.run_directory,
                    analysis_type=run.analysis_type,
                    run_name=run.run_name,
                    status=run.likely_completion_status,
                    reason="Directory contains no generated files.",
                )
            )
            continue
        if _is_diagnostic(run):
            candidates.append(
                ObsoleteCandidate(
                    candidate_type="diagnostic_run_directory",
                    path=run.run_directory,
                    analysis_type=run.analysis_type,
                    run_name=run.run_name,
                    status=run.likely_completion_status,
                    reason="Smoke or diagnostic run is not preferred for supervisor/publication reporting.",
                )
            )
            continue
        selected_directory = selected_by_type.get(run.analysis_type)
        if selected_directory and run.run_directory != selected_directory:
            candidate_type = (
                "partially_completed_run"
                if run.likely_completion_status == COMPLETION_PARTIAL
                else "superseded_run_directory"
            )
            candidates.append(
                ObsoleteCandidate(
                    candidate_type=candidate_type,
                    path=run.run_directory,
                    analysis_type=run.analysis_type,
                    run_name=run.run_name,
                    status=run.likely_completion_status,
                    reason=f"Preferred {run.analysis_type} run is {selected_runs[run.analysis_type].run_name}.",
                    notes="Candidate is reported only; no output is modified or deleted.",
                )
            )

    for record in files:
        if record.size_bytes <= large_file_threshold_bytes:
            continue
        candidates.append(
            ObsoleteCandidate(
                candidate_type="large_generated_file",
                path=record.relative_path,
                analysis_type=record.analysis_type,
                run_name=record.run_name,
                status="REVIEW",
                reason="Large generated output should remain excluded from Git and was not hashed by default.",
                notes=f"size_bytes={record.size_bytes}",
            )
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.candidate_type.casefold(),
            item.analysis_type.casefold(),
            item.path.casefold(),
        ),
    )


def _build_run(
    key: tuple[str, str, str],
    records: list[InventoryFile],
) -> RunInventory:
    analysis_type, run_directory, run_name = key
    filenames = sorted({record.filename for record in records}, key=str.casefold)
    filename_set = {filename.casefold() for filename in filenames}
    rule = RUN_RULES.get(analysis_type)
    expected_files = sorted(
        set(rule.required_files + rule.optional_files) if rule else set(),
        key=str.casefold,
    )
    present_expected = [
        filename
        for filename in expected_files
        if filename.casefold() in filename_set
    ]
    missing_required: list[str] = []
    required_machine_expected = 0
    required_machine_present = 0

    if rule:
        required_machine_expected = len(rule.required_machine_readable)
        required_machine_present = sum(
            1
            for filename in rule.required_machine_readable
            if filename.casefold() in filename_set
        )
        for filename in rule.required_files:
            if filename.casefold() not in filename_set:
                missing_required.append(filename)
        for group_name, alternatives in rule.required_any:
            if not any(filename.casefold() in filename_set for filename in alternatives):
                missing_required.append(group_name)

    warnings: list[str] = []
    if _is_diagnostic_name(run_name, run_directory):
        warnings.append("Diagnostic or smoke run; not preferred for supervisor/publication reporting.")
    if missing_required:
        warnings.append("Missing required outputs: " + ", ".join(missing_required))

    completion_status = _completion_status(
        analysis_type=analysis_type,
        file_count=len(records),
        missing_required=missing_required,
        diagnostic=_is_diagnostic_name(run_name, run_directory),
        has_rule=rule is not None,
    )
    completion_ratio = _completion_ratio(rule, filename_set) if rule else 0.0
    modified_time = max(record.modified_time for record in records)
    run = RunInventory(
        analysis_type=analysis_type,
        run_name=run_name,
        run_directory=run_directory,
        run_version=_run_version(run_name),
        modified_time=modified_time,
        files_present=filenames,
        expected_files_present=present_expected,
        expected_files_missing=missing_required,
        likely_completion_status=completion_status,
        warnings=warnings,
        selection_score=0.0,
        file_count=len(records),
        required_machine_readable_present=required_machine_present,
        required_machine_readable_expected=required_machine_expected,
        figure_count=sum(1 for record in records if record.figure),
        report_count=sum(1 for record in records if record.report),
        completion_ratio=completion_ratio,
    )
    run.selection_score = float(sum(_selection_key(run)))
    return run


def _completion_status(
    *,
    analysis_type: str,
    file_count: int,
    missing_required: list[str],
    diagnostic: bool,
    has_rule: bool,
) -> str:
    if file_count == 0:
        return COMPLETION_EMPTY
    if diagnostic and not missing_required and has_rule:
        return COMPLETION_DIAGNOSTIC
    if has_rule and not missing_required:
        return COMPLETION_COMPLETE
    if has_rule:
        return COMPLETION_PARTIAL
    if analysis_type == "unknown":
        return COMPLETION_UNKNOWN
    return COMPLETION_PARTIAL


def _completion_ratio(rule: RunRule | None, filename_set: set[str]) -> float:
    if rule is None:
        return 0.0
    expected_count = len(rule.required_files) + len(rule.required_any)
    if expected_count == 0:
        return 0.0
    present_required = sum(1 for filename in rule.required_files if filename.casefold() in filename_set)
    present_groups = sum(
        1
        for _, alternatives in rule.required_any
        if any(filename.casefold() in filename_set for filename in alternatives)
    )
    return (present_required + present_groups) / expected_count


def _run_identity(record: InventoryFile) -> tuple[str, str]:
    parent = record.parent_directory
    if parent in {"tables", "figures", "reports"}:
        return parent, f"{parent}:{record.analysis_type}"
    if not parent:
        return "", record.run_name
    return parent, record.run_name


def _run_name_from_directory(directory: str) -> str:
    if not directory:
        return "outputs"
    parts = tuple(PurePosixPath(directory).parts)
    if parts and parts[-1] == "audit" and len(parts) >= 2:
        return f"{parts[-2]}/audit"
    return parts[-1] if parts else directory


def _analysis_type_from_directory(directory: str) -> str:
    parts = tuple(part.casefold() for part in PurePosixPath(directory).parts)
    if not parts:
        return "unknown"
    first = parts[0]
    mapping = {
        "classification": "classification",
        "regression": "regression",
        "exploratory": "exploratory analysis",
        "feature_engineering": "advanced feature engineering",
        "feature_validation": "feature validation",
        "fingerprints": "fingerprint generation",
        "features": "feature extraction",
        "qc": "canonical QC",
        "blind_prediction": "blind prediction",
    }
    if first.startswith("feature_selection"):
        return "feature selection"
    return mapping.get(first, "unknown")


def _selection_key(run: RunInventory) -> tuple[float, ...]:
    complete_score = 1.0 if run.likely_completion_status == COMPLETION_COMPLETE else 0.0
    non_diagnostic_score = 0.0 if _is_diagnostic(run) else 1.0
    machine_score = (
        run.required_machine_readable_present / run.required_machine_readable_expected
        if run.required_machine_readable_expected
        else 0.0
    )
    companion_score = min(1.0, (run.figure_count + run.report_count) / 2.0)
    timestamp_score = _timestamp_score(run.modified_time)
    version_score = _version_score(run.run_name)
    return (
        complete_score,
        non_diagnostic_score,
        machine_score,
        companion_score,
        timestamp_score,
        run.completion_ratio,
        version_score,
        -float(len(run.run_directory)),
    )


def _selection_reason(best: RunInventory, candidates: list[RunInventory]) -> str:
    complete_candidates = [
        run for run in candidates if run.likely_completion_status == COMPLETION_COMPLETE
    ]
    if best.likely_completion_status == COMPLETION_COMPLETE:
        reason = "Selected because it passes completeness requirements"
    else:
        reason = "Selected as the most complete available run"
    if best.required_machine_readable_expected:
        reason += (
            f", has {best.required_machine_readable_present}/"
            f"{best.required_machine_readable_expected} required machine-readable outputs"
        )
    if best.figure_count or best.report_count:
        reason += f", has {best.figure_count} figure(s) and {best.report_count} report(s)"
    reason += f", and was modified at {best.modified_time or 'unknown time'}."
    if complete_candidates and best in complete_candidates:
        reason += f" It was chosen from {len(complete_candidates)} complete candidate run(s)."
    if best.expected_files_missing:
        reason += " Missing required outputs: " + ", ".join(best.expected_files_missing) + "."
    return reason


def _timestamp_score(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _version_score(run_name: str) -> float:
    if "real" in run_name.casefold():
        base = 1000.0
    elif "optimised" in run_name.casefold() or "optimized" in run_name.casefold():
        base = 100.0
    else:
        base = 0.0
    numbers = [int(value) for value in re.findall(r"(\d+)", run_name)]
    return base + (float(numbers[-1]) if numbers else 0.0)


def _run_version(run_name: str) -> str:
    trailing_number = re.search(r"_(\d+)$", run_name)
    if trailing_number:
        return trailing_number.group(1)
    stage_suffix = re.search(r"stage_\d+[a-z]?_(.+)$", run_name)
    if stage_suffix:
        return stage_suffix.group(1)
    return run_name


def _is_diagnostic(run: RunInventory) -> bool:
    return _is_diagnostic_name(run.run_name, run.run_directory)


def _is_diagnostic_name(run_name: str, run_directory: str) -> bool:
    haystack = f"{run_name}/{run_directory}".casefold()
    return any(token in haystack for token in ("smoke", "debug", "tmp", "test"))
