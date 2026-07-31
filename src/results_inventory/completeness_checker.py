"""Supervisor-report completeness and project-health assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.results_inventory.inventory_models import (
    HEALTH_COMPLETE,
    HEALTH_MISSING,
    HEALTH_NOT_APPLICABLE,
    HEALTH_PARTIAL,
    SECTION_FOUND,
    SECTION_MISSING,
    SECTION_NOT_APPLICABLE,
    SECTION_PARTIAL,
    InventoryFile,
    MissingResult,
    RunInventory,
    SelectedResult,
)


CORE_REPORT_SECTIONS = {
    "Dataset summary",
    "Data-quality summary",
    "Fingerprint summary",
    "PCA/exploratory analysis",
    "Chemical similarity or clustering",
    "Classification results",
    "Regression results",
    "Feature engineering results",
    "Feature-selection results",
    "Limitations",
}


def assess_completeness(
    *,
    project_root: str | Path,
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> dict[str, Any]:
    """Assess supervisor-report readiness without generating that report."""

    root = Path(project_root).resolve()
    sections = [
        _dataset_summary(files, selected_runs),
        _data_quality(files, selected_runs),
        _fingerprint_summary(files, selected_runs),
        _exploratory(files, selected_runs),
        _chemical_similarity_or_clustering(files, selected_runs),
        _classification(files, selected_runs),
        _regression(files, selected_runs),
        _feature_engineering(files, selected_runs),
        _feature_selection(files, selected_runs),
        _strain_contribution(files),
        _limitations(root),
        _blind_validation(files),
    ]
    selected_results = [_selected_result_from_missing_result(item, files, selected_runs) for item in sections]
    project_health = _project_health(root, files, selected_runs, sections)
    warnings = _section_warnings(sections)
    return {
        "missing_required_results": sections,
        "selected_results": selected_results,
        "project_health": project_health,
        "warnings": warnings,
        "errors": [],
    }


def _dataset_summary(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("feature extraction")
    candidates = _records_for_run(files, selected) if selected else []
    required = ["feature_dataset.csv", "feature_summary.json", "feature_qc_report.md"]
    found = _relative_paths_named(candidates, required)
    status = _found_status(found, required, selected)
    if status == SECTION_MISSING:
        fallback = [
            record.relative_path
            for record in files
            if record.analysis_type == "canonical ingestion" and record.result_role == "dataset summary"
        ]
        found = sorted(fallback)
        status = SECTION_FOUND if fallback else SECTION_MISSING
    return MissingResult(
        report_section="Dataset summary",
        analysis_type="feature extraction",
        status=status,
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Uses Stage 6B feature output as the primary generated dataset summary.",
    )


def _data_quality(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    required = ["qc_summary.json", "canonical_qc_report.md", "feature_validation_report.md"]
    qc_run = selected_runs.get("canonical QC")
    validation_run = selected_runs.get("feature validation")
    found_records = []
    if qc_run:
        found_records.extend(_records_for_run(files, qc_run))
    if validation_run:
        found_records.extend(_records_for_run(files, validation_run))
    found = _relative_paths_named(found_records, required)
    status = SECTION_FOUND if len(found) >= 3 else SECTION_PARTIAL if found else SECTION_MISSING
    return MissingResult(
        report_section="Data-quality summary",
        analysis_type="canonical QC; feature validation",
        status=status,
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Combines canonical QC and feature-validation evidence.",
    )


def _fingerprint_summary(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("fingerprint generation")
    required = ["fingerprint_dataset.csv", "fingerprint_summary.json", "fingerprint_qc_report.md"]
    found = _relative_paths_named(_records_for_run(files, selected), required) if selected else []
    return MissingResult(
        report_section="Fingerprint summary",
        analysis_type="fingerprint generation",
        status=_found_status(found, required, selected),
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Requires validated Stage 7A fingerprint artifacts.",
    )


def _exploratory(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("exploratory analysis")
    required = ["pca_scores.csv", "pca_loadings.csv", "pca_explained_variance.csv"]
    found = _relative_paths_named(_records_for_run(files, selected), required) if selected else []
    return MissingResult(
        report_section="PCA/exploratory analysis",
        analysis_type="exploratory analysis",
        status=_found_status(found, required, selected),
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Focuses on PCA tables from the selected exploratory run.",
    )


def _chemical_similarity_or_clustering(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("exploratory analysis")
    records = _records_for_run(files, selected) if selected else []
    found_records = [
        record
        for record in records
        if record.result_role in {"chemical similarity", "clustering", "heatmaps"}
    ]
    status = SECTION_FOUND if found_records else SECTION_MISSING
    return MissingResult(
        report_section="Chemical similarity or clustering",
        analysis_type="exploratory analysis",
        status=status,
        required_results=["chemical similarity or clustering output"],
        found_results=[record.relative_path for record in found_records],
        missing_results=[] if found_records else ["chemical similarity or clustering output"],
        notes="Accepts either chemical-similarity heatmaps/tables or clustering tables from Stage 7B.",
    )


def _classification(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("classification")
    required = [
        "classification_summary.csv",
        "best_model_metrics.json",
        "confusion_matrix.csv",
        "per_class_metrics.csv",
        "model_rankings.csv",
        "classification_report.md",
    ]
    found = _relative_paths_named(_records_for_run(files, selected), required) if selected else []
    return MissingResult(
        report_section="Classification results",
        analysis_type="classification",
        status=_found_status(found, required, selected),
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Uses the preferred complete Stage 8A run.",
    )


def _regression(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("regression")
    required = [
        "regression_summary.csv",
        "best_regression_model.json",
        "prediction_vs_actual.csv",
        "residuals.csv",
        "model_rankings.csv",
        "regression_report.md",
    ]
    found = _relative_paths_named(_records_for_run(files, selected), required) if selected else []
    return MissingResult(
        report_section="Regression results",
        analysis_type="regression",
        status=_found_status(found, required, selected),
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Uses the preferred complete Stage 8B run.",
    )


def _feature_engineering(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("advanced feature engineering")
    required = [
        "advanced_feature_dataset.csv",
        "feature_family_ablation_summary.csv",
        "stage_8c_summary.json",
        "stage_8c_feature_engineering_report.md",
    ]
    found = _relative_paths_named(_records_for_run(files, selected), required) if selected else []
    return MissingResult(
        report_section="Feature engineering results",
        analysis_type="advanced feature engineering",
        status=_found_status(found, required, selected),
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Uses Stage 8C advanced feature-family benchmark outputs.",
    )


def _feature_selection(
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> MissingResult:
    selected = selected_runs.get("feature selection")
    required = [
        "selected_features.csv",
        "feature_ranking.csv",
        "feature_selection_summary.csv",
        "classification_after_selection.csv",
        "regression_after_selection.csv",
        "performance_vs_feature_count.csv",
        "feature_selection_report.md",
    ]
    found = _relative_paths_named(_records_for_run(files, selected), required) if selected else []
    return MissingResult(
        report_section="Feature-selection results",
        analysis_type="feature selection",
        status=_found_status(found, required, selected),
        required_results=required,
        found_results=found,
        missing_results=_missing_names(required, found),
        notes="Uses the preferred Stage 8D feature-selection run.",
    )


def _strain_contribution(files: list[InventoryFile]) -> MissingResult:
    records = sorted(
        [
            record
            for record in files
            if record.result_role == "strain ablation"
            or "leave_one_strain" in record.filename.casefold()
            or "single_strain" in record.filename.casefold()
        ],
        key=lambda record: (
            0 if record.result_role == "strain ablation" else 1,
            record.relative_path.casefold(),
        ),
    )
    if not records:
        status = SECTION_NOT_APPLICABLE
        missing = []
        notes = "No strain-contribution or ablation outputs were detected."
    else:
        status = SECTION_FOUND
        missing = []
        notes = "Strain contribution/ablation outputs were detected and can be considered."
    return MissingResult(
        report_section="Strain contribution or ablation, if available",
        analysis_type="strain ablation",
        status=status,
        required_results=["strain contribution or ablation output, if available"],
        found_results=[record.relative_path for record in records],
        missing_results=missing,
        notes=notes,
    )


def _limitations(project_root: Path) -> MissingResult:
    candidates = [
        project_root / "docs" / "LIMITATIONS_AND_RISKS.md",
        project_root / "docs" / "UNCERTAINTY_ANALYSIS.md",
        project_root / "docs" / "ML_VALIDITY_AUDIT.md",
    ]
    found = [
        path.relative_to(project_root).as_posix()
        for path in candidates
        if path.exists()
    ]
    status = SECTION_FOUND if found else SECTION_MISSING
    return MissingResult(
        report_section="Limitations",
        analysis_type="documentation",
        status=status,
        required_results=["docs/LIMITATIONS_AND_RISKS.md or equivalent limitation evidence"],
        found_results=found,
        missing_results=[] if found else ["limitations documentation"],
        notes="Documentation can supply the limitations section; no supervisor report is generated here.",
    )


def _blind_validation(files: list[InventoryFile]) -> MissingResult:
    real_evaluation = [
        record
        for record in files
        if record.analysis_type == "blind prediction"
        and any(
            token in record.relative_path.casefold()
            for token in ("truth", "evaluation", "validated", "reveal")
        )
    ]
    prediction_outputs = [
        record.relative_path
        for record in files
        if record.analysis_type == "blind prediction"
    ]
    if real_evaluation:
        status = SECTION_FOUND
        notes = "Truth-reveal blind-validation evidence was detected."
        found = [record.relative_path for record in real_evaluation]
        missing: list[str] = []
    else:
        status = SECTION_NOT_APPLICABLE
        notes = (
            "Blind-prediction outputs may exist, but no truth-reveal evaluation was detected; "
            "real blind validation remains not yet available."
        )
        found = prediction_outputs
        missing = []
    return MissingResult(
        report_section="Blind validation status",
        analysis_type="real blind validation",
        status=status,
        required_results=["truth-reveal blind validation output, when available"],
        found_results=found,
        missing_results=missing,
        notes=notes,
    )


def _project_health(
    project_root: Path,
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
    sections: list[MissingResult],
) -> dict[str, Any]:
    health = {
        "ingestion": _ingestion_health(files),
        "canonical QC": _run_health(selected_runs.get("canonical QC")),
        "feature extraction": _run_health(selected_runs.get("feature extraction")),
        "feature validation": _run_health(selected_runs.get("feature validation")),
        "fingerprint generation": _run_health(selected_runs.get("fingerprint generation")),
        "exploratory analysis": _run_health(selected_runs.get("exploratory analysis")),
        "classification": _run_health(selected_runs.get("classification")),
        "regression": _run_health(selected_runs.get("regression")),
        "feature engineering": _run_health(selected_runs.get("advanced feature engineering")),
        "feature selection": _run_health(selected_runs.get("feature selection")),
        "blind prediction infrastructure": _blind_infrastructure_health(project_root, files, selected_runs),
        "real blind validation": _real_blind_health(sections),
        "supervisor report": HEALTH_MISSING,
    }
    missing_core_sections = [
        section.report_section
        for section in sections
        if section.report_section in CORE_REPORT_SECTIONS and section.status == SECTION_MISSING
    ]
    health["report_generation_can_proceed"] = not missing_core_sections
    health["report_generation_recommendation"] = (
        "Proceed with supervisor-report generation using selected outputs; note that real blind validation is not yet available."
        if not missing_core_sections
        else "Do not proceed until missing core sections are addressed: " + "; ".join(missing_core_sections)
    )
    return health


def _selected_result_from_missing_result(
    section: MissingResult,
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> SelectedResult:
    selected_file = section.found_results[0] if section.found_results else ""
    companion_files = section.found_results[1:]
    selected_run = _selected_run_for_section(section, selected_runs)
    role = _role_for_selected_file(selected_file, files) if selected_file else section.analysis_type
    include = section.status in {SECTION_FOUND, SECTION_PARTIAL} and bool(selected_file)
    reason = (
        f"{section.status}: {section.notes}"
        if selected_file
        else f"{section.status}: no selectable output file."
    )
    return SelectedResult(
        report_section=section.report_section,
        analysis_type=section.analysis_type,
        selected_file=selected_file,
        selected_run=selected_run,
        status=section.status,
        selection_reason=reason,
        companion_files=companion_files,
        scientific_role=role,
        include_in_supervisor_report=include,
        notes=section.notes,
    )


def _selected_run_for_section(
    section: MissingResult,
    selected_runs: dict[str, RunInventory],
) -> str:
    mapping = {
        "Dataset summary": "feature extraction",
        "Data-quality summary": "canonical QC",
        "Fingerprint summary": "fingerprint generation",
        "PCA/exploratory analysis": "exploratory analysis",
        "Chemical similarity or clustering": "exploratory analysis",
        "Classification results": "classification",
        "Regression results": "regression",
        "Feature engineering results": "advanced feature engineering",
        "Feature-selection results": "feature selection",
    }
    analysis_type = mapping.get(section.report_section)
    if analysis_type and analysis_type in selected_runs:
        return selected_runs[analysis_type].run_name
    if section.report_section == "Limitations":
        return "documentation"
    if section.report_section == "Blind validation status":
        return "not yet available"
    return ""


def _role_for_selected_file(path: str, files: list[InventoryFile]) -> str:
    for record in files:
        if record.relative_path == path:
            return record.result_role
    return "documentation"


def _records_for_run(
    files: list[InventoryFile],
    run: RunInventory | None,
) -> list[InventoryFile]:
    if run is None:
        return []
    return sorted(
        [
            record
            for record in files
            if record.analysis_type == run.analysis_type
            and _record_run_directory(record) == run.run_directory
        ],
        key=lambda record: record.relative_path.casefold(),
    )


def _record_run_directory(record: InventoryFile) -> str:
    if record.parent_directory in {"tables", "figures", "reports"}:
        return record.parent_directory
    return record.parent_directory


def _relative_paths_named(records: list[InventoryFile], filenames: list[str]) -> list[str]:
    wanted = {filename.casefold() for filename in filenames}
    return [
        record.relative_path
        for record in records
        if record.filename.casefold() in wanted
    ]


def _missing_names(required: list[str], found_paths: list[str]) -> list[str]:
    found_names = {Path(path).name.casefold() for path in found_paths}
    return [
        filename
        for filename in required
        if filename.casefold() not in found_names
    ]


def _found_status(
    found: list[str],
    required: list[str],
    selected_run: RunInventory | None,
) -> str:
    if not found:
        return SECTION_MISSING
    if selected_run is not None and selected_run.likely_completion_status == "COMPLETE":
        return SECTION_FOUND
    if len(found) == len(required):
        return SECTION_FOUND
    return SECTION_PARTIAL


def _run_health(run: RunInventory | None) -> str:
    if run is None:
        return HEALTH_MISSING
    if run.likely_completion_status == "COMPLETE":
        return HEALTH_COMPLETE
    if run.likely_completion_status in {"PARTIAL", "DIAGNOSTIC", "UNKNOWN"}:
        return HEALTH_PARTIAL
    return HEALTH_MISSING


def _ingestion_health(files: list[InventoryFile]) -> str:
    names = {record.relative_path.casefold() for record in files}
    if "tables/processed_data.csv" in names or "tables/cleaned_data.csv" in names:
        return HEALTH_COMPLETE
    if any(record.analysis_type == "canonical ingestion" for record in files):
        return HEALTH_PARTIAL
    return HEALTH_MISSING


def _blind_infrastructure_health(
    project_root: Path,
    files: list[InventoryFile],
    selected_runs: dict[str, RunInventory],
) -> str:
    source_present = (project_root / "src" / "blind_prediction").exists()
    script_present = (project_root / "scripts" / "train_blind_prediction_models.py").exists()
    model_metadata = list((project_root / "models" / "blind_prediction").glob("*/model_metadata.json"))
    selected_output = selected_runs.get("blind prediction")
    if source_present and script_present and model_metadata:
        return HEALTH_COMPLETE
    if source_present and script_present and (model_metadata or selected_output or any(record.analysis_type == "blind prediction" for record in files)):
        return HEALTH_PARTIAL
    return HEALTH_MISSING


def _real_blind_health(sections: list[MissingResult]) -> str:
    blind = next(
        (section for section in sections if section.report_section == "Blind validation status"),
        None,
    )
    if blind is None:
        return HEALTH_NOT_APPLICABLE
    if blind.status == SECTION_FOUND:
        return HEALTH_COMPLETE
    return HEALTH_NOT_APPLICABLE


def _section_warnings(sections: list[MissingResult]) -> list[str]:
    warnings: list[str] = []
    for section in sections:
        if section.status == SECTION_MISSING:
            warnings.append(f"Missing supervisor-report section: {section.report_section}")
        elif section.status == SECTION_PARTIAL:
            warnings.append(f"Partial supervisor-report section: {section.report_section}")
    return warnings
