"""Path-aware scientific classification for generated project outputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import PurePosixPath
import re

from src.results_inventory.inventory_models import InventoryFile


MACHINE_READABLE_EXTENSIONS = {
    ".csv",
    ".json",
    ".tsv",
    ".xlsx",
    ".xls",
}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
REPORT_EXTENSIONS = {".md", ".txt", ".docx"}

GENERATOR_BY_ANALYSIS_TYPE = {
    "canonical ingestion": "scripts/build_canonical_dataset.py",
    "canonical QC": "scripts/audit_canonical_dataset.py",
    "feature extraction": "scripts/build_feature_dataset.py",
    "feature validation": "scripts/validate_feature_dataset.py",
    "fingerprint generation": "scripts/build_fingerprint_dataset.py",
    "exploratory analysis": "scripts/run_exploratory_fingerprint_analysis.py",
    "classification": "scripts/run_classification_benchmark.py",
    "regression": "scripts/run_regression_benchmark.py",
    "advanced feature engineering": "scripts/run_feature_engineering_v2_benchmark.py",
    "feature selection": "scripts/run_feature_selection.py",
    "blind prediction": "scripts/predict_blind_sample.py",
    "scientific report": "src/reporting/report.py",
    "results inventory": "scripts/build_results_inventory.py",
}


def classify_files(files: Iterable[InventoryFile]) -> list[InventoryFile]:
    """Classify discovered files using filenames, paths, and nearby companions."""

    records = sorted(files, key=lambda item: item.relative_path.casefold())
    companions_by_parent: dict[str, set[str]] = {}
    for record in records:
        companions_by_parent.setdefault(record.parent_directory, set()).add(record.filename.casefold())
    return [
        classify_file(record, companions_by_parent.get(record.parent_directory, set()))
        for record in records
    ]


def classify_file(
    record: InventoryFile,
    companion_filenames: Iterable[str] | None = None,
) -> InventoryFile:
    """Return a classified copy of one inventory record."""

    companions = {name.casefold() for name in (companion_filenames or [])}
    path = PurePosixPath(record.relative_path)
    parts = tuple(part.casefold() for part in path.parts)
    name = record.filename.casefold()
    stem = PurePosixPath(record.filename).stem.casefold()
    extension = record.extension.casefold()
    path_text = record.relative_path.casefold()

    analysis_stage, analysis_type = _analysis_from_path(parts, stem, companions)
    result_role = _result_role(analysis_type, stem, name, path_text, companions)
    run_name = _run_name(record.parent_directory)
    run_version = _run_version(run_name)
    likely_generator = _generator_script(analysis_type, result_role, stem)

    machine_readable = extension in MACHINE_READABLE_EXTENSIONS
    table = extension in TABLE_EXTENSIONS or stem.endswith("_table")
    report = extension in REPORT_EXTENSIONS or "report" in stem
    figure = extension in FIGURE_EXTENSIONS and not report
    model_metric = _is_model_metric(stem, analysis_type, result_role)
    qc_output = "qc" in path_text or result_role in {
        "canonical QC",
        "feature QC",
        "fingerprint QC",
    }

    diagnostic = _is_diagnostic_run(run_name, path_text)
    include_candidate = (
        result_role != "unknown"
        and not diagnostic
        and analysis_type not in {"unknown", "results inventory"}
        and (machine_readable or figure or report)
    )
    status = "classified" if result_role != "unknown" or analysis_type != "unknown" else "unknown"
    notes = _classification_notes(
        record.notes,
        analysis_type=analysis_type,
        result_role=result_role,
        diagnostic=diagnostic,
        companions=companions,
    )

    return replace(
        record,
        analysis_stage=analysis_stage,
        analysis_type=analysis_type,
        result_role=result_role,
        run_name=run_name,
        run_version=run_version,
        likely_generator_script=likely_generator,
        machine_readable=machine_readable,
        figure=figure,
        table=table,
        report=report,
        model_metric=model_metric,
        QC_output=qc_output,
        include_candidate=include_candidate,
        status=status,
        notes=notes,
    )


def _analysis_from_path(
    parts: tuple[str, ...],
    stem: str,
    companions: set[str],
) -> tuple[str, str]:
    first = parts[0] if parts else ""
    path_text = "/".join(parts)

    if "results_inventory" in parts:
        return "stage_9b1", "results inventory"
    if first == "classification":
        return "stage_8a", "classification"
    if first == "regression":
        return "stage_8b", "regression"
    if first == "exploratory":
        return "stage_7b", "exploratory analysis"
    if first == "feature_engineering":
        return "stage_8c", "advanced feature engineering"
    if first.startswith("feature_selection"):
        return "stage_8d", "feature selection"
    if first == "feature_validation":
        return "stage_6c", "feature validation"
    if first == "fingerprints":
        return "stage_7a", "fingerprint generation"
    if first == "features":
        return "stage_6b", "feature extraction"
    if first == "qc":
        stage = "stage_5b" if "stage_5b" in path_text else "stage_5c"
        return stage, "canonical QC"
    if first == "blind_prediction":
        return "stage_9a", "blind prediction"
    if first == "reports":
        return "legacy_report", "scientific report"

    if first == "tables":
        return _analysis_from_legacy_table(stem, companions)
    if first == "figures":
        return _analysis_from_legacy_figure(stem)

    return "unknown", "unknown"


def _analysis_from_legacy_table(stem: str, companions: set[str]) -> tuple[str, str]:
    if stem in {"processed_data", "cleaned_data", "phase_2a_file_inventory"}:
        return "stage_4b", "canonical ingestion"
    if stem in {"features", "features_normalized"}:
        return "stage_6b", "feature extraction"
    if "panel_optimization" in stem:
        return "legacy_model_evaluation", "reduced-array optimisation"
    if "single_strain" in stem or "leave_one_strain_out" in stem or "strain_rankings" in stem:
        return "legacy_model_evaluation", "strain ablation"
    if stem in {"features_advanced"} or stem.startswith("advanced_"):
        return "legacy_advanced_ml", "advanced feature engineering"
    if "regression" in stem or {"prediction_vs_actual.csv", "residuals.csv"} & companions:
        return "stage_8b", "regression"
    if (
        "loeo" in stem
        or "model_metrics" in stem
        or "feature_importance" in stem
        or "confidence_interval" in stem
        or "repeated_run" in stem
        or "per_chemical" in stem
        or "specialist_ensemble" in stem
    ):
        return "legacy_model_evaluation", "classification"
    return "unknown", "unknown"


def _analysis_from_legacy_figure(stem: str) -> tuple[str, str]:
    if "panel_optimization" in stem or "advanced_panel" in stem:
        return "legacy_model_evaluation", "reduced-array optimisation"
    if "single_strain" in stem or "leave_one_strain" in stem or "strain_heatmap" in stem:
        return "legacy_model_evaluation", "strain ablation"
    if "repeated_run" in stem:
        return "legacy_model_evaluation", "classification"
    if "pca" in stem or "heatmap" in stem or "time_course" in stem or "dose_response" in stem:
        return "legacy_exploratory", "exploratory analysis"
    if "confusion_matrix" in stem or "feature_importance" in stem or "loeo" in stem:
        return "legacy_model_evaluation", "classification"
    return "unknown", "unknown"


def _result_role(
    analysis_type: str,
    stem: str,
    name: str,
    path_text: str,
    companions: set[str],
) -> str:
    if analysis_type == "results inventory":
        return "supplementary material"
    if analysis_type == "canonical ingestion":
        return "dataset summary"
    if analysis_type == "canonical QC":
        return "canonical QC"
    if analysis_type == "feature extraction":
        if "qc" in stem:
            return "feature QC"
        return "dataset summary"
    if analysis_type == "feature validation":
        return "feature QC"
    if analysis_type == "fingerprint generation":
        if "qc" in stem or "summary" in stem:
            return "fingerprint QC"
        if "distance_matrix" in stem:
            return "chemical similarity"
        return "fingerprint QC"
    if analysis_type == "exploratory analysis":
        return _exploratory_role(stem, name, path_text)
    if analysis_type == "classification":
        return _classification_role(stem)
    if analysis_type == "regression":
        return _regression_role(stem)
    if analysis_type == "advanced feature engineering":
        return _feature_engineering_role(stem)
    if analysis_type == "feature selection":
        return "feature selection"
    if analysis_type == "strain ablation":
        return "strain ablation"
    if analysis_type == "reduced-array optimisation":
        return "reduced-array optimisation"
    if analysis_type == "blind prediction":
        return "blind prediction"
    if analysis_type == "scientific report":
        if "supplement" in stem:
            return "supplementary material"
        return "scientific report"

    if "report" in stem and (name.endswith(".md") or name.endswith(".docx")):
        return "scientific report"
    if "summary" in stem and companions:
        return "dataset summary"
    return "unknown"


def _exploratory_role(stem: str, name: str, path_text: str) -> str:
    if "chemical_similarity" in stem:
        return "chemical similarity"
    if "concentration_trajectory" in stem or "concentration_trajectories" in stem:
        return "concentration trajectories"
    if "concentration_response" in stem:
        return "concentration trajectories"
    if "cluster" in stem or "dendrogram" in stem:
        return "clustering"
    if "pca" in stem or "component" in stem or "loading" in stem:
        return "PCA"
    if "heatmap" in stem:
        return "heatmaps"
    if "strain_dispersion" in stem or "replicate_to_consensus" in stem:
        return "fingerprint QC"
    if "exploratory_summary" in stem or "exploratory_analysis_report" in stem:
        return "PCA" if "pca_scores.csv" in path_text else "scientific report"
    if name.endswith(".png") or name.endswith(".pdf"):
        return "heatmaps"
    return "unknown"


def _classification_role(stem: str) -> str:
    if "confusion_matrix" in stem:
        return "confusion matrix"
    if "per_class" in stem:
        return "per-class metrics"
    if "feature_importance" in stem or "permutation_importance" in stem:
        return "classification feature importance"
    if "leave_one_strain_importance" in stem:
        return "classification feature importance"
    if (
        "summary" in stem
        or "model_metrics" in stem
        or "best_model_metrics" in stem
        or "model_rankings" in stem
        or "fold_metrics" in stem
        or "classification_report" in stem
        or "loeo" in stem
        or "confidence_interval" in stem
        or "per_chemical" in stem
        or "specialist_ensemble" in stem
        or "repeated_run" in stem
    ):
        return "classification performance"
    return "unknown"


def _regression_role(stem: str) -> str:
    if "prediction_vs_actual" in stem:
        return "prediction-versus-actual"
    if "residual" in stem:
        return "residual analysis"
    if "feature_importance" in stem or "permutation_importance" in stem:
        return "regression feature importance"
    if "leave_one_strain_importance" in stem:
        return "regression feature importance"
    if (
        "regression_summary" in stem
        or "best_regression_model" in stem
        or "per_model_metrics" in stem
        or "fold_metrics" in stem
        or "model_rankings" in stem
        or "regression_report" in stem
        or "fold_performance" in stem
    ):
        return "regression performance"
    return "unknown"


def _feature_engineering_role(stem: str) -> str:
    if stem == "features_advanced" or stem.startswith("advanced_"):
        return "feature-family benchmark"
    if "advanced_feature" in stem:
        return "feature-family benchmark"
    if "feature_family" in stem:
        return "feature-family benchmark"
    if "stage_8c" in stem:
        return "feature-family benchmark"
    if "classification_improvement" in stem or "regression_improvement" in stem:
        return "feature-family benchmark"
    if "runtime_comparison" in stem:
        return "feature-family benchmark"
    return "unknown"


def _is_model_metric(stem: str, analysis_type: str, result_role: str) -> bool:
    if analysis_type in {"classification", "regression"}:
        return any(
            token in stem
            for token in (
                "metric",
                "summary",
                "ranking",
                "confusion_matrix",
                "prediction_vs_actual",
                "residual",
                "fold",
                "performance",
            )
        )
    return result_role in {
        "classification performance",
        "regression performance",
        "feature-family benchmark",
        "feature selection",
        "strain ablation",
        "reduced-array optimisation",
    } and any(token in stem for token in ("metric", "summary", "performance", "ranking", "importance"))


def _generator_script(analysis_type: str, result_role: str, stem: str) -> str:
    if result_role == "strain ablation":
        return "scripts/run_strain_ablation.py"
    if result_role == "reduced-array optimisation":
        return "scripts/run_panel_optimization.py"
    if "confidence_interval" in stem:
        return "scripts/run_confidence_intervals.py"
    if "per_chemical" in stem:
        return "scripts/run_per_chemical_analysis.py"
    return GENERATOR_BY_ANALYSIS_TYPE.get(analysis_type, "")


def _run_name(parent_directory: str) -> str:
    if not parent_directory or parent_directory == ".":
        return "outputs"
    parts = tuple(part for part in PurePosixPath(parent_directory).parts if part)
    if not parts:
        return "outputs"
    if parts[-1] == "audit" and len(parts) >= 2:
        return f"{parts[-2]}/audit"
    return parts[-1]


def _run_version(run_name: str) -> str:
    if not run_name or run_name == "unknown":
        return ""
    trailing_number = re.search(r"_(\d+)$", run_name)
    if trailing_number:
        return trailing_number.group(1)
    stage_suffix = re.search(r"stage_\d+[a-z]?_(.+)$", run_name)
    if stage_suffix:
        return stage_suffix.group(1)
    feature_selection_suffix = re.search(r"feature_selection_(\d+)$", run_name)
    if feature_selection_suffix:
        return feature_selection_suffix.group(1)
    return run_name


def _is_diagnostic_run(run_name: str, path_text: str) -> bool:
    diagnostic_tokens = ("smoke", "tmp", "debug", "test")
    haystack = f"{run_name}/{path_text}".casefold()
    return any(token in haystack for token in diagnostic_tokens)


def _classification_notes(
    existing_notes: str,
    *,
    analysis_type: str,
    result_role: str,
    diagnostic: bool,
    companions: set[str],
) -> str:
    notes: list[str] = []
    if existing_notes:
        notes.append(existing_notes)
    if result_role != "unknown":
        notes.append("Classified from path, filename, extension, and companion filenames.")
    elif analysis_type != "unknown":
        notes.append("Analysis stage inferred from path; scientific role remains unknown.")
    else:
        notes.append("No known project convention matched this file.")
    if companions:
        notes.append(f"Companion file count in directory: {len(companions)}.")
    if diagnostic:
        notes.append("Diagnostic or smoke run; excluded from publication-preferred selection.")
    return " ".join(notes)
