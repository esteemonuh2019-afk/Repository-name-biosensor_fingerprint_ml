"""Stage 9B.2A evidence extraction orchestration."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.scientific_narrative.evidence_database import (
    EvidenceDatabase,
    EvidenceRecord,
    SourceParseStatus,
)
from src.scientific_narrative.parser_registry import default_registry
from src.scientific_narrative.result_parser import SourceContext


EXPECTED_METRICS: dict[str, dict[str, tuple[str, ...]]] = {
    "classification": {
        "best_model": ("best_model", "model_name"),
        "macro_f1": ("f1_macro_mean", "macro_f1_mean"),
        "weighted_f1": ("f1_weighted_mean", "weighted_f1_mean"),
        "balanced_accuracy": ("balanced_accuracy_mean",),
        "roc_auc": ("roc_auc_ovr_weighted_mean", "roc_auc_mean"),
        "precision": ("precision_macro_mean",),
        "recall": ("recall_macro_mean",),
        "fold_statistics": ("fold_count", "source_row_count"),
    },
    "regression": {
        "best_model": ("best_model", "model_name"),
        "r2": ("r2_mean",),
        "rmse": ("rmse_mean",),
        "mae": ("mae_mean",),
        "fold_statistics": ("fold_count", "source_row_count"),
    },
    "advanced feature engineering": {
        "best_feature_family": ("best_feature_family", "metadata_best_feature_family"),
        "classification_improvement": ("classification_improvement", "classification_macro_f1_gain"),
        "regression_improvement": ("regression_improvement", "regression_r2_gain"),
        "runtime_comparison": ("runtime_increase_seconds", "runtime_increase_seconds_mean"),
    },
    "feature selection": {
        "selected_feature_count": (
            "selected_feature_count",
            "default_classification_selected_feature_count",
            "default_regression_selected_feature_count",
        ),
        "retained_performance": ("retained_performance", "primary_metric"),
        "recommended_feature_set": ("recommended_feature_set",),
    },
    "fingerprint generation": {
        "fingerprint_count": ("fingerprint_rows",),
        "consensus_fingerprint_count": ("consensus_fingerprint_rows",),
        "similarity_summary": ("summary_distance_matrix_rows", "summary_consensus_distance_matrix_rows"),
    },
    "exploratory analysis": {
        "pca_variance_explained": ("explained_variance_ratio", "cumulative_explained_variance_ratio"),
        "clustering_summary": ("cluster_count", "cluster_size", "matrix_value"),
    },
    "canonical QC; feature validation": {
        "qc_pass_status": ("qc_passed", "validation_passed", "passed"),
        "failed_samples": ("failed_rows", "excluded_rows", "error_count"),
        "warnings": ("warning_count",),
    },
    "canonical QC": {
        "qc_pass_status": ("qc_passed",),
        "failed_samples": ("error_count", "conflicting_value_duplicate_count"),
        "warnings": ("warning_count",),
    },
    "real blind validation": {
        "blind_validation_status": ("prediction_passed", "predicted_chemical", "novelty_status"),
    },
}


def build_scientific_evidence(
    project_root: str | Path = ".",
    *,
    selected_results_path: str | Path = "outputs/results_inventory/selected_results.csv",
) -> EvidenceDatabase:
    """Build the unified evidence database from selected inventory outputs only."""

    started_at = _utc_now()
    root = Path(project_root).resolve()
    selected_path = _resolve(root, selected_results_path)
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)

    selected_rows = _read_selected_rows(selected_path)
    source_contexts = _selected_source_contexts(root, selected_rows)
    registry = default_registry()
    database = EvidenceDatabase(
        metadata={
            "project_root": str(root),
            "selected_results_path": str(selected_path),
            "selected_result_rows": len(selected_rows),
            "source_references_listed": len(source_contexts),
            "extraction_started_at": started_at,
        }
    )

    for resolved_path, context in source_contexts:
        suffix = resolved_path.suffix.casefold()
        status_base = {
            "source_file": context.source_file,
            "resolved_path": str(resolved_path),
            "analysis_type": context.analysis_type,
            "source_run": context.source_run,
        }
        parser = registry.parser_for(resolved_path)
        if parser is None:
            notes = (
                "Image ignored in Stage 9B.2A."
                if registry.is_ignored_image(resolved_path)
                else f"Unsupported Stage 9B.2A format: {suffix or '<none>'}."
            )
            database.unsupported_files.append(
                SourceParseStatus(parser_status="UNSUPPORTED", notes=notes, **status_base)
            )
            continue
        if not resolved_path.exists():
            database.unreadable_files.append(
                SourceParseStatus(
                    parser_status="UNREADABLE",
                    notes="Listed file does not exist.",
                    **status_base,
                )
            )
            continue
        try:
            parse_result = parser(resolved_path, context)
        except Exception as error:  # noqa: BLE001 - extraction should report all unreadable files.
            database.unreadable_files.append(
                SourceParseStatus(
                    parser_status="UNREADABLE",
                    notes=f"{type(error).__name__}: {error}",
                    **status_base,
                )
            )
            continue

        database.records.extend(parse_result.records)
        database.warnings.extend(parse_result.warnings)
        database.parsed_files.append(
            SourceParseStatus(
                parser_status="PARSED",
                evidence_count=len(parse_result.records),
                notes="Parsed from selected_results.csv reference.",
                **status_base,
            )
        )

    missing = _missing_expected_evidence(database.records, source_contexts)
    database.records.extend(missing)
    database.missing_evidence.extend(missing)
    database.metadata.update(
        {
            "extraction_finished_at": _utc_now(),
            "supported_files_parsed": len(database.parsed_files),
            "unsupported_files": len(database.unsupported_files),
            "unreadable_files": len(database.unreadable_files),
            "missing_expected_metrics": len(database.missing_evidence),
        }
    )
    return database


def _read_selected_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def _selected_source_contexts(
    project_root: Path,
    selected_rows: list[dict[str, str]],
) -> list[tuple[Path, SourceContext]]:
    contexts: list[tuple[Path, SourceContext]] = []
    seen: set[str] = set()
    for row in selected_rows:
        references = [row.get("selected_file", ""), *_split_companions(row.get("companion_files", ""))]
        for reference in references:
            reference = reference.strip()
            if not reference:
                continue
            if reference in seen:
                continue
            seen.add(reference)
            context = SourceContext(
                analysis_type=row.get("analysis_type", ""),
                source_file=reference,
                source_run=row.get("selected_run", ""),
                report_section=row.get("report_section", ""),
                scientific_role=row.get("scientific_role", ""),
                include_in_supervisor_report=str(row.get("include_in_supervisor_report", "")).casefold() == "true",
            )
            contexts.append((_resolve_selected_reference(project_root, reference), context))
    return sorted(contexts, key=lambda item: item[1].source_file.casefold())


def _split_companions(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def _resolve_selected_reference(project_root: Path, reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path.resolve()
    normalised = reference.replace("\\", "/")
    if normalised.startswith("docs/"):
        return (project_root / normalised).resolve()
    return (project_root / "outputs" / normalised).resolve()


def _resolve(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _missing_expected_evidence(
    records: list[EvidenceRecord],
    contexts: list[tuple[Path, SourceContext]],
) -> list[EvidenceRecord]:
    analysis_types = sorted({context.analysis_type for _, context in contexts if context.analysis_type})
    missing: list[EvidenceRecord] = []
    for analysis_type in analysis_types:
        expected = _expected_for_analysis(analysis_type)
        if not expected:
            continue
        source_run = _first_source_run(analysis_type, contexts)
        for label, aliases in expected.items():
            if _has_any_metric(records, analysis_type, aliases):
                continue
            missing.append(
                EvidenceRecord(
                    analysis_type=analysis_type,
                    source_file="",
                    source_run=source_run,
                    metric_name=label,
                    metric_value=None,
                    confidence="NULL",
                    extraction_status="NULL",
                    notes="Expected headline evidence was unavailable from listed selected sources.",
                )
            )
    return missing


def _expected_for_analysis(analysis_type: str) -> dict[str, tuple[str, ...]]:
    if analysis_type in EXPECTED_METRICS:
        return EXPECTED_METRICS[analysis_type]
    for key, expected in EXPECTED_METRICS.items():
        if key in analysis_type:
            return expected
    return {}


def _has_any_metric(
    records: list[EvidenceRecord],
    analysis_type: str,
    aliases: tuple[str, ...],
) -> bool:
    alias_set = {alias.casefold() for alias in aliases}
    for record in records:
        if record.analysis_type != analysis_type:
            continue
        metric = record.metric_name.casefold()
        if metric in alias_set or any(metric.endswith(f"_{alias}") for alias in alias_set):
            return True
    return False


def _first_source_run(
    analysis_type: str,
    contexts: list[tuple[Path, SourceContext]],
) -> str:
    for _, context in contexts:
        if context.analysis_type == analysis_type:
            return context.source_run
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
