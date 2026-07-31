"""Production factual observation rules for supervisor-results packages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .enums import ConfidenceLevel, ObservationCategory, ObservationStatus, category_id_token
from .models import Observation, ProvenanceRecord, SupportingMetric, ValidationIssue
from .source_loader import SupervisorSourcePayload


CATEGORY_ORDER: tuple[ObservationCategory, ...] = (
    ObservationCategory.DATASET,
    ObservationCategory.QUALITY_CONTROL,
    ObservationCategory.FINGERPRINT,
    ObservationCategory.EXPLORATORY_ANALYSIS,
    ObservationCategory.CLASSIFICATION,
    ObservationCategory.REGRESSION,
    ObservationCategory.FEATURE_ENGINEERING,
    ObservationCategory.FEATURE_SELECTION,
    ObservationCategory.STRAIN_CONTRIBUTION,
    ObservationCategory.BLIND_PREDICTION,
    ObservationCategory.VALIDATION,
)


@dataclass(frozen=True)
class ObservationRuleResult:
    observations: tuple[Observation, ...]
    validation_issues: tuple[ValidationIssue, ...]


class ProvenanceResolver:
    """Resolve matching provenance rows without inventing provenance IDs."""

    def __init__(self, provenance_records: Iterable[ProvenanceRecord]) -> None:
        self.records = tuple(provenance_records)

    def resolve(
        self,
        *,
        metric_name: str,
        metric_value: Any,
        section: str | None = None,
        model_name: str | None = None,
        source_file: str | None = None,
    ) -> tuple[ProvenanceRecord | None, ValidationIssue | None]:
        candidates = [
            record
            for record in self.records
            if record.metric_name == metric_name
            and record.support_status in {"SUPPORTED", "SELECTED"}
        ]
        if section:
            candidates = [record for record in candidates if record.section == section]
        if model_name:
            candidates = [record for record in candidates if record.model_name in {model_name, None, ""}]
        if source_file:
            source_candidates = [record for record in candidates if record.source_file == source_file]
            if source_candidates:
                candidates = source_candidates
        for record in candidates:
            if values_match(record.metric_value, metric_value):
                return record, None
        if candidates:
            return None, ValidationIssue(
                code="PROVENANCE_VALUE_MISMATCH",
                severity="ERROR",
                message=f"No provenance value for {metric_name} matched observation value {metric_value!r}.",
                observation_id=None,
                field="supporting_metrics.metric_value",
                source_file=source_file,
            )
        return None, None


def build_observations(
    payload: SupervisorSourcePayload,
    *,
    software_version: str,
    created_at: str,
) -> ObservationRuleResult:
    """Create deterministic factual observations from loaded supervisor sources."""

    if payload.has_critical_issues or not payload.summary:
        return ObservationRuleResult((), payload.validation_issues)

    resolver = ProvenanceResolver(payload.provenance_records)
    issues: list[ValidationIssue] = list(payload.validation_issues)
    observations = [
        _dataset_observation(payload, resolver, software_version, created_at, issues),
        _quality_control_observation(payload, resolver, software_version, created_at, issues),
        _fingerprint_observation(payload, resolver, software_version, created_at, issues),
        _exploratory_observation(payload, resolver, software_version, created_at, issues),
        _classification_observation(payload, resolver, software_version, created_at, issues),
        _regression_observation(payload, resolver, software_version, created_at, issues),
        _feature_engineering_observation(payload, resolver, software_version, created_at, issues),
        _feature_selection_observation(payload, resolver, software_version, created_at, issues),
        _strain_observation(payload, resolver, software_version, created_at, issues),
        _blind_prediction_observation(payload, resolver, software_version, created_at, issues),
        _validation_observation(payload, software_version, created_at),
    ]
    return ObservationRuleResult(tuple(observations), tuple(issues))


def _dataset_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    data = payload.summary.get("dataset_summary", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Dataset and Experimental Scope",
        specs=[
            ("input_canonical_rows", data.get("input_canonical_rows"), "rows", None, data.get("canonical_qc_source") or data.get("feature_summary_source")),
            ("feature_rows", data.get("feature_rows"), "rows", None, data.get("feature_summary_source")),
            ("core_feature_count", data.get("core_feature_count"), "features", None, data.get("feature_summary_source")),
            ("chemical_count", data.get("chemical_count"), "count", None, data.get("canonical_qc_source")),
            ("strain_count", data.get("strain_count"), "count", None, data.get("canonical_qc_source")),
        ],
    )
    statement = (
        "The supervisor dataset summary listed "
        f"{_value(data.get('input_canonical_rows'))} canonical rows, "
        f"{_value(data.get('feature_rows'))} feature rows, "
        f"{_value(data.get('core_feature_count'))} features, "
        f"{_value(data.get('chemical_count'))} chemical labels, and "
        f"{_value(data.get('strain_count'))} strains."
    )
    metadata = {
        "chemicals_detected": data.get("chemicals_detected", []),
        "strains_detected": data.get("strains_detected", []),
        "concentration_levels": data.get("concentration_levels", []),
        "source_file_count": data.get("source_file_count"),
    }
    return _observation(
        ObservationCategory.DATASET,
        "Dataset summary facts",
        statement,
        "Supervisor dataset summary",
        metrics,
        software_version,
        created_at,
        metadata=metadata,
    )


def _quality_control_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    qc = payload.summary.get("quality_control_summary", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Data Quality and Preprocessing",
        specs=[
            ("canonical_error_count", qc.get("canonical_error_count"), "count", None, qc.get("canonical_qc_source")),
            ("canonical_warning_count", qc.get("canonical_warning_count"), "count", None, qc.get("canonical_qc_source")),
            ("feature_failed_rows", qc.get("feature_failed_rows"), "rows", None, qc.get("feature_summary_source")),
            ("fingerprint_excluded_rows", qc.get("fingerprint_excluded_rows"), "rows", None, qc.get("fingerprint_summary_source")),
        ],
    )
    limitations = tuple(
        item.get("limitation", "")
        for item in payload.summary.get("limitations", [])
        if item.get("status") == "ACTIVE" and item.get("limitation")
    )
    statement = (
        "The supervisor QC summary listed "
        f"{_value(qc.get('canonical_error_count'))} canonical QC errors, "
        f"{_value(qc.get('canonical_warning_count'))} canonical QC warnings, "
        f"{_value(qc.get('feature_failed_rows'))} failed feature rows, and "
        f"{_value(qc.get('fingerprint_excluded_rows'))} excluded fingerprint rows."
    )
    return _observation(
        ObservationCategory.QUALITY_CONTROL,
        "Quality-control summary facts",
        statement,
        "Supervisor QC summary",
        metrics,
        software_version,
        created_at,
        limitations=limitations,
        metadata={"package_validation_passed": payload.report_validation.get("passed")},
    )


def _fingerprint_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    fp = payload.summary.get("fingerprint_summary", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Biosensor Fingerprint Analysis",
        specs=[
            ("fingerprint_rows", fp.get("fingerprint_rows"), "rows", None, fp.get("source_file")),
            ("consensus_fingerprint_rows", fp.get("consensus_fingerprint_rows"), "rows", None, fp.get("source_file")),
            ("feature_count", fp.get("feature_count"), "features", None, fp.get("source_file")),
        ],
    )
    statement = (
        "The supervisor fingerprint summary listed "
        f"{_value(fp.get('fingerprint_rows'))} fingerprint rows, "
        f"{_value(fp.get('consensus_fingerprint_rows'))} consensus fingerprints, and "
        f"normalization method {_value(fp.get('normalization_method'))}."
    )
    return _observation(
        ObservationCategory.FINGERPRINT,
        "Fingerprint output facts",
        statement,
        "Supervisor fingerprint summary",
        metrics,
        software_version,
        created_at,
        metadata={
            "normalization_method": fp.get("normalization_method"),
            "selected_figures": _figures_with_keywords(payload, ("fingerprint",)),
            "selected_tables": _tables_with_ids(payload, ("fingerprint_summary",)),
        },
        contextual_missing=bool(payload.missing_optional_files),
    )


def _exploratory_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    exploratory = payload.summary.get("exploratory_results", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Exploratory Analysis",
        specs=[
            ("cumulative_explained_variance_ratio_pc3", exploratory.get("cumulative_variance_pc3"), None, None, exploratory.get("pca_source")),
            ("cluster_count", exploratory.get("cluster_count"), "count", None, exploratory.get("cluster_source")),
        ],
    )
    statement = (
        "The supervisor exploratory summary listed cumulative explained variance through PC3 as "
        f"{_value(exploratory.get('cumulative_variance_pc3'))} and "
        f"{_value(exploratory.get('cluster_count'))} clusters."
    )
    return _observation(
        ObservationCategory.EXPLORATORY_ANALYSIS,
        "Exploratory output facts",
        statement,
        "Supervisor exploratory summary",
        metrics,
        software_version,
        created_at,
        metadata={
            "selected_figures": _figures_with_keywords(payload, ("similarity", "dendrogram", "fingerprint")),
            "cluster_count": exploratory.get("cluster_count"),
            "cluster_assignment_rows": exploratory.get("cluster_assignment_rows"),
        },
        contextual_missing=bool(payload.missing_optional_files),
    )


def _classification_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    classification = payload.summary.get("classification_results", {})
    selected_model = classification.get("selected_model", {})
    selected_name = selected_model.get("model_name")
    metric_rows = classification.get("selected_metrics", [])
    selected = _selected_metric_map(metric_rows)
    wanted = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_macro_mean",
        "f1_weighted_mean",
        "precision_macro_mean",
        "recall_macro_mean",
        "roc_auc_ovr_weighted_mean",
        "log_loss_mean",
        "fold_count",
        "sample_count",
        "class_count",
    ]
    metrics: list[SupportingMetric] = []
    for name in wanted:
        row = selected.get(name, {})
        metric_model = row.get("model_name") or selected_name
        metric = _metric_from_row(
            payload,
            resolver,
            issues,
            section="Chemical Classification",
            metric_name=name,
            row=row,
            model_name=metric_model,
            selected_model=selected_name,
        )
        metrics.append(metric)
    statement = f"{_value(selected_name)} was listed as the selected classification model."
    return _observation(
        ObservationCategory.CLASSIFICATION,
        "Selected classifier facts",
        statement,
        "Supervisor classification summary",
        metrics,
        software_version,
        created_at,
        metadata={"selected_model": selected_model, "comparison_table_rows": len(classification.get("model_comparison", []))},
    )


def _regression_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    regression = payload.summary.get("regression_results", {})
    selected_model = regression.get("selected_model", {})
    selected_name = selected_model.get("model_name")
    selected = _selected_metric_map(regression.get("selected_metrics", []))
    wanted = [
        "r2_mean",
        "explained_variance_mean",
        "rmse_mean",
        "mae_mean",
        "median_absolute_error_mean",
        "fold_count",
        "sample_count",
        "concentration_min",
        "concentration_max",
    ]
    metrics: list[SupportingMetric] = []
    for name in wanted:
        row = selected.get(name, {})
        metric_model = row.get("model_name") or selected_name
        metrics.append(
            _metric_from_row(
                payload,
                resolver,
                issues,
                section="Concentration Regression",
                metric_name=name,
                row=row,
                model_name=metric_model,
                selected_model=selected_name,
            )
        )
    statement = f"{_value(selected_name)} was listed as the selected regression model."
    return _observation(
        ObservationCategory.REGRESSION,
        "Selected regressor facts",
        statement,
        "Supervisor regression summary",
        metrics,
        software_version,
        created_at,
        metadata={"selected_model": selected_model, "comparison_table_rows": len(regression.get("model_comparison", []))},
    )


def _feature_engineering_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    fe = payload.summary.get("feature_engineering_results", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Advanced Feature Engineering",
        specs=[
            ("classification_improvement", fe.get("classification_improvement"), None, None, fe.get("summary_source")),
            ("regression_improvement", fe.get("regression_improvement"), None, None, fe.get("summary_source")),
            ("runtime_increase_seconds", fe.get("runtime_increase_seconds"), "seconds", None, fe.get("summary_source")),
            ("feature_family_count", fe.get("feature_family_count"), "count", None, fe.get("summary_source")),
        ],
    )
    statement = f"{_value(fe.get('best_feature_family'))} was listed as the selected feature family in the feature-engineering summary."
    return _observation(
        ObservationCategory.FEATURE_ENGINEERING,
        "Feature-engineering summary facts",
        statement,
        "Supervisor feature-engineering summary",
        metrics,
        software_version,
        created_at,
        metadata={"best_feature_family": fe.get("best_feature_family"), "benchmark_source": fe.get("benchmark_source")},
    )


def _feature_selection_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    fs = payload.summary.get("feature_selection_results", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Feature Selection",
        specs=[
            ("selected_feature_rows", fs.get("selected_feature_count"), "rows", None, fs.get("selected_features_source")),
        ],
    )
    summary_row_count = len(fs.get("summary_rows", []))
    statement = (
        "The supervisor feature-selection summary listed "
        f"{_value(summary_row_count)} summary rows and "
        f"{_value(fs.get('selected_feature_count'))} selected-feature records."
    )
    return _observation(
        ObservationCategory.FEATURE_SELECTION,
        "Feature-selection summary facts",
        statement,
        "Supervisor feature-selection summary",
        metrics,
        software_version,
        created_at,
        metadata={
            "summary_row_count": summary_row_count,
            "recommended_default_count": len(fs.get("recommended_defaults", [])),
            "selected_tables": _tables_with_ids(payload, ("feature_selection_results",)),
        },
    )


def _strain_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    strain = payload.summary.get("strain_results", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Strain Contribution",
        specs=[
            ("leave_one_strain_count", strain.get("leave_one_strain_count"), "rows", None, strain.get("loeo_source")),
            ("chemical_specific_count", strain.get("chemical_specific_count"), "rows", None, strain.get("chemical_source")),
        ],
    )
    statement = (
        "The supervisor strain-contribution summary listed "
        f"{_value(strain.get('leave_one_strain_count'))} leave-one-strain records and "
        f"{_value(strain.get('chemical_specific_count'))} chemical-specific strain records."
    )
    return _observation(
        ObservationCategory.STRAIN_CONTRIBUTION,
        "Strain-contribution summary facts",
        statement,
        "Supervisor strain-contribution summary",
        metrics,
        software_version,
        created_at,
        metadata={
            "single_strain_count": strain.get("single_strain_count"),
            "selected_figures": _figures_with_keywords(payload, ("strain", "leave_one")),
            "selected_tables": _tables_with_ids(payload, ("strain_contribution",)),
        },
        contextual_missing=bool(payload.missing_optional_files),
    )


def _blind_prediction_observation(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    software_version: str,
    created_at: str,
    issues: list[ValidationIssue],
) -> Observation:
    blind = payload.summary.get("project_summary", {}).get("blind_prediction_context", {})
    metrics = _metrics(
        payload,
        resolver,
        issues,
        section="Limitations",
        specs=[
            ("true_labels_included", blind.get("true_labels_included"), None, None, blind.get("source_file")),
        ],
    )
    if blind.get("true_labels_included") is False:
        validation_phrase = "Validation performance was not calculated because true labels were absent."
    else:
        validation_phrase = "The blind-prediction source listed true labels as present."
    statement = (
        f"The selected blind-prediction output listed true_labels_included={_value(blind.get('true_labels_included'))}. "
        f"{validation_phrase}"
    )
    return _observation(
        ObservationCategory.BLIND_PREDICTION,
        "Blind-prediction source facts",
        statement,
        "Supervisor blind-prediction context",
        metrics,
        software_version,
        created_at,
        metadata={
            "prediction_passed": blind.get("prediction_passed"),
            "prediction_output_available": bool(blind.get("source_file")),
            "predicted_chemical": blind.get("predicted_chemical"),
            "predicted_concentration": blind.get("predicted_concentration"),
            "concentration_units": blind.get("concentration_units"),
            "novelty_status": blind.get("novelty_status"),
        },
    )


def _validation_observation(
    payload: SupervisorSourcePayload,
    software_version: str,
    created_at: str,
) -> Observation:
    issue_count = len(payload.validation_issues)
    statement = (
        "The supervisor package validation result and source parsing counts are recorded in observation metadata."
    )
    return Observation(
        observation_id=_observation_id(ObservationCategory.VALIDATION),
        category=ObservationCategory.VALIDATION,
        title="Supervisor package validation facts",
        statement=statement,
        status=ObservationStatus.COMPLETE if not payload.validation_issues else ObservationStatus.INCOMPLETE,
        analysis_stage="Supervisor report validation",
        supporting_metrics=tuple(),
        supporting_files=tuple(payload.loaded_files),
        provenance_records=tuple(),
        confidence=ConfidenceLevel.HIGH if not payload.validation_issues else ConfidenceLevel.LOW,
        limitations=tuple(),
        created_at=created_at,
        software_version=software_version,
        source_run=payload.supervisor_results_dir.name,
        tags=("validation",),
        metadata={
            "package_validation_passed": payload.report_validation.get("passed"),
            "provenance_record_count": len(payload.provenance_records),
            "source_files_loaded_count": len(payload.loaded_files),
            "validation_issue_count": issue_count,
            "missing_required_files": list(payload.missing_required_files),
            "missing_optional_files": list(payload.missing_optional_files),
        },
    )


def _metrics(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    issues: list[ValidationIssue],
    *,
    section: str,
    specs: Iterable[tuple[str, Any, str | None, str | None, str | None]],
) -> list[SupportingMetric]:
    return [
        _metric_from_value(
            payload,
            resolver,
            issues,
            section=section,
            metric_name=name,
            metric_value=value,
            units=units,
            model_name=model,
            source_file=source,
        )
        for name, value, units, model, source in specs
    ]


def _metric_from_row(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    issues: list[ValidationIssue],
    *,
    section: str,
    metric_name: str,
    row: dict[str, Any],
    model_name: str | None,
    selected_model: str | None,
) -> SupportingMetric:
    value = row.get("metric_value")
    units = row.get("metric_units")
    source_file = row.get("source_file")
    metric = _metric_from_value(
        payload,
        resolver,
        issues,
        section=section,
        metric_name=metric_name,
        metric_value=value,
        units=units,
        model_name=model_name,
        source_file=source_file,
        source_run=row.get("source_run"),
        fold_count=_metric_value_from_selected(row, "fold_count"),
        sample_count=_metric_value_from_selected(row, "sample_count"),
    )
    if selected_model and model_name and model_name != selected_model:
        issues.append(
            ValidationIssue(
                code="MODEL_COHERENCE_ISSUE",
                severity="ERROR",
                message=f"{metric_name} is linked to {model_name}, not selected model {selected_model}.",
                observation_id=None,
                field="supporting_metrics.model_name",
                source_file=source_file,
            )
        )
    return metric


def _metric_from_value(
    payload: SupervisorSourcePayload,
    resolver: ProvenanceResolver,
    issues: list[ValidationIssue],
    *,
    section: str,
    metric_name: str,
    metric_value: Any,
    units: str | None,
    model_name: str | None,
    source_file: str | None,
    source_run: str | None = None,
    fold_count: int | None = None,
    sample_count: int | None = None,
) -> SupportingMetric:
    provenance, issue = resolver.resolve(
        metric_name=metric_name,
        metric_value=metric_value,
        section=section,
        model_name=model_name,
        source_file=source_file,
    )
    if issue:
        issues.append(issue)
    if provenance:
        units = provenance.units if provenance.units is not None else units
        source_file = provenance.source_file or source_file
        source_run = provenance.source_run or source_run
    return SupportingMetric(
        metric_name=metric_name,
        metric_value=metric_value,
        units=units or None,
        model_name=model_name or None,
        fold_count=fold_count,
        sample_count=sample_count,
        source_file=source_file,
        source_run=source_run or payload.supervisor_results_dir.name,
        provenance_id=provenance.provenance_id if provenance else None,
    )


def _observation(
    category: ObservationCategory,
    title: str,
    statement: str,
    analysis_stage: str,
    metrics: list[SupportingMetric],
    software_version: str,
    created_at: str,
    *,
    limitations: tuple[str, ...] = tuple(),
    metadata: dict[str, Any] | None = None,
    contextual_missing: bool = False,
) -> Observation:
    missing_provenance = [
        metric.metric_name
        for metric in metrics
        if metric.metric_value is not None and metric.provenance_id is None and metric.metric_name != "package_validation_passed"
    ]
    status = ObservationStatus.INCOMPLETE if missing_provenance else ObservationStatus.COMPLETE
    if missing_provenance:
        confidence = ConfidenceLevel.LOW
    elif contextual_missing:
        confidence = ConfidenceLevel.MODERATE
    else:
        confidence = ConfidenceLevel.HIGH
    provenance_records = tuple(
        record
        for record in _provenance_from_metrics(metrics)
    )
    notes = []
    if missing_provenance:
        notes.append(f"missing_provenance={missing_provenance}")
    files = tuple(sorted({metric.source_file for metric in metrics if metric.source_file}))
    return Observation(
        observation_id=_observation_id(category),
        category=category,
        title=title,
        statement=statement,
        status=status,
        analysis_stage=analysis_stage,
        supporting_metrics=tuple(metrics),
        supporting_files=files,
        provenance_records=provenance_records,
        confidence=confidence,
        limitations=limitations,
        created_at=created_at,
        software_version=software_version,
        source_run=None,
        tags=(category.value.lower(),),
        metadata={**(metadata or {}), "notes": "; ".join(notes) if notes else None},
    )


def attach_provenance_records(
    observations: tuple[Observation, ...],
    provenance_records: tuple[ProvenanceRecord, ...],
) -> tuple[Observation, ...]:
    """Replace placeholder provenance IDs with full provenance records."""

    by_id = {record.provenance_id: record for record in provenance_records}
    attached: list[Observation] = []
    for observation in observations:
        records = tuple(
            by_id[metric.provenance_id]
            for metric in observation.supporting_metrics
            if metric.provenance_id in by_id
        )
        attached.append(
            Observation(
                observation_id=observation.observation_id,
                category=observation.category,
                title=observation.title,
                statement=observation.statement,
                status=observation.status,
                analysis_stage=observation.analysis_stage,
                supporting_metrics=observation.supporting_metrics,
                supporting_files=observation.supporting_files,
                provenance_records=records,
                confidence=observation.confidence,
                limitations=observation.limitations,
                created_at=observation.created_at,
                software_version=observation.software_version,
                source_run=observation.source_run,
                tags=observation.tags,
                metadata=observation.metadata,
            )
        )
    return tuple(attached)


def _provenance_from_metrics(metrics: list[SupportingMetric]) -> tuple[ProvenanceRecord, ...]:
    # The full provenance records are attached after all observations are built.
    return tuple()


def _observation_id(category: ObservationCategory) -> str:
    return f"OBS-{category_id_token(category)}-0001"


def _selected_metric_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row.get("metric_name"): row for row in rows if row.get("metric_name")}


def _metric_value_from_selected(row: dict[str, Any], metric_name: str) -> int | None:
    if row.get("metric_name") == metric_name:
        value = row.get("metric_value")
    else:
        value = None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _tables_with_ids(payload: SupervisorSourcePayload, ids: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.selected_tables
        if row.get("table_id") in ids
    ]


def _figures_with_keywords(payload: SupervisorSourcePayload, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.selected_figures
        if any(keyword in (row.get("figure_id", "") + " " + row.get("title", "")).lower() for keyword in keywords)
    ]


def values_match(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    left_number = _to_float(left)
    right_number = _to_float(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-12)
    return str(left) == str(right)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value(value: Any) -> str:
    if value is None or value == "":
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
