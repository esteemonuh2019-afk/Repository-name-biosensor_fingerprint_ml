"""Evidence aggregation engine for Stage 9B.2B."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.scientific_narrative.aggregation_rules import (
    aliases_for,
    classification_model_rankings,
    matching_aliases,
    feature_family_table,
    metric_direction,
    metric_alias_registry_rows,
    parse_matrix_entity,
    parse_metric_value,
    regression_model_rankings,
    select_preferred_records,
    smallest_within_tolerance,
)
from src.scientific_narrative.evidence_traceability import build_traceability_index, traceability_coverage
from src.scientific_narrative.scientific_summary import (
    AggregatedEvidence,
    EvidenceInputRecord,
    SummaryRecord,
)


DATASET_METRICS = (
    ("total_canonical_rows", ("row_count",), "canonical QC; feature validation"),
    ("total_measurement_units", ("measurement_unit_count",), "canonical QC; feature validation"),
    ("total_feature_rows", ("feature_row_count",), "feature extraction"),
    ("total_fingerprint_rows", ("fingerprint_rows",), "fingerprint generation"),
    ("individual_fingerprint_count", ("individual_fingerprint_rows",), "fingerprint generation"),
    ("consensus_fingerprint_count", ("consensus_fingerprint_rows",), "fingerprint generation"),
    ("number_of_strains", ("unique_strain_count", "strains_detected_count"), "canonical QC; feature validation"),
    ("number_of_chemicals", ("unique_chemical_count", "chemicals_detected_count"), "canonical QC; feature validation"),
    ("number_of_concentrations", ("unique_concentration_count", "concentrations_detected_count"), "canonical QC; feature validation"),
    ("experiment_duration_count", ("unique_duration_count", "duration_count"), "canonical QC; feature validation"),
    ("source_file_count", ("source_files_count",), "canonical QC; feature validation"),
)

QC_METRICS = (
    "qc_passed",
    "validation_passed",
    "feature_qc_passed",
    "ambiguous_measurement_identity_count",
    "ambiguous_measurement_identity_group_count",
    "duplicate_timepoint_group_count",
    "logical_duplicate_count",
    "conflicting_value_duplicate_count",
    "missing_required_value_counts_count",
    "missing_identifier_counts_plate_id",
    "missing_identifier_counts_well_id",
    "zero_baseline_count",
    "failed_feature_rows",
    "warning_feature_rows",
    "error_count",
    "warning_count",
)

FINGERPRINT_METRICS = (
    "fingerprint_rows",
    "individual_fingerprint_rows",
    "consensus_fingerprint_rows",
    "distance_matrix_rows",
    "distance_matrix_columns",
    "consensus_distance_matrix_rows",
    "consensus_distance_matrix_columns",
    "duplicate_fingerprint_row_count",
    "missing_feature_cell_count",
    "fingerprint_qc_passed",
    "fingerprint_qc_warning_count",
)

MODEL_RANKING_LIMIT = 3
MATRIX_VALUE_LIMIT = 3
STRAIN_METRIC_LIMIT = 2
REGRESSION_PRIMARY_METRICS = (
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "median_absolute_error_mean",
    "explained_variance_mean",
    "fold_count",
    "sample_count",
    "concentration_min",
    "concentration_max",
)


def aggregate_scientific_evidence(
    evidence_file: str | Path,
    *,
    evidence_json: str | Path | None = None,
) -> AggregatedEvidence:
    """Aggregate a Stage 9B.2A evidence CSV into compact summaries."""

    evidence_path = Path(evidence_file).resolve()
    records = load_evidence_records(evidence_path)
    if not records:
        raise ValueError(f"Evidence input is empty: {evidence_path}")
    json_metadata = _load_optional_json_metadata(evidence_json or evidence_path.with_suffix(".json"))

    supported = [record for record in records if record.extraction_status == "EXTRACTED" and record.metric_value is not None]
    unsupported_or_null = [
        record
        for record in records
        if record.extraction_status != "EXTRACTED" or record.metric_value is None
    ]
    summary_id = _SummaryId()

    aggregation = AggregatedEvidence()
    aggregation.dataset_summary = _dataset_summary(supported, summary_id)
    aggregation.qc_summary = _qc_summary(supported, summary_id)
    aggregation.fingerprint_summary = _fingerprint_summary(supported, summary_id)
    aggregation.exploratory_summary = _exploratory_summary(supported, summary_id)
    aggregation.classification_summary = _classification_summary(supported, summary_id)
    aggregation.regression_summary, aggregation.regression_model_comparison = _regression_summary(supported, summary_id)
    aggregation.feature_engineering_summary = _feature_engineering_summary(supported, summary_id)
    aggregation.feature_selection_summary = _feature_selection_summary(supported, summary_id)
    aggregation.strain_summary = _strain_summary(supported, summary_id)
    aggregation.limitations_summary = _limitations_summary(supported, summary_id)
    aggregation.blind_validation_status = _blind_validation_status(supported, summary_id)

    all_summaries = aggregation.all_summaries
    evidence_by_id = {record.evidence_id: record for record in records}
    aggregation.traceability_index = build_traceability_index(all_summaries, evidence_by_id)
    aggregation.metric_alias_registry = metric_alias_registry_rows()
    aggregation.metric_mapping_audit, aggregation.unmapped_metrics = _metric_mapping_audit(records)
    aggregation.summary_population_audit = _summary_population_audit(all_summaries)
    used_ids = {evidence_id for summary in all_summaries for evidence_id in summary.source_evidence_ids}
    conflict_count = sum(1 for summary in all_summaries if summary.status == "CONFLICT")
    missing_count = sum(1 for summary in all_summaries if summary.status == "MISSING")
    populated_count = sum(1 for summary in all_summaries if summary.status != "MISSING" and summary.source_evidence_ids)
    aggregation.warnings = _aggregation_warnings(all_summaries)
    aggregation.errors = []
    aggregation.aggregation_passed = bool(all_summaries) and traceability_coverage(all_summaries) == 1.0 and not aggregation.errors
    aggregation.metadata = {
        "evidence_file": str(evidence_path),
        "evidence_records_received": len(records),
        "evidence_records_used": len(used_ids),
        "evidence_records_unused": len(records) - len(used_ids),
        "evidence_records_excluded": len(records) - len(used_ids),
        "unsupported_or_null_evidence_records": len(unsupported_or_null),
        "excluded_evidence_reasons": sorted(
            {
                (record.notes or record.extraction_status or "metric_value unavailable")
                for record in unsupported_or_null
            }
        ),
        "unsupported_file_count_from_json": json_metadata.get("unsupported_file_count", 0),
        "unreadable_file_count_from_json": json_metadata.get("unreadable_file_count", 0),
        "summary_records_created": len(all_summaries),
        "summary_records_populated": populated_count,
        "selected_regression_model": _selected_regression_model_name(aggregation.regression_model_comparison),
        "conflicting_summary_count": conflict_count,
        "missing_summary_count": missing_count,
        "mapped_unique_metric_count": sum(1 for row in aggregation.metric_mapping_audit if row["mapping_status"] == "MAPPED"),
        "unmapped_unique_metric_count": len(aggregation.unmapped_metrics),
        "unique_metric_name_count": len({record.metric_name for record in records}),
        "unique_analysis_type_count": len({record.analysis_type for record in records}),
        "traceability_coverage": traceability_coverage(all_summaries),
        "interpretation_can_proceed": missing_count == 0 and conflict_count == 0 and aggregation.aggregation_passed,
        "json_metadata_available": bool(json_metadata),
    }
    return aggregation


def load_evidence_records(evidence_file: str | Path) -> list[EvidenceInputRecord]:
    """Load Stage 9B.2A CSV rows and assign deterministic evidence IDs."""

    path = Path(evidence_file)
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        records = []
        for index, row in enumerate(reader, start=1):
            records.append(
                EvidenceInputRecord(
                    evidence_id=f"EV{index:06d}",
                    analysis_type=row.get("analysis_type", ""),
                    source_file=row.get("source_file", ""),
                    source_run=row.get("source_run", ""),
                    metric_name=row.get("metric_name", ""),
                    metric_value=parse_metric_value(row.get("metric_value")),
                    metric_units=row.get("metric_units", ""),
                    figure_reference=row.get("figure_reference", ""),
                    table_reference=row.get("table_reference", ""),
                    biological_entity=row.get("biological_entity", ""),
                    model_name=row.get("model_name", ""),
                    confidence=row.get("confidence", ""),
                    extraction_status=row.get("extraction_status", ""),
                    notes=row.get("notes", ""),
                )
            )
    return records


class _SummaryId:
    def __init__(self) -> None:
        self._counter = 0

    def next(self) -> str:
        self._counter += 1
        return f"SUM{self._counter:04d}"


def _dataset_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    for output_metric, aliases, preferred_analysis in DATASET_METRICS:
        matches = _find_records(records, aliases=aliases_for("dataset_summary", output_metric) or aliases, analysis_type=preferred_analysis)
        if not matches and preferred_analysis:
            matches = _find_records(records, aliases=aliases_for("dataset_summary", output_metric) or aliases)
        summaries.extend(
            _scalar_summaries(
                summary_id,
                "dataset_summary",
                "dataset_metric",
                output_metric,
                matches,
                notes="Dataset summary metric aggregated from extracted evidence.",
            )
        )
    return summaries


def _qc_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    for metric in QC_METRICS:
        matches = _find_records(records, aliases=aliases_for("quality_control", metric) or (metric,), analysis_contains=("QC", "feature extraction", "fingerprint generation"))
        summaries.extend(
            _scalar_summaries(
                summary_id,
                "quality_control",
                "qc_metric",
                metric,
                matches,
                notes="QC metric aggregated without correction or interpretation.",
            )
        )
    return summaries


def _fingerprint_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    for metric in FINGERPRINT_METRICS:
        matches = _find_records(records, aliases=aliases_for("fingerprint_summary", metric) or (metric,), analysis_type="fingerprint generation")
        summaries.extend(_scalar_summaries(summary_id, "fingerprint_summary", "fingerprint_metric", metric, matches))
    return summaries


def _exploratory_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    pca_records = [
        record
        for record in records
        if record.analysis_type == "exploratory analysis"
        and record.metric_name in {"explained_variance_ratio", "cumulative_explained_variance_ratio", "explained_variance"}
        and record.source_file.endswith("pca_explained_variance.csv")
    ]
    for record in pca_records:
        entity = record.biological_entity or "PCA"
        summaries.append(
            _summary_from_records(
                summary_id,
                "exploratory_analysis",
                "pca_variance",
                record.metric_name,
                [record],
                metric_value=record.metric_value,
                biological_entity=entity,
                direction=metric_direction(record.metric_name, record.metric_value),
                aggregation_method="direct_pca_component_value",
            )
        )
    summaries.extend(
        _scalar_summaries(
            summary_id,
            "exploratory_analysis",
            "clustering_summary",
            "cluster_count",
            _find_records(records, aliases=("cluster_count",), analysis_type="exploratory analysis"),
        )
    )
    summaries.extend(_top_matrix_values(summary_id, records, "exploratory_analysis", "top_pca_loading", "exploratory/stage_7b_3/pca_loadings.csv", limit=MATRIX_VALUE_LIMIT, absolute=True))
    summaries.extend(_top_matrix_values(summary_id, records, "exploratory_analysis", "cluster_composition", "exploratory/stage_7b_3/cluster_composition.csv", column_filter={"count", "cluster_size"}, limit=MATRIX_VALUE_LIMIT))
    summaries.extend(_matrix_extrema(summary_id, records, "exploratory_analysis", "chemical_similarity", "exploratory/stage_7b_3/chemical_similarity_heatmap_table.csv"))
    for missing_metric in ("concentration_trajectory_summary", "strain_dispersion_summary", "replicate_to_consensus_distance_summary"):
        if not any(missing_metric.split("_")[0] in record.source_file for record in records):
            summaries.append(_missing(summary_id, "exploratory_analysis", "missing_expected_summary", missing_metric))
    return summaries


def _classification_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    rankings = classification_model_rankings(records)
    model_metrics = (
        "f1_macro_mean",
        "f1_macro_std",
        "f1_macro_ci95_low",
        "f1_macro_ci95_high",
        "f1_weighted_mean",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "precision_macro_mean",
        "recall_macro_mean",
        "roc_auc_ovr_weighted_mean",
        "fold_count",
    )
    for rank, model in enumerate(rankings[:MODEL_RANKING_LIMIT], start=1):
        evidence = model.get("_evidence", {})
        for metric in model_metrics:
            if metric not in model:
                continue
            source_records = evidence.get(metric, [])
            summaries.extend(
                _model_metric_summaries(
                    summary_id,
                    "classification",
                    metric,
                    source_records,
                    model_name=model["model_name"],
                    rank=rank,
                    aggregation_method="ranked_by_macro_f1_balanced_accuracy_accuracy",
                )
            )
    summaries.extend(_scalar_summaries(summary_id, "classification", "best_model", "best_model", _find_records(records, aliases=aliases_for("classification", "best_model"), analysis_type="classification")))
    summaries.extend(_top_matrix_values(summary_id, records, "classification", "per_class_best_f1", "classification/stage_8a/per_class_metrics.csv", column_filter={"f1"}, limit=2))
    summaries.extend(_top_matrix_values(summary_id, records, "classification", "per_class_worst_f1", "classification/stage_8a/per_class_metrics.csv", column_filter={"f1"}, limit=2, lowest=True))
    summaries.extend(_top_matrix_values(summary_id, records, "classification", "most_frequent_confusions", "classification/stage_8a/confusion_matrix.csv", limit=MATRIX_VALUE_LIMIT, exclude_diagonal=True))
    if not _find_records(records, aliases=("importance",), analysis_type="classification"):
        summaries.append(_missing(summary_id, "classification", "missing_expected_summary", "top_classification_features"))
    return summaries


def _regression_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> tuple[list[SummaryRecord], list[dict[str, Any]]]:
    summaries: list[SummaryRecord] = []
    rankings = regression_model_rankings(records)
    comparison = _regression_model_comparison(rankings)
    if not rankings:
        summaries.append(_missing(summary_id, "regression", "best_model", "best_model"))
        for metric in REGRESSION_PRIMARY_METRICS:
            summaries.append(_missing_selected_model(summary_id, "regression", "selected_model_primary_metric", metric, ""))
        summaries.append(_missing(summary_id, "regression", "missing_expected_summary", "top_regression_features"))
        return summaries, comparison

    selected = rankings[0]
    selected_model = selected["model_name"]
    selected_records = selected.get("_records", [])
    summaries.append(
        _summary_from_records(
            summary_id,
            "regression",
            "best_model",
            "best_model",
            selected_records,
            metric_value=selected_model,
            model_name=selected_model,
            rank=1,
            aggregation_method="selected_once_by_r2_rmse_mae_model_name",
            notes="Regression model selected once, then all primary metrics locked to this model.",
        )
    )

    evidence = selected.get("_evidence", {})
    for metric in REGRESSION_PRIMARY_METRICS:
        metric_records = [
            record
            for record in evidence.get(metric, [])
            if record.model_name == selected_model
        ]
        if metric_records:
            summaries.extend(
                _model_metric_summaries(
                    summary_id,
                    "regression",
                    metric,
                    metric_records,
                    model_name=selected_model,
                    rank=1,
                    aggregation_method="selected_model_primary_metric_locked_to_winning_model",
                )
            )
        else:
            summaries.append(
                _missing_selected_model(
                    summary_id,
                    "regression",
                    "selected_model_primary_metric",
                    metric,
                    selected_model,
                )
            )

    if _as_float(selected.get("r2_mean")) is not None and _as_float(selected.get("r2_mean")) < 0:
        summaries.append(
            _summary_from_records(
                summary_id,
                "regression",
                "negative_r2_model",
                "negative_r2_model",
                selected_records,
                metric_value=selected.get("r2_mean"),
                model_name=selected_model,
                rank=1,
                direction="negative",
                aggregation_method="selected_model_r2_less_than_zero_filter",
            )
        )
    if not _find_records(records, aliases=("importance",), analysis_type="regression"):
        summaries.append(_missing(summary_id, "regression", "missing_expected_summary", "top_regression_features"))
    return summaries, comparison


def _feature_engineering_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    table = feature_family_table(records)
    metrics = (
        "regression_r2",
        "regression_r2_gain",
        "regression_rmse",
        "regression_rmse_delta",
        "regression_mae",
        "regression_mae_delta",
        "regression_sample_count",
        "runtime_increase_seconds",
    )
    for family in sorted(table):
        for metric in metrics:
            if metric not in table[family]:
                continue
            record = table[family][metric]
            summaries.append(
                _summary_from_records(
                    summary_id,
                    "feature_engineering",
                    "feature_family_metric",
                    metric,
                    [record],
                    metric_value=record.metric_value,
                    comparison_group=family,
                    direction=metric_direction(metric, record.metric_value),
                    aggregation_method="feature_family_ablation_direct_value",
                )
            )
    summaries.extend(_feature_family_extrema(summary_id, table))
    for metric in ("best_feature_family", "classification_improvement", "regression_improvement", "runtime_increase_seconds"):
        summaries.extend(_scalar_summaries(summary_id, "feature_engineering", "stage_8c_headline", metric, _find_records(records, aliases=aliases_for("feature_engineering", metric) or (metric,), analysis_type="advanced feature engineering")))
    return summaries


def _feature_selection_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    rows = _feature_selection_rows(records)
    for metric in ("available_feature_count", "default_classification_selected_feature_count", "default_regression_selected_feature_count", "research_selected_feature_count"):
        summaries.extend(_scalar_summaries(summary_id, "feature_selection", "feature_count_metric", metric, _find_records(records, aliases=(metric,), analysis_type="feature selection")))
    for task, metric, higher in (("classification", "macro_f1_mean", True), ("regression", "r2_mean", True), ("regression", "rmse_mean", False), ("regression", "mae_mean", False)):
        best = _best_feature_selection_row(rows, task=task, metric=metric, higher_is_better=higher)
        if best:
            summaries.append(_feature_selection_row_summary(summary_id, best, f"best_{task}_{metric}", metric))
    for task, metric, higher in (("classification", "macro_f1_mean", True), ("regression", "r2_mean", True)):
        smallest = smallest_within_tolerance(rows, task=task, metric=metric, higher_is_better=higher)
        if smallest:
            summaries.append(_feature_selection_row_summary(summary_id, smallest, f"smallest_within_1pct_{task}", metric))
    for record in _find_records(records, aliases=("recommended_feature_set", "retained_performance"), analysis_type="feature selection"):
        summaries.append(
            _summary_from_records(
                summary_id,
                "feature_selection",
                "recommended_feature_set",
                record.metric_name,
                [record],
                metric_value=record.metric_value,
                biological_entity=record.biological_entity,
                model_name=record.model_name,
                aggregation_method="recommended_default_direct_value",
            )
        )
    reductions = sorted({row.get("reduction_level_percent") for row in rows if row.get("reduction_level_percent") is not None})
    if reductions:
        evidence_ids = sorted({eid for row in rows for eid in row.get("_evidence_ids", [])})
        source_files = sorted({source for row in rows for source in row.get("_source_files", [])})
        summaries.append(
            SummaryRecord(
                summary_id=summary_id.next(),
                analysis_section="feature_selection",
                summary_type="tested_reduction_levels",
                metric_name="tested_reduction_levels",
                metric_value=";".join(str(value) for value in reductions),
                evidence_record_count=len(evidence_ids),
                source_evidence_ids=evidence_ids,
                source_files=source_files,
                aggregation_method="distinct_values_from_feature_selection_summary",
                confidence="HIGH",
                status="OK",
            )
        )
    return summaries


def _strain_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    strain_records = [record for record in records if record.analysis_type == "strain ablation"]
    for metric, summary_type, lowest in (
        ("held_out_f1_macro", "leave_one_strain_classification", False),
        ("held_out_balanced_accuracy", "leave_one_strain_balanced_accuracy", False),
        ("held_out_r2", "leave_one_strain_regression_r2", False),
        ("held_out_rmse", "leave_one_strain_regression_rmse", True),
        ("held_out_mae", "leave_one_strain_regression_mae", True),
        ("importance", "strain_importance", False),
    ):
        metric_records = [record for record in strain_records if record.metric_name == metric]
        sorted_records = sorted(
            metric_records,
            key=lambda record: (
                _sort_value(record.metric_value, lowest=lowest),
                record.biological_entity.casefold(),
                record.source_file.casefold(),
            ),
        )[:STRAIN_METRIC_LIMIT]
        for rank, record in enumerate(sorted_records, start=1):
            summaries.append(
                _summary_from_records(
                    summary_id,
                    "strain_evidence",
                    summary_type,
                    metric,
                    [record],
                    metric_value=record.metric_value,
                    biological_entity=record.biological_entity,
                    model_name=record.model_name,
                    rank=rank,
                    direction=metric_direction(metric, record.metric_value),
                    aggregation_method="top_10_strain_metric_values",
                )
            )
    return summaries


def _limitations_summary(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    limitation_metrics = (
        "qc_passed",
        "validation_passed",
        "conflicting_value_duplicate_count",
        "ambiguous_measurement_identity_count",
        "error_count",
        "missing_identifier_counts_plate_id",
        "missing_identifier_counts_well_id",
        "rows_excluded_from_blind_classification_because_selected_features_were_non_finite",
        "rows_excluded_from_blind_regression_because_selected_features_were_non_finite",
    )
    for metric in limitation_metrics:
        matches = _find_records(records, aliases=(metric,), analysis_contains=("QC", "real blind validation", "feature extraction", "advanced feature engineering"))
        if matches:
            summaries.extend(_scalar_summaries(summary_id, "limitations", "evidence_based_limitation", metric, matches))
    blind_status_records = [
        record
        for record in records
        if record.analysis_type == "real blind validation" and record.source_run == "not yet available"
    ]
    if blind_status_records:
        summaries.append(
            _summary_from_records(
                summary_id,
                "limitations",
                "blind_validation_limitation",
                "real_blind_validation_not_yet_available",
                blind_status_records[:10],
                metric_value=True,
                aggregation_method="source_run_not_yet_available_flag",
                confidence="HIGH",
                notes="Infrastructure/prediction evidence exists, but source_run records real validation as not yet available.",
            )
        )
    return summaries


def _blind_validation_status(records: list[EvidenceInputRecord], summary_id: _SummaryId) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    blind_records = [record for record in records if record.analysis_type == "real blind validation"]
    prediction_sources = [record for record in blind_records if "real_subset_prediction" in record.source_file]
    if prediction_sources:
        summaries.append(
            _summary_from_records(
                summary_id,
                "blind_validation_status",
                "infrastructure_status",
                "blind_prediction_infrastructure_evidence_available",
                prediction_sources[:20],
                metric_value=True,
                aggregation_method="selected_prediction_output_presence",
            )
        )
    for metric in ("prediction_passed", "predicted_chemical", "chemical_confidence", "novelty_status", "novelty_score"):
        summaries.extend(_scalar_summaries(summary_id, "blind_validation_status", "blind_prediction_output", metric, _find_records(records, aliases=(metric,), analysis_type="real blind validation")))
    not_available = [record for record in blind_records if record.source_run == "not yet available"]
    if not_available:
        summaries.append(
            _summary_from_records(
                summary_id,
                "blind_validation_status",
                "real_blind_validation_status",
                "real_blind_experimental_validation_status",
                not_available[:20],
                metric_value="not yet available",
                aggregation_method="source_run_status",
                confidence="HIGH",
                notes="Kept distinct from blind-prediction infrastructure.",
            )
        )
    summaries.append(_missing(summary_id, "blind_validation_status", "simulated_blind_test_status", "simulated_blind_test_status"))
    summaries.append(_missing(summary_id, "blind_validation_status", "frozen_model_bundle_status", "frozen_model_bundle_status"))
    return summaries


def _regression_model_comparison(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, model in enumerate(rankings, start=1):
        rows.append(
            {
                "model_name": model.get("model_name", ""),
                "rank": rank,
                "r2_mean": _blank_if_none(model.get("r2_mean")),
                "r2_std": _blank_if_none(model.get("r2_std")),
                "rmse_mean": _blank_if_none(model.get("rmse_mean")),
                "rmse_std": _blank_if_none(model.get("rmse_std")),
                "mae_mean": _blank_if_none(model.get("mae_mean")),
                "mae_std": _blank_if_none(model.get("mae_std")),
                "median_absolute_error_mean": _blank_if_none(model.get("median_absolute_error_mean")),
                "explained_variance_mean": _blank_if_none(model.get("explained_variance_mean")),
                "fold_count": _blank_if_none(model.get("fold_count")),
                "sample_count": _blank_if_none(model.get("sample_count")),
                "selection_status": "SELECTED" if rank == 1 else "NOT_SELECTED",
            }
        )
    return rows


def _model_metric_summaries(
    summary_id: _SummaryId,
    section: str,
    metric_name: str,
    records: list[EvidenceInputRecord],
    *,
    model_name: str,
    rank: int,
    aggregation_method: str,
) -> list[SummaryRecord]:
    if not records:
        return [_missing(summary_id, section, "model_ranking_metric", metric_name)]
    grouped: dict[str, list[EvidenceInputRecord]] = defaultdict(list)
    for record in records:
        grouped[_value_key(record.metric_value)].append(record)
    status = "CONFLICT" if len(grouped) > 1 else "OK"
    summaries = []
    for value_key in sorted(grouped):
        group = grouped[value_key]
        summaries.append(
            _summary_from_records(
                summary_id,
                section,
                "model_ranking_metric",
                metric_name,
                group,
                metric_value=group[0].metric_value,
                model_name=model_name,
                rank=rank,
                direction=metric_direction(metric_name, group[0].metric_value),
                aggregation_method=aggregation_method,
                status=status,
                notes="Conflicting aliases preserved under section-specific precedence." if status == "CONFLICT" else "",
            )
        )
    return summaries


def _scalar_summaries(
    summary_id: _SummaryId,
    section: str,
    summary_type: str,
    metric_name: str,
    records: list[EvidenceInputRecord],
    *,
    notes: str = "",
) -> list[SummaryRecord]:
    if not records:
        return [_missing(summary_id, section, summary_type, metric_name)]
    preferred = select_preferred_records(records)
    grouped: dict[str, list[EvidenceInputRecord]] = defaultdict(list)
    for record in preferred:
        grouped[_value_key(record.metric_value)].append(record)
    status = "CONFLICT" if len(grouped) > 1 else "OK"
    summaries: list[SummaryRecord] = []
    for value_key in sorted(grouped):
        group = grouped[value_key]
        summaries.append(
            _summary_from_records(
                summary_id,
                section,
                summary_type,
                metric_name,
                group,
                metric_value=group[0].metric_value,
                direction=metric_direction(metric_name, group[0].metric_value),
                aggregation_method="preferred_source_direct_value",
                status=status,
                notes=notes if status != "CONFLICT" else f"Conflicting values preserved. {notes}".strip(),
            )
        )
    return summaries


def _summary_from_records(
    summary_id: _SummaryId,
    section: str,
    summary_type: str,
    metric_name: str,
    records: list[EvidenceInputRecord],
    *,
    metric_value: Any | None,
    metric_units: str = "",
    model_name: str = "",
    biological_entity: str = "",
    comparison_group: str = "",
    rank: int | str = "",
    direction: str = "",
    aggregation_method: str = "",
    confidence: str = "HIGH",
    status: str = "OK",
    notes: str = "",
) -> SummaryRecord:
    source_ids = sorted({record.evidence_id for record in records})
    source_files = sorted({record.source_file for record in records if record.source_file})
    units = metric_units or _first_non_empty(record.metric_units for record in records)
    return SummaryRecord(
        summary_id=summary_id.next(),
        analysis_section=section,
        summary_type=summary_type,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_units=units,
        model_name=model_name or _first_non_empty(record.model_name for record in records),
        biological_entity=biological_entity,
        comparison_group=comparison_group,
        rank=rank,
        direction=direction,
        evidence_record_count=len(source_ids),
        source_evidence_ids=source_ids,
        source_files=source_files,
        aggregation_method=aggregation_method,
        confidence=confidence,
        status=status,
        notes=notes,
    )


def _missing(summary_id: _SummaryId, section: str, summary_type: str, metric_name: str) -> SummaryRecord:
    return SummaryRecord(
        summary_id=summary_id.next(),
        analysis_section=section,
        summary_type=summary_type,
        metric_name=metric_name,
        metric_value=None,
        evidence_record_count=0,
        source_evidence_ids=[],
        source_files=[],
        aggregation_method="missing_evidence_check",
        confidence="NULL",
        status="MISSING",
        notes="Expected summary value was not available from input evidence.",
    )


def _missing_selected_model(summary_id: _SummaryId, section: str, summary_type: str, metric_name: str, model_name: str) -> SummaryRecord:
    return SummaryRecord(
        summary_id=summary_id.next(),
        analysis_section=section,
        summary_type=summary_type,
        metric_name=metric_name,
        metric_value=None,
        model_name=model_name,
        evidence_record_count=0,
        source_evidence_ids=[],
        source_files=[],
        aggregation_method="selected_model_metric_missing_check",
        confidence="NULL",
        status="MISSING",
        notes="Selected-model metric was not available with evidence records belonging to the selected model.",
    )


def _find_records(
    records: list[EvidenceInputRecord],
    *,
    aliases: Iterable[str],
    analysis_type: str = "",
    analysis_contains: tuple[str, ...] = (),
) -> list[EvidenceInputRecord]:
    alias_set = {alias.casefold() for alias in aliases}
    results = []
    for record in records:
        if analysis_type and record.analysis_type != analysis_type:
            continue
        if analysis_contains and not any(token.casefold() in record.analysis_type.casefold() for token in analysis_contains):
            continue
        metric = record.metric_name.casefold()
        if metric in alias_set:
            results.append(record)
    return sorted(results, key=lambda record: record.evidence_id)


def _metric_mapping_audit(records: list[EvidenceInputRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[EvidenceInputRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.analysis_type, record.metric_name)].append(record)

    audit_rows: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    for (analysis_type, metric_name), group in sorted(grouped.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())):
        matches = []
        for record in group:
            matches.extend(matching_aliases(record))
        unique_matches = sorted(
            {(match.section, match.summary_metric, match.role, match.notes) for match in matches},
            key=lambda item: (item[0], item[1], item[2]),
        )
        mapping_status = "MAPPED" if unique_matches else "UNMAPPED"
        mapped_sections = "; ".join(dict.fromkeys(match[0] for match in unique_matches))
        mapped_summary_metrics = "; ".join(dict.fromkeys(match[1] for match in unique_matches))
        roles = "; ".join(dict.fromkeys(match[2] for match in unique_matches))
        notes = "; ".join(dict.fromkeys(match[3] for match in unique_matches if match[3]))
        row = {
            "analysis_type": analysis_type,
            "metric_name": metric_name,
            "record_count": len(group),
            "mapping_status": mapping_status,
            "mapped_sections": mapped_sections,
            "mapped_summary_metrics": mapped_summary_metrics,
            "roles": roles,
            "source_files": "; ".join(sorted({record.source_file for record in group if record.source_file})[:25]),
            "notes": notes if notes else ("No explicit alias registry entry for this section/metric." if mapping_status == "UNMAPPED" else ""),
        }
        audit_rows.append(row)
        if mapping_status == "UNMAPPED":
            unmapped_rows.append(
                {
                    "analysis_type": analysis_type,
                    "metric_name": metric_name,
                    "record_count": len(group),
                    "source_files": row["source_files"],
                    "reason": "No explicit alias registry entry for this section/metric.",
                }
            )
    return audit_rows, unmapped_rows


def _summary_population_audit(summaries: list[SummaryRecord]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        populated = summary.status != "MISSING" and bool(summary.source_evidence_ids)
        rows.append(
            {
                "summary_id": summary.summary_id,
                "analysis_section": summary.analysis_section,
                "summary_type": summary.summary_type,
                "metric_name": summary.metric_name,
                "status": summary.status,
                "populated": str(populated),
                "evidence_record_count": summary.evidence_record_count,
                "source_evidence_ids": "; ".join(summary.source_evidence_ids),
                "source_files": "; ".join(summary.source_files),
                "missing_reason": summary.notes if summary.status == "MISSING" else "",
            }
        )
    return rows


def _top_matrix_values(
    summary_id: _SummaryId,
    records: list[EvidenceInputRecord],
    section: str,
    summary_type: str,
    source_file: str,
    *,
    column_filter: set[str] | None = None,
    limit: int = 10,
    lowest: bool = False,
    absolute: bool = False,
    exclude_diagonal: bool = False,
) -> list[SummaryRecord]:
    candidates = []
    for record in records:
        if record.source_file != source_file or record.metric_name != "matrix_value":
            continue
        entity = parse_matrix_entity(record.biological_entity)
        column = entity.get("column", "")
        row = entity.get("row", "")
        if column_filter and column not in column_filter:
            continue
        if exclude_diagonal and row == column:
            continue
        value = _as_float(record.metric_value)
        if value is None:
            continue
        candidates.append((abs(value) if absolute else value, row, column, record))
    candidates = sorted(candidates, key=lambda item: (item[0] if lowest else -item[0], item[1].casefold(), item[2].casefold()))[:limit]
    summaries = []
    for rank, (_, row, column, record) in enumerate(candidates, start=1):
        summaries.append(
            _summary_from_records(
                summary_id,
                section,
                summary_type,
                column or "matrix_value",
                [record],
                metric_value=record.metric_value,
                biological_entity=record.biological_entity,
                comparison_group=row,
                rank=rank,
                direction=metric_direction(column or "matrix_value", record.metric_value),
                aggregation_method="ranked_matrix_values",
            )
        )
    return summaries


def _matrix_extrema(
    summary_id: _SummaryId,
    records: list[EvidenceInputRecord],
    section: str,
    summary_type: str,
    source_file: str,
) -> list[SummaryRecord]:
    matrix = [
        record
        for record in records
        if record.source_file == source_file and record.metric_name == "matrix_value" and _as_float(record.metric_value) is not None
    ]
    if not matrix:
        return [_missing(summary_id, section, summary_type, f"{summary_type}_matrix_values")]
    minimum = min(matrix, key=lambda record: (_as_float(record.metric_value), record.biological_entity.casefold()))
    maximum = max(matrix, key=lambda record: (_as_float(record.metric_value), record.biological_entity.casefold()))
    return [
        _summary_from_records(summary_id, section, summary_type, "minimum_matrix_value", [minimum], metric_value=minimum.metric_value, biological_entity=minimum.biological_entity, aggregation_method="matrix_minimum"),
        _summary_from_records(summary_id, section, summary_type, "maximum_matrix_value", [maximum], metric_value=maximum.metric_value, biological_entity=maximum.biological_entity, aggregation_method="matrix_maximum"),
    ]


def _feature_family_extrema(summary_id: _SummaryId, table: dict[str, dict[str, EvidenceInputRecord]]) -> list[SummaryRecord]:
    summaries: list[SummaryRecord] = []
    extrema_specs = (
        ("highest_r2_family", "regression_r2", False),
        ("lowest_rmse_family", "regression_rmse", True),
        ("lowest_mae_family", "regression_mae", True),
    )
    for metric_name, column, lowest in extrema_specs:
        candidates = [
            (family, record)
            for family, values in table.items()
            for key, record in values.items()
            if key == column and _as_float(record.metric_value) is not None
        ]
        if not candidates:
            summaries.append(_missing(summary_id, "feature_engineering", "feature_family_extrema", metric_name))
            continue
        family, record = sorted(candidates, key=lambda item: (_sort_value(item[1].metric_value, lowest=lowest), item[0].casefold()))[0]
        summaries.append(
            _summary_from_records(
                summary_id,
                "feature_engineering",
                "feature_family_extrema",
                metric_name,
                [record],
                metric_value=record.metric_value,
                comparison_group=family,
                direction=metric_direction(column, record.metric_value),
                aggregation_method="feature_family_extreme_value",
            )
        )
    balanced_candidates = []
    for family, values in table.items():
        required = ["regression_r2", "regression_rmse", "regression_mae", "runtime_increase_seconds"]
        if not all(metric in values and _as_float(values[metric].metric_value) is not None for metric in required):
            continue
        score = (
            -_as_float(values["regression_r2"].metric_value),
            _as_float(values["regression_rmse"].metric_value),
            _as_float(values["regression_mae"].metric_value),
            _as_float(values["runtime_increase_seconds"].metric_value),
            family.casefold(),
        )
        balanced_candidates.append((score, family, [values[metric] for metric in required]))
    if balanced_candidates:
        _, family, evidence = sorted(balanced_candidates, key=lambda item: item[0])[0]
        summaries.append(
            _summary_from_records(
                summary_id,
                "feature_engineering",
                "feature_family_extrema",
                "most_balanced_feature_family",
                evidence,
                metric_value=family,
                comparison_group=family,
                aggregation_method="rank_sum_r2_rmse_mae_runtime",
            )
        )
    for family, values in sorted(table.items()):
        worsened = [
            values[key]
            for key in ("regression_r2_gain", "classification_macro_f1_gain")
            if key in values and _as_float(values[key].metric_value) is not None and _as_float(values[key].metric_value) < 0
        ]
        if worsened:
            summaries.append(
                _summary_from_records(
                    summary_id,
                    "feature_engineering",
                    "feature_family_worsened_metric",
                    "worsened_performance_metric_count",
                    worsened,
                    metric_value=len(worsened),
                    comparison_group=family,
                    aggregation_method="negative_gain_count",
                )
            )
    return summaries


def _feature_selection_rows(records: list[EvidenceInputRecord]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.analysis_type != "feature selection" or not record.source_file.endswith("feature_selection_summary.csv"):
            continue
        entity = parse_matrix_entity(record.biological_entity)
        subset = entity.get("feature_subset_id", "")
        if not subset:
            continue
        row = rows.setdefault(subset, {"feature_subset_id": subset, "_evidence_ids": [], "_source_files": []})
        row["task"] = entity.get("task", row.get("task", ""))
        row["selector_method"] = entity.get("selector_method", row.get("selector_method", ""))
        row[record.metric_name] = record.metric_value
        row["_evidence_ids"].append(record.evidence_id)
        row["_source_files"].append(record.source_file)
        if "reduction_level_percent" not in row:
            match = entity.get("feature_subset_id", "")
            percent = _percent_from_subset(match)
            if percent is not None:
                row["reduction_level_percent"] = percent
    return list(rows.values())


def _best_feature_selection_row(rows: list[dict[str, Any]], *, task: str, metric: str, higher_is_better: bool) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("task") == task and _as_float(row.get(metric)) is not None]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -_as_float(row[metric]) if higher_is_better else _as_float(row[metric]),
            _as_float(row.get("feature_count")) if row.get("feature_count") is not None else float("inf"),
            str(row.get("feature_subset_id", "")).casefold(),
        ),
    )[0]


def _feature_selection_row_summary(summary_id: _SummaryId, row: dict[str, Any], summary_type: str, metric: str) -> SummaryRecord:
    evidence_ids = sorted(set(row.get("_evidence_ids", [])))
    source_files = sorted(set(row.get("_source_files", [])))
    return SummaryRecord(
        summary_id=summary_id.next(),
        analysis_section="feature_selection",
        summary_type=summary_type,
        metric_name=metric,
        metric_value=row.get(metric),
        metric_units="unitless",
        biological_entity=f"task={row.get('task', '')}; selector_method={row.get('selector_method', '')}",
        comparison_group=row.get("feature_subset_id", ""),
        direction=metric_direction(metric, row.get(metric)),
        evidence_record_count=len(evidence_ids),
        source_evidence_ids=evidence_ids,
        source_files=source_files,
        aggregation_method="feature_selection_subset_ranking",
        confidence="HIGH",
        status="OK",
        notes=f"feature_count={row.get('feature_count', '')}",
    )


def _aggregation_warnings(summaries: list[SummaryRecord]) -> list[str]:
    warnings: list[str] = []
    conflict_count = sum(1 for summary in summaries if summary.status == "CONFLICT")
    missing_count = sum(1 for summary in summaries if summary.status == "MISSING")
    metadata_only = sum(1 for summary in summaries if summary.status == "METADATA_ONLY")
    if conflict_count:
        warnings.append(f"Conflicting summary values preserved: {conflict_count}.")
    if missing_count:
        warnings.append(f"Missing expected summaries: {missing_count}.")
    if metadata_only:
        warnings.append(f"Metadata-only summaries without source evidence IDs: {metadata_only}.")
    return warnings


def _load_optional_json_metadata(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        return payload["metadata"]
    return {}


def _first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def _value_key(value: Any) -> str:
    parsed = parse_metric_value(value)
    if isinstance(parsed, float):
        return f"{parsed:.12g}"
    return str(parsed)


def _as_float(value: Any) -> float | None:
    parsed = parse_metric_value(value)
    if isinstance(parsed, bool) or parsed is None:
        return None
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return None


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _selected_regression_model_name(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row.get("selection_status") == "SELECTED":
            return str(row.get("model_name", ""))
    return ""


def _sort_value(value: Any, *, lowest: bool) -> float:
    numeric = _as_float(value)
    if numeric is None:
        return float("inf")
    return numeric if lowest else -numeric


def _percent_from_subset(value: str) -> int | None:
    import re

    match = re.search(r"__(\d+)pct", value)
    if not match:
        return None
    return int(match.group(1))
