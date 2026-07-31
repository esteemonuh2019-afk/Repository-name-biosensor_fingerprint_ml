"""Deterministic aggregation and ranking rules for Stage 9B.2B."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
import math
import re
from typing import Any, Iterable

from src.scientific_narrative.scientific_summary import EvidenceInputRecord


CLASSIFICATION_RANK_METRICS = ("f1_macro_mean", "balanced_accuracy_mean", "accuracy_mean")
REGRESSION_RANK_METRICS = ("r2_mean", "rmse_mean", "mae_mean")
LOWER_IS_BETTER = {"rmse_mean", "mae_mean", "median_absolute_error_mean", "log_loss_mean", "runtime_seconds"}


@dataclass(frozen=True)
class MetricAlias:
    """One explicit evidence-to-summary metric mapping rule."""

    section: str
    summary_metric: str
    alias: str
    analysis_types: tuple[str, ...]
    precedence: int
    role: str
    notes: str = ""


METRIC_ALIAS_REGISTRY: tuple[MetricAlias, ...] = (
    MetricAlias("dataset_summary", "total_canonical_rows", "row_count", ("canonical QC; feature validation",), 1, "dataset"),
    MetricAlias("dataset_summary", "total_canonical_rows", "canonical_qc_row_count", ("canonical QC; feature validation",), 2, "dataset"),
    MetricAlias("dataset_summary", "total_measurement_units", "measurement_unit_count", ("canonical QC; feature validation",), 1, "dataset"),
    MetricAlias("dataset_summary", "total_measurement_units", "canonical_qc_measurement_unit_count", ("canonical QC; feature validation",), 2, "dataset"),
    MetricAlias("dataset_summary", "total_feature_rows", "feature_row_count", ("feature extraction",), 1, "dataset"),
    MetricAlias("dataset_summary", "total_feature_rows", "feature_qc_feature_rows", ("feature extraction",), 2, "dataset"),
    MetricAlias("dataset_summary", "total_fingerprint_rows", "fingerprint_rows", ("fingerprint generation",), 1, "dataset"),
    MetricAlias("dataset_summary", "total_fingerprint_rows", "summary_fingerprint_rows", ("fingerprint generation",), 2, "dataset"),
    MetricAlias("dataset_summary", "individual_fingerprint_count", "individual_fingerprint_rows", ("fingerprint generation",), 1, "dataset"),
    MetricAlias("dataset_summary", "individual_fingerprint_count", "summary_individual_fingerprint_rows", ("fingerprint generation",), 2, "dataset"),
    MetricAlias("dataset_summary", "consensus_fingerprint_count", "consensus_fingerprint_rows", ("fingerprint generation",), 1, "dataset"),
    MetricAlias("dataset_summary", "consensus_fingerprint_count", "summary_consensus_fingerprint_rows", ("fingerprint generation",), 2, "dataset"),
    MetricAlias("dataset_summary", "number_of_strains", "unique_strain_count", ("canonical QC; feature validation",), 1, "dataset"),
    MetricAlias("dataset_summary", "number_of_strains", "strains_detected_count", ("canonical QC; feature validation",), 2, "dataset"),
    MetricAlias("dataset_summary", "number_of_strains", "canonical_qc_strains_detected_count", ("canonical QC; feature validation",), 3, "dataset"),
    MetricAlias("dataset_summary", "number_of_chemicals", "unique_chemical_count", ("canonical QC; feature validation",), 1, "dataset"),
    MetricAlias("dataset_summary", "number_of_chemicals", "chemicals_detected_count", ("canonical QC; feature validation",), 2, "dataset"),
    MetricAlias("dataset_summary", "number_of_chemicals", "canonical_qc_chemicals_detected_count", ("canonical QC; feature validation",), 3, "dataset"),
    MetricAlias("dataset_summary", "number_of_concentrations", "unique_concentration_count", ("canonical QC; feature validation",), 1, "dataset"),
    MetricAlias("dataset_summary", "number_of_concentrations", "concentrations_detected_count", ("canonical QC; feature validation",), 2, "dataset"),
    MetricAlias("dataset_summary", "number_of_concentrations", "canonical_qc_concentrations_detected_count", ("canonical QC; feature validation",), 3, "dataset"),
    MetricAlias("dataset_summary", "source_file_count", "source_files_count", ("canonical QC; feature validation",), 1, "dataset"),
    MetricAlias("dataset_summary", "source_file_count", "canonical_qc_source_files_count", ("canonical QC; feature validation",), 2, "dataset"),
    MetricAlias("quality_control", "qc_passed", "qc_passed", ("canonical QC; feature validation", "feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("quality_control", "qc_passed", "canonical_qc_qc_passed", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "validation_passed", "validation_passed", ("canonical QC; feature validation", "feature extraction"), 1, "qc"),
    MetricAlias("quality_control", "feature_qc_passed", "feature_qc_passed", ("feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("quality_control", "ambiguous_measurement_identity_count", "ambiguous_measurement_identity_count", ("canonical QC; feature validation",), 1, "qc"),
    MetricAlias("quality_control", "ambiguous_measurement_identity_count", "canonical_qc_ambiguous_measurement_identity_count", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "duplicate_timepoint_group_count", "duplicate_timepoint_group_count", ("canonical QC; feature validation",), 1, "qc"),
    MetricAlias("quality_control", "duplicate_timepoint_group_count", "canonical_qc_duplicate_timepoint_group_count", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "logical_duplicate_count", "logical_duplicate_count", ("canonical QC; feature validation",), 1, "qc"),
    MetricAlias("quality_control", "logical_duplicate_count", "canonical_qc_logical_duplicate_count", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "conflicting_value_duplicate_count", "conflicting_value_duplicate_count", ("canonical QC; feature validation",), 1, "qc"),
    MetricAlias("quality_control", "conflicting_value_duplicate_count", "canonical_qc_conflicting_value_duplicate_count", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "missing_identifier_counts_plate_id", "missing_identifier_counts_plate_id", ("canonical QC; feature validation",), 1, "qc"),
    MetricAlias("quality_control", "missing_identifier_counts_plate_id", "canonical_qc_missing_identifier_counts_plate_id", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "missing_identifier_counts_well_id", "missing_identifier_counts_well_id", ("canonical QC; feature validation",), 1, "qc"),
    MetricAlias("quality_control", "missing_identifier_counts_well_id", "canonical_qc_missing_identifier_counts_well_id", ("canonical QC; feature validation",), 2, "qc"),
    MetricAlias("quality_control", "zero_baseline_count", "zero_baseline_count", ("feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("quality_control", "zero_baseline_count", "qc_zero_baseline_count", ("feature extraction", "fingerprint generation"), 2, "qc"),
    MetricAlias("quality_control", "failed_feature_rows", "failed_feature_rows", ("feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("quality_control", "failed_feature_rows", "qc_failed_feature_rows", ("feature extraction", "fingerprint generation"), 2, "qc"),
    MetricAlias("quality_control", "warning_feature_rows", "warning_feature_rows", ("feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("quality_control", "warning_feature_rows", "qc_warning_feature_rows", ("feature extraction", "fingerprint generation"), 2, "qc"),
    MetricAlias("quality_control", "error_count", "error_count", ("canonical QC; feature validation", "feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("quality_control", "warning_count", "warning_count", ("canonical QC; feature validation", "feature extraction", "fingerprint generation"), 1, "qc"),
    MetricAlias("fingerprint_summary", "fingerprint_rows", "fingerprint_rows", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "fingerprint_rows", "summary_fingerprint_rows", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("fingerprint_summary", "individual_fingerprint_rows", "individual_fingerprint_rows", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "consensus_fingerprint_rows", "consensus_fingerprint_rows", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "distance_matrix_rows", "distance_matrix_rows", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "distance_matrix_rows", "summary_distance_matrix_rows", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("fingerprint_summary", "distance_matrix_columns", "distance_matrix_columns", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "distance_matrix_columns", "summary_distance_matrix_columns", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("fingerprint_summary", "consensus_distance_matrix_rows", "consensus_distance_matrix_rows", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "consensus_distance_matrix_columns", "consensus_distance_matrix_columns", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "duplicate_fingerprint_row_count", "duplicate_fingerprint_row_count", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "duplicate_fingerprint_row_count", "summary_duplicate_fingerprint_row_count", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("fingerprint_summary", "missing_feature_cell_count", "missing_feature_cell_count", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "missing_feature_cell_count", "qc_missing_feature_cell_count", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("fingerprint_summary", "fingerprint_qc_passed", "fingerprint_qc_passed", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "fingerprint_qc_passed", "summary_fingerprint_qc_passed", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("fingerprint_summary", "fingerprint_qc_warning_count", "fingerprint_qc_warning_count", ("fingerprint generation",), 1, "fingerprint"),
    MetricAlias("fingerprint_summary", "fingerprint_qc_warning_count", "summary_fingerprint_qc_warning_count", ("fingerprint generation",), 2, "fingerprint"),
    MetricAlias("exploratory_analysis", "explained_variance_ratio", "explained_variance_ratio", ("exploratory analysis",), 1, "exploratory"),
    MetricAlias("exploratory_analysis", "cumulative_explained_variance_ratio", "cumulative_explained_variance_ratio", ("exploratory analysis",), 1, "exploratory"),
    MetricAlias("exploratory_analysis", "explained_variance", "explained_variance", ("exploratory analysis",), 1, "exploratory"),
    MetricAlias("exploratory_analysis", "cluster_count", "cluster_count", ("exploratory analysis",), 1, "exploratory"),
    MetricAlias("exploratory_analysis", "matrix_value", "matrix_value", ("exploratory analysis",), 1, "exploratory"),
    MetricAlias("classification", "best_model", "best_model", ("classification",), 1, "primary"),
    MetricAlias("classification", "f1_macro_mean", "f1_macro_mean", ("classification",), 1, "primary", "Primary classification Macro F1."),
    MetricAlias("classification", "f1_macro_mean", "macro_f1_mean", ("classification",), 2, "primary", "Legacy classification Macro F1 alias."),
    MetricAlias("classification", "f1_macro_std", "f1_macro_std", ("classification",), 1, "primary"),
    MetricAlias("classification", "f1_macro_ci95_low", "f1_macro_ci95_low", ("classification",), 1, "primary"),
    MetricAlias("classification", "f1_macro_ci95_high", "f1_macro_ci95_high", ("classification",), 1, "primary"),
    MetricAlias("classification", "f1_weighted_mean", "f1_weighted_mean", ("classification",), 1, "primary"),
    MetricAlias("classification", "accuracy_mean", "accuracy_mean", ("classification",), 1, "primary"),
    MetricAlias("classification", "balanced_accuracy_mean", "balanced_accuracy_mean", ("classification",), 1, "primary"),
    MetricAlias("classification", "precision_macro_mean", "precision_macro_mean", ("classification",), 1, "primary"),
    MetricAlias("classification", "recall_macro_mean", "recall_macro_mean", ("classification",), 1, "primary"),
    MetricAlias("classification", "roc_auc_ovr_weighted_mean", "roc_auc_ovr_weighted_mean", ("classification",), 1, "primary"),
    MetricAlias("classification", "fold_count", "fold_count", ("classification",), 1, "primary"),
    MetricAlias("classification", "matrix_value", "matrix_value", ("classification",), 1, "secondary"),
    MetricAlias("classification", "confusion_matrix_count", "confusion_matrix_count", ("classification",), 1, "secondary"),
    MetricAlias("regression", "best_model", "best_model", ("regression",), 1, "primary"),
    MetricAlias("regression", "r2_mean", "r2_mean", ("regression",), 1, "primary", "Primary regression R2."),
    MetricAlias("regression", "r2_std", "r2_std", ("regression",), 1, "primary"),
    MetricAlias("regression", "r2_ci95_low", "r2_ci95_low", ("regression",), 1, "primary"),
    MetricAlias("regression", "r2_ci95_high", "r2_ci95_high", ("regression",), 1, "primary"),
    MetricAlias("regression", "rmse_mean", "rmse_mean", ("regression",), 1, "primary"),
    MetricAlias("regression", "rmse_std", "rmse_std", ("regression",), 1, "primary"),
    MetricAlias("regression", "rmse_ci95_low", "rmse_ci95_low", ("regression",), 1, "primary"),
    MetricAlias("regression", "rmse_ci95_high", "rmse_ci95_high", ("regression",), 1, "primary"),
    MetricAlias("regression", "mae_mean", "mae_mean", ("regression",), 1, "primary"),
    MetricAlias("regression", "mae_std", "mae_std", ("regression",), 1, "primary"),
    MetricAlias("regression", "mae_ci95_low", "mae_ci95_low", ("regression",), 1, "primary"),
    MetricAlias("regression", "mae_ci95_high", "mae_ci95_high", ("regression",), 1, "primary"),
    MetricAlias("regression", "median_absolute_error_mean", "median_absolute_error_mean", ("regression",), 1, "primary"),
    MetricAlias("regression", "explained_variance_mean", "explained_variance_mean", ("regression",), 1, "primary"),
    MetricAlias("regression", "fold_count", "fold_count", ("regression",), 1, "primary"),
    MetricAlias("regression", "sample_count", "sample_count", ("regression",), 1, "context"),
    MetricAlias("regression", "concentration_min", "concentration_min", ("regression",), 1, "context"),
    MetricAlias("regression", "concentration_max", "concentration_max", ("regression",), 1, "context"),
    MetricAlias("feature_engineering", "best_feature_family", "best_feature_family", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_r2", "regression_r2", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_r2", "r2_mean", ("advanced feature engineering",), 2, "feature_engineering", "Feature-family R2 alias, not primary regression."),
    MetricAlias("feature_engineering", "regression_r2_gain", "regression_r2_gain", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_rmse", "regression_rmse", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_rmse", "rmse_mean", ("advanced feature engineering",), 2, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_rmse_delta", "regression_rmse_delta", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_mae", "regression_mae", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_mae", "mae_mean", ("advanced feature engineering",), 2, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_mae_delta", "regression_mae_delta", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_sample_count", "regression_sample_count", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "runtime_increase_seconds", "runtime_increase_seconds", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "classification_improvement", "classification_improvement", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "regression_improvement", "regression_improvement", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_engineering", "matrix_value", "matrix_value", ("advanced feature engineering",), 1, "feature_engineering"),
    MetricAlias("feature_selection", "available_feature_count", "available_feature_count", ("feature selection",), 1, "feature_selection"),
    MetricAlias("feature_selection", "selected_feature_count", "selected_feature_count", ("feature selection",), 1, "feature_selection"),
    MetricAlias("feature_selection", "feature_count", "feature_count", ("feature selection",), 1, "feature_selection"),
    MetricAlias("feature_selection", "macro_f1_mean", "macro_f1_mean", ("feature selection",), 1, "feature_selection", "Feature-selection Macro F1, not primary classification."),
    MetricAlias("feature_selection", "balanced_accuracy_mean", "balanced_accuracy_mean", ("feature selection",), 1, "feature_selection"),
    MetricAlias("feature_selection", "r2_mean", "r2_mean", ("feature selection",), 1, "feature_selection", "Feature-selection R2, not primary regression."),
    MetricAlias("feature_selection", "rmse_mean", "rmse_mean", ("feature selection",), 1, "feature_selection"),
    MetricAlias("feature_selection", "mae_mean", "mae_mean", ("feature selection",), 1, "feature_selection"),
    MetricAlias("feature_selection", "recommended_feature_set", "recommended_feature_set", ("feature selection",), 1, "feature_selection"),
    MetricAlias("strain_evidence", "held_out_f1_macro", "held_out_f1_macro", ("strain ablation",), 1, "strain", "Leave-one-strain Macro F1, not primary classification."),
    MetricAlias("strain_evidence", "held_out_balanced_accuracy", "held_out_balanced_accuracy", ("strain ablation",), 1, "strain"),
    MetricAlias("strain_evidence", "held_out_r2", "held_out_r2", ("strain ablation",), 1, "strain", "Leave-one-strain R2, not primary regression."),
    MetricAlias("strain_evidence", "held_out_rmse", "held_out_rmse", ("strain ablation",), 1, "strain"),
    MetricAlias("strain_evidence", "held_out_mae", "held_out_mae", ("strain ablation",), 1, "strain"),
    MetricAlias("strain_evidence", "importance", "importance", ("strain ablation",), 1, "strain"),
    MetricAlias("blind_validation_status", "prediction_passed", "prediction_passed", ("real blind validation",), 1, "blind_validation"),
    MetricAlias("blind_validation_status", "predicted_chemical", "predicted_chemical", ("real blind validation",), 1, "blind_validation"),
    MetricAlias("blind_validation_status", "chemical_confidence", "chemical_confidence", ("real blind validation",), 1, "blind_validation"),
    MetricAlias("blind_validation_status", "novelty_status", "novelty_status", ("real blind validation",), 1, "blind_validation"),
    MetricAlias("blind_validation_status", "novelty_score", "novelty_score", ("real blind validation",), 1, "blind_validation"),
)


def parse_metric_value(value: str | Any) -> Any:
    """Parse an evidence value into bool, number, string, or None."""

    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if text.casefold() == "true":
        return True
    if text.casefold() == "false":
        return False
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer() and not any(token in text.casefold() for token in (".", "e")):
        return int(numeric)
    return numeric


def is_supported_evidence(record: EvidenceInputRecord) -> bool:
    """Return whether an input evidence row is usable for aggregation."""

    return record.extraction_status == "EXTRACTED" and record.metric_value is not None


def source_priority(record: EvidenceInputRecord) -> int:
    """Priority for resolving duplicate metric records from multiple sources."""

    source = record.source_file.casefold()
    if source.endswith("best_model_metrics.json") or source.endswith("best_regression_model.json"):
        return 0
    if source.endswith("model_rankings.csv"):
        return 1
    if source.endswith("classification_summary.csv") or source.endswith("regression_summary.csv"):
        return 2
    if source.endswith("feature_selection_summary.csv") or source.endswith("feature_family_ablation_summary.csv"):
        return 1
    if source.endswith(".json"):
        return 3
    if source.endswith(".csv"):
        return 4
    return 5


def select_preferred_records(records: Iterable[EvidenceInputRecord]) -> list[EvidenceInputRecord]:
    """Select records from the highest-priority source among duplicates."""

    grouped: dict[tuple[str, str, str, str], list[EvidenceInputRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.analysis_type, record.metric_name, record.model_name, record.biological_entity)].append(record)
    selected: list[EvidenceInputRecord] = []
    for group in grouped.values():
        priority = min(source_priority(record) for record in group)
        selected.extend(record for record in group if source_priority(record) == priority)
    return sorted(selected, key=lambda record: record.evidence_id)


def metric_alias_registry_rows() -> list[dict[str, Any]]:
    """Return the explicit alias registry in CSV-ready form."""

    rows = []
    for rule in METRIC_ALIAS_REGISTRY:
        row = asdict(rule)
        row["analysis_types"] = "; ".join(rule.analysis_types)
        rows.append(row)
    return rows


def matching_aliases(record: EvidenceInputRecord) -> list[MetricAlias]:
    """Return registry entries that match an evidence record exactly."""

    metric = record.metric_name.casefold()
    analysis_type = record.analysis_type.casefold()
    matches = [
        rule
        for rule in METRIC_ALIAS_REGISTRY
        if rule.alias.casefold() == metric
        and any(candidate.casefold() == analysis_type for candidate in rule.analysis_types)
    ]
    return sorted(matches, key=lambda rule: (rule.section, rule.summary_metric, rule.precedence, rule.alias.casefold()))


def canonical_metric_for(section: str, record: EvidenceInputRecord | None = None, metric_name: str = "", analysis_type: str = "") -> str | None:
    """Return the canonical summary metric for a section-specific evidence metric."""

    metric = (record.metric_name if record is not None else metric_name).casefold()
    observed_analysis = (record.analysis_type if record is not None else analysis_type).casefold()
    candidates = [
        rule
        for rule in METRIC_ALIAS_REGISTRY
        if rule.section == section
        and rule.alias.casefold() == metric
        and any(candidate.casefold() == observed_analysis for candidate in rule.analysis_types)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda rule: (rule.precedence, rule.summary_metric.casefold(), rule.alias.casefold()))[0].summary_metric


def aliases_for(section: str, summary_metric: str) -> tuple[str, ...]:
    """Return exact aliases for one section summary metric."""

    aliases = [
        rule.alias
        for rule in sorted(METRIC_ALIAS_REGISTRY, key=lambda item: (item.precedence, item.alias.casefold()))
        if rule.section == section and rule.summary_metric == summary_metric
    ]
    return tuple(dict.fromkeys(aliases))


def alias_precedence(section: str, summary_metric: str, record: EvidenceInputRecord) -> int:
    """Return alias precedence for a matching record."""

    matches = [
        rule.precedence
        for rule in METRIC_ALIAS_REGISTRY
        if rule.section == section
        and rule.summary_metric == summary_metric
        and rule.alias.casefold() == record.metric_name.casefold()
        and any(candidate.casefold() == record.analysis_type.casefold() for candidate in rule.analysis_types)
    ]
    return min(matches) if matches else 9999


def detect_conflicts(records: Iterable[EvidenceInputRecord]) -> dict[tuple[str, str, str, str], list[EvidenceInputRecord]]:
    """Return groups where the same metric key has conflicting values."""

    grouped: dict[tuple[str, str, str, str], list[EvidenceInputRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.analysis_type, record.metric_name, record.model_name, record.biological_entity)].append(record)
    conflicts: dict[tuple[str, str, str, str], list[EvidenceInputRecord]] = {}
    for key, group in grouped.items():
        values = {_value_key(record.metric_value) for record in group}
        if len(values) > 1:
            conflicts[key] = group
    return conflicts


def classification_model_rankings(records: list[EvidenceInputRecord]) -> list[dict[str, Any]]:
    """Rank classification models by Macro F1, balanced accuracy, accuracy, then name."""

    model_metrics = _model_metric_table(records, analysis_type="classification")
    return sorted(
        model_metrics.values(),
        key=lambda row: (
            -_numeric_or_low(row.get("f1_macro_mean")),
            -_numeric_or_low(row.get("balanced_accuracy_mean")),
            -_numeric_or_low(row.get("accuracy_mean")),
            str(row.get("model_name", "")).casefold(),
        ),
    )


def regression_model_rankings(records: list[EvidenceInputRecord]) -> list[dict[str, Any]]:
    """Rank regressors by R2 descending, RMSE ascending, MAE ascending, then name."""

    model_metrics = _model_metric_table(records, analysis_type="regression")
    return sorted(
        model_metrics.values(),
        key=lambda row: (
            -_numeric_or_low(row.get("r2_mean")),
            _numeric_or_high(row.get("rmse_mean")),
            _numeric_or_high(row.get("mae_mean")),
            str(row.get("model_name", "")).casefold(),
        ),
    )


def feature_family_table(records: list[EvidenceInputRecord]) -> dict[str, dict[str, EvidenceInputRecord]]:
    """Return feature-family rows extracted from matrix evidence."""

    table: dict[str, dict[str, EvidenceInputRecord]] = defaultdict(dict)
    for record in records:
        if record.analysis_type != "advanced feature engineering":
            continue
        if record.source_file.endswith("feature_family_ablation_summary.csv") and record.metric_name == "matrix_value":
            entity = parse_matrix_entity(record.biological_entity)
            row = entity.get("row", "")
            column = entity.get("column", "")
            canonical = canonical_metric_for("feature_engineering", metric_name=column, analysis_type=record.analysis_type)
            if row and canonical:
                table[row][canonical] = record
            elif row and column:
                table[row][column] = record
            continue
        canonical = canonical_metric_for("feature_engineering", record)
        if not canonical:
            continue
        entity = parse_matrix_entity(record.biological_entity)
        row = entity.get("row") or entity.get("feature_family") or record.model_name or record.source_file
        table[row][canonical] = record
    return dict(table)


def parse_matrix_entity(entity: str) -> dict[str, str]:
    """Parse biological_entity strings like ``row=A; column=B``."""

    result: dict[str, str] = {}
    for part in entity.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def aggregate_fold_metric(records: list[EvidenceInputRecord], metric_name: str) -> tuple[float | None, list[EvidenceInputRecord]]:
    """Aggregate fold-level metric values by arithmetic mean."""

    values: list[float] = []
    used: list[EvidenceInputRecord] = []
    for record in records:
        if record.metric_name != metric_name:
            continue
        value = _as_float(record.metric_value)
        if value is None:
            continue
        values.append(value)
        used.append(record)
    if not values:
        return None, []
    return sum(values) / len(values), used


def smallest_within_tolerance(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    feature_count_key: str = "feature_count",
    task: str | None = None,
    tolerance_fraction: float = 0.01,
    higher_is_better: bool = True,
) -> dict[str, Any] | None:
    """Return the smallest subset within tolerance of the best observed metric."""

    candidates = [
        row
        for row in rows
        if row.get(metric) is not None
        and row.get(feature_count_key) is not None
        and (task is None or row.get("task") == task)
    ]
    if not candidates:
        return None
    values = [float(row[metric]) for row in candidates]
    best = max(values) if higher_is_better else min(values)
    if higher_is_better:
        threshold = best - abs(best) * tolerance_fraction
        acceptable = [row for row in candidates if float(row[metric]) >= threshold]
    else:
        threshold = best + abs(best) * tolerance_fraction
        acceptable = [row for row in candidates if float(row[metric]) <= threshold]
    return sorted(
        acceptable,
        key=lambda row: (float(row[feature_count_key]), str(row.get("feature_subset_id", "")).casefold()),
    )[0]


def metric_direction(metric_name: str, value: Any | None = None) -> str:
    """Return the non-interpretive direction label for a metric."""

    metric = metric_name.casefold()
    if metric in LOWER_IS_BETTER or any(token in metric for token in ("rmse", "mae", "loss", "error", "runtime")):
        return "lower_is_better"
    if any(token in metric for token in ("f1", "accuracy", "precision", "recall", "auc", "r2", "variance")):
        return "higher_is_better"
    if isinstance(value, (int, float)) and value < 0:
        return "negative"
    return ""


def _model_metric_table(records: list[EvidenceInputRecord], *, analysis_type: str) -> dict[str, dict[str, Any]]:
    section = analysis_type
    grouped: dict[tuple[str, str], list[EvidenceInputRecord]] = defaultdict(list)
    for record in records:
        if record.analysis_type != analysis_type or not record.model_name:
            continue
        canonical = canonical_metric_for(section, record)
        if canonical is None:
            continue
        grouped[(record.model_name, canonical)].append(record)

    table: dict[str, dict[str, Any]] = {}
    for (model, canonical), group in grouped.items():
        chosen = select_preferred_records(group)
        chosen = sorted(
            chosen,
            key=lambda record: (
                alias_precedence(section, canonical, record),
                source_priority(record),
                record.evidence_id,
            ),
        )
        if not chosen:
            continue
        table.setdefault(model, {"model_name": model, "_records": [], "_evidence": {}, "_metric_conflicts": {}})
        table[model][canonical] = chosen[0].metric_value
        table[model]["_evidence"][canonical] = chosen
        table[model]["_records"].extend(chosen)
        values = {_value_key(record.metric_value) for record in chosen}
        if len(values) > 1:
            table[model]["_metric_conflicts"][canonical] = chosen
    return table


def _numeric_or_low(value: Any) -> float:
    parsed = _as_float(value)
    return parsed if parsed is not None else float("-inf")


def _numeric_or_high(value: Any) -> float:
    parsed = _as_float(value)
    return parsed if parsed is not None else float("inf")


def _as_float(value: Any) -> float | None:
    parsed = parse_metric_value(value)
    if isinstance(parsed, bool) or parsed is None:
        return None
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return None


def _value_key(value: Any) -> str:
    parsed = parse_metric_value(value)
    if isinstance(parsed, float):
        return f"{parsed:.12g}"
    return str(parsed)
