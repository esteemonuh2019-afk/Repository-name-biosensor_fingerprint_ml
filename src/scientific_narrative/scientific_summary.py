"""Structured evidence aggregation outputs for Stage 9B.2B."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


SUMMARY_FIELDS: tuple[str, ...] = (
    "summary_id",
    "analysis_section",
    "summary_type",
    "metric_name",
    "metric_value",
    "metric_units",
    "model_name",
    "biological_entity",
    "comparison_group",
    "rank",
    "direction",
    "evidence_record_count",
    "source_evidence_ids",
    "source_files",
    "aggregation_method",
    "confidence",
    "status",
    "notes",
)

REGRESSION_MODEL_COMPARISON_FIELDS: tuple[str, ...] = (
    "model_name",
    "rank",
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
    "selection_status",
)

OUTPUT_FILENAMES: tuple[str, ...] = (
    "aggregated_evidence.csv",
    "aggregated_evidence.json",
    "dataset_summary.csv",
    "qc_summary.csv",
    "exploratory_summary.csv",
    "classification_summary.csv",
    "regression_summary.csv",
    "regression_model_comparison.csv",
    "feature_engineering_summary.csv",
    "feature_selection_summary.csv",
    "strain_summary.csv",
    "limitations_summary.csv",
    "blind_validation_status.csv",
    "evidence_traceability.csv",
    "aggregation_report.md",
    "metric_alias_registry.csv",
    "metric_mapping_audit.csv",
    "unmapped_metrics.csv",
    "summary_population_audit.csv",
)


@dataclass(frozen=True)
class EvidenceInputRecord:
    """One input evidence row with a deterministic aggregation ID."""

    evidence_id: str
    analysis_type: str
    source_file: str
    source_run: str
    metric_name: str
    metric_value: Any | None
    metric_units: str = ""
    figure_reference: str = ""
    table_reference: str = ""
    biological_entity: str = ""
    model_name: str = ""
    confidence: str = ""
    extraction_status: str = ""
    notes: str = ""


@dataclass(frozen=True)
class SummaryRecord:
    """One compact, traceable scientific summary row."""

    summary_id: str
    analysis_section: str
    summary_type: str
    metric_name: str
    metric_value: Any | None
    metric_units: str = ""
    model_name: str = ""
    biological_entity: str = ""
    comparison_group: str = ""
    rank: int | str = ""
    direction: str = ""
    evidence_record_count: int = 0
    source_evidence_ids: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    aggregation_method: str = ""
    confidence: str = "HIGH"
    status: str = "OK"
    notes: str = ""


@dataclass
class AggregatedEvidence:
    """Complete Stage 9B.2B aggregation result."""

    dataset_summary: list[SummaryRecord] = field(default_factory=list)
    qc_summary: list[SummaryRecord] = field(default_factory=list)
    fingerprint_summary: list[SummaryRecord] = field(default_factory=list)
    exploratory_summary: list[SummaryRecord] = field(default_factory=list)
    classification_summary: list[SummaryRecord] = field(default_factory=list)
    regression_summary: list[SummaryRecord] = field(default_factory=list)
    regression_model_comparison: list[dict[str, Any]] = field(default_factory=list)
    feature_engineering_summary: list[SummaryRecord] = field(default_factory=list)
    feature_selection_summary: list[SummaryRecord] = field(default_factory=list)
    strain_summary: list[SummaryRecord] = field(default_factory=list)
    limitations_summary: list[SummaryRecord] = field(default_factory=list)
    blind_validation_status: list[SummaryRecord] = field(default_factory=list)
    traceability_index: list[dict[str, Any]] = field(default_factory=list)
    metric_alias_registry: list[dict[str, Any]] = field(default_factory=list)
    metric_mapping_audit: list[dict[str, Any]] = field(default_factory=list)
    unmapped_metrics: list[dict[str, Any]] = field(default_factory=list)
    summary_population_audit: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    aggregation_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_summaries(self) -> list[SummaryRecord]:
        """Return all section summary records in output order."""

        return [
            *self.dataset_summary,
            *self.qc_summary,
            *self.fingerprint_summary,
            *self.exploratory_summary,
            *self.classification_summary,
            *self.regression_summary,
            *self.feature_engineering_summary,
            *self.feature_selection_summary,
            *self.strain_summary,
            *self.limitations_summary,
            *self.blind_validation_status,
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable aggregation object."""

        return _json_ready(self)


def write_aggregation_outputs(
    aggregation: AggregatedEvidence,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    maximum_source_ids_per_cell: int = 50,
) -> list[Path]:
    """Write exact Stage 9B.2B aggregation outputs."""

    target = Path(output_dir)
    _prepare_output_dir(target, overwrite=overwrite)
    all_summaries = aggregation.all_summaries
    paths = {
        "aggregated_evidence.csv": target / "aggregated_evidence.csv",
        "aggregated_evidence.json": target / "aggregated_evidence.json",
        "dataset_summary.csv": target / "dataset_summary.csv",
        "qc_summary.csv": target / "qc_summary.csv",
        "exploratory_summary.csv": target / "exploratory_summary.csv",
        "classification_summary.csv": target / "classification_summary.csv",
        "regression_summary.csv": target / "regression_summary.csv",
        "regression_model_comparison.csv": target / "regression_model_comparison.csv",
        "feature_engineering_summary.csv": target / "feature_engineering_summary.csv",
        "feature_selection_summary.csv": target / "feature_selection_summary.csv",
        "strain_summary.csv": target / "strain_summary.csv",
        "limitations_summary.csv": target / "limitations_summary.csv",
        "blind_validation_status.csv": target / "blind_validation_status.csv",
        "evidence_traceability.csv": target / "evidence_traceability.csv",
        "aggregation_report.md": target / "aggregation_report.md",
        "metric_alias_registry.csv": target / "metric_alias_registry.csv",
        "metric_mapping_audit.csv": target / "metric_mapping_audit.csv",
        "unmapped_metrics.csv": target / "unmapped_metrics.csv",
        "summary_population_audit.csv": target / "summary_population_audit.csv",
    }
    _write_summary_csv(paths["aggregated_evidence.csv"], all_summaries, maximum_source_ids_per_cell)
    paths["aggregated_evidence.json"].write_text(
        json.dumps(aggregation.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(paths["dataset_summary.csv"], aggregation.dataset_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["qc_summary.csv"], aggregation.qc_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["exploratory_summary.csv"], aggregation.exploratory_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["classification_summary.csv"], aggregation.classification_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["regression_summary.csv"], aggregation.regression_summary, maximum_source_ids_per_cell)
    _write_dict_csv(paths["regression_model_comparison.csv"], aggregation.regression_model_comparison, list(REGRESSION_MODEL_COMPARISON_FIELDS))
    _write_summary_csv(paths["feature_engineering_summary.csv"], aggregation.feature_engineering_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["feature_selection_summary.csv"], aggregation.feature_selection_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["strain_summary.csv"], aggregation.strain_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["limitations_summary.csv"], aggregation.limitations_summary, maximum_source_ids_per_cell)
    _write_summary_csv(paths["blind_validation_status.csv"], aggregation.blind_validation_status, maximum_source_ids_per_cell)
    _write_dict_csv(paths["evidence_traceability.csv"], aggregation.traceability_index)
    _write_dict_csv(paths["metric_alias_registry.csv"], aggregation.metric_alias_registry)
    _write_dict_csv(paths["metric_mapping_audit.csv"], aggregation.metric_mapping_audit)
    _write_dict_csv(paths["unmapped_metrics.csv"], aggregation.unmapped_metrics)
    _write_dict_csv(paths["summary_population_audit.csv"], aggregation.summary_population_audit)
    paths["aggregation_report.md"].write_text(render_aggregation_report(aggregation), encoding="utf-8")
    return [paths[name] for name in OUTPUT_FILENAMES]


def render_aggregation_report(aggregation: AggregatedEvidence) -> str:
    """Render a non-interpretive aggregation report."""

    summaries = aggregation.all_summaries
    metadata = aggregation.metadata
    lines = [
        "# Stage 9B.2B Evidence Aggregation Report",
        "",
        "## Summary",
        "",
        f"- Evidence records received: {metadata.get('evidence_records_received', 0)}",
        f"- Evidence records used: {metadata.get('evidence_records_used', 0)}",
        f"- Evidence records excluded: {metadata.get('evidence_records_excluded', metadata.get('evidence_records_unused', 0))}",
        f"- Evidence records not selected into summaries: {metadata.get('evidence_records_unused', metadata.get('evidence_records_excluded', 0))}",
        f"- Unsupported or NULL evidence rows excluded: {metadata.get('unsupported_or_null_evidence_records', 0)}",
        f"- Unsupported files reported by evidence JSON: {metadata.get('unsupported_file_count_from_json', 0)}",
        f"- Unreadable files reported by evidence JSON: {metadata.get('unreadable_file_count_from_json', 0)}",
        f"- Summary records created: {len(summaries)}",
        f"- Summaries populated: {metadata.get('summary_records_populated', 0)}",
        f"- Selected regression model: {metadata.get('selected_regression_model', '')}",
        f"- Regression model comparison rows: {len(aggregation.regression_model_comparison)}",
        f"- Conflicting evidence detected: {metadata.get('conflicting_summary_count', 0)}",
        f"- Missing expected summaries: {metadata.get('missing_summary_count', 0)}",
        f"- Mapped unique metric/type pairs: {metadata.get('mapped_unique_metric_count', 0)}",
        f"- Unmapped unique metric/type pairs: {metadata.get('unmapped_unique_metric_count', 0)}",
        f"- Traceability coverage: {metadata.get('traceability_coverage', 0)}",
        f"- Scientific interpretation can proceed: {metadata.get('interpretation_can_proceed', False)}",
        "",
        "## Summaries By Section",
        "",
    ]
    counts = {
        "dataset": len(aggregation.dataset_summary),
        "qc": len(aggregation.qc_summary),
        "fingerprint": len(aggregation.fingerprint_summary),
        "exploratory": len(aggregation.exploratory_summary),
        "classification": len(aggregation.classification_summary),
        "regression": len(aggregation.regression_summary),
        "feature_engineering": len(aggregation.feature_engineering_summary),
        "feature_selection": len(aggregation.feature_selection_summary),
        "strain": len(aggregation.strain_summary),
        "limitations": len(aggregation.limitations_summary),
        "blind_validation": len(aggregation.blind_validation_status),
    }
    lines.extend(f"- {section}: {count}" for section, count in counts.items())
    lines.extend(["", "## Metric Mapping", ""])
    mapped_names = sorted(
        {
            row.get("metric_name", "")
            for row in aggregation.metric_mapping_audit
            if row.get("mapping_status") == "MAPPED" and row.get("metric_name")
        }
    )
    unmapped_names = sorted(
        {
            row.get("metric_name", "")
            for row in aggregation.unmapped_metrics
            if row.get("metric_name") and row.get("metric_name", "") not in mapped_names
        }
    )
    unmapped_pairs = sorted(
        {
            f"{row.get('analysis_type', '')}: {row.get('metric_name', '')}"
            for row in aggregation.unmapped_metrics
            if row.get("metric_name")
        }
    )
    lines.append(f"- Mapped metric names: {len(mapped_names)}")
    if mapped_names:
        lines.append(f"- Mapped metric name sample: {', '.join(mapped_names[:30])}")
    lines.append(f"- Unmapped metric names: {len(unmapped_names)}")
    if unmapped_names:
        lines.append(f"- Unmapped metric name sample: {', '.join(unmapped_names[:30])}")
    if unmapped_pairs:
        lines.append(f"- Unmapped metric/type pair sample: {', '.join(unmapped_pairs[:20])}")
    populated = [row for row in aggregation.summary_population_audit if row.get("populated") == "True"]
    missing_population = [row for row in aggregation.summary_population_audit if row.get("status") == "MISSING"]
    lines.append(f"- Summaries populated: {len(populated)}")
    lines.append(f"- Summaries still missing: {len(missing_population)}")
    lines.extend(["", "## Conflicts", ""])
    conflict_records = [record for record in summaries if record.status == "CONFLICT"]
    if conflict_records:
        for record in conflict_records[:25]:
            lines.append(f"- {record.summary_id}: {record.analysis_section} {record.metric_name} = {record.metric_value}")
        if len(conflict_records) > 25:
            lines.append(f"- Additional conflicts: {len(conflict_records) - 25}")
    else:
        lines.append("- None")
    lines.extend(["", "## Missing Expected Summaries", ""])
    missing = [record for record in summaries if record.status == "MISSING"]
    if missing:
        lines.extend(f"- {record.analysis_section}: {record.metric_name} ({record.notes})" for record in missing)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "This aggregation report reduces extracted evidence into traceable quantitative summaries only. It does not write Results, Discussion, DOCX, or PDF outputs.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _prepare_output_dir(target: Path, *, overwrite: bool) -> None:
    if target.exists() and target.is_file():
        raise FileExistsError(f"Output path is a file: {target}")
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is not empty: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _write_summary_csv(path: Path, records: Iterable[SummaryRecord], max_ids: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_summary_row(record, max_ids))


def _summary_row(record: SummaryRecord, max_ids: int) -> dict[str, Any]:
    row = asdict(record)
    row["source_evidence_ids"] = "; ".join(record.source_evidence_ids[:max_ids])
    if len(record.source_evidence_ids) > max_ids:
        row["source_evidence_ids"] += f"; ... ({len(record.source_evidence_ids) - max_ids} more)"
    row["source_files"] = "; ".join(record.source_files)
    row["metric_value"] = "" if record.metric_value is None else record.metric_value
    return row


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = _dict_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _dict_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["status"]
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
