"""Result parsers for Stage 9B.2A evidence extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from src.scientific_narrative.evidence_database import EvidenceRecord


SUMMARY_ONLY_FILENAMES = {
    "advanced_feature_dataset.csv",
    "feature_dataset.csv",
    "fingerprint_dataset.csv",
    "prediction_vs_actual.csv",
    "residuals.csv",
}

IDENTIFIER_COLUMNS = {
    "",
    "model_id",
    "model_name",
    "optional_model",
    "feature",
    "feature_name",
    "feature_family",
    "feature_set",
    "feature_subset_id",
    "task",
    "selector_method",
    "component",
    "chemical",
    "Chemical",
    "strain",
    "Strain",
    "Concentration",
    "cluster_id",
    "class",
    "label",
    "actual",
    "predicted",
}

ENTITY_COLUMNS = (
    "component",
    "chemical",
    "Chemical",
    "strain",
    "Strain",
    "feature_name",
    "feature",
    "feature_family",
    "feature_set",
    "feature_subset_id",
    "task",
    "selector_method",
    "cluster_id",
    "class",
    "label",
)

TEXT_FACT_KEYS = {
    "model_name",
    "model_id",
    "best_feature_family",
    "worst_feature_family",
    "selection_metric",
    "target_units",
    "normalization_method",
    "feature_version",
    "fingerprint_version",
    "novelty_status",
    "predicted_chemical",
    "prediction_passed",
    "qc_passed",
    "passed",
}

METRIC_NAME_TOKENS = (
    "accuracy",
    "auc",
    "balanced",
    "ci95",
    "column_count",
    "count",
    "error",
    "excluded",
    "explained_variance",
    "f1",
    "fold",
    "importance",
    "loss",
    "mae",
    "mape",
    "mean",
    "median",
    "metric",
    "missing",
    "passed",
    "precision",
    "probability",
    "r2",
    "rank",
    "recall",
    "residual",
    "rmse",
    "row_count",
    "runtime",
    "sample",
    "seconds",
    "std",
    "variance",
)


@dataclass(frozen=True)
class SourceContext:
    """Context from one selected_results.csv source reference."""

    analysis_type: str
    source_file: str
    source_run: str
    report_section: str = ""
    scientific_role: str = ""
    include_in_supervisor_report: bool = False


@dataclass
class ParseResult:
    """Parser output for one source file."""

    records: list[EvidenceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_csv_source(path: Path, context: SourceContext) -> ParseResult:
    """Extract evidence from a CSV source without interpreting it."""

    filename = path.name
    if _is_summary_only_csv(filename):
        return _parse_summary_only_csv(path, context)

    rows, fieldnames = _read_csv(path)
    records = [
        _record(
            context,
            metric_name="source_row_count",
            metric_value=len(rows),
            metric_units="count",
            table_reference=context.source_file,
            notes="CSV row count extracted from listed selected source.",
        ),
        _record(
            context,
            metric_name="source_column_count",
            metric_value=len(fieldnames),
            metric_units="count",
            table_reference=context.source_file,
            notes="CSV column count extracted from listed selected source.",
        ),
    ]

    if _looks_like_confusion_matrix(filename, fieldnames):
        records.extend(_parse_matrix_csv(rows, fieldnames, context, "confusion_matrix_count"))
    elif _looks_like_numeric_matrix(fieldnames):
        records.extend(_parse_matrix_csv(rows, fieldnames, context, "matrix_value"))
    else:
        records.extend(_parse_tabular_metrics(rows, fieldnames, context))

    records.extend(_special_csv_records(filename, rows, fieldnames, context))
    return ParseResult(records=records)


def parse_json_source(path: Path, context: SourceContext) -> ParseResult:
    """Extract scalar evidence from a JSON source."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[EvidenceRecord] = []
    _extract_json_payload(payload, context, records, path_parts=())
    records.extend(_special_json_records(path.name, payload, context))
    return ParseResult(records=_deduplicate_records(records))


def parse_text_source(path: Path, context: SourceContext) -> ParseResult:
    """Extract conservative scalar facts from Markdown or TXT source."""

    records: list[EvidenceRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        parsed = _parse_text_fact(line)
        if parsed is None:
            continue
        metric_name, metric_value = parsed
        records.append(
            _record(
                context,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_units=_units_for(metric_name, context.analysis_type),
                confidence="MEDIUM",
                notes=f"Extracted from text line {line_number}; no interpretation applied.",
            )
        )
    return ParseResult(records=records)


def _parse_summary_only_csv(path: Path, context: SourceContext) -> ParseResult:
    filename = path.name
    row_count = 0
    fieldnames: list[str] = []
    unique_values: dict[str, set[str]] = {}
    tracked_columns = ("Chemical", "chemical", "Strain", "strain", "Concentration", "model_name", "feature_family")
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            for column in tracked_columns:
                value = row.get(column)
                if value not in {None, ""}:
                    unique_values.setdefault(column, set()).add(str(value))

    records = [
        _record(
            context,
            metric_name="source_row_count",
            metric_value=row_count,
            metric_units="count",
            table_reference=context.source_file,
            notes="CSV row count extracted from listed selected source.",
        ),
        _record(
            context,
            metric_name="source_column_count",
            metric_value=len(fieldnames),
            metric_units="count",
            table_reference=context.source_file,
            notes="CSV column count extracted from listed selected source.",
        ),
    ]
    if filename in {"feature_dataset.csv", "advanced_feature_dataset.csv"}:
        records.append(
            _record(
                context,
                metric_name="feature_row_count",
                metric_value=row_count,
                metric_units="count",
                table_reference=context.source_file,
            )
        )
        records.append(
            _record(
                context,
                metric_name="feature_column_count",
                metric_value=_estimated_feature_columns(fieldnames),
                metric_units="count",
                table_reference=context.source_file,
            )
        )
    if filename == "fingerprint_dataset.csv":
        records.append(
            _record(
                context,
                metric_name="fingerprint_rows",
                metric_value=row_count,
                metric_units="count",
                table_reference=context.source_file,
            )
        )
        records.append(
            _record(
                context,
                metric_name="fingerprint_feature_count",
                metric_value=_estimated_feature_columns(fieldnames),
                metric_units="count",
                table_reference=context.source_file,
            )
        )
    for column, values in sorted(unique_values.items()):
        records.append(
            _record(
                context,
                metric_name=f"unique_{_normalise_metric_name(column)}_count",
                metric_value=len(values),
                metric_units="count",
                table_reference=context.source_file,
            )
        )
    return ParseResult(records=records)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _parse_matrix_csv(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    context: SourceContext,
    metric_name: str,
) -> list[EvidenceRecord]:
    if not fieldnames:
        return []
    label_column = fieldnames[0]
    records: list[EvidenceRecord] = []
    for row in rows:
        row_label = row.get(label_column, "")
        for column in fieldnames[1:]:
            numeric_value = _coerce_scalar(row.get(column))
            if not isinstance(numeric_value, (int, float, bool)):
                continue
            records.append(
                _record(
                    context,
                    metric_name=metric_name,
                    metric_value=numeric_value,
                    metric_units=_units_for(metric_name, context.analysis_type),
                    table_reference=context.source_file,
                    biological_entity=f"row={row_label}; column={column}".strip("; "),
                )
            )
    return records


def _parse_tabular_metrics(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    context: SourceContext,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for row_index, row in enumerate(rows, start=1):
        model_name = str(row.get("model_name", "") or row.get("model_id", ""))
        entity = _biological_entity(row)
        for column in fieldnames:
            if column in IDENTIFIER_COLUMNS:
                continue
            value = _coerce_scalar(row.get(column))
            if not isinstance(value, (int, float, bool)):
                continue
            metric_name = _normalise_metric_name(column)
            if not _looks_like_metric_name(metric_name):
                continue
            records.append(
                _record(
                    context,
                    metric_name=metric_name,
                    metric_value=value,
                    metric_units=_units_for(metric_name, context.analysis_type),
                    table_reference=context.source_file,
                    biological_entity=entity,
                    model_name=model_name,
                    notes=f"Extracted from CSV row {row_index}.",
                )
            )
    return records


def _special_csv_records(
    filename: str,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    context: SourceContext,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    if filename == "selected_features.csv":
        for flag_column, metric_name in (
            ("default_classification_feature_set", "default_classification_selected_feature_count"),
            ("default_regression_feature_set", "default_regression_selected_feature_count"),
            ("research_feature_set", "research_selected_feature_count"),
        ):
            if flag_column not in fieldnames:
                continue
            count = sum(1 for row in rows if str(row.get(flag_column, "")).casefold() == "true")
            records.append(
                _record(
                    context,
                    metric_name=metric_name,
                    metric_value=count,
                    metric_units="count",
                    table_reference=context.source_file,
                    notes=f"Counted rows flagged {flag_column}=True in selected source.",
                )
            )
        records.append(
            _record(
                context,
                metric_name="selected_feature_count",
                metric_value=len(rows),
                metric_units="count",
                table_reference=context.source_file,
            )
        )

    if filename == "feature_selection_summary.csv":
        for row in rows:
            if str(row.get("recommended_default", "")).casefold() != "true":
                continue
            subset = row.get("feature_subset_id", "") or row.get("selector_method", "")
            task = row.get("task", "")
            records.append(
                _record(
                    context,
                    metric_name="recommended_feature_set",
                    metric_value=subset,
                    table_reference=context.source_file,
                    biological_entity=task,
                    model_name=row.get("model_name", ""),
                    notes="Recommended default row extracted directly from feature_selection_summary.csv.",
                )
            )
            primary_metric = _coerce_scalar(row.get("primary_metric"))
            if isinstance(primary_metric, (int, float, bool)):
                records.append(
                    _record(
                        context,
                        metric_name="retained_performance",
                        metric_value=primary_metric,
                        metric_units="unitless",
                        table_reference=context.source_file,
                        biological_entity=task,
                        model_name=row.get("model_name", ""),
                    )
                )

    if filename == "cluster_assignments.csv":
        clusters = {row.get("cluster_id", "") for row in rows if row.get("cluster_id", "")}
        records.append(
            _record(
                context,
                metric_name="cluster_count",
                metric_value=len(clusters),
                metric_units="count",
                table_reference=context.source_file,
            )
        )

    return records


def _extract_json_payload(
    payload: Any,
    context: SourceContext,
    records: list[EvidenceRecord],
    *,
    path_parts: tuple[str, ...],
) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = (*path_parts, str(key))
            if key in {"warnings", "errors"} and isinstance(value, list):
                records.append(
                    _record(
                        context,
                        metric_name=f"{key[:-1]}_count",
                        metric_value=len(value),
                        metric_units="count",
                        notes=f"Count extracted from JSON field {'.'.join(next_path)}.",
                    )
                )
                continue
            _extract_json_payload(value, context, records, path_parts=next_path)
        return
    if isinstance(payload, list):
        if all(not isinstance(item, (dict, list)) for item in payload):
            metric_name = _normalise_metric_name(".".join(path_parts) + "_count")
            records.append(
                _record(
                    context,
                    metric_name=metric_name,
                    metric_value=len(payload),
                    metric_units="count",
                    notes=f"List length extracted from JSON field {'.'.join(path_parts)}.",
                )
            )
        return

    metric_name = _normalise_metric_name(".".join(path_parts))
    scalar = _coerce_scalar(payload)
    if isinstance(scalar, (int, float, bool)) or _is_meaningful_text_fact(metric_name):
        records.append(
            _record(
                context,
                metric_name=metric_name,
                metric_value=scalar,
                metric_units=_units_for(metric_name, context.analysis_type),
                notes=f"Scalar extracted from JSON field {'.'.join(path_parts)}.",
            )
        )


def _special_json_records(filename: str, payload: Any, context: SourceContext) -> list[EvidenceRecord]:
    if not isinstance(payload, dict):
        return []
    records: list[EvidenceRecord] = []
    if filename in {"best_model_metrics.json", "best_regression_model.json"} and payload.get("model_name") is not None:
        records.append(
            _record(
                context,
                metric_name="best_model",
                metric_value=payload.get("model_name"),
                model_name=str(payload.get("model_name", "")),
                notes=f"Best model extracted directly from {filename}.",
            )
        )
    if filename in {"fingerprint_summary.json"}:
        summary = payload.get("summary", {})
        if isinstance(summary, dict):
            for key in ("fingerprint_rows", "consensus_fingerprint_rows", "fingerprint_qc_passed"):
                if key in summary:
                    records.append(
                        _record(
                            context,
                            metric_name=key,
                            metric_value=_coerce_scalar(summary[key]),
                            metric_units=_units_for(key, context.analysis_type),
                            notes="Fingerprint summary value extracted from summary block.",
                        )
                    )
    return records


def _parse_text_fact(line: str) -> tuple[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(("-", "*")):
        return None
    match = re.match(r"^[-*]\s+([^:]+):\s+(.+?)\s*$", stripped)
    if not match:
        return None
    label = _normalise_metric_name(match.group(1))
    value_text = match.group(2).strip()
    value = _coerce_scalar(value_text)
    if isinstance(value, (int, float, bool)) or _is_meaningful_text_fact(label):
        return label, value
    return None


def _record(
    context: SourceContext,
    *,
    metric_name: str,
    metric_value: Any | None,
    metric_units: str = "",
    figure_reference: str = "",
    table_reference: str = "",
    biological_entity: str = "",
    model_name: str = "",
    confidence: str = "HIGH",
    extraction_status: str = "EXTRACTED",
    notes: str = "",
) -> EvidenceRecord:
    return EvidenceRecord(
        analysis_type=context.analysis_type,
        source_file=context.source_file,
        source_run=context.source_run,
        metric_name=_normalise_metric_name(metric_name),
        metric_value=metric_value,
        metric_units=metric_units,
        figure_reference=figure_reference,
        table_reference=table_reference,
        biological_entity=biological_entity,
        model_name=model_name,
        confidence=confidence,
        extraction_status=extraction_status,
        notes=notes,
    )


def _is_summary_only_csv(filename: str) -> bool:
    if filename in SUMMARY_ONLY_FILENAMES:
        return True
    return bool(re.match(r"BL\d+.*\.csv$", filename))


def _estimated_feature_columns(fieldnames: list[str]) -> int:
    metadata = {
        "Experiment_ID",
        "Measurement_Unit_ID",
        "Fingerprint_ID",
        "Source_File",
        "Strain",
        "Chemical",
        "Concentration",
        "Replicate_ID",
        "Duration",
        "QC_Status",
        "Feature_QC_Flags",
        "feature_qc_flags",
    }
    return sum(1 for name in fieldnames if name not in metadata)


def _looks_like_confusion_matrix(filename: str, fieldnames: list[str]) -> bool:
    return filename == "confusion_matrix.csv" and len(fieldnames) >= 2


def _looks_like_numeric_matrix(fieldnames: list[str]) -> bool:
    if len(fieldnames) < 3:
        return False
    if "explained_variance_ratio" in {field.casefold() for field in fieldnames}:
        return False
    first = fieldnames[0].casefold()
    if any(token in first for token in ("chemical", "feature", "strain", "cluster", "unnamed")):
        return True
    return fieldnames[0] == ""


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if text.casefold() == "true":
        return True
    if text.casefold() == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and not any(char in text.casefold() for char in (".", "e")):
        return int(number)
    return number


def _normalise_metric_name(value: str) -> str:
    cleaned = value.strip().replace("%", "percent")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    return cleaned.strip("_").casefold()


def _looks_like_metric_name(metric_name: str) -> bool:
    return any(token in metric_name for token in METRIC_NAME_TOKENS)


def _is_meaningful_text_fact(metric_name: str) -> bool:
    return metric_name.split("_")[-1] in TEXT_FACT_KEYS or metric_name in TEXT_FACT_KEYS


def _biological_entity(row: dict[str, str]) -> str:
    parts: list[str] = []
    for column in ENTITY_COLUMNS:
        value = row.get(column)
        if value not in {None, ""}:
            parts.append(f"{column}={value}")
    return "; ".join(parts)


def _units_for(metric_name: str, analysis_type: str) -> str:
    metric = metric_name.casefold()
    if any(token in metric for token in ("count", "rows", "columns", "fold_count", "rank")):
        return "count"
    if any(token in metric for token in ("runtime", "seconds", "fit_time", "predict_time")):
        return "seconds"
    if any(token in metric for token in ("rmse", "mae", "concentration")) and "regression" in analysis_type:
        return "ug/mL"
    if any(token in metric for token in ("accuracy", "f1", "precision", "recall", "auc", "r2", "variance_ratio", "mape", "loss", "probability")):
        return "unitless"
    return ""


def _deduplicate_records(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[EvidenceRecord] = []
    for record in records:
        key = (
            record.analysis_type,
            record.source_file,
            record.metric_name,
            json.dumps(record.metric_value, sort_keys=True),
            record.biological_entity,
            record.model_name,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique
