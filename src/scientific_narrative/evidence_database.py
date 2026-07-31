"""Evidence database models and writers for Stage 9B.2A."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


EVIDENCE_FIELDS: tuple[str, ...] = (
    "analysis_type",
    "source_file",
    "source_run",
    "metric_name",
    "metric_value",
    "metric_units",
    "figure_reference",
    "table_reference",
    "biological_entity",
    "model_name",
    "confidence",
    "extraction_status",
    "notes",
)

OUTPUT_FILENAMES: tuple[str, ...] = (
    "scientific_evidence.json",
    "scientific_evidence.csv",
    "scientific_evidence_report.md",
)


@dataclass(frozen=True)
class EvidenceRecord:
    """One extracted scientific evidence item."""

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
    confidence: str = "HIGH"
    extraction_status: str = "EXTRACTED"
    notes: str = ""


@dataclass(frozen=True)
class SourceParseStatus:
    """Status for one listed source file."""

    source_file: str
    resolved_path: str
    analysis_type: str
    source_run: str
    parser_status: str
    evidence_count: int = 0
    notes: str = ""


@dataclass
class EvidenceDatabase:
    """Unified Stage 9B.2A scientific evidence database."""

    records: list[EvidenceRecord] = field(default_factory=list)
    parsed_files: list[SourceParseStatus] = field(default_factory=list)
    unsupported_files: list[SourceParseStatus] = field(default_factory=list)
    unreadable_files: list[SourceParseStatus] = field(default_factory=list)
    missing_evidence: list[EvidenceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def extraction_success(self) -> bool:
        """Return whether all supported selected files were read successfully."""

        return not self.errors and not self.unreadable_files and bool(self.records)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable database."""

        return _json_ready(
            {
                "records": self.records,
                "parsed_files": self.parsed_files,
                "unsupported_files": self.unsupported_files,
                "unreadable_files": self.unreadable_files,
                "missing_evidence": self.missing_evidence,
                "warnings": self.warnings,
                "errors": self.errors,
                "metadata": {
                    **self.metadata,
                    "evidence_record_count": len(self.records),
                    "parsed_file_count": len(self.parsed_files),
                    "unsupported_file_count": len(self.unsupported_files),
                    "unreadable_file_count": len(self.unreadable_files),
                    "missing_evidence_count": len(self.missing_evidence),
                    "extraction_success": self.extraction_success,
                },
            }
        )


def write_evidence_outputs(
    database: EvidenceDatabase,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write Stage 9B.2A evidence JSON, CSV, and Markdown report."""

    target = Path(output_dir)
    _prepare_output_dir(target, overwrite=overwrite)
    json_path = target / "scientific_evidence.json"
    csv_path = target / "scientific_evidence.csv"
    report_path = target / "scientific_evidence_report.md"

    json_path.write_text(
        json.dumps(database.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_records_csv(csv_path, database.records)
    report_path.write_text(render_evidence_report(database), encoding="utf-8")
    return [json_path, csv_path, report_path]


def render_evidence_report(database: EvidenceDatabase) -> str:
    """Render a non-interpretive Markdown extraction summary."""

    metadata = database.to_dict()["metadata"]
    lines = [
        "# Stage 9B.2A Scientific Evidence Extraction Report",
        "",
        "## Extraction Summary",
        "",
        f"- Files parsed: {metadata['parsed_file_count']}",
        f"- Evidence records extracted: {metadata['evidence_record_count']}",
        f"- Missing expected evidence records: {metadata['missing_evidence_count']}",
        f"- Unsupported files: {metadata['unsupported_file_count']}",
        f"- Unreadable files: {metadata['unreadable_file_count']}",
        f"- Extraction success: {metadata['extraction_success']}",
        "",
        "## Evidence By Analysis Type",
        "",
    ]
    by_analysis: dict[str, int] = {}
    for record in database.records:
        by_analysis[record.analysis_type] = by_analysis.get(record.analysis_type, 0) + 1
    if by_analysis:
        for analysis_type, count in sorted(by_analysis.items()):
            lines.append(f"- {analysis_type}: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Evidence Missing", ""])
    if database.missing_evidence:
        for record in database.missing_evidence:
            lines.append(
                f"- {record.analysis_type}: {record.metric_name} "
                f"({record.notes or 'expected metric unavailable'})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Unreadable Files", ""])
    if database.unreadable_files:
        for status in database.unreadable_files:
            lines.append(f"- {status.source_file}: {status.notes}")
    else:
        lines.append("- None")

    lines.extend(["", "## Unsupported Formats", ""])
    if database.unsupported_files:
        for status in database.unsupported_files:
            lines.append(f"- {status.source_file}: {status.notes}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "This report summarises extracted evidence only. It does not interpret results, write a Results section, write a Discussion section, or generate DOCX/PDF outputs.",
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


def _write_records_csv(path: Path, records: Iterable[EvidenceRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({field: _csv_value(row.get(field)) for field in EVIDENCE_FIELDS})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


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
