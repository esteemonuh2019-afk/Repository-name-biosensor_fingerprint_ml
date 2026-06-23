"""Traceability matrix generation for V&V requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.validation.requirements import (
    ALL_REQUIREMENT_IDS,
    REQUIREMENT_ACCEPTANCE_CRITERIA,
    REQUIREMENT_DESCRIPTIONS,
    REQUIREMENT_VALIDATION_METHODS,
)


@dataclass(frozen=True)
class TraceabilityRecord:
    """One requirement-to-evidence row in the traceability matrix."""

    requirement_id: str
    description: str
    validation_method: str
    acceptance_criteria: str
    test_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence: str = "Pending"


def default_traceability_records() -> tuple[TraceabilityRecord, ...]:
    """Build traceability records from the V&V plan requirement matrix."""

    return tuple(
        TraceabilityRecord(
            requirement_id=requirement_id,
            description=REQUIREMENT_DESCRIPTIONS[requirement_id],
            validation_method=REQUIREMENT_VALIDATION_METHODS[requirement_id],
            acceptance_criteria=REQUIREMENT_ACCEPTANCE_CRITERIA[requirement_id],
        )
        for requirement_id in ALL_REQUIREMENT_IDS
    )


def generate_traceability_matrix(
    output_path: str | Path,
    records: Sequence[TraceabilityRecord] | None = None,
) -> Path:
    """Write a markdown traceability matrix and return its path."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    matrix_records = tuple(records) if records is not None else default_traceability_records()
    lines = [
        "# Traceability Matrix",
        "",
        "| Requirement ID | Description | Validation Method | Acceptance Criteria | Test IDs | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for record in matrix_records:
        test_ids = ", ".join(record.test_ids) if record.test_ids else "Pending"
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_markdown_table_cell(record.requirement_id),
                    _escape_markdown_table_cell(record.description),
                    _escape_markdown_table_cell(record.validation_method),
                    _escape_markdown_table_cell(record.acceptance_criteria),
                    _escape_markdown_table_cell(test_ids),
                    _escape_markdown_table_cell(record.evidence),
                )
            )
            + " |"
        )

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _escape_markdown_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
