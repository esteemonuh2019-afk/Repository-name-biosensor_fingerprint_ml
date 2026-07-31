"""Provenance helpers for quantitative report claims."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .report_models import SelectedSource


class ProvenanceTracker:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        record_type: str,
        section: str,
        claim: str,
        source: Optional[SelectedSource],
        metric_name: Optional[str] = None,
        metric_value: Any = None,
        metric_units: Optional[str] = None,
        model_name: Optional[str] = None,
        table_reference: Optional[str] = None,
        figure_reference: Optional[str] = None,
        status: str = "SUPPORTED",
        notes: str = "",
    ) -> Dict[str, Any]:
        record = {
            "provenance_id": f"P{len(self.records) + 1:04d}",
            "record_type": record_type,
            "section": section,
            "claim": claim,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_units": metric_units,
            "model_name": model_name,
            "source_file": source.source_file if source else None,
            "source_run": source.selected_run if source else None,
            "table_reference": table_reference,
            "figure_reference": figure_reference,
            "status": status,
            "notes": notes,
        }
        self.records.append(record)
        return record


def write_provenance_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "provenance_id",
        "record_type",
        "section",
        "claim",
        "metric_name",
        "metric_value",
        "metric_units",
        "model_name",
        "source_file",
        "source_run",
        "table_reference",
        "figure_reference",
        "status",
        "notes",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})
