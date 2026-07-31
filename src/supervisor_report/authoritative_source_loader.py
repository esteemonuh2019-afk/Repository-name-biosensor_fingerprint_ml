"""Load only files listed in a selected-results inventory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .report_models import SelectedSource


MACHINE_READABLE_SUFFIXES = {".csv", ".json", ".md", ".txt"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REFERENCE_SUFFIXES = {".pdf", ".svg"}


def source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".txt"}:
        return "text"
    if suffix in FIGURE_SUFFIXES:
        return "figure"
    if suffix in REFERENCE_SUFFIXES:
        return "reference"
    return "unsupported"


def resolve_listed_path(project_root: Path, listed_path: str) -> Path:
    raw = Path(listed_path)
    if raw.is_absolute():
        return raw
    outputs_candidate = project_root / "outputs" / raw
    if outputs_candidate.exists():
        return outputs_candidate
    return project_root / raw


def _split_companions(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def load_selected_sources(project_root: Path, selected_results_path: Path) -> List[SelectedSource]:
    project_root = Path(project_root)
    selected_results_path = Path(selected_results_path)
    with selected_results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    sources: List[SelectedSource] = []
    seen = set()
    for row in rows:
        listed_files = [row.get("selected_file", "").strip()] + _split_companions(
            row.get("companion_files", "")
        )
        for index, listed in enumerate(listed_files):
            if not listed:
                continue
            key = (row.get("analysis_type", ""), listed)
            if key in seen:
                continue
            seen.add(key)
            resolved = resolve_listed_path(project_root, listed)
            kind = source_kind(resolved)
            sources.append(
                SelectedSource(
                    analysis_type=row.get("analysis_type", ""),
                    scientific_role=row.get("scientific_role", ""),
                    source_file=listed,
                    selected_run=row.get("selected_run", ""),
                    status=row.get("status", ""),
                    selection_reason=row.get("selection_reason", ""),
                    notes=row.get("notes", ""),
                    resolved_path=str(resolved),
                    exists=resolved.exists(),
                    source_kind=kind,
                    is_primary_selection=index == 0,
                )
            )
    return sources


def filter_sources(
    sources: Iterable[SelectedSource],
    analysis_type: Optional[str] = None,
    suffix: Optional[str] = None,
    contains: Optional[str] = None,
    primary: Optional[bool] = None,
) -> List[SelectedSource]:
    matches: List[SelectedSource] = []
    for source in sources:
        if analysis_type and source.analysis_type != analysis_type:
            continue
        if suffix and not source.source_file.lower().endswith(suffix.lower()):
            continue
        if contains and contains.lower() not in source.source_file.lower():
            continue
        if primary is not None and source.is_primary_selection is not primary:
            continue
        matches.append(source)
    return matches


def first_source(
    sources: Iterable[SelectedSource],
    analysis_type: str,
    filename: str,
    primary: Optional[bool] = None,
) -> Optional[SelectedSource]:
    for source in sources:
        if source.analysis_type == analysis_type and filename.lower() in source.source_file.lower():
            if primary is None or source.is_primary_selection is primary:
                return source
    return None


def read_json(source: Optional[SelectedSource]) -> Dict[str, Any]:
    if not source or not source.exists:
        return {}
    with Path(source.resolved_path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def read_csv_rows(source: Optional[SelectedSource], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not source or not source.exists:
        return []
    with Path(source.resolved_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        return rows[:limit]
    return rows


def read_text(source: Optional[SelectedSource], limit_chars: Optional[int] = None) -> str:
    if not source or not source.exists:
        return ""
    text = Path(source.resolved_path).read_text(encoding="utf-8", errors="replace")
    if limit_chars is not None:
        return text[:limit_chars]
    return text


def rows_to_count(rows: List[Dict[str, Any]]) -> int:
    return len(rows) if rows else 0
