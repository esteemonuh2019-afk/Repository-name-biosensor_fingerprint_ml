"""Writers for Stage 9B.1 inventory tables, JSON, and Markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from src.results_inventory.inventory_models import (
    DuplicateCandidate,
    InventoryFile,
    MissingResult,
    ObsoleteCandidate,
    ResultsInventory,
    RunInventory,
    SelectedResult,
)


REQUIRED_OUTPUT_FILENAMES: tuple[str, ...] = (
    "output_inventory.csv",
    "output_inventory.json",
    "detected_runs.csv",
    "selected_results.csv",
    "duplicate_candidates.csv",
    "obsolete_candidates.csv",
    "missing_results.csv",
    "project_results_health.json",
    "results_inventory_report.md",
)

SELECTED_RESULTS_FIELDS: tuple[str, ...] = (
    "report_section",
    "analysis_type",
    "selected_file",
    "selected_run",
    "status",
    "selection_reason",
    "companion_files",
    "scientific_role",
    "include_in_supervisor_report",
    "notes",
)

OUTPUT_INVENTORY_FIELDS: tuple[str, ...] = (
    "full_path",
    "relative_path",
    "filename",
    "extension",
    "size_bytes",
    "modified_time",
    "parent_directory",
    "analysis_stage",
    "analysis_type",
    "result_role",
    "run_name",
    "run_version",
    "likely_generator_script",
    "machine_readable",
    "figure",
    "table",
    "report",
    "model_metric",
    "QC_output",
    "include_candidate",
    "selection_reason",
    "status",
    "notes",
    "content_hash",
    "hash_status",
)

DETECTED_RUNS_FIELDS: tuple[str, ...] = (
    "analysis_type",
    "run_name",
    "run_directory",
    "run_version",
    "modified_time",
    "file_count",
    "files_present",
    "expected_files_present",
    "expected_files_missing",
    "likely_completion_status",
    "warnings",
    "selection_score",
    "required_machine_readable_present",
    "required_machine_readable_expected",
    "figure_count",
    "report_count",
    "completion_ratio",
    "selected",
    "selection_reason",
)

DUPLICATE_FIELDS: tuple[str, ...] = (
    "filename",
    "duplicate_count",
    "analysis_types",
    "run_names",
    "paths",
    "newest_modified_time",
    "notes",
)

OBSOLETE_FIELDS: tuple[str, ...] = (
    "candidate_type",
    "path",
    "analysis_type",
    "run_name",
    "status",
    "reason",
    "notes",
)

MISSING_FIELDS: tuple[str, ...] = (
    "report_section",
    "analysis_type",
    "status",
    "required_results",
    "found_results",
    "missing_results",
    "notes",
)


def write_inventory_outputs(
    inventory: ResultsInventory,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write the exact Stage 9B.1 output files."""

    target = Path(output_dir)
    _prepare_output_dir(target, overwrite=overwrite)
    paths = {
        "output_inventory.csv": target / "output_inventory.csv",
        "output_inventory.json": target / "output_inventory.json",
        "detected_runs.csv": target / "detected_runs.csv",
        "selected_results.csv": target / "selected_results.csv",
        "duplicate_candidates.csv": target / "duplicate_candidates.csv",
        "obsolete_candidates.csv": target / "obsolete_candidates.csv",
        "missing_results.csv": target / "missing_results.csv",
        "project_results_health.json": target / "project_results_health.json",
        "results_inventory_report.md": target / "results_inventory_report.md",
    }

    _write_csv(
        paths["output_inventory.csv"],
        inventory.classified_files,
        _inventory_file_row,
        fields=OUTPUT_INVENTORY_FIELDS,
    )
    _write_json(paths["output_inventory.json"], inventory.to_dict())
    _write_csv(
        paths["detected_runs.csv"],
        inventory.detected_runs,
        _run_row,
        fields=DETECTED_RUNS_FIELDS,
    )
    _write_csv(
        paths["selected_results.csv"],
        inventory.selected_results,
        _selected_result_row,
        fields=SELECTED_RESULTS_FIELDS,
    )
    _write_csv(
        paths["duplicate_candidates.csv"],
        inventory.duplicate_candidates,
        _duplicate_row,
        fields=DUPLICATE_FIELDS,
    )
    _write_csv(
        paths["obsolete_candidates.csv"],
        inventory.obsolete_candidates,
        _obsolete_row,
        fields=OBSOLETE_FIELDS,
    )
    _write_csv(
        paths["missing_results.csv"],
        inventory.missing_required_results,
        _missing_row,
        fields=MISSING_FIELDS,
    )
    _write_json(paths["project_results_health.json"], inventory.project_health)
    paths["results_inventory_report.md"].write_text(render_inventory_report(inventory), encoding="utf-8")
    return [paths[name] for name in REQUIRED_OUTPUT_FILENAMES]


def render_inventory_report(inventory: ResultsInventory) -> str:
    """Render the Markdown inventory report."""

    metadata = inventory.scan_metadata
    categories = metadata.get("analysis_categories_found", [])
    total_size = int(metadata.get("total_size_bytes", 0))
    lines = [
        "# Stage 9B.1 Results Inventory Report",
        "",
        "## Summary",
        "",
        f"- Total output files: {metadata.get('total_files', len(inventory.classified_files))}",
        f"- Total size: {_format_bytes(total_size)} ({total_size} bytes)",
        f"- Analysis categories found: {', '.join(categories) if categories else 'None'}",
        f"- Runs detected: {len(inventory.detected_runs)}",
        f"- Duplicate filename candidates: {len(inventory.duplicate_candidates)}",
        f"- Obsolete or review candidates: {len(inventory.obsolete_candidates)}",
        "",
        "## Preferred Runs",
        "",
    ]
    if inventory.selected_runs:
        for analysis_type, run in sorted(inventory.selected_runs.items()):
            lines.append(
                f"- {analysis_type}: {run.run_name} ({run.run_directory}) - {run.selection_reason}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Report Components", ""])
    missing_or_partial = [
        item
        for item in inventory.missing_required_results
        if item.status not in {"FOUND", "NOT YET APPLICABLE"}
    ]
    if missing_or_partial:
        for item in missing_or_partial:
            missing = ", ".join(item.missing_results) if item.missing_results else "None"
            lines.append(f"- {item.report_section}: {item.status}; missing: {missing}")
    else:
        lines.append("- No missing core components were detected.")

    lines.extend(["", "## Blind Validation", ""])
    blind = next(
        (
            item
            for item in inventory.missing_required_results
            if item.report_section == "Blind validation status"
        ),
        None,
    )
    if blind:
        lines.append(f"- {blind.status}: {blind.notes}")
    else:
        lines.append("- Not assessed.")

    lines.extend(["", "## Duplicate Or Obsolete Runs", ""])
    if inventory.duplicate_candidates:
        lines.append(f"- Filename duplicate groups: {len(inventory.duplicate_candidates)}")
    else:
        lines.append("- Filename duplicate groups: 0")
    if inventory.obsolete_candidates:
        for candidate in inventory.obsolete_candidates[:25]:
            lines.append(
                f"- {candidate.candidate_type}: {candidate.path} ({candidate.status}) - {candidate.reason}"
            )
        if len(inventory.obsolete_candidates) > 25:
            lines.append(f"- Additional obsolete/review candidates: {len(inventory.obsolete_candidates) - 25}")
    else:
        lines.append("- Obsolete/review candidates: 0")

    lines.extend(["", "## Large-File Warnings", ""])
    large_warnings = [
        warning
        for warning in inventory.warnings
        if "was not hashed" in warning or "Large generated output" in warning
    ]
    if large_warnings:
        lines.extend(f"- {warning}" for warning in large_warnings)
    else:
        lines.append("- None")

    lines.extend(["", "## Selected Files For Future Supervisor Report", ""])
    included = [
        row
        for row in inventory.selected_results
        if row.include_in_supervisor_report
    ]
    if included:
        for row in included:
            lines.append(
                f"- {row.report_section}: {row.selected_file} ({row.status})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Unresolved Ambiguities", ""])
    ambiguity_warnings = [
        warning
        for warning in inventory.warnings
        if "Missing supervisor-report section" in warning
        or "Partial supervisor-report section" in warning
    ]
    if ambiguity_warnings:
        lines.extend(f"- {warning}" for warning in ambiguity_warnings)
    else:
        lines.append("- None beyond upstream QC limitations already documented in stage reports.")

    lines.extend(["", "## Recommendation", ""])
    lines.append(f"- {inventory.project_health.get('report_generation_recommendation', 'Not assessed.')}")
    lines.append(f"- Inventory passed: {inventory.inventory_passed}")
    return "\n".join(lines).rstrip() + "\n"


def _prepare_output_dir(target: Path, *, overwrite: bool) -> None:
    if target.exists() and target.is_file():
        raise FileExistsError(f"Output path is a file: {target}")
    if _looks_like_project_outputs_dir(target):
        raise ValueError(
            "Refusing to write inventory files directly into a broad outputs directory; "
            "choose a dedicated output directory such as outputs/results_inventory."
        )
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is not empty: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _looks_like_project_outputs_dir(path: Path) -> bool:
    return path.name.casefold() == "outputs"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Iterable[Any],
    row_mapper,
    *,
    fields: tuple[str, ...] | None = None,
) -> None:
    mapped_rows = [row_mapper(row) for row in rows]
    fieldnames = list(fields or _fields_from_rows(mapped_rows))
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapped_rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _fields_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _csv_value(value: Any) -> str | int | float | bool:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _inventory_file_row(record: InventoryFile) -> dict[str, Any]:
    return {
        "full_path": record.full_path,
        "relative_path": record.relative_path,
        "filename": record.filename,
        "extension": record.extension,
        "size_bytes": record.size_bytes,
        "modified_time": record.modified_time,
        "parent_directory": record.parent_directory,
        "analysis_stage": record.analysis_stage,
        "analysis_type": record.analysis_type,
        "result_role": record.result_role,
        "run_name": record.run_name,
        "run_version": record.run_version,
        "likely_generator_script": record.likely_generator_script,
        "machine_readable": record.machine_readable,
        "figure": record.figure,
        "table": record.table,
        "report": record.report,
        "model_metric": record.model_metric,
        "QC_output": record.QC_output,
        "include_candidate": record.include_candidate,
        "selection_reason": record.selection_reason,
        "status": record.status,
        "notes": record.notes,
        "content_hash": record.content_hash,
        "hash_status": record.hash_status,
    }


def _run_row(run: RunInventory) -> dict[str, Any]:
    return {
        "analysis_type": run.analysis_type,
        "run_name": run.run_name,
        "run_directory": run.run_directory,
        "run_version": run.run_version,
        "modified_time": run.modified_time,
        "file_count": run.file_count,
        "files_present": run.files_present,
        "expected_files_present": run.expected_files_present,
        "expected_files_missing": run.expected_files_missing,
        "likely_completion_status": run.likely_completion_status,
        "warnings": run.warnings,
        "selection_score": run.selection_score,
        "required_machine_readable_present": run.required_machine_readable_present,
        "required_machine_readable_expected": run.required_machine_readable_expected,
        "figure_count": run.figure_count,
        "report_count": run.report_count,
        "completion_ratio": run.completion_ratio,
        "selected": run.selected,
        "selection_reason": run.selection_reason,
    }


def _selected_result_row(row: SelectedResult) -> dict[str, Any]:
    return {
        "report_section": row.report_section,
        "analysis_type": row.analysis_type,
        "selected_file": row.selected_file,
        "selected_run": row.selected_run,
        "status": row.status,
        "selection_reason": row.selection_reason,
        "companion_files": row.companion_files,
        "scientific_role": row.scientific_role,
        "include_in_supervisor_report": row.include_in_supervisor_report,
        "notes": row.notes,
    }


def _duplicate_row(row: DuplicateCandidate) -> dict[str, Any]:
    return {
        "filename": row.filename,
        "duplicate_count": row.duplicate_count,
        "analysis_types": row.analysis_types,
        "run_names": row.run_names,
        "paths": row.paths,
        "newest_modified_time": row.newest_modified_time,
        "notes": row.notes,
    }


def _obsolete_row(row: ObsoleteCandidate) -> dict[str, Any]:
    return {
        "candidate_type": row.candidate_type,
        "path": row.path,
        "analysis_type": row.analysis_type,
        "run_name": row.run_name,
        "status": row.status,
        "reason": row.reason,
        "notes": row.notes,
    }


def _missing_row(row: MissingResult) -> dict[str, Any]:
    return {
        "report_section": row.report_section,
        "analysis_type": row.analysis_type,
        "status": row.status,
        "required_results": row.required_results,
        "found_results": row.found_results,
        "missing_results": row.missing_results,
        "notes": row.notes,
    }


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "bytes" else f"{int(value)} bytes"
        value /= 1024.0
    return f"{size} bytes"
