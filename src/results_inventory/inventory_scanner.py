"""Recursive output scanner and Stage 9B.1 inventory orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from src.results_inventory.completeness_checker import assess_completeness
from src.results_inventory.inventory_models import InventoryFile, ResultsInventory, ScanResult
from src.results_inventory.result_classifier import classify_files
from src.results_inventory.run_selector import (
    detect_runs,
    identify_duplicate_candidates,
    identify_obsolete_candidates,
    select_preferred_runs,
)


DEFAULT_MAXIMUM_HASH_SIZE_MB = 100.0


def build_results_inventory(
    project_root: str | Path = ".",
    *,
    outputs_dir: str | Path = "outputs",
    output_dir: str | Path | None = "outputs/results_inventory",
    include_large_files: bool = False,
    minimum_file_size: int = 0,
    maximum_hash_size_mb: float = DEFAULT_MAXIMUM_HASH_SIZE_MB,
) -> ResultsInventory:
    """Build a complete read-only inventory for the project's generated outputs."""

    scan = scan_output_files(
        project_root=project_root,
        outputs_dir=outputs_dir,
        output_dir=output_dir,
        include_large_files=include_large_files,
        minimum_file_size=minimum_file_size,
        maximum_hash_size_mb=maximum_hash_size_mb,
    )
    classified = classify_files(scan.all_files)
    detected_runs = detect_runs(classified, empty_directories=scan.empty_directories)
    selected_runs = select_preferred_runs(detected_runs)
    duplicate_candidates = identify_duplicate_candidates(classified)
    obsolete_candidates = identify_obsolete_candidates(
        detected_runs,
        selected_runs,
        classified,
        large_file_threshold_bytes=int(maximum_hash_size_mb * 1024 * 1024),
    )
    completeness = assess_completeness(
        project_root=project_root,
        files=classified,
        selected_runs=selected_runs,
    )

    warnings = [
        *scan.warnings,
        *completeness.get("warnings", []),
    ]
    large_file_warnings = [
        f"{record.relative_path} is {record.size_bytes} bytes and was not hashed."
        for record in classified
        if record.hash_status == "skipped_large_file"
    ]
    warnings.extend(large_file_warnings)

    errors = [*scan.errors, *completeness.get("errors", [])]
    inventory_passed = not errors and bool(classified)
    metadata = {
        **scan.metadata,
        "analysis_categories_found": sorted(
            {record.analysis_type for record in classified if record.analysis_type != "unknown"}
        ),
        "scientific_roles_found": sorted(
            {record.result_role for record in classified if record.result_role != "unknown"}
        ),
        "detected_run_count": len(detected_runs),
        "selected_run_count": len(selected_runs),
        "duplicate_candidate_count": len(duplicate_candidates),
        "obsolete_candidate_count": len(obsolete_candidates),
        "large_file_warning_count": len(large_file_warnings),
        "report_generation_can_proceed": completeness["project_health"].get(
            "report_generation_can_proceed",
            False,
        ),
    }

    return ResultsInventory(
        all_files=scan.all_files,
        classified_files=classified,
        detected_runs=detected_runs,
        selected_runs=selected_runs,
        duplicate_candidates=duplicate_candidates,
        obsolete_candidates=obsolete_candidates,
        missing_required_results=completeness["missing_required_results"],
        selected_results=completeness["selected_results"],
        project_health=completeness["project_health"],
        warnings=warnings,
        errors=errors,
        inventory_passed=inventory_passed,
        scan_metadata=metadata,
    )


def scan_output_files(
    project_root: str | Path = ".",
    *,
    outputs_dir: str | Path = "outputs",
    output_dir: str | Path | None = "outputs/results_inventory",
    include_large_files: bool = False,
    minimum_file_size: int = 0,
    maximum_hash_size_mb: float = DEFAULT_MAXIMUM_HASH_SIZE_MB,
) -> ScanResult:
    """Recursively scan generated output files without modifying them."""

    started_at = _utc_now()
    root = Path(project_root).resolve()
    outputs_path = _resolve_under_project(root, outputs_dir)
    inventory_output_path = (
        _resolve_under_project(root, output_dir) if output_dir is not None else None
    )
    warnings: list[str] = []
    errors: list[str] = []

    if minimum_file_size < 0:
        raise ValueError("--minimum-file-size must be zero or greater.")
    if maximum_hash_size_mb <= 0:
        raise ValueError("--maximum-hash-size-mb must be greater than zero.")

    if not outputs_path.exists():
        errors.append(f"Outputs directory does not exist: {outputs_path}")
        return ScanResult(
            all_files=[],
            empty_directories=[],
            warnings=warnings,
            errors=errors,
            metadata=_scan_metadata(
                root=root,
                outputs_path=outputs_path,
                output_path=inventory_output_path,
                started_at=started_at,
                finished_at=_utc_now(),
                total_files=0,
                total_size=0,
                minimum_file_size=minimum_file_size,
                maximum_hash_size_mb=maximum_hash_size_mb,
                include_large_files=include_large_files,
                excluded_output_dir=False,
            ),
        )
    if not outputs_path.is_dir():
        errors.append(f"Outputs path is not a directory: {outputs_path}")
        return ScanResult(
            all_files=[],
            empty_directories=[],
            warnings=warnings,
            errors=errors,
            metadata=_scan_metadata(
                root=root,
                outputs_path=outputs_path,
                output_path=inventory_output_path,
                started_at=started_at,
                finished_at=_utc_now(),
                total_files=0,
                total_size=0,
                minimum_file_size=minimum_file_size,
                maximum_hash_size_mb=maximum_hash_size_mb,
                include_large_files=include_large_files,
                excluded_output_dir=False,
            ),
        )

    excluded_output_dir = (
        inventory_output_path is not None
        and inventory_output_path.exists()
        and _is_relative_to(inventory_output_path, outputs_path)
    )
    max_hash_size_bytes = int(maximum_hash_size_mb * 1024 * 1024)
    records: list[InventoryFile] = []

    for path in sorted(outputs_path.rglob("*"), key=lambda item: _stable_path(item, outputs_path)):
        if inventory_output_path is not None and _is_relative_to(path, inventory_output_path):
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        size_bytes = int(stat.st_size)
        if size_bytes < minimum_file_size:
            continue
        content_hash, hash_status = _hash_if_allowed(
            path,
            size_bytes=size_bytes,
            max_hash_size_bytes=max_hash_size_bytes,
            include_large_files=include_large_files,
        )
        relative_path = _stable_path(path, outputs_path)
        parent = _parent_directory(relative_path)
        records.append(
            InventoryFile(
                full_path=str(path.resolve()),
                relative_path=relative_path,
                filename=path.name,
                extension=path.suffix.casefold(),
                size_bytes=size_bytes,
                modified_time=_timestamp(stat.st_mtime),
                parent_directory=parent,
                content_hash=content_hash,
                hash_status=hash_status,
            )
        )

    empty_directories = _empty_directories(outputs_path, inventory_output_path)
    total_size = sum(record.size_bytes for record in records)
    if not records:
        errors.append(
            f"No generated result files found under {outputs_path}"
            if minimum_file_size == 0
            else f"No generated result files under {outputs_path} matched --minimum-file-size={minimum_file_size}."
        )
    if excluded_output_dir:
        warnings.append(
            f"Excluded existing inventory output directory from scan: {inventory_output_path}"
        )

    finished_at = _utc_now()
    return ScanResult(
        all_files=records,
        empty_directories=empty_directories,
        warnings=warnings,
        errors=errors,
        metadata=_scan_metadata(
            root=root,
            outputs_path=outputs_path,
            output_path=inventory_output_path,
            started_at=started_at,
            finished_at=finished_at,
            total_files=len(records),
            total_size=total_size,
            minimum_file_size=minimum_file_size,
            maximum_hash_size_mb=maximum_hash_size_mb,
            include_large_files=include_large_files,
            excluded_output_dir=excluded_output_dir,
        ),
    )


def _resolve_under_project(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=0).isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parent_directory(relative_path: str) -> str:
    parent = PurePosixPath(relative_path).parent.as_posix()
    return "" if parent == "." else parent


def _hash_if_allowed(
    path: Path,
    *,
    size_bytes: int,
    max_hash_size_bytes: int,
    include_large_files: bool,
) -> tuple[str, str]:
    if size_bytes > max_hash_size_bytes and not include_large_files:
        return "", "skipped_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "sha256"


def _empty_directories(outputs_path: Path, excluded_output_dir: Path | None) -> list[str]:
    directories: list[str] = []
    for directory in sorted(outputs_path.rglob("*"), key=lambda item: _stable_path(item, outputs_path)):
        if not directory.is_dir():
            continue
        if excluded_output_dir is not None and _is_relative_to(directory, excluded_output_dir):
            continue
        has_file = any(child.is_file() for child in directory.rglob("*"))
        if not has_file:
            directories.append(_stable_path(directory, outputs_path))
    return directories


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _scan_metadata(
    *,
    root: Path,
    outputs_path: Path,
    output_path: Path | None,
    started_at: str,
    finished_at: str,
    total_files: int,
    total_size: int,
    minimum_file_size: int,
    maximum_hash_size_mb: float,
    include_large_files: bool,
    excluded_output_dir: bool,
) -> dict[str, Any]:
    return {
        "project_root": str(root),
        "outputs_dir": str(outputs_path),
        "inventory_output_dir": str(output_path) if output_path is not None else "",
        "scan_started_at": started_at,
        "scan_finished_at": finished_at,
        "total_files": total_files,
        "total_size_bytes": total_size,
        "minimum_file_size": int(minimum_file_size),
        "maximum_hash_size_mb": float(maximum_hash_size_mb),
        "include_large_files": bool(include_large_files),
        "excluded_existing_inventory_output_dir": bool(excluded_output_dir),
    }
