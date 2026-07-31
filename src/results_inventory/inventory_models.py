"""Structured models for Stage 9B.1 results inventory outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


COMPLETION_COMPLETE = "COMPLETE"
COMPLETION_PARTIAL = "PARTIAL"
COMPLETION_MISSING = "MISSING"
COMPLETION_EMPTY = "EMPTY"
COMPLETION_UNKNOWN = "UNKNOWN"
COMPLETION_DIAGNOSTIC = "DIAGNOSTIC"

SECTION_FOUND = "FOUND"
SECTION_PARTIAL = "PARTIAL"
SECTION_MISSING = "MISSING"
SECTION_NOT_APPLICABLE = "NOT YET APPLICABLE"

HEALTH_COMPLETE = "COMPLETE"
HEALTH_PARTIAL = "PARTIAL"
HEALTH_MISSING = "MISSING"
HEALTH_NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class InventoryFile:
    """One generated file discovered under the configured outputs tree."""

    full_path: str
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    modified_time: str
    parent_directory: str
    analysis_stage: str = "unknown"
    analysis_type: str = "unknown"
    result_role: str = "unknown"
    run_name: str = "unknown"
    run_version: str = ""
    likely_generator_script: str = ""
    machine_readable: bool = False
    figure: bool = False
    table: bool = False
    report: bool = False
    model_metric: bool = False
    QC_output: bool = False
    include_candidate: bool = False
    selection_reason: str = ""
    status: str = "discovered"
    notes: str = ""
    content_hash: str = ""
    hash_status: str = "not_hashed"


@dataclass
class RunInventory:
    """A detected output run or run-like output directory."""

    analysis_type: str
    run_name: str
    run_directory: str
    run_version: str
    modified_time: str
    files_present: list[str] = field(default_factory=list)
    expected_files_present: list[str] = field(default_factory=list)
    expected_files_missing: list[str] = field(default_factory=list)
    likely_completion_status: str = COMPLETION_UNKNOWN
    warnings: list[str] = field(default_factory=list)
    selection_score: float = 0.0
    file_count: int = 0
    required_machine_readable_present: int = 0
    required_machine_readable_expected: int = 0
    figure_count: int = 0
    report_count: int = 0
    completion_ratio: float = 0.0
    selected: bool = False
    selection_reason: str = ""


@dataclass(frozen=True)
class DuplicateCandidate:
    """Filename duplicate found across distinct runs or output folders."""

    filename: str
    duplicate_count: int
    analysis_types: list[str]
    run_names: list[str]
    paths: list[str]
    newest_modified_time: str
    notes: str = ""


@dataclass(frozen=True)
class ObsoleteCandidate:
    """Output directory or file that appears superseded, partial, or non-reportable."""

    candidate_type: str
    path: str
    analysis_type: str
    run_name: str
    status: str
    reason: str
    notes: str = ""


@dataclass(frozen=True)
class MissingResult:
    """Supervisor-report section completeness result."""

    report_section: str
    analysis_type: str
    status: str
    required_results: list[str]
    found_results: list[str]
    missing_results: list[str]
    notes: str = ""


@dataclass(frozen=True)
class SelectedResult:
    """Selected artifact row for a future supervisor report."""

    report_section: str
    analysis_type: str
    selected_file: str
    selected_run: str
    status: str
    selection_reason: str
    companion_files: list[str]
    scientific_role: str
    include_in_supervisor_report: bool
    notes: str = ""


@dataclass(frozen=True)
class ScanResult:
    """Raw recursive scan result before scientific classification."""

    all_files: list[InventoryFile]
    empty_directories: list[str]
    warnings: list[str]
    errors: list[str]
    metadata: dict[str, Any]


@dataclass
class ResultsInventory:
    """Complete Stage 9B.1 inventory object."""

    all_files: list[InventoryFile] = field(default_factory=list)
    classified_files: list[InventoryFile] = field(default_factory=list)
    detected_runs: list[RunInventory] = field(default_factory=list)
    selected_runs: dict[str, RunInventory] = field(default_factory=dict)
    duplicate_candidates: list[DuplicateCandidate] = field(default_factory=list)
    obsolete_candidates: list[ObsoleteCandidate] = field(default_factory=list)
    missing_required_results: list[MissingResult] = field(default_factory=list)
    selected_results: list[SelectedResult] = field(default_factory=list)
    project_health: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    inventory_passed: bool = False
    scan_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return _json_ready(self)


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
