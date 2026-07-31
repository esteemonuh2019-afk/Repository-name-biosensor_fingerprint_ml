"""Data models for the supervisor results package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SelectedSource:
    analysis_type: str
    scientific_role: str
    source_file: str
    selected_run: str
    status: str
    selection_reason: str
    notes: str
    resolved_path: str
    exists: bool
    source_kind: str
    is_primary_selection: bool = False


@dataclass
class SupervisorResultsPackage:
    title: str
    author: str = ""
    supervisor_name: str = ""
    generated_at: str = field(default_factory=utc_now_iso)
    selected_results_file: str = ""
    project_summary: Dict[str, Any] = field(default_factory=dict)
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    quality_control_summary: Dict[str, Any] = field(default_factory=dict)
    fingerprint_summary: Dict[str, Any] = field(default_factory=dict)
    exploratory_results: Dict[str, Any] = field(default_factory=dict)
    classification_results: Dict[str, Any] = field(default_factory=dict)
    regression_results: Dict[str, Any] = field(default_factory=dict)
    feature_engineering_results: Dict[str, Any] = field(default_factory=dict)
    feature_selection_results: Dict[str, Any] = field(default_factory=dict)
    strain_results: Dict[str, Any] = field(default_factory=dict)
    limitations: List[Dict[str, Any]] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    selected_figures: List[Dict[str, Any]] = field(default_factory=list)
    selected_tables: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    package_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def empty_metric(metric_name: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    return {
        "metric_name": metric_name,
        "metric_value": None,
        "metric_units": None,
        "model_name": model_name,
        "source_file": None,
        "source_run": None,
        "status": "MISSING",
        "notes": "Metric unavailable in authoritative selected-model output.",
    }
