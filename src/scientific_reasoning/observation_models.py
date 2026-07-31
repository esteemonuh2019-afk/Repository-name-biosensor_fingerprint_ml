"""Data models for factual scientific observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


OBSERVATION_FIELDS: tuple[str, ...] = (
    "id",
    "category",
    "title",
    "statement",
    "analysis_stage",
    "supporting_files",
    "supporting_metrics",
    "confidence",
    "notes",
    "status",
)


@dataclass
class Observation:
    """One factual observation with traceable evidence."""

    id: str
    category: str
    title: str
    statement: str
    analysis_stage: str
    supporting_files: list[str] = field(default_factory=list)
    supporting_metrics: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "High"
    notes: str = ""
    status: str = "COMPLETE"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class ObservationDatabase:
    """Collection of factual observations and engine status metadata."""

    observations: list[Observation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def complete_count(self) -> int:
        return sum(1 for observation in self.observations if observation.status == "COMPLETE")

    @property
    def incomplete_count(self) -> int:
        return sum(1 for observation in self.observations if observation.status == "INCOMPLETE")

    @property
    def extraction_success(self) -> bool:
        return bool(self.observations) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "observations": self.observations,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": {
                **self.metadata,
                "observation_count": len(self.observations),
                "complete_observation_count": self.complete_count,
                "incomplete_observation_count": self.incomplete_count,
                "extraction_success": self.extraction_success,
            },
        }
        return _json_ready(payload)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
