"""Public abstract interface for future Observation Engine implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .models import Observation, ValidationIssue


@dataclass(frozen=True)
class ObservationRunResult:
    observations: tuple[Observation, ...] = field(default_factory=tuple)
    validation_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationEngine(ABC):
    """Stable contract for factual observation builders.

    Concrete implementations may read completed analysis outputs, but this
    interface deliberately contains no file-reading behavior.
    """

    @abstractmethod
    def load_sources(self) -> Mapping[str, Any]:
        """Return validated source payloads keyed by source identifier."""

    @abstractmethod
    def build_observations(self, sources: Mapping[str, Any]) -> tuple[Observation, ...]:
        """Create factual observations from already loaded source payloads."""

    @abstractmethod
    def validate_observations(
        self, observations: tuple[Observation, ...]
    ) -> tuple[ValidationIssue, ...]:
        """Return structured validation issues for candidate observations."""

    @abstractmethod
    def write_outputs(self, observations: tuple[Observation, ...]) -> tuple[Path, ...]:
        """Write observations using a future implementation-specific output policy."""

    @abstractmethod
    def run(self) -> ObservationRunResult:
        """Execute the load, build, validate, and write lifecycle."""
