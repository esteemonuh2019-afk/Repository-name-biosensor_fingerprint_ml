"""Public abstract interface for future Interpretation Engine implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.scientific_reasoning.observation import Observation

from .models import Interpretation, InterpretationValidationIssue


@dataclass(frozen=True)
class InterpretationRunResult:
    interpretations: tuple[Interpretation, ...] = field(default_factory=tuple)
    validation_issues: tuple[InterpretationValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InterpretationEngine(ABC):
    """Stable contract for Level 3 scientific interpretation builders.

    Concrete implementations must consume validated Observation objects. This
    interface deliberately contains no raw output parsing or interpretation
    production behavior.
    """

    @abstractmethod
    def load_observations(self) -> tuple[Observation, ...]:
        """Return validated Observation objects for downstream interpretation."""

    @abstractmethod
    def validate_input_observations(
        self, observations: tuple[Observation, ...]
    ) -> tuple[InterpretationValidationIssue, ...]:
        """Return structured validation issues for source observations."""

    @abstractmethod
    def build_interpretations(
        self, observations: tuple[Observation, ...]
    ) -> tuple[Interpretation, ...]:
        """Create candidate interpretations from Observation objects."""

    @abstractmethod
    def validate_interpretations(
        self,
        interpretations: tuple[Interpretation, ...],
        observations: tuple[Observation, ...],
    ) -> tuple[InterpretationValidationIssue, ...]:
        """Return structured validation issues for candidate interpretations."""

    @abstractmethod
    def write_outputs(self, interpretations: tuple[Interpretation, ...]) -> tuple[Path, ...]:
        """Write interpretation outputs using an implementation-specific policy."""

    @abstractmethod
    def run(self) -> InterpretationRunResult:
        """Execute the load, validate, build, validate, and write lifecycle."""
