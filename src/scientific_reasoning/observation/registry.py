"""Minimal deterministic registry for Observation contract objects."""

from __future__ import annotations

from .enums import ObservationCategory
from .models import Observation, ValidationIssue
from .validators import validate_observation, validate_observations


class DuplicateObservationError(ValueError):
    """Raised when an observation ID is registered more than once."""


class ObservationRegistry:
    """Register observations and expose deterministic contract behavior."""

    def __init__(self) -> None:
        self._observations: dict[str, Observation] = {}
        self._validation_issues: list[ValidationIssue] = []

    def register(self, observation: Observation) -> None:
        issues = list(validate_observation(observation))
        if issues:
            self._validation_issues.extend(issues)
            raise ValueError(f"Observation failed validation: {observation.observation_id}")
        if observation.observation_id in self._observations:
            issue = ValidationIssue(
                code="DUPLICATE_OBSERVATION_ID",
                severity="ERROR",
                message=f"Duplicate observation ID: {observation.observation_id}",
                observation_id=observation.observation_id,
                field="observation_id",
                source_file=None,
            )
            self._validation_issues.append(issue)
            raise DuplicateObservationError(issue.message)
        self._observations[observation.observation_id] = observation

    def register_many(self, observations: tuple[Observation, ...]) -> None:
        for observation in observations:
            self.register(observation)

    def get(self, observation_id: str) -> Observation | None:
        return self._observations.get(observation_id)

    def by_category(self, category: ObservationCategory | str) -> tuple[Observation, ...]:
        category_value = ObservationCategory(category)
        return tuple(
            observation
            for observation in self.ordered()
            if observation.category == category_value
        )

    def ordered(self) -> tuple[Observation, ...]:
        return tuple(self._observations[key] for key in sorted(self._observations))

    def to_records(self) -> list[dict]:
        return [observation.to_record() for observation in self.ordered()]

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues = tuple(self._validation_issues) + validate_observations(self.ordered())
        return issues

    @property
    def validation_issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(self._validation_issues)
