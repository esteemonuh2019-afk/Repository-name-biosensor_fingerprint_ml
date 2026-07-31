"""Minimal deterministic registry for Interpretation contract objects."""

from __future__ import annotations

from .enums import EvidenceDirection, InterpretationCategory
from .models import Interpretation, InterpretationValidationIssue
from .validators import validate_interpretation, validate_interpretations


class DuplicateInterpretationError(ValueError):
    """Raised when an interpretation ID is registered more than once."""


class InterpretationRegistry:
    """Register interpretations and expose deterministic contract behavior."""

    def __init__(self) -> None:
        self._interpretations: dict[str, Interpretation] = {}
        self._validation_issues: list[InterpretationValidationIssue] = []

    def register(self, interpretation: Interpretation) -> None:
        issues = list(validate_interpretation(interpretation))
        if issues:
            self._validation_issues.extend(issues)
            raise ValueError(f"Interpretation failed validation: {interpretation.interpretation_id}")
        if interpretation.interpretation_id in self._interpretations:
            issue = InterpretationValidationIssue(
                code="DUPLICATE_INTERPRETATION_ID",
                severity="CRITICAL",
                message=f"Duplicate interpretation ID: {interpretation.interpretation_id}",
                interpretation_id=interpretation.interpretation_id,
                field="interpretation_id",
                observation_id=None,
                rule_id=None,
            )
            self._validation_issues.append(issue)
            raise DuplicateInterpretationError(issue.message)
        self._interpretations[interpretation.interpretation_id] = interpretation

    def register_many(self, interpretations: tuple[Interpretation, ...]) -> None:
        for interpretation in interpretations:
            self.register(interpretation)

    def get(self, interpretation_id: str) -> Interpretation | None:
        return self._interpretations.get(interpretation_id)

    def by_category(self, category: InterpretationCategory | str) -> tuple[Interpretation, ...]:
        category_value = InterpretationCategory(category)
        return tuple(
            interpretation
            for interpretation in self.ordered()
            if interpretation.category == category_value
        )

    def by_supporting_observation_id(self, observation_id: str) -> tuple[Interpretation, ...]:
        matches = []
        for interpretation in self.ordered():
            supporting_ids = set(interpretation.supporting_observation_ids)
            supporting_ids.update(
                link.observation_id
                for link in interpretation.evidence_summary
                if link.direction == EvidenceDirection.SUPPORTING
            )
            if observation_id in supporting_ids:
                matches.append(interpretation)
        return tuple(matches)

    def ordered(self) -> tuple[Interpretation, ...]:
        return tuple(self._interpretations[key] for key in sorted(self._interpretations))

    def to_records(self) -> list[dict]:
        return [interpretation.to_record() for interpretation in self.ordered()]

    def validate(self) -> tuple[InterpretationValidationIssue, ...]:
        issues = tuple(self._validation_issues) + validate_interpretations(self.ordered())
        return issues

    @property
    def validation_issues(self) -> tuple[InterpretationValidationIssue, ...]:
        return tuple(self._validation_issues)
