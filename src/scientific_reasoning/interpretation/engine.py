"""Concrete BSIP Scientific Interpretation Engine implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.scientific_reasoning.observation import Observation, validate_observations

from .interfaces import InterpretationEngine, InterpretationRunResult
from .models import Interpretation, InterpretationValidationIssue, utc_now_iso
from .rules import build_interpretations
from .source_loader import ObservationSourcePackage, load_observation_package
from .validators import validate_interpretations
from .writers import summarize_interpretations, summarize_validation, write_interpretation_outputs


DEFAULT_SOFTWARE_VERSION = "BSIP-2.1.0-interpretation-engine"
INTERPRETATION_SCHEMA_VERSION = "BSIP-2.1.0"


class ScientificInterpretationEngine(InterpretationEngine):
    """Generate conservative interpretations from validated observations."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        observations_dir: Path | str = "outputs/scientific_observations",
        output_dir: Path | str = "outputs/scientific_interpretations",
        overwrite: bool = False,
        software_version: str = DEFAULT_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.observations_dir = Path(observations_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.software_version = software_version
        self._source_package: ObservationSourcePackage | None = None
        self._generated_at: str | None = None

    def load_observations(self) -> tuple[Observation, ...]:
        self._source_package = load_observation_package(self.project_root, self.observations_dir)
        return self._source_package.observations

    def validate_input_observations(
        self, observations: tuple[Observation, ...]
    ) -> tuple[InterpretationValidationIssue, ...]:
        issues: list[InterpretationValidationIssue] = []
        if self._source_package is not None:
            issues.extend(self._source_package.validation_issues)
        issues.extend(_convert_observation_issues(validate_observations(observations)))
        return tuple(issues)

    def build_interpretations(
        self, observations: tuple[Observation, ...]
    ) -> tuple[Interpretation, ...]:
        package = self._source_package
        source_schema_version = None if package is None else package.schema_version
        self._generated_at = self._generated_at or utc_now_iso()
        return build_interpretations(
            observations,
            software_version=self.software_version,
            source_observation_schema_version=source_schema_version,
            created_at=self._generated_at,
            metadata={
                "observation_validation_passed": None
                if package is None
                else package.validation_document.get("validation_passed"),
                "source_observation_count": len(observations),
            },
        )

    def validate_interpretations(
        self,
        interpretations: tuple[Interpretation, ...],
        observations: tuple[Observation, ...],
    ) -> tuple[InterpretationValidationIssue, ...]:
        return validate_interpretations(interpretations, observations)

    def write_outputs(self, interpretations: tuple[Interpretation, ...]) -> tuple[Path, ...]:
        package = self._source_package
        if package is None:
            raise RuntimeError("Observation package must be loaded before writing outputs.")
        validation_issues = validate_interpretations(interpretations, package.observations)
        return write_interpretation_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            interpretations=interpretations,
            validation_issues=validation_issues,
            schema_version=INTERPRETATION_SCHEMA_VERSION,
            software_version=self.software_version,
            source_observation_dir=package.observations_dir,
            generated_at=self._generated_at or utc_now_iso(),
            source_observations_loaded=tuple(observation.observation_id for observation in package.observations),
            source_observations_missing=package.source_files_missing,
            overwrite=self.overwrite,
        )

    def run(self) -> InterpretationRunResult:
        self._generated_at = utc_now_iso()
        observations = self.load_observations()
        input_issues = self.validate_input_observations(observations)
        if _has_critical_issues(input_issues):
            return InterpretationRunResult(
                interpretations=tuple(),
                validation_issues=input_issues,
                output_paths=tuple(),
                metadata=_run_metadata(
                    observations=observations,
                    interpretations=tuple(),
                    validation_issues=input_issues,
                    package=self._source_package,
                    output_paths=tuple(),
                ),
            )

        interpretations = self.build_interpretations(observations)
        interpretation_issues = self.validate_interpretations(interpretations, observations)
        all_issues = input_issues + interpretation_issues
        if _has_critical_issues(interpretation_issues):
            return InterpretationRunResult(
                interpretations=interpretations,
                validation_issues=all_issues,
                output_paths=tuple(),
                metadata=_run_metadata(
                    observations=observations,
                    interpretations=interpretations,
                    validation_issues=all_issues,
                    package=self._source_package,
                    output_paths=tuple(),
                ),
            )

        output_paths = write_interpretation_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            interpretations=interpretations,
            validation_issues=all_issues,
            schema_version=INTERPRETATION_SCHEMA_VERSION,
            software_version=self.software_version,
            source_observation_dir=self._source_package.observations_dir if self._source_package else self.observations_dir,
            generated_at=self._generated_at,
            source_observations_loaded=tuple(observation.observation_id for observation in observations),
            source_observations_missing=self._source_package.source_files_missing if self._source_package else tuple(),
            overwrite=self.overwrite,
        )
        return InterpretationRunResult(
            interpretations=interpretations,
            validation_issues=all_issues,
            output_paths=output_paths,
            metadata=_run_metadata(
                observations=observations,
                interpretations=interpretations,
                validation_issues=all_issues,
                package=self._source_package,
                output_paths=output_paths,
            ),
        )


def _has_critical_issues(issues: tuple[InterpretationValidationIssue, ...]) -> bool:
    return any(issue.severity.value == "CRITICAL" for issue in issues)


def _convert_observation_issues(observation_issues) -> tuple[InterpretationValidationIssue, ...]:
    converted: list[InterpretationValidationIssue] = []
    for issue in observation_issues:
        severity = "WARNING" if str(issue.severity).upper() == "WARNING" else "CRITICAL"
        converted.append(
            InterpretationValidationIssue(
                code=f"OBSERVATION_{issue.code}",
                severity=severity,
                message=issue.message,
                interpretation_id=None,
                field=issue.field,
                observation_id=issue.observation_id,
                rule_id=None,
            )
        )
    return tuple(converted)


def _run_metadata(
    *,
    observations: tuple[Observation, ...],
    interpretations: tuple[Interpretation, ...],
    validation_issues: tuple[InterpretationValidationIssue, ...],
    package: ObservationSourcePackage | None,
    output_paths: tuple[Path, ...],
) -> dict[str, Any]:
    validation_summary = summarize_validation(validation_issues, output_readability_checks={})
    interpretation_summary = summarize_interpretations(
        interpretations,
        source_observations_loaded=tuple(observation.observation_id for observation in observations),
        source_observations_missing=tuple() if package is None else package.source_files_missing,
        validation_passed=validation_summary["validation_passed"],
    )
    return {
        "validation_passed": validation_summary["validation_passed"],
        "critical_issue_count": validation_summary["critical_issue_count"],
        "warning_count": validation_summary["warning_count"],
        "interpretation_count": len(interpretations),
        "source_observation_count": len(observations),
        "source_observations_dir": None if package is None else str(package.observations_dir),
        "output_paths": [str(path) for path in output_paths],
        "interpretation_summary": interpretation_summary,
        "validation_summary": validation_summary,
    }
