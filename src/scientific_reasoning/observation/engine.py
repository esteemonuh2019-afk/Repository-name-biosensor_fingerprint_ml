"""Production Scientific Observation Engine implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .interfaces import ObservationEngine
from .models import Observation, ValidationIssue
from .rules import attach_provenance_records, build_observations
from .source_loader import SupervisorSourcePayload, load_supervisor_sources
from .validators import validate_observations
from .writers import ObservationWriteResult, write_observation_outputs


DEFAULT_SOFTWARE_VERSION = "BSIP-2.0-observation-engine"


@dataclass(frozen=True)
class ScientificObservationRunResult:
    observations: tuple[Observation, ...] = field(default_factory=tuple)
    validation_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    source_payload: SupervisorSourcePayload | None = None
    write_result: ObservationWriteResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def validation_passed(self) -> bool:
        if self.write_result:
            return bool(self.write_result.validation_summary.get("validation_passed"))
        return not any(issue.severity in {"CRITICAL", "ERROR"} for issue in self.validation_issues)


class ScientificObservationEngine(ObservationEngine):
    """Build factual observations from a validated supervisor-results package."""

    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        supervisor_results_dir: str | Path = "outputs/supervisor_results_2",
        output_dir: str | Path = "outputs/scientific_observations",
        overwrite: bool = False,
        software_version: str = DEFAULT_SOFTWARE_VERSION,
        generated_at: str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.supervisor_results_dir = Path(supervisor_results_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.software_version = software_version
        self.generated_at = generated_at or utc_now_iso()
        self._payload: SupervisorSourcePayload | None = None
        self._rule_issues: tuple[ValidationIssue, ...] = tuple()

    def load_sources(self) -> Mapping[str, Any]:
        payload = load_supervisor_sources(self.project_root, self.supervisor_results_dir)
        self._payload = payload
        return {"payload": payload}

    def build_observations(self, sources: Mapping[str, Any]) -> tuple[Observation, ...]:
        payload = sources["payload"]
        if not isinstance(payload, SupervisorSourcePayload):
            raise TypeError("sources['payload'] must be a SupervisorSourcePayload")
        result = build_observations(
            payload,
            software_version=self.software_version,
            created_at=self.generated_at,
        )
        observations = attach_provenance_records(result.observations, payload.provenance_records)
        self._rule_issues = result.validation_issues
        return _sort_observations(observations)

    def validate_observations(
        self, observations: tuple[Observation, ...]
    ) -> tuple[ValidationIssue, ...]:
        payload = self._payload
        issues: list[ValidationIssue] = []
        if payload:
            issues.extend(payload.validation_issues)
            issues.extend(_optional_warning_issues(payload))
        issues.extend(self._rule_issues)
        issues.extend(validate_observations(observations))
        return _dedupe_issues(tuple(issues))

    def write_outputs(self, observations: tuple[Observation, ...]) -> tuple[Path, ...]:
        payload = self._payload
        if payload is None:
            raise RuntimeError("load_sources() must be called before write_outputs().")
        issues = self.validate_observations(observations)
        result = write_observation_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            supervisor_payload=payload,
            observations=observations,
            validation_issues=issues,
            software_version=self.software_version,
            generated_at=self.generated_at,
            overwrite=self.overwrite,
        )
        return result.output_paths

    def run(self) -> ScientificObservationRunResult:
        sources = self.load_sources()
        payload = sources["payload"]
        observations = self.build_observations(sources)
        validation_issues = self.validate_observations(observations)
        write_result = write_observation_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            supervisor_payload=payload,
            observations=observations,
            validation_issues=validation_issues,
            software_version=self.software_version,
            generated_at=self.generated_at,
            overwrite=self.overwrite,
        )
        return ScientificObservationRunResult(
            observations=observations,
            validation_issues=validation_issues,
            output_paths=write_result.output_paths,
            source_payload=payload,
            write_result=write_result,
            metadata={
                "generated_at": self.generated_at,
                "software_version": self.software_version,
                "supervisor_results_dir": str(payload.supervisor_results_dir),
            },
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sort_observations(observations: tuple[Observation, ...]) -> tuple[Observation, ...]:
    return tuple(sorted(observations, key=lambda observation: observation.observation_id))


def _optional_warning_issues(payload: SupervisorSourcePayload) -> tuple[ValidationIssue, ...]:
    return tuple(
        ValidationIssue(
            code="OPTIONAL_SOURCE_MISSING",
            severity="WARNING",
            message=f"Optional supervisor-results source is missing: {filename}",
            observation_id=None,
            field="source_file",
            source_file=str(payload.supervisor_results_dir / filename),
        )
        for filename in payload.missing_optional_files
    )


def _dedupe_issues(issues: tuple[ValidationIssue, ...]) -> tuple[ValidationIssue, ...]:
    seen = set()
    unique: list[ValidationIssue] = []
    for issue in issues:
        key = (
            issue.code,
            issue.severity,
            issue.message,
            issue.observation_id,
            issue.field,
            issue.source_file,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return tuple(unique)
