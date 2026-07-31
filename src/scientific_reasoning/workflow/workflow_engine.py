"""BSIP v3.0.0 Workflow Engine orchestration layer."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from src.scientific_reasoning.hypothesis import (
    DEFAULT_SOFTWARE_VERSION as HYPOTHESIS_VERSION,
    HypothesisEngine,
)
from src.scientific_reasoning.interpretation import (
    DEFAULT_SOFTWARE_VERSION as INTERPRETATION_VERSION,
    ScientificInterpretationEngine,
)
from src.scientific_reasoning.observation import (
    DEFAULT_SOFTWARE_VERSION as OBSERVATION_VERSION,
    ScientificObservationEngine,
)

from .workflow_manifest import overall_status_from_stage_records, write_workflow_manifest
from .workflow_models import (
    WORKFLOW_SOFTWARE_VERSION,
    StageOutputValidation,
    WorkflowRunResult,
    WorkflowStageName,
    WorkflowStageRecord,
    WorkflowStageStatus,
    utc_now_iso,
    workflow_id_from_timestamp,
)
from .workflow_report import write_workflow_report
from .workflow_validator import validate_stage_outputs


class WorkflowEngine:
    """Coordinate released BSIP reasoning engines without doing reasoning itself."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        output_root: Path | str = "outputs",
        supervisor_results_dir: Path | str = "outputs/supervisor_results_2",
        overwrite: bool = False,
        resume: bool = False,
        software_version: str = WORKFLOW_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.output_root = self._resolve_path(output_root)
        self.supervisor_results_dir = self._resolve_path(supervisor_results_dir)
        self.overwrite = overwrite
        self.resume = resume
        self.software_version = software_version

    @property
    def observation_output_dir(self) -> Path:
        return self.output_root / "scientific_observations"

    @property
    def interpretation_output_dir(self) -> Path:
        return self.output_root / "scientific_interpretations"

    @property
    def hypothesis_output_dir(self) -> Path:
        return self.output_root / "scientific_hypotheses"

    @property
    def workflow_output_dir(self) -> Path:
        return self.output_root / "workflow"

    def run(self) -> WorkflowRunResult:
        started_at = utc_now_iso()
        workflow_id = workflow_id_from_timestamp(started_at)
        if self.overwrite and not self.resume:
            self._prepare_stage_output_root()
        stage_records: list[WorkflowStageRecord] = []

        for stage_name, runner in (
            (WorkflowStageName.OBSERVATION, self._run_observation_stage),
            (WorkflowStageName.INTERPRETATION, self._run_interpretation_stage),
            (WorkflowStageName.HYPOTHESIS, self._run_hypothesis_stage),
        ):
            record = runner()
            stage_records.append(record)
            if record.status == WorkflowStageStatus.FAILED:
                break

        completed_at = utc_now_iso()
        result = WorkflowRunResult(
            workflow_id=workflow_id,
            started_at=started_at,
            completed_at=completed_at,
            software_version=self.software_version,
            overall_status=overall_status_from_stage_records(tuple(stage_records)),
            stage_records=tuple(stage_records),
            metadata={
                "project_root": str(self.project_root),
                "output_root": str(self.output_root),
                "resume": self.resume,
                "overwrite": self.overwrite,
                "source_dataset": str(self.supervisor_results_dir),
            },
        )
        manifest_path = write_workflow_manifest(
            self.workflow_output_dir,
            result,
            source_dataset=str(self.supervisor_results_dir),
        )
        report_path = write_workflow_report(self.workflow_output_dir, result)
        return WorkflowRunResult(
            workflow_id=result.workflow_id,
            started_at=result.started_at,
            completed_at=result.completed_at,
            software_version=result.software_version,
            overall_status=result.overall_status,
            stage_records=result.stage_records,
            manifest_path=manifest_path,
            report_path=report_path,
            metadata=dict(result.metadata),
        )

    def _run_observation_stage(self) -> WorkflowStageRecord:
        return self._run_stage(
            stage_name=WorkflowStageName.OBSERVATION,
            software_version=OBSERVATION_VERSION,
            input_directory=self.supervisor_results_dir,
            output_directory=self.observation_output_dir,
            engine_factory=lambda: ScientificObservationEngine(
                project_root=self.project_root,
                supervisor_results_dir=self.supervisor_results_dir,
                output_dir=self.observation_output_dir,
                overwrite=self.overwrite,
            ),
        )

    def _run_interpretation_stage(self) -> WorkflowStageRecord:
        return self._run_stage(
            stage_name=WorkflowStageName.INTERPRETATION,
            software_version=INTERPRETATION_VERSION,
            input_directory=self.observation_output_dir,
            output_directory=self.interpretation_output_dir,
            engine_factory=lambda: ScientificInterpretationEngine(
                project_root=self.project_root,
                observations_dir=self.observation_output_dir,
                output_dir=self.interpretation_output_dir,
                overwrite=self.overwrite,
            ),
        )

    def _run_hypothesis_stage(self) -> WorkflowStageRecord:
        return self._run_stage(
            stage_name=WorkflowStageName.HYPOTHESIS,
            software_version=HYPOTHESIS_VERSION,
            input_directory=self.interpretation_output_dir,
            output_directory=self.hypothesis_output_dir,
            engine_factory=lambda: HypothesisEngine(
                project_root=self.project_root,
                interpretations_dir=self.interpretation_output_dir,
                output_dir=self.hypothesis_output_dir,
                overwrite=self.overwrite,
            ),
        )

    def _run_stage(
        self,
        *,
        stage_name: WorkflowStageName,
        software_version: str,
        input_directory: Path,
        output_directory: Path,
        engine_factory: Callable[[], Any],
    ) -> WorkflowStageRecord:
        started_at = utc_now_iso()
        started = time.perf_counter()
        if self.resume:
            existing = validate_stage_outputs(stage_name, output_directory)
            if existing.validation_passed:
                return self._stage_record(
                    stage_name=stage_name,
                    status=WorkflowStageStatus.SKIPPED,
                    started_at=started_at,
                    started=started,
                    software_version=software_version,
                    input_directory=input_directory,
                    output_directory=output_directory,
                    validation=existing,
                    metadata={"resume_skipped": True},
                )

        try:
            engine_result = engine_factory().run()
            validation = validate_stage_outputs(stage_name, output_directory)
            status = WorkflowStageStatus.COMPLETED if validation.validation_passed else WorkflowStageStatus.FAILED
            metadata = dict(getattr(engine_result, "metadata", {}) or {})
            return self._stage_record(
                stage_name=stage_name,
                status=status,
                started_at=started_at,
                started=started,
                software_version=software_version,
                input_directory=input_directory,
                output_directory=output_directory,
                validation=validation,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - workflow must record and stop on stage failures.
            validation = validate_stage_outputs(stage_name, output_directory)
            return self._stage_record(
                stage_name=stage_name,
                status=WorkflowStageStatus.FAILED,
                started_at=started_at,
                started=started,
                software_version=software_version,
                input_directory=input_directory,
                output_directory=output_directory,
                validation=validation,
                error=str(exc),
            )

    def _stage_record(
        self,
        *,
        stage_name: WorkflowStageName,
        status: WorkflowStageStatus,
        started_at: str,
        started: float,
        software_version: str,
        input_directory: Path,
        output_directory: Path,
        validation: StageOutputValidation,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowStageRecord:
        completed_at = utc_now_iso()
        critical_count = validation.critical_issue_count
        if status == WorkflowStageStatus.FAILED and error and critical_count == 0:
            critical_count = 1
        record_metadata = dict(metadata or {})
        record_metadata["workflow_stage_validation"] = validation.to_dict()
        return WorkflowStageRecord(
            stage_name=stage_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=round(time.perf_counter() - started, 6),
            software_version=software_version,
            input_directory=str(input_directory),
            output_directory=str(output_directory),
            generated_files=tuple(str(path) for path in validation.generated_files),
            validation_passed=validation.validation_passed,
            critical_issue_count=critical_count,
            warning_count=validation.warning_count,
            validation_summary=dict(validation.validation_summary),
            error=error,
            metadata=record_metadata,
        )

    def _resolve_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()

    def _prepare_stage_output_root(self) -> None:
        for directory in (
            self.observation_output_dir,
            self.interpretation_output_dir,
            self.hypothesis_output_dir,
        ):
            self._remove_safe_directory(directory)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _remove_safe_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        resolved = directory.resolve()
        if resolved == self.project_root or resolved == self.output_root.resolve():
            raise ValueError(f"Refusing to remove protected workflow directory: {resolved}")
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"Refusing to remove directory outside project root: {resolved}") from exc
        shutil.rmtree(resolved)
