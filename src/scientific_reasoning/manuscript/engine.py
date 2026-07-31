"""Service layer for the BSIP v4.2.0 Manuscript Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .composer import DEFAULT_TITLE, compose_manuscript
from .models import (
    MANUSCRIPT_SOFTWARE_VERSION,
    ManuscriptDocument,
    ManuscriptRunResult,
    ManuscriptSourcePackage,
    ManuscriptValidationIssue,
    utc_now_iso,
)
from .source_loader import has_fatal_source_issue, load_source_package
from .validators import validation_summary
from .writers import summarize_document, write_manuscript_outputs


class ManuscriptEngine:
    """Generate a conservative manuscript draft from validated BSIP reasoning artifacts."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        observations_dir: Path | str = "outputs/scientific_observations",
        interpretations_dir: Path | str = "outputs/scientific_interpretations",
        hypotheses_dir: Path | str = "outputs/scientific_hypotheses",
        claims_dir: Path | str = "outputs/scientific_claims",
        evidence_dir: Path | str = "outputs/evidence_scoring",
        review_dir: Path | str = "outputs/scientific_review",
        graph_dir: Path | str = "outputs/reasoning_graph",
        supervisor_results: Path | str = "outputs/supervisor_results_2",
        output_dir: Path | str = "outputs/scientific_manuscript",
        overwrite: bool = False,
        strict: bool = False,
        software_version: str = MANUSCRIPT_SOFTWARE_VERSION,
        title: str = DEFAULT_TITLE,
        author: str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.observations_dir = Path(observations_dir)
        self.interpretations_dir = Path(interpretations_dir)
        self.hypotheses_dir = Path(hypotheses_dir)
        self.claims_dir = Path(claims_dir)
        self.evidence_dir = Path(evidence_dir)
        self.review_dir = Path(review_dir)
        self.graph_dir = Path(graph_dir)
        self.supervisor_results = Path(supervisor_results)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.strict = strict
        self.software_version = software_version
        self.title = title
        self.author = author
        self._source_package: ManuscriptSourcePackage | None = None
        self._generated_at: str | None = None

    def load_sources(self) -> ManuscriptSourcePackage:
        self._source_package = load_source_package(
            project_root=self.project_root,
            observations_dir=self.observations_dir,
            interpretations_dir=self.interpretations_dir,
            hypotheses_dir=self.hypotheses_dir,
            claims_dir=self.claims_dir,
            evidence_dir=self.evidence_dir,
            review_dir=self.review_dir,
            graph_dir=self.graph_dir,
            supervisor_dir=self.supervisor_results,
        )
        return self._source_package

    def validate_inputs(self) -> tuple[ManuscriptValidationIssue, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before validation.")
        return self._source_package.validation_issues

    def build_document(self) -> ManuscriptDocument:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before manuscript composition.")
        return compose_manuscript(
            self._source_package,
            title=self.title,
            author=self.author,
            created_at=self._generated_at or utc_now_iso(),
            software_version=self.software_version,
        )

    def write_outputs(
        self,
        document: ManuscriptDocument,
    ) -> tuple[tuple[Path, ...], tuple[ManuscriptValidationIssue, ...], dict[str, Any]]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before writing outputs.")
        return write_manuscript_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            document=document,
            source=self._source_package,
            generated_at=self._generated_at or utc_now_iso(),
            overwrite=self.overwrite,
            software_version=self.software_version,
        )

    def run(self) -> ManuscriptRunResult:
        self._generated_at = utc_now_iso()
        source = self.load_sources()
        source_issues = self.validate_inputs()
        if has_fatal_source_issue(source_issues):
            summary_validation = validation_summary(None, source_issues, output_readability_checks={})
            metadata = {
                "validation_passed": False,
                "critical_issue_count": summary_validation["critical_issue_count"],
                "warning_count": summary_validation["warning_count"],
                "document_status": "NOT_GENERATED",
                "output_paths": [],
                "summary": summary_validation,
            }
            return ManuscriptRunResult(document=None, validation_issues=source_issues, output_paths=tuple(), metadata=metadata)
        document = self.build_document()
        output_paths, validation_issues, summary = self.write_outputs(document)
        metadata = _run_metadata(document, source, validation_issues, output_paths=output_paths, summary=summary)
        return ManuscriptRunResult(document=document, validation_issues=validation_issues, output_paths=output_paths, metadata=metadata)


def _run_metadata(
    document: ManuscriptDocument,
    source: ManuscriptSourcePackage,
    issues: tuple[ManuscriptValidationIssue, ...],
    *,
    output_paths: tuple[Path, ...],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if summary is None:
        validation_passed = not any(issue.severity.value == "CRITICAL" for issue in issues)
        summary = summarize_document(document, source, validation_passed=validation_passed)
    return {
        "validation_passed": summary.get("validation_passed", False),
        "critical_issue_count": sum(1 for issue in issues if issue.severity.value == "CRITICAL"),
        "warning_count": sum(1 for issue in issues if issue.severity.value == "WARNING"),
        "document_status": document.document_status.value,
        "sentence_count": summary.get("sentence_count"),
        "figure_caption_count": summary.get("figure_caption_count"),
        "table_caption_count": summary.get("table_caption_count"),
        "unresolved_revision_flag_count": summary.get("unresolved_revision_flag_count"),
        "overall_reviewer_recommendation": summary.get("overall_reviewer_recommendation"),
        "output_paths": [str(path) for path in output_paths],
        "summary": summary,
    }


ManuscriptService = ManuscriptEngine
