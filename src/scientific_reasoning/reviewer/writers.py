"""Output writers for the BSIP v4.1.0 Reviewer Engine."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .enums import OverallRecommendation, PublicationRisk, ReviewCategory, ReviewerType, Severity
from .models import (
    REVIEW_RULE_VERSION,
    REVIEW_SCHEMA_VERSION,
    REVIEW_SOFTWARE_VERSION,
    ReviewContext,
    ReviewFinding,
    ReviewValidationIssue,
    json_ready,
)
from .policies import determine_recommendation, extract_claim_links, extract_row_identifier, json_compact, severity_counts
from .validators import validate_review_package, validation_summary


OUTPUT_FILENAMES: tuple[str, ...] = (
    "review_findings.json",
    "review_findings.csv",
    "reviewer_report.md",
    "reviewer_validation.json",
    "reviewer_summary.json",
    "reviewer_blockers.csv",
    "reviewer_claim_matrix.csv",
    "reviewer_revision_requirements.csv",
    "reviewer_figure_matrix.csv",
    "reviewer_publication_assessment.json",
)


def write_review_outputs(
    *,
    project_root: Path | str,
    output_dir: Path | str,
    findings: tuple[ReviewFinding, ...],
    context: ReviewContext,
    generated_at: str,
    overwrite: bool = False,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[tuple[Path, ...], tuple[ReviewValidationIssue, ...], dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    directory = _resolve_output_directory(root, output_dir)
    _prepare_output_directory(root, directory, overwrite=overwrite)
    ordered = tuple(sorted(findings, key=lambda finding: finding.finding_id))
    paths = {name: directory / name for name in OUTPUT_FILENAMES}

    assessment = publication_assessment(context, ordered, generated_at=generated_at, software_version=software_version)
    output_issues = validate_review_package(
        ordered,
        context=context,
        overall_recommendation=assessment["overall_recommendation"],
        has_results_ready_claim=bool(assessment["results_claim_ids"]),
        output_readability_checks={},
    )
    all_issues = context.validation_issues + output_issues
    validation = validation_summary(ordered, all_issues, output_readability_checks={})
    summary = summarize_findings(
        ordered,
        validation_passed=validation["validation_passed"],
        assessment=assessment,
        context=context,
        software_version=software_version,
    )
    _write_all(paths, ordered, context, assessment, summary, validation, generated_at, software_version=software_version)

    readability = _readability_checks(paths)
    output_issues = validate_review_package(
        ordered,
        context=context,
        overall_recommendation=assessment["overall_recommendation"],
        has_results_ready_claim=bool(assessment["results_claim_ids"]),
        output_readability_checks=readability,
    )
    all_issues = context.validation_issues + output_issues
    validation = validation_summary(ordered, all_issues, output_readability_checks=readability)
    summary = summarize_findings(
        ordered,
        validation_passed=validation["validation_passed"],
        assessment=assessment,
        context=context,
        software_version=software_version,
    )
    _write_all(paths, ordered, context, assessment, summary, validation, generated_at, software_version=software_version)
    return tuple(paths[name] for name in OUTPUT_FILENAMES), all_issues, summary, assessment


def publication_assessment(
    context: ReviewContext,
    findings: tuple[ReviewFinding, ...],
    *,
    generated_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> dict[str, Any]:
    non_publication = tuple(finding for finding in findings if finding.reviewer_type is not ReviewerType.PUBLICATION)
    results_claim_ids = tuple(
        str(score.get("claim_id"))
        for score in context.evidence_scores
        if score.get("publication_readiness") in {"RESULTS_READY", "HIGH_CONFIDENCE_RESULTS_READY"}
    )
    discussion_claim_ids = tuple(
        str(score.get("claim_id"))
        for score in context.evidence_scores
        if score.get("publication_readiness") == "DISCUSSION_READY"
    )
    limitation_claim_ids = tuple(
        str(score.get("claim_id"))
        for score in context.evidence_scores
        if score.get("publication_readiness") == "LIMITATION_ONLY"
    )
    recommendation = determine_recommendation(non_publication, has_results_ready_claim=bool(results_claim_ids))
    blockers = tuple(finding for finding in non_publication if finding.blocking)
    major_findings = tuple(finding for finding in non_publication if finding.severity is Severity.MAJOR)
    dimension_scores = {
        "scientific_soundness": _dimension_score(non_publication, reviewer_types={ReviewerType.SCIENTIFIC}),
        "statistical_adequacy": _dimension_score(non_publication, reviewer_types={ReviewerType.STATISTICAL}),
        "evidence_quality": _dimension_score(non_publication, reviewer_types={ReviewerType.EVIDENCE}),
        "validation_strength": _dimension_score(non_publication, reviewer_types={ReviewerType.VALIDATION}),
        "reproducibility": _dimension_score(non_publication, reviewer_types={ReviewerType.REPRODUCIBILITY}),
        "figure_support": _dimension_score(non_publication, reviewer_types={ReviewerType.FIGURE}),
        "writing_safety": _dimension_score(non_publication, reviewer_types={ReviewerType.WRITING}),
        "traceability": _dimension_score(non_publication, categories={ReviewCategory.TRACEABILITY}),
    }
    readiness_score = round(mean(dimension_scores.values()), 2) if dimension_scores else 0.0
    definitive_generalization_blocked = any(
        finding.blocking and finding.category in {ReviewCategory.EXTERNAL_VALIDATION, ReviewCategory.GENERALIZATION}
        for finding in non_publication
    )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "software_version": software_version,
        "review_rule_version": REVIEW_RULE_VERSION,
        "generated_at": generated_at,
        **dimension_scores,
        "overall_readiness_score": readiness_score,
        "overall_recommendation": recommendation.value,
        "blocking_reasons": [finding.title for finding in blockers],
        "major_revision_reasons": [finding.title for finding in major_findings],
        "manuscript_drafting_allowed": recommendation is not OverallRecommendation.INTERNAL_REVIEW_ONLY,
        "definitive_generalization_allowed": not definitive_generalization_blocked,
        "results_claim_ids": list(sorted(results_claim_ids)),
        "discussion_claim_ids": list(sorted(discussion_claim_ids)),
        "limitation_claim_ids": list(sorted(limitation_claim_ids)),
        "assessment_notice": "Scores are deterministic review-support indices and not acceptance probabilities, p-values, novelty evidence, mechanism proof, or journal predictions.",
    }


def summarize_findings(
    findings: tuple[ReviewFinding, ...],
    *,
    validation_passed: bool,
    assessment: dict[str, Any],
    context: ReviewContext,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> dict[str, Any]:
    reviewer_counts = Counter(finding.reviewer_type.value for finding in findings)
    category_counts = Counter(finding.category.value for finding in findings)
    risk_counts = Counter(finding.publication_risk.value for finding in findings)
    blockers = tuple(finding for finding in findings if finding.blocking)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "software_version": software_version,
        "review_rule_version": REVIEW_RULE_VERSION,
        "finding_count": len(findings),
        "blocking_finding_count": len(blockers),
        "count_by_reviewer_type": {item.value: reviewer_counts[item.value] for item in ReviewerType},
        "count_by_category": {item.value: category_counts[item.value] for item in ReviewCategory},
        "count_by_severity": severity_counts(findings),
        "count_by_publication_risk": {item.value: risk_counts[item.value] for item in PublicationRisk},
        "overall_recommendation": assessment["overall_recommendation"],
        "overall_readiness_score": assessment["overall_readiness_score"],
        "manuscript_drafting_allowed": assessment["manuscript_drafting_allowed"],
        "definitive_generalization_allowed": assessment["definitive_generalization_allowed"],
        "source_claim_count": len(context.claims),
        "source_evidence_score_count": len(context.evidence_scores),
        "source_graph_node_count": len(context.graph_node_ids),
        "selected_figure_count": len(context.selected_figures),
        "selected_table_count": len(context.selected_tables),
        "source_files_loaded": list(context.source_files_loaded),
        "source_files_missing": list(context.source_files_missing),
        "validation_passed": validation_passed,
    }


def markdown_report(
    findings: tuple[ReviewFinding, ...],
    *,
    context: ReviewContext,
    assessment: dict[str, Any],
    validation: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Scientific Review",
        "",
        "## Executive Review Summary",
        "",
        f"- Findings generated: {summary.get('finding_count')}",
        f"- Blocking findings: {summary.get('blocking_finding_count')}",
        f"- Validation passed: {validation.get('validation_passed')}",
        "",
        "## Overall Recommendation",
        "",
        f"`{assessment.get('overall_recommendation')}`",
        "",
        "## Publication Blockers",
        "",
    ]
    blockers = tuple(finding for finding in findings if finding.blocking)
    if blockers:
        for finding in blockers:
            lines.append(f"- `{finding.finding_id}` ({finding.severity.value}): {finding.title}")
    else:
        lines.append("- None")
    for severity, heading in (
        (Severity.MAJOR, "Major Findings"),
        (Severity.MODERATE, "Moderate Findings"),
        (Severity.MINOR, "Minor Findings"),
    ):
        lines.extend(("", f"## {heading}", ""))
        subset = tuple(finding for finding in findings if finding.severity is severity)
        if subset:
            for finding in subset:
                lines.append(f"- `{finding.finding_id}`: {finding.title}")
        else:
            lines.append("- None")
    for reviewer_type, heading in (
        (ReviewerType.SCIENTIFIC, "Scientific Review"),
        (ReviewerType.STATISTICAL, "Statistical Review"),
        (ReviewerType.EVIDENCE, "Evidence Review"),
        (ReviewerType.VALIDATION, "Validation Review"),
        (ReviewerType.REPRODUCIBILITY, "Reproducibility Review"),
        (ReviewerType.FIGURE, "Figure and Table Review"),
        (ReviewerType.WRITING, "Writing Review"),
    ):
        lines.extend(("", f"## {heading}", ""))
        subset = tuple(finding for finding in findings if finding.reviewer_type is reviewer_type)
        _append_finding_details(lines, subset)
    lines.extend(("", "## Claim-by-Claim Assessment", ""))
    lines.append("| Claim ID | Findings | Blocking | Highest Severity |")
    lines.append("|---|---:|---:|---|")
    for row in _claim_matrix_rows(findings, context):
        lines.append(f"| {row['claim_id']} | {row['finding_count']} | {row['blocking_finding_count']} | {row['highest_severity']} |")
    lines.extend(("", "## Revision Requirements", ""))
    revision_rows = _revision_requirement_rows(findings)
    if revision_rows:
        for row in revision_rows:
            lines.append(f"- `{row['finding_id']}`: {row['revision_requirement']}")
    else:
        lines.append("- None")
    lines.extend(
        (
            "",
            "## Strengths",
            "",
            f"- Source claims loaded: {summary.get('source_claim_count')}",
            f"- Evidence score records loaded: {summary.get('source_evidence_score_count')}",
            f"- Reasoning graph nodes loaded: {summary.get('source_graph_node_count')}",
            "",
            "## Remaining Limitations",
            "",
        )
    )
    if blockers:
        for finding in blockers:
            for limitation in finding.limitations:
                lines.append(f"- `{finding.finding_id}`: {limitation}")
    else:
        lines.append("- No blocking reviewer limitations recorded.")
    lines.extend(
        (
            "",
            "## Traceability Statement",
            "",
            "Each review finding links to source claim IDs, evidence score IDs, reasoning graph node IDs, source validation IDs, and selected figure or table IDs where those references are available.",
            "",
            "No manuscript prose, raw-data analysis, model retraining, scientific interpretation, mechanism claim, novelty claim, or journal prediction is generated by this report.",
            "",
        )
    )
    return "\n".join(lines)


def _write_all(
    paths: dict[str, Path],
    findings: tuple[ReviewFinding, ...],
    context: ReviewContext,
    assessment: dict[str, Any],
    summary: dict[str, Any],
    validation: dict[str, Any],
    generated_at: str,
    *,
    software_version: str,
) -> None:
    _write_json(paths["review_findings.json"], _findings_document(findings, context, assessment, summary, validation, generated_at, software_version=software_version))
    _write_csv(paths["review_findings.csv"], [_flatten_record(finding.to_record()) for finding in findings], fieldnames=_finding_fieldnames())
    paths["reviewer_report.md"].write_text(markdown_report(findings, context=context, assessment=assessment, validation=validation, summary=summary), encoding="utf-8")
    _write_json(paths["reviewer_validation.json"], validation)
    _write_json(paths["reviewer_summary.json"], summary)
    _write_csv(paths["reviewer_blockers.csv"], _blocker_rows(findings), fieldnames=_blocker_fieldnames())
    _write_csv(paths["reviewer_claim_matrix.csv"], _claim_matrix_rows(findings, context), fieldnames=_claim_matrix_fieldnames())
    _write_csv(paths["reviewer_revision_requirements.csv"], _revision_requirement_rows(findings), fieldnames=_revision_fieldnames())
    _write_csv(paths["reviewer_figure_matrix.csv"], _figure_matrix_rows(context), fieldnames=_figure_matrix_fieldnames())
    _write_json(paths["reviewer_publication_assessment.json"], assessment)


def _findings_document(
    findings: tuple[ReviewFinding, ...],
    context: ReviewContext,
    assessment: dict[str, Any],
    summary: dict[str, Any],
    validation: dict[str, Any],
    generated_at: str,
    *,
    software_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "software_version": software_version,
        "review_rule_version": REVIEW_RULE_VERSION,
        "generated_at": generated_at,
        "source_validation_status": {
            "claim_validation_passed": context.claim_validation_document.get("validation_passed") is True,
            "evidence_scoring_validation_passed": context.evidence_validation_document.get("validation_passed") is True,
            "reasoning_graph_validation_passed": context.graph_validation_document.get("validation_passed") is True,
            "source_files_loaded": list(context.source_files_loaded),
            "source_files_missing": list(context.source_files_missing),
        },
        "publication_assessment": assessment,
        "summary": summary,
        "validation_summary": validation,
        "review_findings": [finding.to_record() for finding in findings],
        "review_notice": "Reviewer findings are deterministic structured review checks, not new scientific analyses or manuscript text.",
    }


def _dimension_score(
    findings: tuple[ReviewFinding, ...],
    *,
    reviewer_types: set[ReviewerType] | None = None,
    categories: set[ReviewCategory] | None = None,
) -> float:
    selected = [
        finding
        for finding in findings
        if (reviewer_types is None or finding.reviewer_type in reviewer_types)
        and (categories is None or finding.category in categories)
        and finding.severity is not Severity.INFORMATION
    ]
    score = 100.0
    penalties = {
        Severity.CRITICAL: 45.0,
        Severity.MAJOR: 25.0,
        Severity.MODERATE: 12.0,
        Severity.MINOR: 4.0,
        Severity.INFORMATION: 0.0,
    }
    for finding in selected:
        score -= penalties[finding.severity]
        if finding.blocking:
            score -= 10.0
    return max(0.0, round(score, 2))


def _append_finding_details(lines: list[str], findings: tuple[ReviewFinding, ...]) -> None:
    if not findings:
        lines.append("- None")
        return
    for finding in findings:
        lines.extend(
            (
                f"### {finding.finding_id}",
                "",
                f"- Category: `{finding.category.value}`",
                f"- Severity: `{finding.severity.value}`",
                f"- Blocking: `{finding.blocking}`",
                f"- Finding: {finding.finding_text}",
                f"- Evidence: {finding.evidence_summary or 'Recorded in linked source artifacts.'}",
                f"- Confidence: `{finding.confidence.value}`",
                f"- Revision requirement: {finding.revision_requirement or 'None'}",
                "",
            )
        )


def _finding_fieldnames() -> tuple[str, ...]:
    return (
        "finding_id",
        "reviewer_type",
        "category",
        "title",
        "finding_text",
        "severity",
        "blocking",
        "confidence",
        "affected_claim_ids",
        "affected_hypothesis_ids",
        "affected_interpretation_ids",
        "affected_observation_ids",
        "affected_figure_ids",
        "affected_table_ids",
        "evidence_score_ids",
        "reasoning_graph_node_ids",
        "source_validation_ids",
        "rationale",
        "evidence_summary",
        "publication_risk",
        "revision_requirement",
        "limitations",
        "rule_ids",
        "created_at",
        "software_version",
        "tags",
        "metadata",
    )


def _blocker_fieldnames() -> tuple[str, ...]:
    return ("finding_id", "reviewer_type", "category", "severity", "title", "affected_claim_ids", "publication_risk", "revision_requirement")


def _blocker_rows(findings: tuple[ReviewFinding, ...]) -> list[dict[str, Any]]:
    return [
        {
            "finding_id": finding.finding_id,
            "reviewer_type": finding.reviewer_type.value,
            "category": finding.category.value,
            "severity": finding.severity.value,
            "title": finding.title,
            "affected_claim_ids": json_compact(list(finding.affected_claim_ids)),
            "publication_risk": finding.publication_risk.value,
            "revision_requirement": finding.revision_requirement,
        }
        for finding in findings
        if finding.blocking
    ]


def _claim_matrix_fieldnames() -> tuple[str, ...]:
    return (
        "claim_id",
        "category",
        "claim_type",
        "claim_status",
        "claim_publication_use",
        "evidence_level",
        "uncertainty_level",
        "publication_readiness",
        "review_finding_ids",
        "blocking_finding_ids",
        "finding_count",
        "blocking_finding_count",
        "critical_count",
        "major_count",
        "moderate_count",
        "minor_count",
        "information_count",
        "highest_severity",
    )


def _claim_matrix_rows(findings: tuple[ReviewFinding, ...], context: ReviewContext) -> list[dict[str, Any]]:
    rows = []
    finding_by_claim: dict[str, list[ReviewFinding]] = {}
    for finding in findings:
        for claim_id in finding.affected_claim_ids:
            finding_by_claim.setdefault(claim_id, []).append(finding)
    claims = context.claims
    if not claims:
        claims = tuple({"claim_id": claim_id} for claim_id in sorted(finding_by_claim))
    for claim in claims:
        claim_id = str(claim.get("claim_id"))
        score = context.evidence_by_claim_id.get(claim_id, {})
        related = tuple(sorted(finding_by_claim.get(claim_id, ()), key=lambda finding: finding.finding_id))
        counts = Counter(finding.severity.value for finding in related)
        highest = _highest_severity(related)
        rows.append(
            {
                "claim_id": claim_id,
                "category": claim.get("category") or score.get("claim_category"),
                "claim_type": claim.get("claim_type") or score.get("claim_type"),
                "claim_status": claim.get("claim_status") or score.get("claim_status"),
                "claim_publication_use": claim.get("publication_use") or score.get("claim_publication_use"),
                "evidence_level": score.get("evidence_level"),
                "uncertainty_level": score.get("uncertainty_level"),
                "publication_readiness": score.get("publication_readiness"),
                "review_finding_ids": json_compact([finding.finding_id for finding in related]),
                "blocking_finding_ids": json_compact([finding.finding_id for finding in related if finding.blocking]),
                "finding_count": len(related),
                "blocking_finding_count": sum(1 for finding in related if finding.blocking),
                "critical_count": counts[Severity.CRITICAL.value],
                "major_count": counts[Severity.MAJOR.value],
                "moderate_count": counts[Severity.MODERATE.value],
                "minor_count": counts[Severity.MINOR.value],
                "information_count": counts[Severity.INFORMATION.value],
                "highest_severity": highest.value if highest else "",
            }
        )
    return rows


def _revision_fieldnames() -> tuple[str, ...]:
    return ("finding_id", "severity", "blocking", "reviewer_type", "category", "affected_claim_ids", "revision_requirement", "rule_ids")


def _revision_requirement_rows(findings: tuple[ReviewFinding, ...]) -> list[dict[str, Any]]:
    return [
        {
            "finding_id": finding.finding_id,
            "severity": finding.severity.value,
            "blocking": finding.blocking,
            "reviewer_type": finding.reviewer_type.value,
            "category": finding.category.value,
            "affected_claim_ids": json_compact(list(finding.affected_claim_ids)),
            "revision_requirement": finding.revision_requirement,
            "rule_ids": json_compact(list(finding.rule_ids)),
        }
        for finding in findings
        if finding.severity is not Severity.INFORMATION
    ]


def _figure_matrix_fieldnames() -> tuple[str, ...]:
    return ("claim_id", "figure_ids", "table_ids", "visual_support_status", "selected_figure_count", "selected_table_count")


def _figure_matrix_rows(context: ReviewContext) -> list[dict[str, Any]]:
    figure_map = _claim_visual_map(context.selected_figures, "figure_id")
    table_map = _claim_visual_map(context.selected_tables, "table_id")
    metadata_available = bool(context.selected_figures or context.selected_tables)
    claim_links_available = bool(figure_map or table_map)
    rows = []
    for claim in context.claims:
        claim_id = str(claim.get("claim_id"))
        figure_ids = tuple(sorted(figure_map.get(claim_id, ())))
        table_ids = tuple(sorted(table_map.get(claim_id, ())))
        if not metadata_available:
            status = "METADATA_UNAVAILABLE"
        elif not claim_links_available:
            status = "CLAIM_LINK_METADATA_UNAVAILABLE"
        elif figure_ids or table_ids:
            status = "LINKED"
        else:
            status = "UNLINKED"
        rows.append(
            {
                "claim_id": claim_id,
                "figure_ids": json_compact(list(figure_ids)),
                "table_ids": json_compact(list(table_ids)),
                "visual_support_status": status,
                "selected_figure_count": len(context.selected_figures),
                "selected_table_count": len(context.selected_tables),
            }
        )
    return rows


def _claim_visual_map(rows: tuple[dict[str, str], ...], id_field: str) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, set[str]] = {}
    for row in rows:
        item_id = extract_row_identifier(dict(row), id_field, "id")
        for claim_id in extract_claim_links(dict(row)):
            mapping.setdefault(claim_id, set()).add(item_id)
    return {claim_id: tuple(sorted(values)) for claim_id, values in sorted(mapping.items())}


def _highest_severity(findings: tuple[ReviewFinding, ...]) -> Severity | None:
    if not findings:
        return None
    return max((finding.severity for finding in findings), key=lambda severity: _severity_value(severity))


def _severity_value(severity: Severity) -> int:
    return {
        Severity.CRITICAL: 4,
        Severity.MAJOR: 3,
        Severity.MODERATE: 2,
        Severity.MINOR: 1,
        Severity.INFORMATION: 0,
    }[severity]


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            flattened[key] = json_compact(json_ready(value))
        else:
            flattened[key] = value
    return flattened


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _readability_checks(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    checks = {}
    for name, path in sorted(paths.items()):
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    list(csv.DictReader(handle))
            else:
                path.read_text(encoding="utf-8")
            checks[name] = {"readable": True, "path": str(path)}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            checks[name] = {"readable": False, "path": str(path), "error": str(exc)}
    return checks


def _resolve_output_directory(project_root: Path, output_dir: Path | str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _prepare_output_directory(project_root: Path, directory: Path, *, overwrite: bool) -> None:
    try:
        directory.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Reviewer output directory must be inside project root: {directory}") from exc
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Reviewer output directory is not empty: {directory}. Use --overwrite to replace it.")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
