"""Output writers for the BSIP v4.2.0 Manuscript Engine."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .docx_writer import write_docx
from .enums import SectionType, SentenceType, TraceabilityStatus
from .models import (
    MANUSCRIPT_RULE_VERSION,
    MANUSCRIPT_SCHEMA_VERSION,
    MANUSCRIPT_SOFTWARE_VERSION,
    ManuscriptDocument,
    ManuscriptSourcePackage,
    ManuscriptValidationIssue,
    json_ready,
)
from .policies import boundary_for_claim, json_compact
from .validators import readability_checks, validate_manuscript_document, validation_summary


OUTPUT_FILENAMES: tuple[str, ...] = (
    "manuscript_draft.md",
    "manuscript_draft.docx",
    "manuscript_results.md",
    "manuscript_discussion.md",
    "manuscript_limitations.md",
    "manuscript_conclusion.md",
    "figure_captions.md",
    "table_captions.md",
    "manuscript_sentence_traceability.csv",
    "manuscript_claim_matrix.csv",
    "manuscript_figure_matrix.csv",
    "manuscript_table_matrix.csv",
    "manuscript_validation.json",
    "manuscript_summary.json",
    "manuscript_revision_flags.csv",
    "manuscript_manifest.json",
)


def write_manuscript_outputs(
    *,
    project_root: Path | str,
    output_dir: Path | str,
    document: ManuscriptDocument,
    source: ManuscriptSourcePackage,
    generated_at: str,
    overwrite: bool = False,
    software_version: str = MANUSCRIPT_SOFTWARE_VERSION,
) -> tuple[tuple[Path, ...], tuple[ManuscriptValidationIssue, ...], dict[str, Any]]:
    root = Path(project_root).resolve()
    directory = _resolve_output_directory(root, output_dir)
    _prepare_output_directory(root, directory, overwrite=overwrite)
    paths = {name: directory / name for name in OUTPUT_FILENAMES}

    output_issues = validate_manuscript_document(document, source, output_readability_checks={})
    all_issues = source.validation_issues + output_issues
    validation = validation_summary(document, all_issues, output_readability_checks={})
    summary = summarize_document(document, source, validation_passed=validation["validation_passed"], software_version=software_version)
    manifest = manifest_document(document, source, paths=paths, validation=validation, summary=summary, generated_at=generated_at, software_version=software_version)
    _write_all(paths, document, source, validation, summary, manifest)

    checks = readability_checks(paths)
    output_issues = validate_manuscript_document(document, source, output_readability_checks=checks)
    all_issues = source.validation_issues + output_issues
    validation = validation_summary(document, all_issues, output_readability_checks=checks)
    summary = summarize_document(document, source, validation_passed=validation["validation_passed"], software_version=software_version)
    manifest = manifest_document(document, source, paths=paths, validation=validation, summary=summary, generated_at=generated_at, software_version=software_version)
    _write_all(paths, document, source, validation, summary, manifest)
    return tuple(paths[name] for name in OUTPUT_FILENAMES), all_issues, summary


def summarize_document(
    document: ManuscriptDocument,
    source: ManuscriptSourcePackage,
    *,
    validation_passed: bool,
    software_version: str = MANUSCRIPT_SOFTWARE_VERSION,
) -> dict[str, Any]:
    sentences = document.sentences
    trace_counts = Counter(sentence.traceability_status.value for sentence in sentences)
    type_counts = Counter(sentence.sentence_type.value for sentence in sentences)
    return {
        "manuscript_id": document.manuscript_id,
        "document_status": document.document_status.value,
        "manuscript_drafting_allowed": source.reviewer_publication_assessment_document.get("manuscript_drafting_allowed"),
        "overall_reviewer_recommendation": source.reviewer_publication_assessment_document.get("overall_recommendation"),
        "section_count": len(document.sections),
        "paragraph_count": len(document.paragraphs),
        "sentence_count": len(sentences),
        "quantitative_sentence_count": sum(1 for sentence in sentences if sentence.sentence_type is SentenceType.RESULT and any(char.isdigit() for char in sentence.text)),
        "interpretive_sentence_count": type_counts[SentenceType.INTERPRETATION.value],
        "limitation_sentence_count": type_counts[SentenceType.LIMITATION.value],
        "conclusion_sentence_count": type_counts[SentenceType.CONCLUSION.value],
        "figure_caption_count": len(document.figure_captions),
        "table_caption_count": len(document.table_captions),
        "fully_traceable_sentence_count": trace_counts[TraceabilityStatus.COMPLETE.value],
        "partially_traceable_sentence_count": trace_counts[TraceabilityStatus.PARTIAL.value],
        "withheld_sentence_count": sum(1 for sentence in sentences if sentence.language_policy_status == "WITHHELD"),
        "unresolved_revision_flag_count": sum(1 for flag in document.unresolved_flags if flag.author_action_required),
        "blocking_reviewer_finding_count": sum(1 for finding in source.review_findings if finding.get("blocking") is True),
        "source_observation_count": len(source.observations),
        "source_interpretation_count": len(source.interpretations),
        "source_hypothesis_count": len(source.hypotheses),
        "source_claim_count": len(source.claims),
        "source_evidence_score_count": len(source.evidence_scores),
        "source_reviewer_finding_count": len(source.review_findings),
        "validation_passed": validation_passed,
        "schema_version": MANUSCRIPT_SCHEMA_VERSION,
        "software_version": software_version,
    }


def manifest_document(
    document: ManuscriptDocument,
    source: ManuscriptSourcePackage,
    *,
    paths: dict[str, Path],
    validation: dict[str, Any],
    summary: dict[str, Any],
    generated_at: str,
    software_version: str,
) -> dict[str, Any]:
    generated_files = [str(path) for _, path in sorted(paths.items()) if path.exists()]
    return {
        "generated_at": generated_at,
        "software_version": software_version,
        "schema_version": MANUSCRIPT_SCHEMA_VERSION,
        "rule_versions": {
            "manuscript_rule_version": MANUSCRIPT_RULE_VERSION,
            "review_rule_version": source.reviewer_publication_assessment_document.get("review_rule_version"),
        },
        "source_directories": {
            "observations": str(source.observations_dir),
            "interpretations": str(source.interpretations_dir),
            "hypotheses": str(source.hypotheses_dir),
            "claims": str(source.claims_dir),
            "evidence": str(source.evidence_dir),
            "review": str(source.review_dir),
            "reasoning_graph": str(source.graph_dir),
            "supervisor_results": str(source.supervisor_dir),
        },
        "source_files": list(source.source_files_loaded),
        "source_validation_summaries": {
            "observation": _validation_status(source.observation_validation_document),
            "interpretation": _validation_status(source.interpretation_validation_document),
            "hypothesis": _validation_status(source.hypothesis_validation_document),
            "claim": _validation_status(source.claim_validation_document),
            "evidence_scoring": _validation_status(source.evidence_validation_document),
            "reviewer": _validation_status(source.reviewer_validation_document),
            "reasoning_graph": _validation_status(source.graph_validation_document),
            "supervisor": _validation_status(source.supervisor_validation_document),
        },
        "generated_output_files": generated_files,
        "generated_output_checksums": _checksums(generated_files),
        "document_status": document.document_status.value,
        "unresolved_blocker_count": sum(1 for flag in document.unresolved_flags if flag.blocking and flag.author_action_required),
        "summary": summary,
        "validation_summary": validation,
    }


def markdown_document(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> str:
    lines = [
        f"# {document.title}",
        "",
        "Generated under BSIP Reviewer Constraints",
        "",
        f"Document status: `{document.document_status.value}`",
        f"Overall reviewer recommendation: `{source.reviewer_publication_assessment_document.get('overall_recommendation')}`",
        "",
        "> This is an internal scientific draft. Reviewer blockers and revision flags remain unresolved until author review.",
        "",
    ]
    for section in document.sections:
        if section.section_type is SectionType.TITLE:
            continue
        lines.extend((f"## {section.title}", ""))
        for sentence in section.sentences:
            lines.append(sentence.text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def section_markdown(document: ManuscriptDocument, section_type: SectionType) -> str:
    lines = []
    for section in document.sections:
        if section.section_type is section_type:
            lines.append(f"# {section.title}")
            lines.append("")
            for sentence in section.sentences:
                lines.append(sentence.text)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def captions_markdown(title: str, captions: tuple) -> str:
    lines = [f"# {title}", ""]
    if not captions:
        lines.append("No selected metadata was available.")
    for caption in captions:
        lines.append(caption.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_all(
    paths: dict[str, Path],
    document: ManuscriptDocument,
    source: ManuscriptSourcePackage,
    validation: dict[str, Any],
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    paths["manuscript_draft.md"].write_text(markdown_document(document, source), encoding="utf-8")
    write_docx(paths["manuscript_draft.docx"], document, source)
    paths["manuscript_results.md"].write_text(section_markdown(document, SectionType.RESULTS), encoding="utf-8")
    paths["manuscript_discussion.md"].write_text(section_markdown(document, SectionType.DISCUSSION), encoding="utf-8")
    paths["manuscript_limitations.md"].write_text(section_markdown(document, SectionType.LIMITATIONS), encoding="utf-8")
    paths["manuscript_conclusion.md"].write_text(section_markdown(document, SectionType.CONCLUSION), encoding="utf-8")
    paths["figure_captions.md"].write_text(captions_markdown("Figure Captions", document.figure_captions), encoding="utf-8")
    paths["table_captions.md"].write_text(captions_markdown("Table Captions", document.table_captions), encoding="utf-8")
    _write_csv(paths["manuscript_sentence_traceability.csv"], _traceability_rows(document), fieldnames=_traceability_fieldnames())
    _write_csv(paths["manuscript_claim_matrix.csv"], _claim_matrix_rows(document, source), fieldnames=_claim_matrix_fieldnames())
    _write_csv(paths["manuscript_figure_matrix.csv"], _figure_matrix_rows(document, source), fieldnames=_figure_matrix_fieldnames())
    _write_csv(paths["manuscript_table_matrix.csv"], _table_matrix_rows(document, source), fieldnames=_table_matrix_fieldnames())
    _write_json(paths["manuscript_validation.json"], validation)
    _write_json(paths["manuscript_summary.json"], summary)
    _write_csv(paths["manuscript_revision_flags.csv"], [flag.to_dict() for flag in document.unresolved_flags], fieldnames=_revision_flag_fieldnames())
    _write_json(paths["manuscript_manifest.json"], manifest)


def _traceability_fieldnames() -> tuple[str, ...]:
    return (
        "sentence_id",
        "section_id",
        "sentence_type",
        "text",
        "traceability_status",
        "language_policy_status",
        "source_ids",
        "claim_ids",
        "observation_ids",
        "interpretation_ids",
        "hypothesis_ids",
        "evidence_score_ids",
        "reviewer_finding_ids",
        "figure_ids",
        "table_ids",
        "reasoning_graph_node_ids",
        "limitations",
        "metadata",
    )


def _traceability_rows(document: ManuscriptDocument) -> list[dict[str, Any]]:
    rows = []
    for sentence in document.sentences:
        record = sentence.to_record()
        rows.append(
            {
                "sentence_id": record["sentence_id"],
                "section_id": record["section_id"],
                "sentence_type": record["sentence_type"],
                "text": record["text"],
                "traceability_status": record["traceability_status"],
                "language_policy_status": record["language_policy_status"],
                "source_ids": json_compact(record["source_ids"]),
                "claim_ids": json_compact(record["claim_ids"]),
                "observation_ids": json_compact(record["observation_ids"]),
                "interpretation_ids": json_compact(record["interpretation_ids"]),
                "hypothesis_ids": json_compact(record["hypothesis_ids"]),
                "evidence_score_ids": json_compact(record["evidence_score_ids"]),
                "reviewer_finding_ids": json_compact(record["reviewer_finding_ids"]),
                "figure_ids": json_compact(record["figure_ids"]),
                "table_ids": json_compact(record["table_ids"]),
                "reasoning_graph_node_ids": json_compact(record["reasoning_graph_node_ids"]),
                "limitations": json_compact(record["limitations"]),
                "metadata": json_compact(record["metadata"]),
            }
        )
    return rows


def _claim_matrix_fieldnames() -> tuple[str, ...]:
    return (
        "claim_id",
        "category",
        "claim_type",
        "claim_status",
        "claim_publication_use",
        "publication_readiness",
        "publication_boundary",
        "result_sentence_ids",
        "discussion_sentence_ids",
        "limitation_sentence_ids",
        "reviewer_finding_ids",
        "withheld",
    )


def _claim_matrix_rows(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> list[dict[str, Any]]:
    rows = []
    for claim in source.claims:
        claim_id = str(claim.get("claim_id"))
        score = source.evidence_by_claim_id.get(claim_id, {})
        boundary = boundary_for_claim(claim, score)
        claim_sentences = tuple(sentence for sentence in document.sentences if claim_id in sentence.claim_ids)
        rows.append(
            {
                "claim_id": claim_id,
                "category": claim.get("category"),
                "claim_type": claim.get("claim_type"),
                "claim_status": claim.get("claim_status"),
                "claim_publication_use": claim.get("publication_use"),
                "publication_readiness": score.get("publication_readiness"),
                "publication_boundary": boundary.value,
                "result_sentence_ids": json_compact([sentence.sentence_id for sentence in claim_sentences if sentence.sentence_type is SentenceType.RESULT]),
                "discussion_sentence_ids": json_compact([sentence.sentence_id for sentence in claim_sentences if sentence.sentence_type is SentenceType.INTERPRETATION]),
                "limitation_sentence_ids": json_compact([sentence.sentence_id for sentence in claim_sentences if sentence.sentence_type is SentenceType.LIMITATION]),
                "reviewer_finding_ids": json_compact(sorted({finding for sentence in claim_sentences for finding in sentence.reviewer_finding_ids})),
                "withheld": claim.get("claim_type") == "WITHHELD",
            }
        )
    return rows


def _figure_matrix_fieldnames() -> tuple[str, ...]:
    return ("figure_id", "title", "source_file", "source_run", "caption_id", "caption_sentence_id", "caption_status")


def _figure_matrix_rows(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> list[dict[str, Any]]:
    captions = {caption.source_id: caption for caption in document.figure_captions}
    return [
        {
            "figure_id": row.get("figure_id"),
            "title": row.get("title"),
            "source_file": row.get("source_file") or row.get("output_file"),
            "source_run": row.get("source_run"),
            "caption_id": captions.get(str(row.get("figure_id"))).caption_id if captions.get(str(row.get("figure_id"))) else "",
            "caption_sentence_id": captions.get(str(row.get("figure_id"))).sentence_id if captions.get(str(row.get("figure_id"))) else "",
            "caption_status": "GENERATED" if captions.get(str(row.get("figure_id"))) else "MISSING_METADATA",
        }
        for row in source.selected_figures
    ]


def _table_matrix_fieldnames() -> tuple[str, ...]:
    return ("table_id", "title", "source_file", "row_count", "caption_id", "caption_sentence_id", "caption_status")


def _table_matrix_rows(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> list[dict[str, Any]]:
    captions = {caption.source_id: caption for caption in document.table_captions}
    return [
        {
            "table_id": row.get("table_id"),
            "title": row.get("title"),
            "source_file": row.get("source_file"),
            "row_count": row.get("row_count"),
            "caption_id": captions.get(str(row.get("table_id"))).caption_id if captions.get(str(row.get("table_id"))) else "",
            "caption_sentence_id": captions.get(str(row.get("table_id"))).sentence_id if captions.get(str(row.get("table_id"))) else "",
            "caption_status": "GENERATED" if captions.get(str(row.get("table_id"))) else "MISSING_METADATA",
        }
        for row in source.selected_tables
    ]


def _revision_flag_fieldnames() -> tuple[str, ...]:
    return (
        "flag_id",
        "reviewer_finding_id",
        "severity",
        "blocking",
        "affected_section",
        "affected_sentence_ids",
        "applied_action",
        "resolution_status",
        "author_action_required",
        "notes",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_csv_row(row))


def _flatten_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flattened[key] = json_compact(json_ready(value))
        else:
            flattened[key] = value
    return flattened


def _validation_status(document: dict[str, Any]) -> dict[str, Any]:
    passed = document.get("validation_passed")
    if passed is None and "passed" in document:
        passed = document.get("passed")
    return {
        "validation_passed": passed,
        "critical_issue_count": document.get("critical_issue_count", 0),
        "warning_count": document.get("warning_count", 0),
    }


def _checksums(files: list[str]) -> dict[str, str]:
    checksums = {}
    for filename in files:
        path = Path(filename)
        if path.exists() and path.is_file():
            checksums[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def _resolve_output_directory(project_root: Path, output_dir: Path | str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _prepare_output_directory(project_root: Path, directory: Path, *, overwrite: bool) -> None:
    try:
        directory.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Manuscript output directory must be inside project root: {directory}") from exc
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Manuscript output directory is not empty: {directory}. Use --overwrite to replace it.")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
