"""Validation for BSIP manuscript documents and outputs."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from .enums import ManuscriptIssueSeverity, SectionType, SentenceType, TraceabilityStatus
from .models import ManuscriptDocument, ManuscriptSentence, ManuscriptSourcePackage, ManuscriptValidationIssue
from .policies import language_issue_codes, normalize_number_token, number_tokens, source_number_tokens


def validate_manuscript_document(
    document: ManuscriptDocument,
    source: ManuscriptSourcePackage,
    *,
    output_readability_checks: dict[str, dict[str, Any]],
) -> tuple[ManuscriptValidationIssue, ...]:
    issues: list[ManuscriptValidationIssue] = []
    issues.extend(_unique_id_issues(document))
    issues.extend(_sentence_traceability_issues(document))
    issues.extend(_sentence_support_issues(document, source))
    issues.extend(_language_issues(document))
    issues.extend(_reviewer_constraint_issues(document, source))
    issues.extend(_publication_boundary_issues(document, source))
    issues.extend(_section_separation_issues(document, source))
    issues.extend(_figure_table_reference_issues(document, source))
    issues.extend(_deterministic_ordering_issues(document))
    issues.extend(_output_readability_issues(output_readability_checks))
    return tuple(issues)


def validation_summary(
    document: ManuscriptDocument | None,
    issues: Iterable[ManuscriptValidationIssue],
    *,
    output_readability_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    issue_tuple = tuple(issues)
    code_counts = Counter(issue.code for issue in issue_tuple)
    critical_count = sum(1 for issue in issue_tuple if issue.severity is ManuscriptIssueSeverity.CRITICAL)
    warning_count = sum(1 for issue in issue_tuple if issue.severity is ManuscriptIssueSeverity.WARNING)
    readability_failed = sum(1 for check in output_readability_checks.values() if not check.get("readable"))
    return {
        "validation_passed": critical_count == 0 and readability_failed == 0,
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "duplicate_sentence_id_count": code_counts["DUPLICATE_SENTENCE_ID"],
        "missing_traceability_count": code_counts["MISSING_TRACEABILITY"],
        "unsupported_quantitative_statement_count": code_counts["UNSUPPORTED_QUANTITATIVE_STATEMENT"],
        "unsupported_interpretive_statement_count": code_counts["UNSUPPORTED_INTERPRETIVE_STATEMENT"],
        "missing_limitation_source_count": code_counts["MISSING_LIMITATION_SOURCE"],
        "causal_language_issue_count": code_counts["CAUSAL_LANGUAGE_ISSUE"],
        "mechanism_language_issue_count": code_counts["MECHANISM_LANGUAGE_ISSUE"],
        "novelty_language_issue_count": code_counts["NOVELTY_LANGUAGE_ISSUE"],
        "external_validation_overclaim_count": code_counts["EXTERNAL_VALIDATION_OVERCLAIM"],
        "statistical_significance_issue_count": code_counts["STATISTICAL_SIGNIFICANCE_ISSUE"],
        "fabricated_number_issue_count": code_counts["FABRICATED_NUMBER_ISSUE"],
        "reviewer_constraint_issue_count": code_counts["REVIEWER_CONSTRAINT_ISSUE"],
        "publication_boundary_issue_count": code_counts["PUBLICATION_BOUNDARY_ISSUE"],
        "section_separation_issue_count": code_counts["SECTION_SEPARATION_ISSUE"],
        "missing_figure_reference_count": code_counts["MISSING_FIGURE_REFERENCE"],
        "missing_table_reference_count": code_counts["MISSING_TABLE_REFERENCE"],
        "deterministic_ordering_issue_count": code_counts["DETERMINISTIC_ORDERING_ISSUE"],
        "output_readability_checks": output_readability_checks,
        "structured_validation_issues": [issue.to_record() for issue in issue_tuple],
        "sentence_count": 0 if document is None else len(document.sentences),
    }


def readability_checks(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    checks = {}
    for name, path in sorted(paths.items()):
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    list(csv.DictReader(handle))
            elif path.suffix == ".docx":
                _read_docx(path)
            else:
                path.read_text(encoding="utf-8")
            checks[name] = {"readable": True, "path": str(path)}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error, zipfile.BadZipFile, ElementTree.ParseError, KeyError) as exc:
            checks[name] = {"readable": False, "path": str(path), "error": str(exc)}
    return checks


def _unique_id_issues(document: ManuscriptDocument) -> tuple[ManuscriptValidationIssue, ...]:
    issues: list[ManuscriptValidationIssue] = []
    for field_name, values, code in (
        ("section_id", [section.section_id for section in document.sections], "DUPLICATE_SECTION_ID"),
        ("paragraph_id", [paragraph.paragraph_id for paragraph in document.paragraphs], "DUPLICATE_PARAGRAPH_ID"),
        ("sentence_id", [sentence.sentence_id for sentence in document.sentences], "DUPLICATE_SENTENCE_ID"),
    ):
        for value, count in sorted(Counter(values).items()):
            if count > 1:
                issues.append(_issue(code, f"Duplicate {field_name}: {value}", field=field_name))
    return tuple(issues)


def _sentence_traceability_issues(document: ManuscriptDocument) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    for sentence in document.sentences:
        if sentence.metadata.get("placeholder"):
            continue
        if not sentence.source_ids or sentence.traceability_status is TraceabilityStatus.MISSING:
            issues.append(_sentence_issue("MISSING_TRACEABILITY", "Non-placeholder sentence lacks source traceability.", sentence, field="source_ids"))
    return tuple(issues)


def _sentence_support_issues(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> tuple[ManuscriptValidationIssue, ...]:
    issues: list[ManuscriptValidationIssue] = []
    source_numbers = _source_numbers_by_id(source)
    for sentence in document.sentences:
        if sentence.metadata.get("placeholder"):
            continue
        if sentence.sentence_type is SentenceType.RESULT and _has_quantitative_text(sentence.text):
            if not sentence.observation_ids:
                issues.append(_sentence_issue("UNSUPPORTED_QUANTITATIVE_STATEMENT", "Quantitative Results sentence lacks observation support.", sentence))
            available = set()
            for source_id in sentence.source_ids:
                available.update(source_numbers.get(source_id, set()))
            for token in _semantic_number_tokens(sentence):
                normalized = normalize_number_token(token)
                if token not in available and normalized not in available:
                    issues.append(
                        _sentence_issue(
                            "FABRICATED_NUMBER_ISSUE",
                            f"Number is not present in linked source records: {token}",
                            sentence,
                            field="text",
                        )
                    )
        if sentence.sentence_type is SentenceType.INTERPRETATION and not (
            sentence.claim_ids or sentence.interpretation_ids or sentence.hypothesis_ids
        ):
            issues.append(_sentence_issue("UNSUPPORTED_INTERPRETIVE_STATEMENT", "Interpretive sentence lacks claim, interpretation, or hypothesis support.", sentence))
        if sentence.sentence_type is SentenceType.LIMITATION and not (
            sentence.reviewer_finding_ids or _has_gap_node(sentence.reasoning_graph_node_ids)
        ):
            issues.append(_sentence_issue("MISSING_LIMITATION_SOURCE", "Limitation sentence lacks reviewer finding or evidence-gap support.", sentence))
    return tuple(issues)


def _language_issues(document: ManuscriptDocument) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    for sentence in document.sentences:
        if sentence.metadata.get("placeholder"):
            continue
        for code in language_issue_codes(sentence.text):
            if code == "LANGUAGE_STRENGTH_ISSUE":
                code = "PUBLICATION_BOUNDARY_ISSUE"
            issues.append(_sentence_issue(code, f"Sentence violates manuscript language policy: {code}", sentence, field="text"))
    return tuple(issues)


def _reviewer_constraint_issues(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    sentence_finding_ids = {finding_id for sentence in document.sentences for finding_id in sentence.reviewer_finding_ids}
    for finding in source.review_findings:
        finding_id = str(finding.get("finding_id"))
        if finding.get("blocking") is True and finding_id not in sentence_finding_ids:
            issues.append(
                ManuscriptValidationIssue(
                    code="REVIEWER_CONSTRAINT_ISSUE",
                    severity=ManuscriptIssueSeverity.CRITICAL,
                    message=f"Blocking reviewer finding is not represented in manuscript sentences: {finding_id}",
                    reviewer_finding_id=finding_id,
                    field="reviewer_finding_ids",
                )
            )
    flag_finding_ids = {flag.reviewer_finding_id for flag in document.unresolved_flags}
    for finding in source.review_findings:
        finding_id = str(finding.get("finding_id"))
        if finding.get("severity") in {"CRITICAL", "MAJOR"} and finding_id not in flag_finding_ids:
            issues.append(
                ManuscriptValidationIssue(
                    code="REVIEWER_CONSTRAINT_ISSUE",
                    severity=ManuscriptIssueSeverity.CRITICAL,
                    message=f"Major or blocking reviewer finding lacks a manuscript revision flag: {finding_id}",
                    reviewer_finding_id=finding_id,
                    field="unresolved_flags",
                )
            )
    return tuple(issues)


def _publication_boundary_issues(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    results_section_ids = {section.section_id for section in document.sections if section.section_type is SectionType.RESULTS}
    for sentence in document.sentences:
        if sentence.section_id not in results_section_ids:
            continue
        for claim_id in sentence.claim_ids:
            score = source.evidence_by_claim_id.get(claim_id, {})
            if score.get("publication_readiness") not in {"RESULTS_READY", "HIGH_CONFIDENCE_RESULTS_READY"}:
                issues.append(_sentence_issue("PUBLICATION_BOUNDARY_ISSUE", f"Claim not Results-ready appears in Results: {claim_id}", sentence, claim_id=claim_id))
    if source.reviewer_publication_assessment_document.get("definitive_generalization_allowed") is False:
        for sentence in document.sentences:
            text = sentence.text.lower()
            if "generalizes" in text and "cannot" not in text:
                issues.append(_sentence_issue("EXTERNAL_VALIDATION_OVERCLAIM", "Definitive generalization language is not allowed.", sentence))
    return tuple(issues)


def _section_separation_issues(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    discussion_claim_ids = set(str(item) for item in source.reviewer_publication_assessment_document.get("discussion_claim_ids", ()) or ())
    results_sentences = tuple(sentence for section in document.sections if section.section_type is SectionType.RESULTS for sentence in section.sentences)
    for sentence in results_sentences:
        if set(sentence.claim_ids) & discussion_claim_ids:
            issues.append(_sentence_issue("SECTION_SEPARATION_ISSUE", "Discussion-only claim appears in Results.", sentence))
        if sentence.sentence_type is SentenceType.INTERPRETATION:
            issues.append(_sentence_issue("SECTION_SEPARATION_ISSUE", "Interpretive sentence appears in Results.", sentence))
    return tuple(issues)


def _figure_table_reference_issues(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> tuple[ManuscriptValidationIssue, ...]:
    figure_ids = {str(row.get("figure_id")) for row in source.selected_figures if row.get("figure_id")}
    table_ids = {str(row.get("table_id")) for row in source.selected_tables if row.get("table_id")}
    issues = []
    for sentence in document.sentences:
        for figure_id in sentence.figure_ids:
            if figure_id not in figure_ids:
                issues.append(_sentence_issue("MISSING_FIGURE_REFERENCE", f"Figure reference is absent from selected_figures.csv: {figure_id}", sentence))
        for table_id in sentence.table_ids:
            if table_id not in table_ids:
                issues.append(_sentence_issue("MISSING_TABLE_REFERENCE", f"Table reference is absent from selected_tables.csv: {table_id}", sentence))
    return tuple(issues)


def _deterministic_ordering_issues(document: ManuscriptDocument) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    for section in document.sections:
        actual = [sentence.sentence_id for sentence in section.sentences]
        if actual != sorted(actual):
            issues.append(
                ManuscriptValidationIssue(
                    code="DETERMINISTIC_ORDERING_ISSUE",
                    severity=ManuscriptIssueSeverity.WARNING,
                    message=f"Sentences are not deterministically ordered in section: {section.section_id}",
                    section_id=section.section_id,
                    field="sentence_id",
                )
            )
    return tuple(issues)


def _source_numbers_by_id(source: ManuscriptSourcePackage) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for observation in source.observations:
        observation_id = str(observation.get("observation_id"))
        _add_source_numbers(mapping, observation_id, observation)
    for claim in source.claims:
        claim_id = str(claim.get("claim_id"))
        _add_source_numbers(mapping, claim_id, claim, source.evidence_by_claim_id.get(claim_id, {}))
    for row in source.selected_figures:
        figure_id = str(row.get("figure_id") or "")
        if figure_id:
            _add_source_numbers(mapping, figure_id, dict(row))
    for row in source.selected_tables:
        table_id = str(row.get("table_id") or "")
        if table_id:
            _add_source_numbers(mapping, table_id, dict(row))
    for finding in source.review_findings:
        finding_id = str(finding.get("finding_id"))
        _add_source_numbers(mapping, finding_id, finding)
    for node in source.graph_nodes:
        node_id = str(node.get("node_id"))
        _add_source_numbers(mapping, node_id, node)
    return mapping


def _add_source_numbers(mapping: dict[str, set[str]], source_id: str, *records: dict[str, Any]) -> None:
    mapping.setdefault(source_id, set()).update(source_number_tokens(records))


def _semantic_number_tokens(sentence: ManuscriptSentence) -> tuple[str, ...]:
    text = re.sub(r"\b(Figure|Table|SENT|REV|MSFLAG)\s*-?\s*\d+(?:\.\d+)?", "", sentence.text, flags=re.IGNORECASE)
    return number_tokens(text)


def _has_quantitative_text(text: str) -> bool:
    return bool(number_tokens(text))


def _has_gap_node(node_ids: tuple[str, ...]) -> bool:
    return any(node_id.startswith("GAP-") for node_id in node_ids)


def _read_docx(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for name in ("[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"):
            if name not in archive.namelist():
                raise KeyError(name)
        ElementTree.fromstring(archive.read("word/document.xml"))


def _output_readability_issues(checks: dict[str, dict[str, Any]]) -> tuple[ManuscriptValidationIssue, ...]:
    issues = []
    for filename, check in sorted(checks.items()):
        if not check.get("readable"):
            issues.append(
                ManuscriptValidationIssue(
                    code="OUTPUT_READABILITY_FAILURE",
                    severity=ManuscriptIssueSeverity.CRITICAL,
                    message=f"Manuscript output is not readable: {filename}",
                    source_file=filename,
                    field="output_readability_checks",
                )
            )
    return tuple(issues)


def _sentence_issue(
    code: str,
    message: str,
    sentence: ManuscriptSentence,
    *,
    claim_id: str | None = None,
    field: str | None = None,
) -> ManuscriptValidationIssue:
    return ManuscriptValidationIssue(
        code=code,
        severity=ManuscriptIssueSeverity.CRITICAL,
        message=message,
        sentence_id=sentence.sentence_id,
        section_id=sentence.section_id,
        claim_id=claim_id,
        field=field,
    )


def _issue(code: str, message: str, *, field: str | None = None) -> ManuscriptValidationIssue:
    return ManuscriptValidationIssue(
        code=code,
        severity=ManuscriptIssueSeverity.CRITICAL if code != "DETERMINISTIC_ORDERING_ISSUE" else ManuscriptIssueSeverity.WARNING,
        message=message,
        field=field,
    )
