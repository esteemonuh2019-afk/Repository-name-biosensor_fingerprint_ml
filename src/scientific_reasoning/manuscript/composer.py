"""Conservative manuscript composition from validated BSIP artifacts."""

from __future__ import annotations

from typing import Any, Iterable

from .enums import (
    DocumentStatus,
    PublicationBoundary,
    ResolutionStatus,
    SectionType,
    SentenceStatus,
    SentenceType,
)
from .models import (
    MANUSCRIPT_RULE_VERSION,
    MANUSCRIPT_SOFTWARE_VERSION,
    Caption,
    ManuscriptDocument,
    ManuscriptParagraph,
    ManuscriptSection,
    ManuscriptSentence,
    ManuscriptSourcePackage,
    RevisionFlag,
)
from .policies import (
    DISCUSSION_CATEGORY_ORDER,
    PLACEHOLDER_TEXT,
    RESULT_CATEGORY_ORDER,
    boundary_for_claim,
    conclusion_sentences,
    document_status_from_review,
    evidence_gap_sentence,
    extract_json_list,
    language_issue_codes,
    limitation_sentence_from_finding,
    normalize_sentence,
    paragraph_id,
    qualified_claim_sentence,
    section_id,
    sentence_id,
    sentence_status_for_boundary,
    traceability_status_for,
)


DEFAULT_TITLE = "Internal Scientific Manuscript Draft: Generated under BSIP Reviewer Constraints"


def compose_manuscript(
    source: ManuscriptSourcePackage,
    *,
    title: str = DEFAULT_TITLE,
    author: str | None = None,
    created_at: str,
    software_version: str = MANUSCRIPT_SOFTWARE_VERSION,
) -> ManuscriptDocument:
    assessment = dict(source.reviewer_publication_assessment_document)
    document_status = document_status_from_review(assessment, dict(source.reviewer_summary_document))
    sections: list[ManuscriptSection] = []
    sections.append(_title_section(title, author=author, source=source))
    sections.extend(_placeholder_sections())
    results = _results_section(source)
    if results.sentences:
        sections.append(results)
    discussion = _discussion_section(source)
    if discussion.sentences:
        sections.append(discussion)
    limitations = _limitations_section(source)
    if limitations.sentences:
        sections.append(limitations)
    conclusion = _conclusion_section(source)
    if conclusion.sentences:
        sections.append(conclusion)
    figure_captions, figure_section = _figure_caption_section(source)
    table_captions, table_section = _table_caption_section(source)
    if figure_section.sentences:
        sections.append(figure_section)
    if table_section.sentences:
        sections.append(table_section)
    flags = _revision_flags(source, tuple(sentence for section in sections for sentence in section.sentences))
    revision_section = _revision_notes_section(flags)
    if revision_section.sentences:
        sections.append(revision_section)
    return ManuscriptDocument(
        manuscript_id="MS-BSIP-4.2.0-0001",
        title=title,
        document_status=document_status,
        sections=tuple(sections),
        figure_captions=figure_captions,
        table_captions=table_captions,
        unresolved_flags=flags,
        source_schema_versions=_source_schema_versions(source),
        software_version=software_version,
        created_at=created_at,
        metadata={
            "author": author,
            "manuscript_rule_version": MANUSCRIPT_RULE_VERSION,
            "overall_reviewer_recommendation": assessment.get("overall_recommendation"),
            "manuscript_drafting_allowed": assessment.get("manuscript_drafting_allowed"),
            "definitive_generalization_allowed": assessment.get("definitive_generalization_allowed"),
        },
    )


def _title_section(title: str, *, author: str | None, source: ManuscriptSourcePackage) -> ManuscriptSection:
    text = normalize_sentence(
        f"{title}. This internal draft is generated from validated BSIP artifacts and retains Reviewer Engine constraints"
    )
    sentence = ManuscriptSentence(
        sentence_id=sentence_id(SectionType.TITLE, 1),
        text=text,
        sentence_type=SentenceType.CONTEXT,
        section_id=section_id(SectionType.TITLE),
        reviewer_finding_ids=tuple(str(finding.get("finding_id")) for finding in source.review_findings if finding.get("blocking")),
        traceability_status=traceability_status_for(source_ids=source.review_findings),
        metadata={"author": author},
    )
    return _section_from_sentences(
        SectionType.TITLE,
        "Title Page",
        (sentence,),
        publication_boundary=PublicationBoundary.INTERNAL_ONLY,
    )


def _placeholder_sections() -> tuple[ManuscriptSection, ...]:
    sections = []
    for section_type in (
        SectionType.ABSTRACT_PLACEHOLDER,
        SectionType.INTRODUCTION_PLACEHOLDER,
        SectionType.METHODS_PLACEHOLDER,
    ):
        sentence = ManuscriptSentence(
            sentence_id=sentence_id(section_type, 1),
            text=PLACEHOLDER_TEXT[section_type],
            sentence_type=SentenceType.CONTEXT,
            section_id=section_id(section_type),
            traceability_status=traceability_status_for(source_ids=(), placeholder=True),
            metadata={"placeholder": True},
        )
        sections.append(
            _section_from_sentences(
                section_type,
                section_type.value.replace("_", " ").title(),
                (sentence,),
                publication_boundary=PublicationBoundary.INTERNAL_ONLY,
            )
        )
    return tuple(sections)


def _results_section(source: ManuscriptSourcePackage) -> ManuscriptSection:
    sentences = []
    by_category = {str(observation.get("category")): observation for observation in source.observations}
    index = 1
    for category, heading in RESULT_CATEGORY_ORDER:
        observation = by_category.get(category)
        if not observation:
            continue
        text = normalize_sentence(f"{heading}: {observation.get('statement')}")
        obs_id = str(observation.get("observation_id"))
        metrics = tuple(str(metric.get("metric_name")) for metric in observation.get("supporting_metrics", ()) or ())
        limitations = tuple(str(item) for item in observation.get("limitations", ()) or ())
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.RESULTS, index),
                text=text,
                sentence_type=SentenceType.RESULT,
                section_id=section_id(SectionType.RESULTS),
                observation_ids=(obs_id,),
                reasoning_graph_node_ids=_graph_ids_for_observation(source, obs_id),
                traceability_status=traceability_status_for(source_ids=(obs_id,)),
                language_policy_status=_language_status(text),
                limitations=limitations,
                metadata={
                    "observation_category": category,
                    "metric_names": list(metrics),
                    "supporting_files": list(str(item) for item in observation.get("supporting_files", ()) or ()),
                },
            )
        )
        index += 1
    boundary_note_ids = tuple(str(finding.get("finding_id")) for finding in source.review_findings if finding.get("category") == "EXTERNAL_VALIDATION")
    if boundary_note_ids:
        text = "Results are reported as descriptive internal-evaluation outputs because the reviewer assessment did not identify results-ready claim IDs."
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.RESULTS, index),
                text=text,
                sentence_type=SentenceType.CONTEXT,
                section_id=section_id(SectionType.RESULTS),
                reviewer_finding_ids=boundary_note_ids,
                traceability_status=traceability_status_for(source_ids=boundary_note_ids),
                language_policy_status=_language_status(text),
                limitations=("No interpretive claim is presented as a definitive Results conclusion.",),
            )
        )
    return _section_from_sentences(
        SectionType.RESULTS,
        "Results",
        tuple(sentences),
        publication_boundary=PublicationBoundary.RESULTS_ALLOWED,
    )


def _discussion_section(source: ManuscriptSourcePackage) -> ManuscriptSection:
    assessment = source.reviewer_publication_assessment_document
    discussion_claim_ids = set(str(item) for item in assessment.get("discussion_claim_ids", ()) or ())
    eligible_claims = [
        claim
        for claim in source.claims
        if str(claim.get("claim_id")) in discussion_claim_ids
        and boundary_for_claim(claim, source.evidence_by_claim_id.get(str(claim.get("claim_id")))) is PublicationBoundary.DISCUSSION_ONLY
    ]
    order = {category: index for index, category in enumerate(DISCUSSION_CATEGORY_ORDER)}
    eligible_claims = sorted(eligible_claims, key=lambda claim: (order.get(str(claim.get("category")), 999), str(claim.get("claim_id"))))
    sentences = []
    for index, claim in enumerate(eligible_claims, start=1):
        claim_id = str(claim.get("claim_id"))
        score = source.evidence_by_claim_id.get(claim_id, {})
        text = qualified_claim_sentence(claim, score)
        reviewer_ids = _reviewer_ids_for_claim(source, claim_id)
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.DISCUSSION, index),
                text=text,
                sentence_type=SentenceType.INTERPRETATION,
                section_id=section_id(SectionType.DISCUSSION),
                claim_ids=(claim_id,),
                interpretation_ids=tuple(str(item) for item in claim.get("supporting_interpretation_ids", ()) or ()),
                hypothesis_ids=tuple(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ()),
                observation_ids=tuple(str(item) for item in claim.get("supporting_observation_ids", ()) or ()),
                evidence_score_ids=(claim_id,),
                reviewer_finding_ids=reviewer_ids,
                reasoning_graph_node_ids=tuple(str(item) for item in claim.get("reasoning_graph_node_ids", ()) or ()),
                traceability_status=traceability_status_for(source_ids=(claim_id, *reviewer_ids)),
                language_policy_status=_language_status(text),
                limitations=tuple(str(item) for item in claim.get("limitations", ()) or ()),
                metadata={
                    "publication_boundary": PublicationBoundary.DISCUSSION_ONLY.value,
                    "publication_readiness": score.get("publication_readiness"),
                    "uncertainty_level": score.get("uncertainty_level"),
                },
            )
        )
    return _section_from_sentences(
        SectionType.DISCUSSION,
        "Discussion",
        tuple(sentences),
        publication_boundary=PublicationBoundary.DISCUSSION_ONLY,
    )


def _limitations_section(source: ManuscriptSourcePackage) -> ManuscriptSection:
    sentences: list[ManuscriptSentence] = []
    index = 1
    material_findings = tuple(
        finding
        for finding in source.review_findings
        if finding.get("severity") in {"CRITICAL", "MAJOR", "MODERATE"} or finding.get("blocking") is True
    )
    for finding in material_findings:
        text = limitation_sentence_from_finding(finding)
        finding_id = str(finding.get("finding_id"))
        claim_ids = tuple(str(item) for item in finding.get("affected_claim_ids", ()) or ())
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.LIMITATIONS, index),
                text=text,
                sentence_type=SentenceType.LIMITATION,
                section_id=section_id(SectionType.LIMITATIONS),
                claim_ids=claim_ids,
                evidence_score_ids=tuple(str(item) for item in finding.get("evidence_score_ids", ()) or ()),
                reviewer_finding_ids=(finding_id,),
                reasoning_graph_node_ids=tuple(str(item) for item in finding.get("reasoning_graph_node_ids", ()) or ()),
                traceability_status=traceability_status_for(source_ids=(finding_id,)),
                language_policy_status=_language_status(text),
                limitations=tuple(str(item) for item in finding.get("limitations", ()) or ()),
                metadata={"reviewer_severity": finding.get("severity"), "blocking": finding.get("blocking")},
            )
        )
        index += 1
    figure_findings = tuple(finding for finding in source.review_findings if finding.get("reviewer_type") == "FIGURE")
    for finding in figure_findings:
        text = "Claim-level figure and table links remain incomplete in the selected supervisor metadata."
        finding_id = str(finding.get("finding_id"))
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.LIMITATIONS, index),
                text=text,
                sentence_type=SentenceType.LIMITATION,
                section_id=section_id(SectionType.LIMITATIONS),
                reviewer_finding_ids=(finding_id,),
                figure_ids=tuple(str(item) for item in finding.get("affected_figure_ids", ()) or ()),
                table_ids=tuple(str(item) for item in finding.get("affected_table_ids", ()) or ()),
                traceability_status=traceability_status_for(source_ids=(finding_id,)),
                language_policy_status=_language_status(text),
                limitations=("Figure and table support is limited to selected metadata and explicit claim links.",),
            )
        )
        index += 1
    limitation_claims = tuple(
        claim
        for claim in source.claims
        if boundary_for_claim(claim, source.evidence_by_claim_id.get(str(claim.get("claim_id")))) is PublicationBoundary.LIMITATION_ONLY
    )
    for claim in limitation_claims:
        claim_id = str(claim.get("claim_id"))
        text = normalize_sentence(str(claim.get("claim_text") or f"Limitation claim {claim_id} is recorded."))
        reviewer_ids = _reviewer_ids_for_claim(source, claim_id)
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.LIMITATIONS, index),
                text=text,
                sentence_type=SentenceType.LIMITATION,
                section_id=section_id(SectionType.LIMITATIONS),
                claim_ids=(claim_id,),
                evidence_score_ids=(claim_id,),
                reviewer_finding_ids=reviewer_ids,
                observation_ids=tuple(str(item) for item in claim.get("supporting_observation_ids", ()) or ()),
                interpretation_ids=tuple(str(item) for item in claim.get("supporting_interpretation_ids", ()) or ()),
                hypothesis_ids=tuple(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ()),
                reasoning_graph_node_ids=tuple(str(item) for item in claim.get("reasoning_graph_node_ids", ()) or ()),
                traceability_status=traceability_status_for(source_ids=(claim_id, *reviewer_ids)),
                language_policy_status=_language_status(text),
                limitations=tuple(str(item) for item in claim.get("limitations", ()) or ()),
                metadata={"publication_boundary": PublicationBoundary.LIMITATION_ONLY.value},
            )
        )
        index += 1
    gap_nodes = tuple(node for node in source.graph_nodes if node.get("node_type") == "EvidenceGap")
    unique_gap_texts: set[str] = set()
    for node in gap_nodes:
        text = evidence_gap_sentence(node)
        normalized_key = text.lower()
        if normalized_key in unique_gap_texts:
            continue
        unique_gap_texts.add(normalized_key)
        node_id = str(node.get("node_id"))
        hyp_id = str((node.get("attributes") or {}).get("hypothesis_id") or node.get("source_id") or "")
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.LIMITATIONS, index),
                text=text,
                sentence_type=SentenceType.LIMITATION,
                section_id=section_id(SectionType.LIMITATIONS),
                hypothesis_ids=(hyp_id,) if hyp_id else tuple(),
                reasoning_graph_node_ids=(node_id,),
                traceability_status=traceability_status_for(source_ids=(node_id,)),
                language_policy_status=_language_status(text),
                metadata={"evidence_gap_node": node_id},
            )
        )
        index += 1
    return _section_from_sentences(
        SectionType.LIMITATIONS,
        "Limitations",
        tuple(sentences),
        publication_boundary=PublicationBoundary.LIMITATION_ONLY,
    )


def _conclusion_section(source: ManuscriptSourcePackage) -> ManuscriptSection:
    reviewer_ids = tuple(str(finding.get("finding_id")) for finding in source.review_findings if finding.get("blocking"))
    observation_ids = tuple(str(obs.get("observation_id")) for obs in source.observations if obs.get("category") in {"CLASSIFICATION", "REGRESSION", "FEATURE_ENGINEERING", "STRAIN_CONTRIBUTION"})
    sentences = []
    for index, text in enumerate(conclusion_sentences(dict(source.reviewer_publication_assessment_document)), start=1):
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.CONCLUSION, index),
                text=normalize_sentence(text),
                sentence_type=SentenceType.CONCLUSION,
                section_id=section_id(SectionType.CONCLUSION),
                observation_ids=observation_ids,
                reviewer_finding_ids=reviewer_ids,
                traceability_status=traceability_status_for(source_ids=(*observation_ids, *reviewer_ids)),
                language_policy_status=_language_status(text),
                limitations=("Conclusions are restricted to the current dataset and analysis design.",),
            )
        )
    return _section_from_sentences(
        SectionType.CONCLUSION,
        "Conclusion",
        tuple(sentences),
        publication_boundary=PublicationBoundary.DISCUSSION_ONLY,
    )


def _figure_caption_section(source: ManuscriptSourcePackage) -> tuple[tuple[Caption, ...], ManuscriptSection]:
    captions = []
    sentences = []
    for index, row in enumerate(source.selected_figures, start=1):
        figure_id = _row_id(row, "figure_id")
        if not figure_id:
            continue
        title = str(row.get("title") or figure_id)
        source_run = str(row.get("source_run") or "not recorded")
        source_file = str(row.get("source_file") or row.get("output_file") or "")
        text = normalize_sentence(
            f"Figure {index}. {title} ({figure_id}) is a selected figure from analysis stage {source_run}; source file {source_file or 'not recorded'}"
        )
        sent_id = sentence_id(SectionType.FIGURE_CAPTIONS, index)
        caption = Caption(
            caption_id=f"FIGCAP-{index:04d}",
            caption_type=SentenceType.FIGURE_REFERENCE,
            title=title,
            text=text,
            source_id=figure_id,
            source_file=source_file,
            source_run=source_run,
            sentence_id=sent_id,
            claim_ids=_row_claim_ids(row),
            limitations=("Caption is generated only from selected figure metadata.",),
        )
        captions.append(caption)
        sentences.append(
            ManuscriptSentence(
                sentence_id=sent_id,
                text=text,
                sentence_type=SentenceType.FIGURE_REFERENCE,
                section_id=section_id(SectionType.FIGURE_CAPTIONS),
                claim_ids=caption.claim_ids,
                figure_ids=(figure_id,),
                traceability_status=traceability_status_for(source_ids=(figure_id,)),
                language_policy_status=_language_status(text),
                limitations=caption.limitations,
                metadata={"source_file": source_file, "source_run": source_run},
            )
        )
    return tuple(captions), _section_from_sentences(SectionType.FIGURE_CAPTIONS, "Figure Captions", tuple(sentences), publication_boundary=PublicationBoundary.INTERNAL_ONLY)


def _table_caption_section(source: ManuscriptSourcePackage) -> tuple[tuple[Caption, ...], ManuscriptSection]:
    captions = []
    sentences = []
    for index, row in enumerate(source.selected_tables, start=1):
        table_id = _row_id(row, "table_id")
        if not table_id:
            continue
        title = str(row.get("title") or table_id)
        source_file = str(row.get("source_file") or "")
        row_count = str(row.get("row_count") or "not recorded")
        text = normalize_sentence(f"Table {index}. {title} ({table_id}) is a selected table with row_count={row_count}; source file {source_file or 'not recorded'}")
        sent_id = sentence_id(SectionType.TABLE_CAPTIONS, index)
        caption = Caption(
            caption_id=f"TBLCAP-{index:04d}",
            caption_type=SentenceType.TABLE_REFERENCE,
            title=title,
            text=text,
            source_id=table_id,
            source_file=source_file,
            sentence_id=sent_id,
            claim_ids=_row_claim_ids(row),
            limitations=("Caption is generated only from selected table metadata.",),
        )
        captions.append(caption)
        sentences.append(
            ManuscriptSentence(
                sentence_id=sent_id,
                text=text,
                sentence_type=SentenceType.TABLE_REFERENCE,
                section_id=section_id(SectionType.TABLE_CAPTIONS),
                claim_ids=caption.claim_ids,
                table_ids=(table_id,),
                traceability_status=traceability_status_for(source_ids=(table_id,)),
                language_policy_status=_language_status(text),
                limitations=caption.limitations,
                metadata={"source_file": source_file, "row_count": row_count},
            )
        )
    return tuple(captions), _section_from_sentences(SectionType.TABLE_CAPTIONS, "Table Captions", tuple(sentences), publication_boundary=PublicationBoundary.INTERNAL_ONLY)


def _revision_flags(source: ManuscriptSourcePackage, sentences: tuple[ManuscriptSentence, ...]) -> tuple[RevisionFlag, ...]:
    flags = []
    sentence_lookup = {sentence.sentence_id: sentence for sentence in sentences}
    for index, finding in enumerate(
        (item for item in source.review_findings if item.get("severity") in {"CRITICAL", "MAJOR"} or item.get("blocking") is True),
        start=1,
    ):
        finding_id = str(finding.get("finding_id"))
        linked_sentences = tuple(
            sentence_id
            for sentence_id, sentence in sentence_lookup.items()
            if finding_id in sentence.reviewer_finding_ids
        )
        if finding.get("category") == "PUBLICATION_READINESS":
            action = "Restricted downgraded Results-eligible claims to Discussion and Limitations."
            resolution = ResolutionStatus.MOVED_TO_DISCUSSION.value
            affected_section = "DISCUSSION"
        elif finding.get("category") in {"EXTERNAL_VALIDATION", "COMPETING_EXPLANATIONS"}:
            action = "Preserved the reviewer finding as a limitation and qualifying constraint."
            resolution = ResolutionStatus.MOVED_TO_LIMITATIONS.value
            affected_section = "LIMITATIONS"
        else:
            action = "Applied reviewer qualification in manuscript text."
            resolution = ResolutionStatus.APPLIED_AS_QUALIFICATION.value
            affected_section = "LIMITATIONS"
        flags.append(
            RevisionFlag(
                flag_id=f"MSFLAG-{index:04d}",
                reviewer_finding_id=finding_id,
                severity=str(finding.get("severity")),
                blocking=bool(finding.get("blocking")),
                affected_section=affected_section,
                affected_sentence_ids=linked_sentences,
                applied_action=action,
                resolution_status=resolution,
                author_action_required=True,
                notes=str(finding.get("revision_requirement") or "Author review is required."),
            )
        )
    return tuple(flags)


def _revision_notes_section(flags: tuple[RevisionFlag, ...]) -> ManuscriptSection:
    sentences = []
    for index, flag in enumerate(flags, start=1):
        text = normalize_sentence(
            f"Revision flag {flag.flag_id} preserves reviewer finding {flag.reviewer_finding_id} with status {flag.resolution_status}"
        )
        sentences.append(
            ManuscriptSentence(
                sentence_id=sentence_id(SectionType.REVISION_NOTES, index),
                text=text,
                sentence_type=SentenceType.CONTEXT,
                section_id=section_id(SectionType.REVISION_NOTES),
                reviewer_finding_ids=(flag.reviewer_finding_id,),
                traceability_status=traceability_status_for(source_ids=(flag.reviewer_finding_id,)),
                language_policy_status=_language_status(text),
                metadata={"flag_id": flag.flag_id},
            )
        )
    return _section_from_sentences(
        SectionType.REVISION_NOTES,
        "Revision Notes",
        tuple(sentences),
        publication_boundary=PublicationBoundary.INTERNAL_ONLY,
    )


def _section_from_sentences(
    section_type: SectionType,
    title: str,
    sentences: tuple[ManuscriptSentence, ...],
    *,
    publication_boundary: PublicationBoundary,
) -> ManuscriptSection:
    paragraphs = tuple(
        ManuscriptParagraph(
            paragraph_id=paragraph_id(section_type, index),
            text=sentence.text,
            sentence_ids=(sentence.sentence_id,),
            source_ids=sentence.source_ids,
            claim_ids=sentence.claim_ids,
            reviewer_finding_ids=sentence.reviewer_finding_ids,
            confidence="GUARDED" if sentence.reviewer_finding_ids else "HIGH",
            publication_use=publication_boundary.value,
            status=SentenceStatus.QUALIFIED if sentence.reviewer_finding_ids else SentenceStatus.ALLOWED,
        )
        for index, sentence in enumerate(sentences, start=1)
    )
    return ManuscriptSection(
        section_id=section_id(section_type),
        section_type=section_type,
        title=title,
        paragraphs=paragraphs,
        sentences=sentences,
        source_claim_ids=_combined(sentences, "claim_ids"),
        source_observation_ids=_combined(sentences, "observation_ids"),
        source_interpretation_ids=_combined(sentences, "interpretation_ids"),
        source_hypothesis_ids=_combined(sentences, "hypothesis_ids"),
        source_figure_ids=_combined(sentences, "figure_ids"),
        source_table_ids=_combined(sentences, "table_ids"),
        reviewer_finding_ids=_combined(sentences, "reviewer_finding_ids"),
        publication_boundary=publication_boundary,
        limitations=tuple(sorted({item for sentence in sentences for item in sentence.limitations})),
        status=SentenceStatus.QUALIFIED if any(sentence.reviewer_finding_ids for sentence in sentences) else SentenceStatus.ALLOWED,
    )


def _combined(sentences: Iterable[ManuscriptSentence], field: str) -> tuple[str, ...]:
    return tuple(sorted({item for sentence in sentences for item in getattr(sentence, field)}))


def _source_schema_versions(source: ManuscriptSourcePackage) -> dict[str, Any]:
    return {
        "observations": source.observations_document.get("schema_version"),
        "interpretations": source.interpretations_document.get("schema_version"),
        "hypotheses": source.hypotheses_document.get("schema_version"),
        "claims": source.claims_document.get("schema_version"),
        "evidence_scoring": source.evidence_scores_document.get("schema_version"),
        "reviewer": source.review_findings_document.get("schema_version"),
        "reasoning_graph": source.graph_document.get("schema_version"),
    }


def _language_status(text: str) -> str:
    issues = language_issue_codes(text)
    return "PASSED" if not issues else ",".join(issues)


def _graph_ids_for_observation(source: ManuscriptSourcePackage, observation_id: str) -> tuple[str, ...]:
    if observation_id in source.graph_node_by_id:
        return (observation_id,)
    return tuple()


def _reviewer_ids_for_claim(source: ManuscriptSourcePackage, claim_id: str) -> tuple[str, ...]:
    ids = set()
    for row in source.reviewer_claim_rows:
        if str(row.get("claim_id")) == claim_id:
            ids.update(extract_json_list(row.get("review_finding_ids")))
            ids.update(extract_json_list(row.get("blocking_finding_ids")))
    for finding in source.review_findings:
        if claim_id in tuple(str(item) for item in finding.get("affected_claim_ids", ()) or ()):
            ids.add(str(finding.get("finding_id")))
    return tuple(sorted(ids))


def _row_id(row: dict[str, str], preferred_key: str) -> str:
    if row.get(preferred_key):
        return str(row[preferred_key])
    for key in sorted(row):
        if key.endswith("_id") and row.get(key):
            return str(row[key])
    return ""


def _row_claim_ids(row: dict[str, str]) -> tuple[str, ...]:
    ids = set()
    for key in ("claim_id", "claim_ids", "affected_claim_ids", "linked_claim_ids"):
        ids.update(item for item in extract_json_list(row.get(key)) if item.startswith("CLM-"))
    return tuple(sorted(ids))
