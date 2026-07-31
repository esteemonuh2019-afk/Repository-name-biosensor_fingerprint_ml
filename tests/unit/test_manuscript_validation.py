from src.scientific_reasoning.manuscript import (
    ManuscriptDocument,
    ManuscriptSection,
    ManuscriptSentence,
    ManuscriptSourcePackage,
    SectionType,
    SentenceType,
)
from src.scientific_reasoning.manuscript.validators import validate_manuscript_document


def source() -> ManuscriptSourcePackage:
    return ManuscriptSourcePackage(
        observations_dir=".",
        interpretations_dir=".",
        hypotheses_dir=".",
        claims_dir=".",
        evidence_dir=".",
        review_dir=".",
        graph_dir=".",
        supervisor_dir=".",
        observations_document={"observations": [{"observation_id": "OBS-1", "statement": "The source listed 10 rows."}]},
        review_findings_document={"review_findings": [{"finding_id": "REV-1", "blocking": True, "severity": "MAJOR"}]},
        reviewer_publication_assessment_document={"discussion_claim_ids": [], "definitive_generalization_allowed": False},
    )


def document(sentence: ManuscriptSentence) -> ManuscriptDocument:
    section = ManuscriptSection(
        section_id=sentence.section_id,
        section_type=SectionType.RESULTS if sentence.sentence_type is SentenceType.RESULT else SectionType.LIMITATIONS,
        title="Section",
        sentences=(sentence,),
    )
    return ManuscriptDocument(
        manuscript_id="MS-1",
        title="Internal Scientific Manuscript Draft",
        document_status="REVISION_REQUIRED",
        sections=(section,),
    )


def test_missing_sentence_traceability_is_flagged() -> None:
    sent = ManuscriptSentence("SENT-RESULTS-0001", "The source listed 10 rows.", "RESULT", "SEC-RESULTS-0001")

    issues = validate_manuscript_document(document(sent), source(), output_readability_checks={})

    assert any(issue.code == "MISSING_TRACEABILITY" for issue in issues)


def test_fabricated_number_is_flagged() -> None:
    sent = ManuscriptSentence("SENT-RESULTS-0001", "The source listed 999 rows.", "RESULT", "SEC-RESULTS-0001", observation_ids=("OBS-1",))

    issues = validate_manuscript_document(document(sent), source(), output_readability_checks={})

    assert any(issue.code == "FABRICATED_NUMBER_ISSUE" for issue in issues)


def test_limitation_without_reviewer_or_gap_source_is_flagged() -> None:
    sent = ManuscriptSentence("SENT-LIMITATIONS-0001", "A limitation is present.", "LIMITATION", "SEC-LIMITATIONS-0001", claim_ids=("CLM-1",))

    issues = validate_manuscript_document(document(sent), source(), output_readability_checks={})

    assert any(issue.code == "MISSING_LIMITATION_SOURCE" for issue in issues)
