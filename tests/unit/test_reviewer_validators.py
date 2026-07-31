from src.scientific_reasoning.reviewer import ReviewCategory, ReviewContext, ReviewerType, Severity
from src.scientific_reasoning.reviewer.policies import build_finding
from src.scientific_reasoning.reviewer.validators import validate_review_findings, validate_source_documents


def context() -> ReviewContext:
    return ReviewContext(
        claims_document={"schema_version": "BSIP-3.2.0", "claims": [{"claim_id": "CLM-1"}]},
        evidence_scores_document={"schema_version": "BSIP-4.0.0", "evidence_scores": [{"claim_id": "CLM-1"}]},
        graph_document={"schema_version": "BSIP-3.1.0", "nodes": [{"node_id": "OBS-1"}]},
    )


def test_missing_claim_reference_is_flagged() -> None:
    finding = build_finding(
        reviewer_type=ReviewerType.EVIDENCE,
        index=1,
        category=ReviewCategory.CLAIM_SUPPORT,
        title="Missing claim",
        finding_text="A missing claim is referenced.",
        severity=Severity.MODERATE,
        affected_claim_ids=("CLM-MISSING",),
        rule_ids=("REVIEW-EVIDENCE-UNCERTAINTY-001",),
        created_at="2026-07-31T00:00:00+00:00",
    )

    issues = validate_review_findings((finding,), context=context())

    assert any(issue.code == "MISSING_CLAIM_REFERENCE" for issue in issues)


def test_source_validation_accepts_supervisor_passed_key() -> None:
    issues = validate_source_documents(
        claims_document={"schema_version": "BSIP-3.2.0", "claims": [{"claim_id": "CLM-1"}]},
        claim_validation_document={"validation_passed": True, "critical_issue_count": 0},
        evidence_scores_document={"schema_version": "BSIP-4.0.0", "evidence_scores": [{"claim_id": "CLM-1"}]},
        evidence_validation_document={"validation_passed": True, "critical_issue_count": 0},
        graph_document={"schema_version": "BSIP-3.1.0", "nodes": []},
        graph_validation_document={"validation_passed": True, "critical_issue_count": 0},
        supervisor_validation_document={"passed": True},
    )

    assert not issues
