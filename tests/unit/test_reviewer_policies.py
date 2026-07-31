from src.scientific_reasoning.reviewer import OverallRecommendation, ReviewCategory, ReviewerType, Severity
from src.scientific_reasoning.reviewer.policies import build_finding, determine_recommendation, finding_id, revision_requirement_for


def test_finding_id_is_deterministic_per_reviewer_type() -> None:
    assert finding_id(ReviewerType.STATISTICAL, 3) == "REV-STATISTICAL-0003"


def test_revision_requirement_uses_rule_registry() -> None:
    requirement = revision_requirement_for(("REVIEW-VALIDATION-EXTERNAL-001",), Severity.MAJOR)

    assert "external-validation" in requirement


def test_recommendation_policy_ready_requires_no_material_findings_and_results_ready_claim() -> None:
    assert determine_recommendation(tuple(), has_results_ready_claim=True) is OverallRecommendation.READY_FOR_DRAFT_MANUSCRIPT


def test_recommendation_policy_major_for_single_blocking_major() -> None:
    finding = build_finding(
        reviewer_type=ReviewerType.VALIDATION,
        index=1,
        category=ReviewCategory.EXTERNAL_VALIDATION,
        title="External validation absent",
        finding_text="External validation is absent.",
        severity=Severity.MAJOR,
        rule_ids=("REVIEW-VALIDATION-EXTERNAL-001",),
        created_at="2026-07-31T00:00:00+00:00",
    )

    assert finding.blocking is True
    assert determine_recommendation((finding,), has_results_ready_claim=False) is OverallRecommendation.NEEDS_MAJOR_REVISION
