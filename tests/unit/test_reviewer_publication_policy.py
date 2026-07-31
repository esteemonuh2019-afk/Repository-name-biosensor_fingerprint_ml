from src.scientific_reasoning.reviewer import OverallRecommendation, ReviewCategory, ReviewerType, Severity
from src.scientific_reasoning.reviewer.policies import build_finding, determine_recommendation


def finding(severity: Severity, *, blocking: bool = False):
    return build_finding(
        reviewer_type=ReviewerType.EVIDENCE,
        index=1,
        category=ReviewCategory.CLAIM_SUPPORT,
        title="Finding",
        finding_text="A finding is present.",
        severity=severity,
        blocking=blocking,
        rule_ids=("REVIEW-EVIDENCE-UNCERTAINTY-001",),
        created_at="2026-07-31T00:00:00+00:00",
    )


def test_critical_policy_returns_internal_review_only() -> None:
    assert determine_recommendation((finding(Severity.CRITICAL, blocking=True),), has_results_ready_claim=True) is OverallRecommendation.INTERNAL_REVIEW_ONLY


def test_major_policy_returns_major_revision_even_without_blocker() -> None:
    assert determine_recommendation((finding(Severity.MAJOR),), has_results_ready_claim=True) is OverallRecommendation.NEEDS_MAJOR_REVISION


def test_moderate_policy_returns_moderate_revision() -> None:
    assert determine_recommendation((finding(Severity.MODERATE),), has_results_ready_claim=True) is OverallRecommendation.NEEDS_MODERATE_REVISION


def test_minor_policy_returns_minor_revision() -> None:
    assert determine_recommendation((finding(Severity.MINOR),), has_results_ready_claim=True) is OverallRecommendation.NEEDS_MINOR_REVISION
