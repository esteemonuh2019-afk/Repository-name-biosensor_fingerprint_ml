from src.scientific_reasoning.manuscript import PublicationBoundary
from src.scientific_reasoning.manuscript.policies import boundary_for_claim, language_issue_codes


def test_boundary_for_results_ready_claim() -> None:
    assert boundary_for_claim({"publication_use": "RESULTS_ELIGIBLE"}, {"publication_readiness": "RESULTS_READY"}) is PublicationBoundary.RESULTS_ALLOWED


def test_boundary_for_discussion_only_claim() -> None:
    assert boundary_for_claim({"publication_use": "RESULTS_ELIGIBLE"}, {"publication_readiness": "DISCUSSION_READY"}) is PublicationBoundary.DISCUSSION_ONLY


def test_boundary_for_limitation_only_claim() -> None:
    assert boundary_for_claim({"publication_use": "LIMITATION_ONLY", "claim_type": "LIMITATION"}, {"publication_readiness": "LIMITATION_ONLY"}) is PublicationBoundary.LIMITATION_ONLY


def test_boundary_for_withheld_claim() -> None:
    assert boundary_for_claim({"claim_type": "WITHHELD"}, {"publication_readiness": "NOT_READY"}) is PublicationBoundary.WITHHELD


def test_language_policy_flags_forbidden_terms() -> None:
    assert "CAUSAL_LANGUAGE_ISSUE" in language_issue_codes("The response caused classification.")
    assert "MECHANISM_LANGUAGE_ISSUE" in language_issue_codes("The mechanism explains the response.")
    assert "NOVELTY_LANGUAGE_ISSUE" in language_issue_codes("This is a novel platform.")
    assert "STATISTICAL_SIGNIFICANCE_ISSUE" in language_issue_codes("The effect was statistically significant.")


def test_language_policy_allows_negated_external_validation_boundary() -> None:
    assert "EXTERNAL_VALIDATION_OVERCLAIM" not in language_issue_codes("The model cannot yet generalize to independently labelled unknown samples.")
