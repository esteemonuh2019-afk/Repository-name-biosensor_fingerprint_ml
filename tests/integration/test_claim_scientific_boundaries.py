from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine, ClaimType
from src.scientific_reasoning.claim.policies import (
    causal_overclaim_terms,
    external_validation_overclaim_terms,
    mechanistic_overclaim_terms,
    novelty_overclaim_terms,
    recommendation_terms,
)
from tests.integration.claim_fixture import create_claim_source_fixture


def test_generated_claims_avoid_forbidden_scientific_overclaims(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    result = ClaimEngine(project_root=project_root, overwrite=True).run()

    for claim in result.claims:
        assert not causal_overclaim_terms(claim.claim_text)
        assert not mechanistic_overclaim_terms(claim.claim_text)
        assert not novelty_overclaim_terms(claim.claim_text)
        assert not external_validation_overclaim_terms(claim.claim_text)
        assert not recommendation_terms(claim.claim_text)
        assert "clinical" not in claim.claim_text.lower()
        assert "regulatory" not in claim.claim_text.lower()


def test_critically_invalid_hypothesis_package_withholds_claims(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path, invalid_hypothesis_validation=True)
    result = ClaimEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["critical_issue_count"] > 0
    assert all(claim.claim_type is ClaimType.WITHHELD for claim in result.claims)
    assert any(issue.code == "SOURCE_VALIDATION_FAILURE" for issue in result.validation_issues)


def test_critically_invalid_reasoning_graph_withholds_claims(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path, invalid_graph_validation=True)
    result = ClaimEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["critical_issue_count"] > 0
    assert all(claim.claim_type is ClaimType.WITHHELD for claim in result.claims)
    assert any(issue.code == "SOURCE_VALIDATION_FAILURE" for issue in result.validation_issues)
