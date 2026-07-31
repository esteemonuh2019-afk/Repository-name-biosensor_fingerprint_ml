from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine, ClaimType, PublicationUse
from tests.integration.claim_fixture import create_claim_source_fixture


def test_claim_engine_generates_claims_from_synthetic_validated_packages(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    result = ClaimEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["validation_passed"] is True
    assert len(result.claims) == 8
    assert not [claim for claim in result.claims if claim.claim_type is ClaimType.WITHHELD]
    assert all(claim.supporting_hypothesis_ids for claim in result.claims)
    assert all(claim.supporting_interpretation_ids for claim in result.claims)
    assert all(claim.supporting_observation_ids for claim in result.claims)
    assert any(claim.competing_hypothesis_ids for claim in result.claims)
    assert any(claim.publication_use is PublicationUse.RESULTS_ELIGIBLE for claim in result.claims)


def test_competing_hypotheses_are_preserved_not_suppressed(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    claims = ClaimEngine(project_root=project_root, overwrite=True).run().claims
    chemical = next(claim for claim in claims if claim.claim_id == "CLM-CHEMICAL_DISCRIMINATION-0001")

    assert chemical.competing_hypothesis_ids == ("HYP-CHEMICAL_DISCRIMINATION-0002",)
    assert chemical.claim_status.value == "PARTIALLY_SUPPORTED"
