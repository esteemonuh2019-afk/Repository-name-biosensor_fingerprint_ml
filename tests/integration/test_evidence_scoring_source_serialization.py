from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from src.scientific_reasoning.evidence_scoring import EvidenceScoringEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def test_malformed_json_fails_safely_without_outputs(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    (project_root / "outputs" / "scientific_claims" / "claims.json").write_text("{malformed", encoding="utf-8")

    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    assert not result.output_paths
    assert any(issue.code == "UNREADABLE_SOURCE_FILE" for issue in result.validation_issues)


def test_non_overwrite_refuses_non_empty_output_directory(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    try:
        EvidenceScoringEngine(project_root=project_root, overwrite=False).run()
    except FileExistsError as exc:
        assert "Use --overwrite" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError")
