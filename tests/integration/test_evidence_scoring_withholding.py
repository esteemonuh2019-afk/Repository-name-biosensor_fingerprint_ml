import json
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from src.scientific_reasoning.evidence_scoring import EvidenceScoringEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def create_claim_outputs(tmp_path: Path) -> Path:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    return project_root


def test_failed_source_validation_withholds_records_and_fails_validation(tmp_path: Path) -> None:
    project_root = create_claim_outputs(tmp_path)
    validation_path = project_root / "outputs" / "scientific_claims" / "claim_validation.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["validation_passed"] = False
    payload["critical_issue_count"] = 1
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["critical_issue_count"] > 0
    assert all(record.is_withheld for record in result.records)
    assert all(record.normalized_score == 0 for record in result.records)


def test_missing_graph_dependency_withholds_affected_claim(tmp_path: Path) -> None:
    project_root = create_claim_outputs(tmp_path)
    claims_path = project_root / "outputs" / "scientific_claims" / "claims.json"
    payload = json.loads(claims_path.read_text(encoding="utf-8"))
    payload["claims"][0]["reasoning_graph_node_ids"].append("MISSING-GRAPH-NODE")
    claims_path.write_text(json.dumps(payload), encoding="utf-8")

    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["critical_issue_count"] > 0
    assert any(record.is_withheld for record in result.records)
    assert any("critical reasoning-graph dependency is missing" in record.withholding_reasons for record in result.records)


def test_duplicate_claim_ids_are_critical(tmp_path: Path) -> None:
    project_root = create_claim_outputs(tmp_path)
    claims_path = project_root / "outputs" / "scientific_claims" / "claims.json"
    payload = json.loads(claims_path.read_text(encoding="utf-8"))
    payload["claims"][1]["claim_id"] = payload["claims"][0]["claim_id"]
    claims_path.write_text(json.dumps(payload), encoding="utf-8")

    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    assert any(issue.code == "DUPLICATE_CLAIM_ID" for issue in result.validation_issues)
    assert result.metadata["critical_issue_count"] > 0
