import json
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from src.scientific_reasoning.evidence_scoring import EvidenceScoringEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def test_genuine_external_validation_fixture_can_raise_generalization_dimension(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    claims_path = project_root / "outputs" / "scientific_claims" / "claims.json"
    claims_payload = json.loads(claims_path.read_text(encoding="utf-8"))
    claim = claims_payload["claims"][0]
    claim["limitations"] = ["External independent labels are traceable in this synthetic fixture."]
    claim["evidence_gap_ids"] = []
    claims_path.write_text(json.dumps(claims_payload), encoding="utf-8")

    graph_path = project_root / "outputs" / "reasoning_graph" / "reasoning_graph.json"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    for node in graph_payload["nodes"]:
        if node["node_id"] == claim["supporting_observation_ids"][0]:
            node["attributes"] = {"external": "independent true_labels available"}
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")

    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()
    first = next(record for record in result.records if record.claim_id == claim["claim_id"])

    assert "genuine external-validation signal is traceable" in first.positive_factors
    assert first.dimension_scores[next(d for d in first.dimension_scores if d.value == "GENERALIZATION_SUPPORT")].raw_score == 100
