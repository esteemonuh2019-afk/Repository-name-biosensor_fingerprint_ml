import json

from src.scientific_reasoning.reviewer import ReviewerEngine
from tests.integration.test_reviewer_engine_integration import create_reviewer_project


def test_review_findings_reference_existing_claims_and_graph_nodes(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path)
    ReviewerEngine(project_root=project_root, overwrite=True).run()

    findings = json.loads((project_root / "outputs" / "scientific_review" / "review_findings.json").read_text(encoding="utf-8"))["review_findings"]
    claims = json.loads((project_root / "outputs" / "scientific_claims" / "claims.json").read_text(encoding="utf-8"))["claims"]
    graph = json.loads((project_root / "outputs" / "reasoning_graph" / "reasoning_graph.json").read_text(encoding="utf-8"))
    claim_ids = {claim["claim_id"] for claim in claims}
    node_ids = {node["node_id"] for node in graph["nodes"]}

    for finding in findings:
        assert set(finding["affected_claim_ids"]).issubset(claim_ids)
        assert set(finding["reasoning_graph_node_ids"]).issubset(node_ids)
