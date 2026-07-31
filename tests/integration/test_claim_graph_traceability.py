import json
from collections import defaultdict, deque
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine, ClaimType
from tests.integration.claim_fixture import create_claim_source_fixture


def test_active_claims_reference_existing_graph_nodes_and_complete_support_chains(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    result = ClaimEngine(project_root=project_root, overwrite=True).run()
    graph = json.loads((project_root / "outputs" / "reasoning_graph" / "reasoning_graph.json").read_text(encoding="utf-8"))
    node_types = {node["node_id"]: node["node_type"] for node in graph["nodes"]}
    support_parents = defaultdict(set)
    for edge in graph["edges"]:
        if edge["edge_type"] == "supports":
            support_parents[edge["target_id"]].add(edge["source_id"])

    for claim in result.claims:
        assert all(node_id in node_types for node_id in claim.reasoning_graph_node_ids)
        if claim.claim_type is ClaimType.WITHHELD:
            continue
        assert claim.validation_summary_ids
        assert claim.evidence_gap_ids
        for hypothesis_id in claim.supporting_hypothesis_ids:
            ancestors = _ancestors(hypothesis_id, support_parents)
            assert any(node_types.get(node_id) == "Interpretation" for node_id in ancestors)
            assert any(node_types.get(node_id) == "Observation" for node_id in ancestors)


def test_missing_required_hypothesis_produces_structured_withheld_claim(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path, remove_hypothesis_id="HYP-TEMPORAL_INFORMATION-0001")
    result = ClaimEngine(project_root=project_root, overwrite=True).run()
    temporal = next(claim for claim in result.claims if claim.claim_id == "CLM-TEMPORAL_INFORMATION-0001")

    assert temporal.claim_type is ClaimType.WITHHELD
    assert "Required hypothesis source is missing" in temporal.rationale


def _ancestors(node_id: str, support_parents) -> set[str]:
    seen = set()
    queue = deque(sorted(support_parents.get(node_id, ())))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(parent for parent in sorted(support_parents.get(current, ())) if parent not in seen)
    return seen
