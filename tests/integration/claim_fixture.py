import csv
import json
from pathlib import Path
from typing import Any


def create_claim_source_fixture(
    tmp_path: Path,
    *,
    remove_hypothesis_id: str | None = None,
    invalid_hypothesis_validation: bool = False,
    invalid_graph_validation: bool = False,
) -> Path:
    project_root = tmp_path / "project"
    hypotheses_dir = project_root / "outputs" / "scientific_hypotheses"
    graph_dir = project_root / "outputs" / "reasoning_graph"
    hypotheses_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    hypotheses = [record for record in _hypotheses() if record["hypothesis_id"] != remove_hypothesis_id]
    graph = _graph(hypotheses)
    _write_json(
        hypotheses_dir / "hypotheses.json",
        {
            "schema_version": "BSIP-2.2.0",
            "software_version": "fixture",
            "generated_at": "2026-07-31T00:00:00+00:00",
            "hypotheses": hypotheses,
        },
    )
    _write_json(
        hypotheses_dir / "hypothesis_validation.json",
        {
            "validation_passed": not invalid_hypothesis_validation,
            "critical_issue_count": 1 if invalid_hypothesis_validation else 0,
            "warning_count": 0,
            "structured_validation_issues": [],
        },
    )
    _write_json(
        hypotheses_dir / "hypothesis_summary.json",
        {"total_hypotheses": len(hypotheses), "validation_passed": not invalid_hypothesis_validation},
    )
    _write_csv(hypotheses_dir / "hypothesis_dependencies.csv", _dependency_rows(hypotheses))
    _write_csv(hypotheses_dir / "hypothesis_competition_map.csv", _competition_rows(hypotheses))

    _write_json(graph_dir / "reasoning_graph.json", graph)
    _write_json(
        graph_dir / "reasoning_graph_validation.json",
        {
            "validation_passed": not invalid_graph_validation,
            "critical_issue_count": 1 if invalid_graph_validation else 0,
            "warning_count": 0,
            "structured_validation_issues": [],
        },
    )
    _write_json(
        graph_dir / "reasoning_graph_summary.json",
        {
            "graph_id": graph["graph_id"],
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "validation_passed": not invalid_graph_validation,
        },
    )
    return project_root


def _hypotheses() -> list[dict[str, Any]]:
    return [
        _hyp(
            "HYP-CHEMICAL_DISCRIMINATION-0001",
            "CHEMICAL_DISCRIMINATION",
            "PLAUSIBLE",
            "MODERATE",
            ("INT-CHEMICAL_CLASSIFICATION-0001", "INT-FINGERPRINT_STRUCTURE-0001"),
            ("OBS-CLASSIFICATION-0001", "OBS-FINGERPRINT-0001"),
            alternatives=("HYP-CHEMICAL_DISCRIMINATION-0002",),
            gaps=("No independent external validation is available.", "Correlated structure remains plausible."),
        ),
        _hyp(
            "HYP-CHEMICAL_DISCRIMINATION-0002",
            "CHEMICAL_DISCRIMINATION",
            "COMPETING",
            "MODERATE",
            ("INT-CHEMICAL_CLASSIFICATION-0001", "INT-FINGERPRINT_STRUCTURE-0001"),
            ("OBS-CLASSIFICATION-0001", "OBS-FINGERPRINT-0001"),
            alternatives=("HYP-CHEMICAL_DISCRIMINATION-0001",),
            gaps=("No independent external validation is available.",),
        ),
        _hyp(
            "HYP-CONCENTRATION_ENCODING-0001",
            "CONCENTRATION_ENCODING",
            "WEAKLY_SUPPORTED",
            "LOW",
            ("INT-CONCENTRATION_REGRESSION-0001",),
            ("OBS-REGRESSION-0001",),
            alternatives=("HYP-CONCENTRATION_ENCODING-0002",),
            gaps=("Only limited regression support is available.",),
        ),
        _hyp(
            "HYP-CONCENTRATION_ENCODING-0002",
            "CONCENTRATION_ENCODING",
            "COMPETING",
            "LOW",
            ("INT-CONCENTRATION_REGRESSION-0001",),
            ("OBS-REGRESSION-0001",),
            alternatives=("HYP-CONCENTRATION_ENCODING-0001",),
            gaps=("No independent external validation is available.",),
        ),
        _hyp(
            "HYP-DATA_QUALITY_EFFECT-0001",
            "DATA_QUALITY_EFFECT",
            "PLAUSIBLE",
            "MODERATE",
            ("INT-DATA_QUALITY-0001", "INT-OVERALL_EVIDENCE-0001"),
            ("OBS-QC-0001", "OBS-VALIDATION-0001"),
            gaps=("QC warnings remain active.",),
        ),
        _hyp(
            "HYP-FEATURE_REPRESENTATION-0001",
            "FEATURE_REPRESENTATION",
            "PLAUSIBLE",
            "MODERATE",
            ("INT-FEATURE_ENGINEERING-0001", "INT-FEATURE_SELECTION-0001"),
            ("OBS-FEATURE_ENGINEERING-0001", "OBS-FEATURE_SELECTION-0001"),
            alternatives=("HYP-FEATURE_REPRESENTATION-0002",),
            gaps=("Dimensionality remains a plausible alternative explanation.",),
        ),
        _hyp(
            "HYP-FEATURE_REPRESENTATION-0002",
            "FEATURE_REPRESENTATION",
            "COMPETING",
            "MODERATE",
            ("INT-FEATURE_ENGINEERING-0001", "INT-FEATURE_SELECTION-0001"),
            ("OBS-FEATURE_ENGINEERING-0001", "OBS-FEATURE_SELECTION-0001"),
            alternatives=("HYP-FEATURE_REPRESENTATION-0001",),
            gaps=("No causal temporal-feature ablation is available.",),
        ),
        _hyp(
            "HYP-GENERALIZATION-0001",
            "GENERALIZATION",
            "WEAKLY_SUPPORTED",
            "LOW",
            ("INT-BLIND_VALIDATION-0001", "INT-CHEMICAL_CLASSIFICATION-0001", "INT-CONCENTRATION_REGRESSION-0001"),
            ("OBS-BLIND_PREDICTION-0001", "OBS-CLASSIFICATION-0001", "OBS-REGRESSION-0001"),
            gaps=("No true external validation is available.",),
        ),
        _hyp(
            "HYP-OVERALL_SYSTEM_BEHAVIOR-0001",
            "OVERALL_SYSTEM_BEHAVIOR",
            "PLAUSIBLE",
            "MODERATE",
            ("INT-CHEMICAL_CLASSIFICATION-0001", "INT-CONCENTRATION_REGRESSION-0001", "INT-OVERALL_EVIDENCE-0001"),
            ("OBS-CLASSIFICATION-0001", "OBS-REGRESSION-0001", "OBS-QC-0001"),
            gaps=("Tasks use different metrics.",),
        ),
        _hyp(
            "HYP-STRAIN_CONTRIBUTION-0001",
            "STRAIN_CONTRIBUTION",
            "PLAUSIBLE",
            "MODERATE",
            ("INT-STRAIN_CONTRIBUTION-0001",),
            ("OBS-STRAIN_CONTRIBUTION-0001",),
            alternatives=("HYP-STRAIN_CONTRIBUTION-0002",),
            gaps=("Sampling variability remains plausible.",),
        ),
        _hyp(
            "HYP-STRAIN_CONTRIBUTION-0002",
            "STRAIN_CONTRIBUTION",
            "COMPETING",
            "MODERATE",
            ("INT-STRAIN_CONTRIBUTION-0001",),
            ("OBS-STRAIN_CONTRIBUTION-0001",),
            alternatives=("HYP-STRAIN_CONTRIBUTION-0001",),
            gaps=("Uneven chemical-response coverage remains plausible.",),
        ),
        _hyp(
            "HYP-TEMPORAL_INFORMATION-0001",
            "TEMPORAL_INFORMATION",
            "PLAUSIBLE",
            "MODERATE",
            ("INT-FEATURE_ENGINEERING-0001",),
            ("OBS-FEATURE_ENGINEERING-0001", "OBS-FEATURE_SELECTION-0001"),
            gaps=("No direct temporal-feature ablation is available.",),
        ),
    ]


def _hyp(
    hypothesis_id: str,
    category: str,
    status: str,
    confidence: str,
    interpretation_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    *,
    alternatives: tuple[str, ...] = (),
    gaps: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "category": category,
        "title": hypothesis_id,
        "statement": f"{hypothesis_id} may be plausible under current evidence.",
        "status": status,
        "confidence": confidence,
        "supporting_interpretation_ids": list(interpretation_ids),
        "contradicting_interpretation_ids": [],
        "supporting_observation_ids": list(observation_ids),
        "assumptions": ["Synthetic fixture hypothesis."],
        "alternative_hypothesis_ids": list(alternatives),
        "evidence_gaps": list(gaps),
        "rationale": "Synthetic rationale.",
        "reasoning_rule_ids": [f"RULE-{category}-001"],
        "priority_score": 50,
        "priority": "MEDIUM",
        "created_at": "2026-07-31T00:00:00+00:00",
        "software_version": "fixture",
        "source_interpretation_schema_version": "fixture",
        "tags": [category.lower()],
        "metadata": {},
    }


def _graph(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    observations = sorted({obs for hyp in hypotheses for obs in hyp["supporting_observation_ids"]})
    interpretations = sorted({item for hyp in hypotheses for item in hyp["supporting_interpretation_ids"]})
    nodes = [{"node_id": "DATASET-fixture", "node_type": "Dataset", "label": "Dataset"}]
    nodes.append({"node_id": "BSIP-WF-fixture", "node_type": "Workflow", "label": "Workflow"})
    nodes.extend({"node_id": node_id, "node_type": "ValidationSummary", "label": node_id} for node_id in ("VAL:observation", "VAL:interpretation", "VAL:hypothesis", "VAL:workflow"))
    nodes.extend({"node_id": obs, "node_type": "Observation", "label": obs} for obs in observations)
    nodes.extend({"node_id": interpretation, "node_type": "Interpretation", "label": interpretation} for interpretation in interpretations)
    nodes.extend({"node_id": hyp["hypothesis_id"], "node_type": "Hypothesis", "label": hyp["hypothesis_id"]} for hyp in hypotheses)
    edges = []
    for obs in observations:
        edges.append(_edge(obs, "DATASET-fixture", "derived_from"))
        edges.append(_edge(obs, "VAL:observation", "validated_by"))
    for interpretation in interpretations:
        matching_obs = sorted({obs for hyp in hypotheses if interpretation in hyp["supporting_interpretation_ids"] for obs in hyp["supporting_observation_ids"]})
        for obs in matching_obs:
            edges.append(_edge(obs, interpretation, "supports"))
        edges.append(_edge(interpretation, "VAL:interpretation", "validated_by"))
    for hyp in hypotheses:
        hypothesis_id = hyp["hypothesis_id"]
        for interpretation in hyp["supporting_interpretation_ids"]:
            edges.append(_edge(interpretation, hypothesis_id, "supports"))
        for index, _gap in enumerate(hyp["evidence_gaps"], start=1):
            gap_id = f"GAP-{hypothesis_id}-{index:04d}"
            nodes.append({"node_id": gap_id, "node_type": "EvidenceGap", "label": gap_id})
            edges.append(_edge(hypothesis_id, gap_id, "limited_by"))
        for alternative_id in hyp["alternative_hypothesis_ids"]:
            edges.append(_edge(hypothesis_id, alternative_id, "competes_with"))
        edges.append(_edge(hypothesis_id, "VAL:hypothesis", "validated_by"))
    return {
        "graph_id": "BSIP-GRAPH-fixture",
        "schema_version": "BSIP-3.1.0",
        "software_version": "fixture",
        "generated_at": "2026-07-31T00:00:00+00:00",
        "nodes": nodes,
        "edges": edges,
        "metadata": {},
    }


def _edge(source: str, target: str, edge_type: str) -> dict[str, Any]:
    return {"edge_id": f"E:{edge_type}:{source}->{target}", "source_id": source, "target_id": target, "edge_type": edge_type}


def _dependency_rows(hypotheses: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "hypothesis_id": hyp["hypothesis_id"],
            "category": hyp["category"],
            "dependency_type": "supporting",
            "interpretation_id": interpretation_id,
            "supporting_observation_ids": json.dumps(hyp["supporting_observation_ids"]),
            "reasoning_rule_ids": json.dumps(hyp["reasoning_rule_ids"]),
        }
        for hyp in hypotheses
        for interpretation_id in hyp["supporting_interpretation_ids"]
    ]


def _competition_rows(hypotheses: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "hypothesis_id": hyp["hypothesis_id"],
            "alternative_hypothesis_id": alternative_id,
            "relationship": "competing_or_alternative",
            "reciprocal_link": "True",
        }
        for hyp in hypotheses
        for alternative_id in hyp["alternative_hypothesis_ids"]
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = tuple(rows[0]) if rows else ("placeholder",)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
