import csv
import json
from pathlib import Path
from typing import Any


def create_manuscript_source_fixture(
    tmp_path: Path,
    *,
    drafting_allowed: bool = True,
    empty_figures: bool = False,
    empty_tables: bool = False,
    results_ready: bool = False,
    include_withheld: bool = False,
) -> Path:
    project_root = tmp_path / "project"
    outputs = project_root / "outputs"
    obs_dir = outputs / "scientific_observations"
    int_dir = outputs / "scientific_interpretations"
    hyp_dir = outputs / "scientific_hypotheses"
    claim_dir = outputs / "scientific_claims"
    evidence_dir = outputs / "evidence_scoring"
    review_dir = outputs / "scientific_review"
    graph_dir = outputs / "reasoning_graph"
    supervisor_dir = outputs / "supervisor_results_2"
    for directory in (obs_dir, int_dir, hyp_dir, claim_dir, evidence_dir, review_dir, graph_dir, supervisor_dir):
        directory.mkdir(parents=True)

    observations = [
        {
            "observation_id": "OBS-DATASET-0001",
            "category": "DATASET",
            "title": "Dataset facts",
            "statement": "The supervisor dataset summary listed 10 canonical rows and 2 strains.",
            "status": "COMPLETE",
            "confidence": "HIGH",
            "analysis_stage": "Fixture dataset",
            "supporting_metrics": [{"metric_name": "canonical_rows", "metric_value": 10, "units": None}],
            "supporting_files": ["dataset_summary.json"],
        },
        {
            "observation_id": "OBS-CLASSIFICATION-0001",
            "category": "CLASSIFICATION",
            "title": "Classification facts",
            "statement": "Extra Trees was listed as the selected classification model.",
            "status": "COMPLETE",
            "confidence": "HIGH",
            "analysis_stage": "Fixture classification",
            "supporting_metrics": [{"metric_name": "accuracy_mean", "metric_value": 0.74, "units": None}],
            "supporting_files": ["classification_summary.json"],
        },
    ]
    _write_json(obs_dir / "observations.json", {"schema_version": "bsip_observation_v2", "observations": observations})
    _write_json(obs_dir / "observation_validation.json", _passed_validation())
    _write_json(obs_dir / "observation_summary.json", {"total_observations": len(observations), "validation_passed": True})

    interpretations = [
        {
            "interpretation_id": "INT-CHEMICAL_CLASSIFICATION-0001",
            "category": "CHEMICAL_CLASSIFICATION",
            "claim": "Chemical-classification evidence is internally supported.",
            "status": "SUPPORTED",
            "confidence": "MODERATE",
            "supporting_observation_ids": ["OBS-CLASSIFICATION-0001"],
        }
    ]
    _write_json(int_dir / "interpretations.json", {"schema_version": "BSIP-2.1.0", "interpretations": interpretations})
    _write_json(int_dir / "interpretation_validation.json", _passed_validation())
    _write_json(int_dir / "interpretation_summary.json", {"total_interpretations": len(interpretations), "validation_passed": True})

    hypotheses = [
        {
            "hypothesis_id": "HYP-CHEMICAL_DISCRIMINATION-0001",
            "category": "CHEMICAL_DISCRIMINATION",
            "statement": "Chemical discrimination may be internally plausible.",
            "status": "PLAUSIBLE",
            "confidence": "MODERATE",
            "supporting_interpretation_ids": ["INT-CHEMICAL_CLASSIFICATION-0001"],
            "supporting_observation_ids": ["OBS-CLASSIFICATION-0001"],
            "evidence_gaps": ["No external validation is available."],
        }
    ]
    _write_json(hyp_dir / "hypotheses.json", {"schema_version": "BSIP-2.2.0", "hypotheses": hypotheses})
    _write_json(hyp_dir / "hypothesis_validation.json", _passed_validation())
    _write_json(hyp_dir / "hypothesis_summary.json", {"total_hypotheses": len(hypotheses), "validation_passed": True})

    claims = [
        _claim(
            "CLM-CHEMICAL_DISCRIMINATION-0001",
            "CHEMICAL_DISCRIMINATION",
            "PRIMARY_FINDING",
            "RESULTS_ELIGIBLE",
            "The current evidence supports partial discrimination of chemical identity under internal evaluation conditions.",
        ),
        _claim(
            "CLM-DATA_QUALITY-0001",
            "DATA_QUALITY",
            "LIMITATION",
            "LIMITATION_ONLY",
            "Active quality-control limitations increase uncertainty in downstream interpretation.",
        ),
    ]
    if include_withheld:
        claims.append(
            _claim(
                "CLM-WITHHELD-0001",
                "SYSTEM_LEVEL_PERFORMANCE",
                "WITHHELD",
                "NOT_ELIGIBLE",
                "This claim is withheld.",
            )
        )
    _write_json(claim_dir / "claims.json", {"schema_version": "BSIP-3.2.0", "claims": claims})
    _write_json(claim_dir / "claim_validation.json", _passed_validation())
    _write_json(claim_dir / "claim_summary.json", {"total_claims": len(claims), "validation_passed": True})
    _write_csv(claim_dir / "claim_publication_matrix.csv", [{"claim_id": claim["claim_id"], "publication_use": claim["publication_use"]} for claim in claims])

    readiness = "RESULTS_READY" if results_ready else "DISCUSSION_READY"
    evidence_scores = [
        _score("CLM-CHEMICAL_DISCRIMINATION-0001", "CHEMICAL_DISCRIMINATION", readiness),
        _score("CLM-DATA_QUALITY-0001", "DATA_QUALITY", "LIMITATION_ONLY"),
    ]
    if include_withheld:
        evidence_scores.append(_score("CLM-WITHHELD-0001", "SYSTEM_LEVEL_PERFORMANCE", "NOT_READY"))
    _write_json(evidence_dir / "evidence_scores.json", {"schema_version": "BSIP-4.0.0", "evidence_scores": evidence_scores})
    _write_json(evidence_dir / "evidence_scoring_validation.json", _passed_validation())
    _write_json(evidence_dir / "evidence_scoring_summary.json", {"validation_passed": True, "claims_with_external_validation": 0})
    _write_json(evidence_dir / "reviewer_confidence_summary.json", {"validation_passed": True})
    _write_json(evidence_dir / "uncertainty_report.json", {"uncertainty_model": "fixture"})
    _write_json(evidence_dir / "evidence_traceability.json", {"traceability": []})

    findings = [
        {
            "finding_id": "REV-VALIDATION-0001",
            "reviewer_type": "VALIDATION",
            "category": "EXTERNAL_VALIDATION",
            "title": "True blind-label validation is absent",
            "finding_text": "External validation performance has not been established because independently labelled unknown samples were not evaluated.",
            "severity": "MAJOR",
            "blocking": True,
            "affected_claim_ids": ["CLM-CHEMICAL_DISCRIMINATION-0001"],
            "evidence_score_ids": ["CLM-CHEMICAL_DISCRIMINATION-0001"],
            "reasoning_graph_node_ids": ["OBS-CLASSIFICATION-0001", "HYP-CHEMICAL_DISCRIMINATION-0001"],
            "limitations": ["Absence of external validation blocks definitive generalization claims."],
            "revision_requirement": "Restrict definitive generalization language.",
        }
    ]
    _write_json(review_dir / "review_findings.json", {"schema_version": "BSIP-4.1.0", "review_findings": findings})
    _write_json(review_dir / "reviewer_validation.json", _passed_validation())
    _write_json(review_dir / "reviewer_summary.json", {"validation_passed": True, "blocking_finding_count": 1})
    _write_json(
        review_dir / "reviewer_publication_assessment.json",
        {
            "schema_version": "BSIP-4.1.0",
            "review_rule_version": "BSIP-REVIEW-RULES-4.1.0",
            "manuscript_drafting_allowed": drafting_allowed,
            "overall_recommendation": "NEEDS_MAJOR_REVISION",
            "definitive_generalization_allowed": False,
            "results_claim_ids": ["CLM-CHEMICAL_DISCRIMINATION-0001"] if results_ready else [],
            "discussion_claim_ids": [] if results_ready else ["CLM-CHEMICAL_DISCRIMINATION-0001"],
            "limitation_claim_ids": ["CLM-DATA_QUALITY-0001"],
        },
    )
    _write_csv(
        review_dir / "reviewer_claim_matrix.csv",
        [
            {
                "claim_id": "CLM-CHEMICAL_DISCRIMINATION-0001",
                "review_finding_ids": json.dumps(["REV-VALIDATION-0001"]),
                "blocking_finding_ids": json.dumps(["REV-VALIDATION-0001"]),
            },
            {
                "claim_id": "CLM-DATA_QUALITY-0001",
                "review_finding_ids": json.dumps(["REV-VALIDATION-0001"]),
                "blocking_finding_ids": json.dumps([]),
            },
        ],
    )
    _write_csv(review_dir / "reviewer_revision_requirements.csv", [{"finding_id": "REV-VALIDATION-0001", "revision_requirement": "Restrict definitive generalization language."}])

    nodes = [
        {"node_id": "OBS-DATASET-0001", "node_type": "Observation"},
        {"node_id": "OBS-CLASSIFICATION-0001", "node_type": "Observation"},
        {"node_id": "INT-CHEMICAL_CLASSIFICATION-0001", "node_type": "Interpretation"},
        {"node_id": "HYP-CHEMICAL_DISCRIMINATION-0001", "node_type": "Hypothesis"},
        {
            "node_id": "GAP-HYP-CHEMICAL_DISCRIMINATION-0001-0001",
            "node_type": "EvidenceGap",
            "attributes": {"hypothesis_id": "HYP-CHEMICAL_DISCRIMINATION-0001", "text": "No external validation is available."},
        },
    ]
    _write_json(graph_dir / "reasoning_graph.json", {"schema_version": "BSIP-3.1.0", "nodes": nodes, "edges": []})
    _write_json(graph_dir / "reasoning_graph_validation.json", _passed_validation())
    _write_json(graph_dir / "reasoning_graph_summary.json", {"validation_passed": True, "node_count": len(nodes)})

    _write_csv(supervisor_dir / "selected_figures.csv", [] if empty_figures else [{"figure_id": "fig-1", "title": "Classification heatmap", "source_file": "figures/fig-1.png", "source_run": "fixture"}])
    _write_csv(supervisor_dir / "selected_tables.csv", [] if empty_tables else [{"table_id": "tbl-1", "title": "Classification table", "source_file": "tables/tbl-1.csv", "row_count": "2"}])
    _write_json(supervisor_dir / "report_validation.json", {"passed": True, "critical_issue_count": 0, "warning_count": 0})
    return project_root


def _claim(claim_id: str, category: str, claim_type: str, publication_use: str, claim_text: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "category": category,
        "claim_type": claim_type,
        "publication_use": publication_use,
        "claim_status": "PARTIALLY_SUPPORTED",
        "claim_text": claim_text,
        "supporting_hypothesis_ids": ["HYP-CHEMICAL_DISCRIMINATION-0001"],
        "supporting_interpretation_ids": ["INT-CHEMICAL_CLASSIFICATION-0001"],
        "supporting_observation_ids": ["OBS-CLASSIFICATION-0001"],
        "reasoning_graph_node_ids": ["OBS-CLASSIFICATION-0001", "INT-CHEMICAL_CLASSIFICATION-0001", "HYP-CHEMICAL_DISCRIMINATION-0001"],
        "limitations": ["No independent external validation is available."],
    }


def _score(claim_id: str, category: str, readiness: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_category": category,
        "claim_type": "LIMITATION" if readiness == "LIMITATION_ONLY" else "PRIMARY_FINDING",
        "claim_status": "PARTIALLY_SUPPORTED",
        "claim_publication_use": "LIMITATION_ONLY" if readiness == "LIMITATION_ONLY" else "RESULTS_ELIGIBLE",
        "evidence_level": "STRONG",
        "uncertainty_level": "HIGH",
        "reviewer_confidence": "GUARDED",
        "publication_readiness": readiness,
        "supporting_observation_ids": ["OBS-CLASSIFICATION-0001"],
        "supporting_interpretation_ids": ["INT-CHEMICAL_CLASSIFICATION-0001"],
        "supporting_hypothesis_ids": ["HYP-CHEMICAL_DISCRIMINATION-0001"],
        "reasoning_graph_node_ids": ["OBS-CLASSIFICATION-0001", "INT-CHEMICAL_CLASSIFICATION-0001", "HYP-CHEMICAL_DISCRIMINATION-0001"],
        "evidence_gaps": ["No external validation is available."],
    }


def _passed_validation() -> dict[str, Any]:
    return {"validation_passed": True, "critical_issue_count": 0, "warning_count": 0, "structured_validation_issues": []}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = tuple(rows[0]) if rows else ("placeholder",)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
