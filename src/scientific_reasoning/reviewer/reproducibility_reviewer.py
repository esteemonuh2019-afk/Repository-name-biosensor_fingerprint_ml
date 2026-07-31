"""Reproducibility reviewer for traceability and replicate boundaries."""

from __future__ import annotations

from typing import Any

from .enums import ReviewCategory, ReviewerConfidence, ReviewerType, Severity
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding
from .policies import build_finding


def review(
    context: ReviewContext,
    *,
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[ReviewFinding, ...]:
    findings: list[ReviewFinding] = []
    index = 1
    workflow_nodes = tuple(
        str(node.get("node_id"))
        for node in context.graph_document.get("nodes", ()) or ()
        if node.get("node_type") == "Workflow" and node.get("node_id")
    )
    if not workflow_nodes:
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.REPRODUCIBILITY,
                index=index,
                category=ReviewCategory.REPRODUCIBILITY,
                title="Workflow metadata is not represented in the reasoning graph",
                finding_text="No Workflow node is present in reasoning_graph.json.",
                severity=Severity.MAJOR,
                confidence=ReviewerConfidence.HIGH,
                source_validation_ids=context.source_validation_ids,
                rationale="Workflow metadata is required for deterministic downstream traceability.",
                evidence_summary="workflow_node_count=0",
                limitations=("The reviewer does not reconstruct workflow metadata.",),
                rule_ids=("REVIEW-REPRO-WORKFLOW-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("workflow", "traceability"),
            )
        )
        index += 1
    if not _contains_independent_wet_lab_replicate_evidence(context):
        claim_ids = tuple(str(claim.get("claim_id")) for claim in context.claims)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.REPRODUCIBILITY,
                index=index,
                category=ReviewCategory.REPRODUCIBILITY,
                title="Biological replicate evidence is not represented",
                finding_text="Independent wet-lab replicate evidence is not represented in the reviewed artifacts.",
                severity=Severity.MODERATE,
                confidence=ReviewerConfidence.MODERATE,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=workflow_nodes,
                source_validation_ids=context.source_validation_ids,
                rationale="Computational traceability and biological reproducibility are separate evidence categories.",
                evidence_summary="independent_wet_lab_replicate_evidence=not represented",
                limitations=("Computational reproducibility metadata does not establish biological reproducibility.",),
                rule_ids=("REVIEW-REPRO-REPLICATE-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("biological-reproducibility",),
            )
        )
    return tuple(findings)


def _contains_independent_wet_lab_replicate_evidence(context: ReviewContext) -> bool:
    text = " ".join(_strings_from(context.claims_document))
    text += " " + " ".join(_strings_from(context.evidence_scores_document))
    text += " " + " ".join(_strings_from(context.graph_document))
    lower = text.lower()
    return "independent wet-lab replicate" in lower or "independent biological replicate" in lower


def _strings_from(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(item for nested in value.values() for item in _strings_from(nested))
    if isinstance(value, (list, tuple)):
        return tuple(item for nested in value for item in _strings_from(nested))
    if isinstance(value, str):
        return (value,)
    return tuple()
