"""Scientific-boundary reviewer for structured BSIP claims."""

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
    for claim in context.claims:
        if claim.get("claim_type") != "PRIMARY_FINDING":
            continue
        competing = tuple(str(item) for item in claim.get("competing_hypothesis_ids", ()) or ())
        evidence = context.evidence_by_claim_id.get(str(claim.get("claim_id")), {})
        gap_count = len(evidence.get("evidence_gaps", ()) or claim.get("evidence_gap_ids", ()) or ())
        if competing:
            findings.append(
                build_finding(
                    reviewer_type=ReviewerType.SCIENTIFIC,
                    index=index,
                    category=ReviewCategory.COMPETING_EXPLANATIONS,
                    title="Primary claim retains competing explanations",
                    finding_text=(
                        f"Claim {claim.get('claim_id')} retains {len(competing)} competing hypothesis link(s) "
                        f"and {gap_count} recorded evidence gap(s)."
                    ),
                    severity=Severity.MAJOR,
                    confidence=ReviewerConfidence.HIGH,
                    affected_claim_ids=(str(claim.get("claim_id")),),
                    affected_hypothesis_ids=competing + tuple(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ()),
                    affected_interpretation_ids=_ids(claim, "supporting_interpretation_ids"),
                    affected_observation_ids=_ids(claim, "supporting_observation_ids"),
                    evidence_score_ids=(str(claim.get("claim_id")),),
                    reasoning_graph_node_ids=_ids(claim, "reasoning_graph_node_ids"),
                    rationale="The upstream claim and evidence score explicitly preserve competing hypotheses.",
                    evidence_summary=f"competing_hypothesis_count={len(competing)}; evidence_gap_count={gap_count}",
                    limitations=tuple(str(item) for item in claim.get("limitations", ()) or ()),
                    rule_ids=("REVIEW-SCI-CONFOUNDING-001",),
                    created_at=created_at,
                    software_version=software_version,
                    tags=("scientific-boundary", "competing-explanations"),
                )
            )
            index += 1
    conflicted = tuple(claim for claim in context.claims if claim.get("claim_status") == "CONFLICTED")
    if conflicted:
        claim_ids = tuple(str(claim.get("claim_id")) for claim in conflicted)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.SCIENTIFIC,
                index=index,
                category=ReviewCategory.CLAIM_SUPPORT,
                title="Conflicted claims remain present",
                finding_text=f"{len(claim_ids)} claim(s) are marked CONFLICTED in the validated claim package.",
                severity=Severity.MODERATE,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_combined_nodes(conflicted),
                rationale="The reviewer preserves the upstream claim status without changing the claim.",
                evidence_summary=f"conflicted_claim_ids={', '.join(claim_ids)}",
                limitations=("Conflicted claim status is inherited from the Claim Engine.",),
                rule_ids=("REVIEW-SCI-LIMITATION-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("claim-status",),
            )
        )
        index += 1
    missing_limitations = tuple(
        claim
        for claim in context.claims
        if claim.get("publication_use") in {"RESULTS_ELIGIBLE", "DISCUSSION_ELIGIBLE"}
        and (claim.get("evidence_gap_ids") or context.evidence_by_claim_id.get(str(claim.get("claim_id")), {}).get("evidence_gaps"))
        and not claim.get("limitations")
    )
    if missing_limitations:
        claim_ids = tuple(str(claim.get("claim_id")) for claim in missing_limitations)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.SCIENTIFIC,
                index=index,
                category=ReviewCategory.LIMITATION_COMPLETENESS,
                title="Publication-facing claims lack recorded limitations",
                finding_text=f"{len(claim_ids)} publication-facing claim(s) have evidence gaps without recorded limitations.",
                severity=Severity.MAJOR,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_combined_nodes(missing_limitations),
                rationale="Every publication-facing claim with evidence gaps must retain limitation text.",
                evidence_summary=f"affected_claim_count={len(claim_ids)}",
                rule_ids=("REVIEW-SCI-LIMITATION-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("limitations",),
            )
        )
    return tuple(findings)


def _ids(record: dict[str, Any], field: str) -> tuple[str, ...]:
    return tuple(str(item) for item in record.get(field, ()) or ())


def _combined_nodes(records: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(sorted({node for record in records for node in _ids(record, "reasoning_graph_node_ids")}))
