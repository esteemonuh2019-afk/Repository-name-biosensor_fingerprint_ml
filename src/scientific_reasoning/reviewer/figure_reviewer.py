"""Figure and table support reviewer for selected supervisor outputs."""

from __future__ import annotations

from .enums import ReviewCategory, ReviewerConfidence, ReviewerType, Severity
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding
from .policies import build_finding, extract_claim_links, extract_row_identifier


def review(
    context: ReviewContext,
    *,
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[ReviewFinding, ...]:
    findings: list[ReviewFinding] = []
    figure_ids = tuple(sorted(filter(None, (extract_row_identifier(dict(row), "figure_id", "id") for row in context.selected_figures))))
    table_ids = tuple(sorted(filter(None, (extract_row_identifier(dict(row), "table_id", "id") for row in context.selected_tables))))
    if not figure_ids and not table_ids:
        return (
            build_finding(
                reviewer_type=ReviewerType.FIGURE,
                index=1,
                category=ReviewCategory.FIGURE_SUPPORT,
                title="Figure and table metadata is unavailable",
                finding_text="No selected figure or table metadata was available for reviewer assessment.",
                severity=Severity.INFORMATION,
                confidence=ReviewerConfidence.GUARDED,
                rationale="Supervisor figure and table inputs are optional for this reviewer stage.",
                evidence_summary="selected_figure_count=0; selected_table_count=0",
                rule_ids=("REVIEW-FIGURE-METADATA-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("figures", "metadata"),
            ),
        )
    figure_map = _claim_visual_map(context.selected_figures, "figure_id")
    table_map = _claim_visual_map(context.selected_tables, "table_id")
    if not figure_map and not table_map:
        return (
            build_finding(
                reviewer_type=ReviewerType.FIGURE,
                index=1,
                category=ReviewCategory.FIGURE_SUPPORT,
                title="Claim-level visual links are unavailable",
                finding_text="Selected figure and table metadata is present, but claim-level figure or table links were not recorded.",
                severity=Severity.INFORMATION,
                confidence=ReviewerConfidence.GUARDED,
                affected_figure_ids=figure_ids,
                affected_table_ids=table_ids,
                rationale="The reviewer cannot infer claim support from visual titles or filenames alone.",
                evidence_summary=f"selected_figure_count={len(figure_ids)}; selected_table_count={len(table_ids)}",
                rule_ids=("REVIEW-FIGURE-METADATA-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("figures", "tables", "metadata"),
            ),
        )
    affected = []
    for claim in context.claims:
        claim_id = str(claim.get("claim_id"))
        if claim.get("publication_use") not in {"RESULTS_ELIGIBLE", "DISCUSSION_ELIGIBLE"}:
            continue
        if claim_id not in figure_map and claim_id not in table_map:
            affected.append(claim)
    if not affected:
        return tuple()
    claim_ids = tuple(str(claim.get("claim_id")) for claim in affected)
    return (
        build_finding(
            reviewer_type=ReviewerType.FIGURE,
            index=1,
            category=ReviewCategory.FIGURE_SUPPORT,
            title="Publication-facing claims lack claim-level visual support links",
            finding_text=f"{len(claim_ids)} publication-facing claim(s) lack selected figure or table links in supervisor metadata.",
            severity=Severity.MAJOR,
            confidence=ReviewerConfidence.MODERATE,
            affected_claim_ids=claim_ids,
            affected_figure_ids=figure_ids,
            affected_table_ids=table_ids,
            evidence_score_ids=claim_ids,
            reasoning_graph_node_ids=_nodes_for_claims(context, claim_ids),
            rationale="The figure reviewer uses only explicit claim-to-figure or claim-to-table metadata.",
            evidence_summary=f"claims_without_visual_links={len(claim_ids)}",
            limitations=("Visual-support assessment is limited to supervisor metadata and explicit claim links.",),
            rule_ids=("REVIEW-FIGURE-CLAIM-LINK-001",),
            created_at=created_at,
            software_version=software_version,
            tags=("figures", "tables", "claim-links"),
        ),
    )


def _claim_visual_map(rows: tuple[dict, ...], id_field: str) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, set[str]] = {}
    for row in rows:
        item_id = extract_row_identifier(dict(row), id_field, "id")
        for claim_id in extract_claim_links(dict(row)):
            mapping.setdefault(claim_id, set()).add(item_id)
    return {claim_id: tuple(sorted(ids)) for claim_id, ids in sorted(mapping.items())}


def _nodes_for_claims(context: ReviewContext, claim_ids: tuple[str, ...]) -> tuple[str, ...]:
    nodes: set[str] = set()
    for claim_id in claim_ids:
        nodes.update(str(item) for item in context.claim_by_id.get(claim_id, {}).get("reasoning_graph_node_ids", ()) or ())
    return tuple(sorted(nodes))
