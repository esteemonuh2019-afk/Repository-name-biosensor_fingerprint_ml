"""Markdown reporting for BSIP evidence scoring."""

from __future__ import annotations

from .models import EvidenceScoreRecord
from .rules import DIMENSION_WEIGHTS, EVIDENCE_LEVEL_THRESHOLDS, EVIDENCE_SCORING_RULE_VERSION


def markdown_report(
    records: tuple[EvidenceScoreRecord, ...],
    *,
    validation_summary: dict,
    source_validation_status: dict,
) -> str:
    lines = [
        "# Evidence Scoring",
        "",
        "## Evidence Scoring Engine overview",
        "",
        "The BSIP v4.0.0 Evidence Scoring Engine independently evaluates the evidence support behind existing scientific claims. Scores are deterministic support indices. They are not probabilities, Bayesian posterior probabilities, p-values, causal certainty, mechanism proof, novelty evidence, or evidence of external validity.",
        "",
        "## Source validation status",
        "",
        f"- Claim package validation passed: {source_validation_status.get('claim_validation_passed')}",
        f"- Reasoning graph validation passed: {source_validation_status.get('graph_validation_passed')}",
        f"- Source claim schema: {source_validation_status.get('claim_schema_version')}",
        f"- Source graph schema: {source_validation_status.get('graph_schema_version')}",
        "",
        "## Scoring framework",
        "",
        f"Rule version: `{EVIDENCE_SCORING_RULE_VERSION}`. Dimension weights sum to 1.0 and each dimension is inspectable before weighted aggregation.",
        "",
        "## Dimension definitions",
        "",
    ]
    for dimension, weight in sorted(DIMENSION_WEIGHTS.items(), key=lambda item: item[0].value):
        lines.append(f"- `{dimension.value}`: weight `{weight}`")
    lines.extend(
        (
            "",
            "Evidence-level thresholds:",
            "",
        )
    )
    for level, minimum, maximum in EVIDENCE_LEVEL_THRESHOLDS:
        lines.append(f"- `{level.value}`: {minimum}-{maximum}")
    lines.extend(("", "## Claim-level evidence table", ""))
    lines.append("| Claim ID | Score | Evidence level | Uncertainty | Reviewer confidence | Publication readiness | Withheld |")
    lines.append("|---|---:|---|---|---|---|---|")
    for record in records:
        lines.append(
            f"| {record.claim_id} | {record.normalized_score} | {record.evidence_level.value} | {record.uncertainty_level.value} | {record.reviewer_confidence.value} | {record.publication_readiness.value} | {record.is_withheld} |"
        )
    strongest = sorted(records, key=lambda record: (-record.normalized_score, record.claim_id))[:3]
    caution = sorted(
        records,
        key=lambda record: (
            record.uncertainty_level.value not in {"VERY_HIGH", "HIGH"},
            record.normalized_score,
            record.claim_id,
        ),
    )[:3]
    lines.extend(("", "## Strongest supported claims", ""))
    for record in strongest:
        lines.append(f"- `{record.claim_id}`: {record.normalized_score} ({record.evidence_level.value})")
    lines.extend(("", "## Claims requiring caution", ""))
    for record in caution:
        lines.append(f"- `{record.claim_id}`: uncertainty `{record.uncertainty_level.value}`; {record.publication_readiness_explanation}")
    lines.extend(("", "## Uncertainty analysis", ""))
    for record in records:
        lines.append(f"- `{record.claim_id}`: {record.uncertainty_level.value}; {record.uncertainty_explanation}")
    lines.extend(("", "## Publication-readiness assessment", ""))
    for record in records:
        lines.append(f"- `{record.claim_id}`: {record.publication_readiness.value}; {record.publication_readiness_explanation}")
    lines.extend(
        (
            "",
            "## External-validation boundary",
            "",
            "Internal validation is not external validation. Cross-validation, held-out samples from the same experimental structure, and simulated unknown samples are not treated as external validation. In the absence of genuine external validation, readiness cannot exceed policy ceilings.",
            "",
            "## Withheld claims",
            "",
        )
    )
    withheld = [record for record in records if record.is_withheld]
    if withheld:
        for record in withheld:
            lines.append(f"- `{record.claim_id}`: {', '.join(record.withholding_reasons)}")
    else:
        lines.append("- None")
    lines.extend(
        (
            "",
            "## Validation summary",
            "",
            f"- Validation passed: {validation_summary.get('validation_passed')}",
            f"- Critical issues: {validation_summary.get('critical_issue_count')}",
            f"- Warnings: {validation_summary.get('warning_count')}",
            f"- Missing traceability: {validation_summary.get('missing_traceability_count')}",
            "",
            "## Interpretation limitations",
            "",
            "Evidence scores summarize support completeness and readiness boundaries. They do not generate scientific interpretations, establish causality, establish mechanism, establish novelty, or replace claim-level limitations.",
            "",
            "## Traceability statement",
            "",
            "Each evidence score links to dimension scores, scoring rules, source claim IDs, hypothesis, interpretation, observation, evidence-gap, validation-summary, and reasoning-graph node references where available.",
            "",
        )
    )
    return "\n".join(lines)
