"""Deterministic claim construction rules for BSIP v3.2.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import ClaimCategory, ClaimStatus, ClaimType, EvidenceStrength, PublicationUse
from .models import DEFAULT_CLAIM_SOFTWARE_VERSION, ScientificClaim
from .policies import (
    LANGUAGE_POLICY_RULE_IDS,
    calculate_evidence_score,
    confidence_label_from_strength,
    evidence_strength_from_score,
    publication_use_for_claim,
)


CLAIM_RULE_VERSION = "BSIP-CLAIM-RULES-3.2.0"
TRACEABILITY_RULE_ID = "CLAIM-TRACEABILITY-001"
WITHHOLDING_RULE_ID = "CLAIM-WITHHOLDING-001"


@dataclass(frozen=True)
class ClaimRule:
    claim_id: str
    category: ClaimCategory
    title: str
    claim_text: str
    claim_type: ClaimType
    supporting_hypothesis_ids: tuple[str, ...]
    competing_hypothesis_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


CLAIM_RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        claim_id="CLM-CHEMICAL_DISCRIMINATION-0001",
        category=ClaimCategory.CHEMICAL_DISCRIMINATION,
        title="Chemical-identity discrimination under internal evaluation",
        claim_text=(
            "The current evidence supports partial discrimination of chemical identity from multistrain biosensor "
            "response patterns under internal evaluation conditions."
        ),
        claim_type=ClaimType.PRIMARY_FINDING,
        supporting_hypothesis_ids=("HYP-CHEMICAL_DISCRIMINATION-0001",),
        competing_hypothesis_ids=("HYP-CHEMICAL_DISCRIMINATION-0002",),
        limitations=(
            "Chemical identity may not be the only source of classification structure.",
            "Concentration, batch, or correlated experimental structure remain plausible alternatives.",
            "No independent external validation is available.",
        ),
        tags=("chemical-discrimination", "classification", "internal-evaluation"),
    ),
    ClaimRule(
        claim_id="CLM-CONCENTRATION_INFORMATION-0001",
        category=ClaimCategory.CONCENTRATION_INFORMATION,
        title="Limited concentration-related information",
        claim_text="The biosensor response profiles contain limited concentration-related information under the current feature representation.",
        claim_type=ClaimType.CONTEXTUAL_FINDING,
        supporting_hypothesis_ids=("HYP-CONCENTRATION_ENCODING-0001",),
        competing_hypothesis_ids=("HYP-CONCENTRATION_ENCODING-0002",),
        limitations=(
            "The claim does not establish precise concentration prediction.",
            "The source hypotheses are low-confidence or weakly supported.",
            "No independent external validation is available.",
        ),
        tags=("concentration-information", "regression", "limited-evidence"),
    ),
    ClaimRule(
        claim_id="CLM-TEMPORAL_INFORMATION-0001",
        category=ClaimCategory.TEMPORAL_INFORMATION,
        title="Temporal response information",
        claim_text=(
            "Temporally resolved response characteristics are associated with discriminatory information beyond static "
            "summary measurements in the current internal analyses."
        ),
        claim_type=ClaimType.SUPPORTING_FINDING,
        supporting_hypothesis_ids=("HYP-TEMPORAL_INFORMATION-0001",),
        limitations=(
            "No direct temporal-feature ablation establishes causality.",
            "No independent external validation is available.",
        ),
        tags=("temporal-information", "internal-analysis"),
    ),
    ClaimRule(
        claim_id="CLM-FEATURE_REPRESENTATION-0001",
        category=ClaimCategory.FEATURE_REPRESENTATION,
        title="Window-based temporal feature representation",
        claim_text=(
            "Window-based temporal features are associated with improved internal benchmark performance relative to "
            "the reference feature configuration."
        ),
        claim_type=ClaimType.SUPPORTING_FINDING,
        supporting_hypothesis_ids=("HYP-FEATURE_REPRESENTATION-0001",),
        competing_hypothesis_ids=("HYP-FEATURE_REPRESENTATION-0002",),
        limitations=(
            "The evidence does not distinguish temporal information content from increased dimensionality or model flexibility.",
            "No independent external validation is available.",
        ),
        tags=("feature-representation", "window-features"),
    ),
    ClaimRule(
        claim_id="CLM-STRAIN_CONTRIBUTION-0001",
        category=ClaimCategory.STRAIN_CONTRIBUTION,
        title="Differential strain contribution",
        claim_text=(
            "The multistrain array contains evidence of differential strain contribution to chemical-classification "
            "performance."
        ),
        claim_type=ClaimType.SUPPORTING_FINDING,
        supporting_hypothesis_ids=("HYP-STRAIN_CONTRIBUTION-0001",),
        competing_hypothesis_ids=("HYP-STRAIN_CONTRIBUTION-0002",),
        limitations=(
            "Sampling variability and uneven chemical-response coverage remain plausible explanations.",
            "No specific strain is identified because the source hypotheses do not explicitly name one.",
        ),
        tags=("strain-contribution", "classification"),
    ),
    ClaimRule(
        claim_id="CLM-DATA_QUALITY-0001",
        category=ClaimCategory.DATA_QUALITY,
        title="Quality-control limitations",
        claim_text=(
            "Active quality-control limitations increase uncertainty in the interpretation of downstream classification "
            "and regression estimates."
        ),
        claim_type=ClaimType.LIMITATION,
        supporting_hypothesis_ids=("HYP-DATA_QUALITY_EFFECT-0001",),
        limitations=(
            "The claim does not state that quality-control limitations caused any specific model result.",
            "The limitation applies to interpretation of downstream estimates.",
        ),
        tags=("data-quality", "limitation"),
    ),
    ClaimRule(
        claim_id="CLM-GENERALIZATION-0001",
        category=ClaimCategory.GENERALIZATION,
        title="Generalization boundary",
        claim_text="Internal evaluation performance cannot yet be assumed to generalize to independently labelled unknown samples.",
        claim_type=ClaimType.LIMITATION,
        supporting_hypothesis_ids=("HYP-GENERALIZATION-0001",),
        limitations=(
            "No true external validation with independently labelled unknown samples has occurred.",
            "The claim does not state external-validation failure.",
        ),
        tags=("generalization", "limitation"),
    ),
    ClaimRule(
        claim_id="CLM-SYSTEM_LEVEL_PERFORMANCE-0001",
        category=ClaimCategory.SYSTEM_LEVEL_PERFORMANCE,
        title="System-level performance balance",
        claim_text=(
            "Under the current dataset and feature representation, the multistrain array provides stronger evidence "
            "for chemical-identity discrimination than for precise concentration estimation."
        ),
        claim_type=ClaimType.PRIMARY_FINDING,
        supporting_hypothesis_ids=("HYP-OVERALL_SYSTEM_BEHAVIOR-0001",),
        limitations=(
            "The comparison is based on internal evaluation.",
            "The tasks use different metrics and may not be directly equivalent.",
            "No independent external validation is available.",
        ),
        tags=("system-level-performance", "classification", "regression"),
    ),
)


def build_claims_from_sources(
    hypotheses: tuple[dict[str, Any], ...],
    graph_document: dict[str, Any],
    *,
    hypothesis_validation_passed: bool,
    graph_validation_passed: bool,
    created_at: str,
    software_version: str = DEFAULT_CLAIM_SOFTWARE_VERSION,
    source_hypothesis_schema_version: str | None = None,
    source_graph_schema_version: str | None = None,
) -> tuple[ScientificClaim, ...]:
    hypotheses_by_id = {str(record.get("hypothesis_id")): record for record in hypotheses}
    graph = GraphIndex(graph_document)
    claims = [
        _build_claim(
            rule,
            hypotheses_by_id=hypotheses_by_id,
            graph=graph,
            hypothesis_validation_passed=hypothesis_validation_passed,
            graph_validation_passed=graph_validation_passed,
            created_at=created_at,
            software_version=software_version,
            source_hypothesis_schema_version=source_hypothesis_schema_version,
            source_graph_schema_version=source_graph_schema_version,
        )
        for rule in CLAIM_RULES
    ]
    return tuple(sorted(claims, key=lambda claim: claim.claim_id))


class GraphIndex:
    def __init__(self, graph_document: dict[str, Any]) -> None:
        self.graph_document = graph_document
        self.nodes_by_id = {str(node.get("node_id")): node for node in graph_document.get("nodes", ())}
        self.edges = tuple(graph_document.get("edges", ()))
        self.reverse_support: dict[str, set[str]] = {}
        self.forward_limited_by: dict[str, set[str]] = {}
        for edge in self.edges:
            source = str(edge.get("source_id"))
            target = str(edge.get("target_id"))
            if edge.get("edge_type") == "supports":
                self.reverse_support.setdefault(target, set()).add(source)
            if edge.get("edge_type") == "limited_by":
                self.forward_limited_by.setdefault(source, set()).add(target)

    def node_type(self, node_id: str) -> str | None:
        node = self.nodes_by_id.get(node_id)
        return None if node is None else str(node.get("node_type"))

    def validation_summary_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(node_id for node_id, node in self.nodes_by_id.items() if node.get("node_type") == "ValidationSummary")
        )

    def evidence_gap_ids_for(self, hypothesis_ids: tuple[str, ...]) -> tuple[str, ...]:
        gap_ids = set()
        for hypothesis_id in hypothesis_ids:
            for node_id in self.forward_limited_by.get(hypothesis_id, set()):
                if self.node_type(node_id) == "EvidenceGap":
                    gap_ids.add(node_id)
        return tuple(sorted(gap_ids))

    def support_ancestors(self, node_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        queue = sorted(self.reverse_support.get(node_id, ()))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            queue.extend(parent for parent in sorted(self.reverse_support.get(current, ())) if parent not in seen)
        return tuple(sorted(seen))

    def complete_support_chain(self, hypothesis_id: str) -> bool:
        if self.node_type(hypothesis_id) != "Hypothesis":
            return False
        ancestors = self.support_ancestors(hypothesis_id)
        return any(self.node_type(node_id) == "Interpretation" for node_id in ancestors) and any(
            self.node_type(node_id) == "Observation" for node_id in ancestors
        )


def _build_claim(
    rule: ClaimRule,
    *,
    hypotheses_by_id: dict[str, dict[str, Any]],
    graph: GraphIndex,
    hypothesis_validation_passed: bool,
    graph_validation_passed: bool,
    created_at: str,
    software_version: str,
    source_hypothesis_schema_version: str | None,
    source_graph_schema_version: str | None,
) -> ScientificClaim:
    supporting_ids = tuple(hypothesis_id for hypothesis_id in rule.supporting_hypothesis_ids if hypothesis_id in hypotheses_by_id)
    competing_ids = tuple(hypothesis_id for hypothesis_id in rule.competing_hypothesis_ids if hypothesis_id in hypotheses_by_id)
    missing_ids = tuple(
        hypothesis_id
        for hypothesis_id in rule.supporting_hypothesis_ids + rule.competing_hypothesis_ids
        if hypothesis_id not in hypotheses_by_id
    )
    required_ids = supporting_ids + competing_ids
    required_records = tuple(hypotheses_by_id[hypothesis_id] for hypothesis_id in required_ids)
    graph_traceable = (
        bool(required_ids)
        and graph_validation_passed
        and all(graph.complete_support_chain(hypothesis_id) for hypothesis_id in required_ids)
    )
    source_validation_passed = hypothesis_validation_passed and graph_validation_passed
    status_blocks_claim = any(
        record.get("status") in {"NOT_ASSESSABLE", "INSUFFICIENT_EVIDENCE"}
        for record in required_records
    )
    withhold_reasons = []
    if missing_ids:
        withhold_reasons.append("Required hypothesis source is missing: " + ", ".join(missing_ids))
    if not hypothesis_validation_passed:
        withhold_reasons.append("Hypothesis package validation failed critically.")
    if not graph_validation_passed:
        withhold_reasons.append("Reasoning graph validation failed critically.")
    if status_blocks_claim:
        withhold_reasons.append("Required hypothesis status is not assessable or insufficiently supported.")
    if required_ids and not graph_traceable:
        withhold_reasons.append("No complete observation-to-interpretation-to-hypothesis support chain is available.")

    interpretation_ids = _collect_interpretations(required_records, graph)
    observation_ids = _collect_observations(required_records, graph)
    evidence_gap_ids = graph.evidence_gap_ids_for(required_ids)
    validation_summary_ids = graph.validation_summary_ids()
    graph_node_ids = _graph_node_ids(
        supporting_ids=supporting_ids,
        competing_ids=competing_ids,
        interpretation_ids=interpretation_ids,
        observation_ids=observation_ids,
        evidence_gap_ids=evidence_gap_ids,
        validation_summary_ids=validation_summary_ids,
    )
    score = calculate_evidence_score(
        required_records,
        supporting_interpretation_count=len(interpretation_ids),
        supporting_observation_count=len(observation_ids),
        competing_hypothesis_count=len(competing_ids),
        evidence_gap_count=len(evidence_gap_ids),
        graph_traceable=graph_traceable,
        source_validation_passed=source_validation_passed,
    )
    strength = evidence_strength_from_score(score)
    claim_type = _claim_type_for_rule(rule, graph, graph_traceable)
    claim_status = _status_for_rule(rule, required_records, competing_ids, withheld=bool(withhold_reasons))
    if withhold_reasons:
        claim_type = ClaimType.WITHHELD
        claim_status = ClaimStatus.WITHHELD
        strength = EvidenceStrength.NOT_ASSESSABLE
        score = 0.0
        publication_use = PublicationUse.NOT_ELIGIBLE
        confidence_label = confidence_label_from_strength(strength)
        claim_text = f"Claim withheld for {rule.category.value} because required evidence is unavailable or invalid."
        limitations = tuple(withhold_reasons)
        rationale = " ".join(withhold_reasons)
    else:
        publication_use = publication_use_for_claim(
            claim_type=claim_type,
            claim_status=claim_status,
            evidence_strength=strength,
            has_critical_issue=False,
            category=rule.category.value,
        )
        confidence_label = confidence_label_from_strength(strength)
        claim_text = rule.claim_text
        limitations = rule.limitations
        rationale = (
            "Claim generated from validated hypothesis and reasoning-graph support chains. "
            "The evidence score is a deterministic support index and is not a probability that the claim is true."
        )
    return ScientificClaim(
        claim_id=rule.claim_id,
        category=rule.category,
        title=rule.title,
        claim_text=claim_text,
        claim_type=claim_type,
        claim_status=claim_status,
        evidence_strength=strength,
        publication_use=publication_use,
        supporting_hypothesis_ids=supporting_ids,
        competing_hypothesis_ids=competing_ids,
        supporting_interpretation_ids=interpretation_ids,
        supporting_observation_ids=observation_ids,
        evidence_gap_ids=evidence_gap_ids,
        validation_summary_ids=validation_summary_ids,
        reasoning_graph_node_ids=graph_node_ids,
        assumptions=rule.assumptions,
        limitations=limitations,
        rationale=rationale,
        evidence_score=score,
        confidence_label=confidence_label,
        language_policy_rule_ids=LANGUAGE_POLICY_RULE_IDS,
        reasoning_rule_ids=_reasoning_rule_ids(required_records),
        created_at=created_at,
        software_version=software_version,
        source_hypothesis_schema_version=source_hypothesis_schema_version,
        source_graph_schema_version=source_graph_schema_version,
        tags=rule.tags,
        metadata={
            "claim_rule_version": CLAIM_RULE_VERSION,
            "graph_traceable": graph_traceable,
            "source_validation_passed": source_validation_passed,
            "withholding_reasons": withhold_reasons,
        },
    )


def _claim_type_for_rule(rule: ClaimRule, graph: GraphIndex, graph_traceable: bool) -> ClaimType:
    if rule.category is ClaimCategory.SYSTEM_LEVEL_PERFORMANCE and rule.claim_type is ClaimType.PRIMARY_FINDING:
        ancestors = set()
        for hypothesis_id in rule.supporting_hypothesis_ids:
            ancestors.update(graph.support_ancestors(hypothesis_id))
        if not graph_traceable or not {"INT-CHEMICAL_CLASSIFICATION-0001", "INT-CONCENTRATION_REGRESSION-0001"}.issubset(ancestors):
            return ClaimType.SUPPORTING_FINDING
    return rule.claim_type


def _status_for_rule(
    rule: ClaimRule,
    records: tuple[dict[str, Any], ...],
    competing_ids: tuple[str, ...],
    *,
    withheld: bool,
) -> ClaimStatus:
    if withheld:
        return ClaimStatus.WITHHELD
    if rule.category is ClaimCategory.CHEMICAL_DISCRIMINATION:
        return ClaimStatus.PARTIALLY_SUPPORTED
    if rule.category is ClaimCategory.CONCENTRATION_INFORMATION:
        return ClaimStatus.TENTATIVE
    if rule.category is ClaimCategory.FEATURE_REPRESENTATION and competing_ids:
        return ClaimStatus.CONFLICTED
    if rule.category is ClaimCategory.GENERALIZATION:
        return ClaimStatus.TENTATIVE
    if any(record.get("status") == "CONFLICTED" for record in records):
        return ClaimStatus.CONFLICTED
    if any(record.get("status") == "WEAKLY_SUPPORTED" for record in records):
        return ClaimStatus.TENTATIVE
    return ClaimStatus.PARTIALLY_SUPPORTED


def _collect_interpretations(records: tuple[dict[str, Any], ...], graph: GraphIndex) -> tuple[str, ...]:
    interpretation_ids = set()
    for record in records:
        interpretation_ids.update(str(item) for item in record.get("supporting_interpretation_ids", ()) or ())
        for ancestor in graph.support_ancestors(str(record.get("hypothesis_id"))):
            if graph.node_type(ancestor) == "Interpretation":
                interpretation_ids.add(ancestor)
    return tuple(sorted(interpretation_ids))


def _collect_observations(records: tuple[dict[str, Any], ...], graph: GraphIndex) -> tuple[str, ...]:
    observation_ids = set()
    for record in records:
        observation_ids.update(str(item) for item in record.get("supporting_observation_ids", ()) or ())
        for ancestor in graph.support_ancestors(str(record.get("hypothesis_id"))):
            if graph.node_type(ancestor) == "Observation":
                observation_ids.add(ancestor)
    return tuple(sorted(observation_ids))


def _graph_node_ids(
    *,
    supporting_ids: tuple[str, ...],
    competing_ids: tuple[str, ...],
    interpretation_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    evidence_gap_ids: tuple[str, ...],
    validation_summary_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(supporting_ids)
            | set(competing_ids)
            | set(interpretation_ids)
            | set(observation_ids)
            | set(evidence_gap_ids)
            | set(validation_summary_ids)
        )
    )


def _reasoning_rule_ids(records: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    ids = set()
    for record in records:
        ids.update(str(item) for item in record.get("reasoning_rule_ids", ()) or ())
    ids.update({TRACEABILITY_RULE_ID})
    return tuple(sorted(ids))
