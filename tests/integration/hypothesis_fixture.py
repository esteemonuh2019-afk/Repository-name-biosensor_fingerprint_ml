from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationStatus,
)


FIXED_TIME = "2026-07-31T00:00:00+00:00"


def make_interpretation(
    *,
    interpretation_id: str,
    category: InterpretationCategory,
    title: str,
    claim: str,
    observation_ids: tuple[str, ...],
    confidence: InterpretationConfidence = InterpretationConfidence.MODERATE,
    status: InterpretationStatus = InterpretationStatus.SUPPORTED,
) -> Interpretation:
    evidence = tuple(
        InterpretationEvidenceLink(
            observation_id=observation_id,
            direction=EvidenceDirection.SUPPORTING,
            rationale="Synthetic interpretation evidence link.",
            metric_names=("synthetic_metric",),
            provenance_ids=(f"P-{observation_id}",),
            source_files=(f"observations/{observation_id}.json",),
        )
        for observation_id in observation_ids
    )
    return Interpretation(
        interpretation_id=interpretation_id,
        category=category,
        title=title,
        claim=claim,
        status=status,
        confidence=confidence,
        supporting_observation_ids=tuple(sorted(observation_ids)),
        contradicting_observation_ids=tuple(),
        assumptions=("Synthetic interpretation fixture.",),
        limitations=tuple(),
        evidence_summary=evidence,
        reasoning_rule_ids=(f"RULE-{category.value}-001",),
        created_at=FIXED_TIME,
        software_version="BSIP-2.1.0-test",
        source_observation_schema_version="BSIP-2.0",
        tags=("synthetic",),
        metadata={"fixture": True},
    )


def realistic_interpretations() -> tuple[Interpretation, ...]:
    return tuple(
        sorted(
            (
                make_interpretation(
                    interpretation_id="INT-BLIND_VALIDATION-0001",
                    category=InterpretationCategory.BLIND_VALIDATION,
                    title="Blind-validation boundary",
                    claim=(
                        "The available blind-prediction observations do not establish external validation "
                        "performance because true labels were absent."
                    ),
                    observation_ids=("OBS-BLIND_PREDICTION-0001",),
                    status=InterpretationStatus.PARTIALLY_SUPPORTED,
                ),
                make_interpretation(
                    interpretation_id="INT-CHEMICAL_CLASSIFICATION-0001",
                    category=InterpretationCategory.CHEMICAL_CLASSIFICATION,
                    title="Chemical classification",
                    claim="The classification observations suggest chemical-class discrimination information.",
                    observation_ids=("OBS-CLASSIFICATION-0001",),
                ),
                make_interpretation(
                    interpretation_id="INT-CONCENTRATION_REGRESSION-0001",
                    category=InterpretationCategory.CONCENTRATION_REGRESSION,
                    title="Concentration regression",
                    claim="The regression observations indicate concentration-related information is present.",
                    observation_ids=("OBS-REGRESSION-0001",),
                ),
                make_interpretation(
                    interpretation_id="INT-DATA_QUALITY-0001",
                    category=InterpretationCategory.DATA_QUALITY,
                    title="Data quality",
                    claim="The quality-control observations indicate active data-quality limitations.",
                    observation_ids=("OBS-QC-0001", "OBS-VALIDATION-0001"),
                    confidence=InterpretationConfidence.HIGH,
                ),
                make_interpretation(
                    interpretation_id="INT-FEATURE_ENGINEERING-0001",
                    category=InterpretationCategory.FEATURE_ENGINEERING,
                    title="Feature engineering",
                    claim="The feature-engineering observations indicate temporal feature associations.",
                    observation_ids=("OBS-FEATURE_ENGINEERING-0001",),
                ),
                make_interpretation(
                    interpretation_id="INT-FEATURE_SELECTION-0001",
                    category=InterpretationCategory.FEATURE_SELECTION,
                    title="Feature selection",
                    claim="The feature-selection outputs document selected feature sets.",
                    observation_ids=("OBS-FEATURE_SELECTION-0001",),
                ),
                make_interpretation(
                    interpretation_id="INT-FINGERPRINT_STRUCTURE-0001",
                    category=InterpretationCategory.FINGERPRINT_STRUCTURE,
                    title="Fingerprint structure",
                    claim="The fingerprint and exploratory observations indicate structured variation.",
                    observation_ids=("OBS-FINGERPRINT-0001", "OBS-EXPLORATORY_ANALYSIS-0001"),
                    confidence=InterpretationConfidence.HIGH,
                ),
                make_interpretation(
                    interpretation_id="INT-OVERALL_EVIDENCE-0001",
                    category=InterpretationCategory.OVERALL_EVIDENCE,
                    title="Overall evidence",
                    claim="The available evidence supports further evaluation under current limitations.",
                    observation_ids=(
                        "OBS-BLIND_PREDICTION-0001",
                        "OBS-CLASSIFICATION-0001",
                        "OBS-QC-0001",
                        "OBS-REGRESSION-0001",
                    ),
                ),
                make_interpretation(
                    interpretation_id="INT-STRAIN_CONTRIBUTION-0001",
                    category=InterpretationCategory.STRAIN_CONTRIBUTION,
                    title="Strain contribution",
                    claim="The strain-contribution outputs indicate differential contribution was evaluated.",
                    observation_ids=("OBS-STRAIN_CONTRIBUTION-0001",),
                ),
            ),
            key=lambda interpretation: interpretation.interpretation_id,
        )
    )


def write_interpretation_package(
    directory: Path,
    interpretations: tuple[Interpretation, ...] | None = None,
    *,
    validation_passed: bool = True,
    critical_issue_count: int = 0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    interpretations = interpretations or realistic_interpretations()
    ordered = tuple(sorted(interpretations, key=lambda interpretation: interpretation.interpretation_id))
    (directory / "interpretations.json").write_text(
        json.dumps(
            {
                "schema_version": "BSIP-2.1.0",
                "software_version": "BSIP-2.1.0-test",
                "generated_at": FIXED_TIME,
                "interpretations": [interpretation.to_record() for interpretation in ordered],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "interpretation_validation.json").write_text(
        json.dumps(
            {
                "validation_passed": validation_passed,
                "critical_issue_count": critical_issue_count,
                "warning_count": 0,
                "structured_validation_issues": [
                    {
                        "code": "SYNTHETIC_CRITICAL",
                        "severity": "CRITICAL",
                        "message": "Synthetic critical interpretation issue.",
                        "interpretation_id": None,
                        "field": "interpretation_validation.json",
                        "observation_id": None,
                        "rule_id": None,
                    }
                ]
                if critical_issue_count
                else [],
                "output_readability_checks": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    category_counts = Counter(interpretation.category.value for interpretation in ordered)
    (directory / "interpretation_summary.json").write_text(
        json.dumps(
            {
                "total_interpretations": len(ordered),
                "count_by_category": dict(sorted(category_counts.items())),
                "source_interpretations_missing": [],
                "validation_passed": validation_passed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (directory / "interpretation_dependencies.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "interpretation_id",
            "category",
            "dependency_type",
            "observation_id",
            "evidence_direction",
            "metric_names",
            "provenance_ids",
            "source_files",
            "reasoning_rule_ids",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for interpretation in ordered:
            for observation_id in interpretation.supporting_observation_ids:
                writer.writerow(
                    {
                        "interpretation_id": interpretation.interpretation_id,
                        "category": interpretation.category.value,
                        "dependency_type": "supporting",
                        "observation_id": observation_id,
                        "evidence_direction": "SUPPORTING",
                        "metric_names": "[]",
                        "provenance_ids": "[]",
                        "source_files": "[]",
                        "reasoning_rule_ids": json.dumps(list(interpretation.reasoning_rule_ids), sort_keys=True),
                    }
                )
    return directory
