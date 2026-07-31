"""Serializers for BSIP evidence scoring outputs."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from .enums import EvidenceDimension
from .models import (
    EVIDENCE_SCORING_SCHEMA_VERSION,
    EVIDENCE_SCORING_SOFTWARE_VERSION,
    EvidenceScoreRecord,
    EvidenceScoringValidationIssue,
    json_ready,
)
from .reporting import markdown_report
from .rules import DIMENSION_WEIGHTS, EVIDENCE_SCORING_RULE_VERSION
from .validation import validation_summary


OUTPUT_FILENAMES: tuple[str, ...] = (
    "evidence_scores.json",
    "evidence_scores.csv",
    "claim_confidence_matrix.csv",
    "evidence_dimension_breakdown.csv",
    "uncertainty_report.json",
    "reviewer_confidence_summary.json",
    "evidence_traceability.json",
    "evidence_scoring_validation.json",
    "evidence_scoring_summary.json",
    "evidence_scoring.md",
)


def write_evidence_scoring_outputs(
    *,
    project_root: Path | str,
    output_dir: Path | str,
    records: tuple[EvidenceScoreRecord, ...],
    validation_issues: tuple[EvidenceScoringValidationIssue, ...],
    generated_at: str,
    source_validation_status: dict[str, Any],
    overwrite: bool = False,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    directory = _resolve_output_directory(root, output_dir)
    _prepare_output_directory(root, directory, overwrite=overwrite)
    ordered = tuple(sorted(records, key=lambda record: record.claim_id))
    paths = {name: directory / name for name in OUTPUT_FILENAMES}

    summary = summarize_records(ordered, validation_passed=False, source_validation_status=source_validation_status)
    validation = validation_summary(ordered, validation_issues, output_readability_checks={})
    _write_all(paths, ordered, summary, validation, generated_at, source_validation_status)

    readability = _readability_checks(paths)
    validation = validation_summary(ordered, validation_issues, output_readability_checks=readability)
    summary = summarize_records(ordered, validation_passed=validation["validation_passed"], source_validation_status=source_validation_status)
    _write_all(paths, ordered, summary, validation, generated_at, source_validation_status)
    return tuple(paths[name] for name in OUTPUT_FILENAMES)


def summarize_records(
    records: tuple[EvidenceScoreRecord, ...],
    *,
    validation_passed: bool,
    source_validation_status: dict[str, Any],
) -> dict[str, Any]:
    scores = [record.normalized_score for record in records]
    level_counts = Counter(record.evidence_level.value for record in records)
    uncertainty_counts = Counter(record.uncertainty_level.value for record in records)
    reviewer_counts = Counter(record.reviewer_confidence.value for record in records)
    readiness_counts = Counter(record.publication_readiness.value for record in records)
    lowest = min(records, key=lambda record: (record.normalized_score, record.claim_id), default=None)
    highest = max(records, key=lambda record: (record.normalized_score, "".join(chr(255 - ord(ch)) for ch in record.claim_id)), default=None)
    external = [record for record in records if "genuine external-validation signal is traceable" in record.positive_factors]
    major_gaps = [record for record in records if any("major evidence gaps" in factor for factor in record.negative_factors)]
    return {
        "total_claims_loaded": source_validation_status.get("claims_loaded", len(records)),
        "total_claims_scored": len(records),
        "withheld_claim_count": sum(1 for record in records if record.is_withheld),
        "count_by_evidence_level": dict(sorted(level_counts.items())),
        "count_by_uncertainty_level": dict(sorted(uncertainty_counts.items())),
        "count_by_reviewer_confidence": dict(sorted(reviewer_counts.items())),
        "count_by_publication_readiness": dict(sorted(readiness_counts.items())),
        "mean_normalized_score": None if not scores else round(mean(scores), 2),
        "median_normalized_score": None if not scores else round(median(scores), 2),
        "minimum_normalized_score": None if not scores else min(scores),
        "maximum_normalized_score": None if not scores else max(scores),
        "lowest_scoring_claim_id": None if lowest is None else lowest.claim_id,
        "highest_scoring_claim_id": None if highest is None else highest.claim_id,
        "claims_with_external_validation": len(external),
        "claims_without_external_validation": len(records) - len(external),
        "claims_with_competing_hypotheses": sum(1 for record in records if record.competing_hypothesis_ids),
        "claims_with_major_evidence_gaps": len(major_gaps),
        "validation_passed": validation_passed,
        "source_claim_schema_version": source_validation_status.get("claim_schema_version"),
        "source_graph_schema_version": source_validation_status.get("graph_schema_version"),
        "evidence_scoring_rule_version": EVIDENCE_SCORING_RULE_VERSION,
        "software_version": source_validation_status.get("software_version", EVIDENCE_SCORING_SOFTWARE_VERSION),
    }


def _write_all(
    paths: dict[str, Path],
    records: tuple[EvidenceScoreRecord, ...],
    summary: dict[str, Any],
    validation: dict[str, Any],
    generated_at: str,
    source_validation_status: dict[str, Any],
) -> None:
    _write_json(paths["evidence_scores.json"], _evidence_scores_document(records, summary, validation, generated_at, source_validation_status))
    _write_csv(paths["evidence_scores.csv"], [_score_row(record) for record in records], fieldnames=_score_fieldnames())
    _write_csv(paths["claim_confidence_matrix.csv"], [_confidence_row(record) for record in records], fieldnames=_confidence_fieldnames())
    _write_csv(paths["evidence_dimension_breakdown.csv"], _dimension_rows(records), fieldnames=_dimension_fieldnames())
    _write_json(paths["uncertainty_report.json"], _uncertainty_document(records, generated_at))
    _write_json(paths["reviewer_confidence_summary.json"], _reviewer_document(records, generated_at))
    _write_json(paths["evidence_traceability.json"], _traceability_document(records, generated_at))
    _write_json(paths["evidence_scoring_validation.json"], validation)
    _write_json(paths["evidence_scoring_summary.json"], summary)
    paths["evidence_scoring.md"].write_text(markdown_report(records, validation_summary=validation, source_validation_status=source_validation_status), encoding="utf-8")


def _evidence_scores_document(
    records: tuple[EvidenceScoreRecord, ...],
    summary: dict[str, Any],
    validation: dict[str, Any],
    generated_at: str,
    source_validation_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCORING_SCHEMA_VERSION,
        "software_version": source_validation_status.get("software_version", EVIDENCE_SCORING_SOFTWARE_VERSION),
        "evidence_scoring_rule_version": EVIDENCE_SCORING_RULE_VERSION,
        "generated_at": generated_at,
        "source_validation_status": source_validation_status,
        "summary": summary,
        "validation_summary": validation,
        "evidence_scores": [record.to_record() for record in records],
        "scoring_notice": "Scores are deterministic evidence-support indices, not probabilities, p-values, causal certainty, mechanism proof, novelty evidence, or evidence of external validity.",
    }


def _score_fieldnames() -> tuple[str, ...]:
    return (
        "claim_id",
        "category",
        "claim_type",
        "claim_status",
        "claim_publication_use",
        "normalized_score",
        "evidence_level",
        "uncertainty_level",
        "reviewer_confidence",
        "publication_readiness",
        "is_withheld",
        "withholding_reasons",
        "positive_factor_count",
        "negative_factor_count",
        "evidence_gap_count",
        "competing_hypothesis_count",
        "traceability_node_count",
    )


def _score_row(record: EvidenceScoreRecord) -> dict[str, Any]:
    return {
        "claim_id": record.claim_id,
        "category": record.claim_category,
        "claim_type": record.claim_type,
        "claim_status": record.claim_status,
        "claim_publication_use": record.claim_publication_use,
        "normalized_score": record.normalized_score,
        "evidence_level": record.evidence_level.value,
        "uncertainty_level": record.uncertainty_level.value,
        "reviewer_confidence": record.reviewer_confidence.value,
        "publication_readiness": record.publication_readiness.value,
        "is_withheld": record.is_withheld,
        "withholding_reasons": json.dumps(list(record.withholding_reasons), sort_keys=True),
        "positive_factor_count": len(record.positive_factors),
        "negative_factor_count": len(record.negative_factors),
        "evidence_gap_count": len(record.evidence_gaps),
        "competing_hypothesis_count": len(record.competing_hypothesis_ids),
        "traceability_node_count": len(record.reasoning_graph_node_ids),
    }


def _confidence_fieldnames() -> tuple[str, ...]:
    return (
        "claim_id",
        *tuple(dimension.value for dimension in sorted(EvidenceDimension, key=lambda item: item.value) if dimension in DIMENSION_WEIGHTS),
        "normalized_score",
        "evidence_level",
        "uncertainty_level",
        "reviewer_confidence",
        "publication_readiness",
    )


def _confidence_row(record: EvidenceScoreRecord) -> dict[str, Any]:
    row = {"claim_id": record.claim_id}
    for dimension in sorted(record.dimension_scores, key=lambda item: item.value):
        row[dimension.value] = record.dimension_scores[dimension].raw_score
    row.update(
        {
            "normalized_score": record.normalized_score,
            "evidence_level": record.evidence_level.value,
            "uncertainty_level": record.uncertainty_level.value,
            "reviewer_confidence": record.reviewer_confidence.value,
            "publication_readiness": record.publication_readiness.value,
        }
    )
    return row


def _dimension_fieldnames() -> tuple[str, ...]:
    return (
        "claim_id",
        "dimension",
        "raw_score",
        "weight",
        "weighted_contribution",
        "positive_factors",
        "penalties",
        "rule_ids",
        "source_node_ids",
        "explanation",
    )


def _dimension_rows(records: tuple[EvidenceScoreRecord, ...]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for dimension, score in sorted(record.dimension_scores.items(), key=lambda item: item[0].value):
            rows.append(
                {
                    "claim_id": record.claim_id,
                    "dimension": dimension.value,
                    "raw_score": score.raw_score,
                    "weight": score.weight,
                    "weighted_contribution": score.weighted_contribution,
                    "positive_factors": json.dumps(list(score.positive_factors), sort_keys=True),
                    "penalties": json.dumps(list(score.penalties), sort_keys=True),
                    "rule_ids": json.dumps(list(score.rule_ids), sort_keys=True),
                    "source_node_ids": json.dumps(list(score.source_node_ids), sort_keys=True),
                    "explanation": score.explanation,
                }
            )
    return rows


def _uncertainty_document(records: tuple[EvidenceScoreRecord, ...], generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCORING_SCHEMA_VERSION,
        "generated_at": generated_at,
        "uncertainty_model": "Uncertainty is assessed from competing hypotheses, evidence gaps, internal-only validation, data-quality limitations, unresolved confounding, missing replication, and traceability.",
        "records": [
            {
                "claim_id": record.claim_id,
                "uncertainty_level": record.uncertainty_level.value,
                "uncertainty_sources": list(record.uncertainty_sources),
                "uncertainty_penalties": list(record.uncertainty_penalties),
                "uncertainty_explanation": record.uncertainty_explanation,
            }
            for record in records
        ],
    }


def _reviewer_document(records: tuple[EvidenceScoreRecord, ...], generated_at: str) -> dict[str, Any]:
    counts = Counter(record.reviewer_confidence.value for record in records)
    return {
        "schema_version": EVIDENCE_SCORING_SCHEMA_VERSION,
        "generated_at": generated_at,
        "count_by_reviewer_confidence": dict(sorted(counts.items())),
        "records": [
            {
                "claim_id": record.claim_id,
                "reviewer_confidence": record.reviewer_confidence.value,
                "publication_readiness": record.publication_readiness.value,
                "reviewer_confidence_explanation": record.reviewer_confidence_explanation,
                "publication_readiness_explanation": record.publication_readiness_explanation,
            }
            for record in records
        ],
    }


def _traceability_document(records: tuple[EvidenceScoreRecord, ...], generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCORING_SCHEMA_VERSION,
        "generated_at": generated_at,
        "traceability_notice": "Evidence score -> dimension score -> scoring rule -> claim -> hypothesis -> interpretation -> observation -> reasoning-graph path -> validation source.",
        "records": [
            {
                "claim_id": record.claim_id,
                "normalized_score": record.normalized_score,
                "supporting_hypothesis_ids": list(record.supporting_hypothesis_ids),
                "supporting_interpretation_ids": list(record.supporting_interpretation_ids),
                "supporting_observation_ids": list(record.supporting_observation_ids),
                "evidence_gap_ids": list(record.evidence_gaps),
                "reasoning_graph_node_ids": list(record.reasoning_graph_node_ids),
                "validation_node_ids": [
                    node_id for node_id in record.reasoning_graph_node_ids if node_id.startswith("VAL:")
                ],
                "dimension_traceability": {
                    dimension.value: {
                        "rule_ids": list(score.rule_ids),
                        "source_node_ids": list(score.source_node_ids),
                        "score_contribution": score.weighted_contribution,
                        "penalties": list(score.penalties),
                        "ceilings": list(score.ceilings),
                        "explanatory_text": score.explanation,
                    }
                    for dimension, score in sorted(record.dimension_scores.items(), key=lambda item: item[0].value)
                },
            }
            for record in records
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _readability_checks(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    checks = {}
    for name, path in sorted(paths.items()):
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    next(csv.reader(handle), None)
            else:
                path.read_text(encoding="utf-8")
            checks[name] = {"readable": True, "reason": ""}
        except (OSError, json.JSONDecodeError, csv.Error, UnicodeError) as exc:
            checks[name] = {"readable": False, "reason": str(exc)}
    return checks


def _resolve_output_directory(project_root: Path, output_dir: Path | str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _prepare_output_directory(project_root: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite to replace it.")
        _assert_safe_output_directory(project_root, output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _assert_safe_output_directory(project_root: Path, output_dir: Path) -> None:
    resolved_root = project_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_root:
        raise ValueError("Refusing to overwrite the project root.")
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Refusing to overwrite an output directory outside the project root.") from exc
