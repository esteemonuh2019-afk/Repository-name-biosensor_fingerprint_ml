"""Output writers for BSIP scientific claims."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .enums import ClaimType, PublicationUse
from .models import CLAIM_SCHEMA_VERSION, ScientificClaim, ClaimValidationIssue, json_ready


OUTPUT_FILENAMES: tuple[str, ...] = (
    "claims.json",
    "claims.csv",
    "claims.md",
    "claim_validation.json",
    "claim_summary.json",
    "claim_dependencies.csv",
    "claim_evidence_scores.csv",
    "claim_publication_matrix.csv",
)


def write_claim_outputs(
    *,
    project_root: Path | str,
    output_dir: Path | str,
    claims: tuple[ScientificClaim, ...],
    validation_issues: tuple[ClaimValidationIssue, ...],
    schema_version: str,
    software_version: str,
    generated_at: str,
    hypotheses_dir: Path | str,
    reasoning_graph_dir: Path | str,
    source_hypotheses_loaded: tuple[str, ...],
    source_hypotheses_missing: tuple[str, ...],
    graph_nodes_loaded: int,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    directory = _resolve_output_directory(root, output_dir)
    _prepare_output_directory(root, directory, overwrite=overwrite)
    ordered = tuple(sorted(claims, key=lambda claim: claim.claim_id))
    paths = {name: directory / name for name in OUTPUT_FILENAMES}

    validation_summary = summarize_validation(ordered, validation_issues, output_readability_checks={})
    summary = summarize_claims(
        ordered,
        source_hypotheses_loaded=source_hypotheses_loaded,
        source_hypotheses_missing=source_hypotheses_missing,
        graph_nodes_loaded=graph_nodes_loaded,
        validation_passed=validation_summary["validation_passed"],
    )
    _write_json(
        paths["claims.json"],
        _claims_document(
            ordered,
            validation_summary=validation_summary,
            schema_version=schema_version,
            software_version=software_version,
            generated_at=generated_at,
            hypotheses_dir=hypotheses_dir,
            reasoning_graph_dir=reasoning_graph_dir,
        ),
    )
    _write_csv(paths["claims.csv"], [_flatten_claim(claim) for claim in ordered], fieldnames=_claim_fieldnames())
    paths["claims.md"].write_text(_markdown_report(ordered), encoding="utf-8")
    _write_json(paths["claim_validation.json"], validation_summary)
    _write_json(paths["claim_summary.json"], summary)
    _write_dependencies(paths["claim_dependencies.csv"], ordered)
    _write_evidence_scores(paths["claim_evidence_scores.csv"], ordered)
    _write_publication_matrix(paths["claim_publication_matrix.csv"], ordered)

    readability = _readability_checks(paths)
    validation_summary = summarize_validation(ordered, validation_issues, output_readability_checks=readability)
    summary = summarize_claims(
        ordered,
        source_hypotheses_loaded=source_hypotheses_loaded,
        source_hypotheses_missing=source_hypotheses_missing,
        graph_nodes_loaded=graph_nodes_loaded,
        validation_passed=validation_summary["validation_passed"],
    )
    _write_json(paths["claim_validation.json"], validation_summary)
    _write_json(paths["claim_summary.json"], summary)
    _write_json(
        paths["claims.json"],
        _claims_document(
            ordered,
            validation_summary=validation_summary,
            schema_version=schema_version,
            software_version=software_version,
            generated_at=generated_at,
            hypotheses_dir=hypotheses_dir,
            reasoning_graph_dir=reasoning_graph_dir,
        ),
    )
    return tuple(paths[name] for name in OUTPUT_FILENAMES)


def summarize_validation(
    claims: tuple[ScientificClaim, ...],
    validation_issues: tuple[ClaimValidationIssue, ...],
    *,
    output_readability_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    code_counts = Counter(issue.code for issue in validation_issues)
    critical_issue_count = sum(1 for issue in validation_issues if issue.severity.value == "CRITICAL")
    warning_count = sum(1 for issue in validation_issues if issue.severity.value == "WARNING")
    readability_failed = sum(1 for check in output_readability_checks.values() if not check["readable"])
    return {
        "validation_passed": critical_issue_count == 0 and readability_failed == 0,
        "critical_issue_count": critical_issue_count,
        "warning_count": warning_count,
        "missing_dependency_count": code_counts["MISSING_HYPOTHESIS_DEPENDENCY"] + code_counts["MISSING_GRAPH_DEPENDENCY"],
        "unsupported_claim_count": code_counts["UNSUPPORTED_CLAIM"],
        "causal_overclaim_count": code_counts["CAUSAL_OVERCLAIM"],
        "mechanism_overclaim_count": code_counts["MECHANISM_OVERCLAIM"],
        "novelty_claim_issue_count": code_counts["NOVELTY_CLAIM_ISSUE"],
        "external_validation_overclaim_count": code_counts["EXTERNAL_VALIDATION_OVERCLAIM"],
        "missing_traceability_count": code_counts["MISSING_TRACEABILITY"],
        "missing_limitation_count": code_counts["MISSING_LIMITATION"],
        "publication_use_policy_issue_count": code_counts["PUBLICATION_USE_POLICY_ISSUE"],
        "evidence_score_policy_issue_count": code_counts["EVIDENCE_SCORE_POLICY_ISSUE"],
        "deterministic_ordering_issue_count": code_counts["DETERMINISTIC_ORDERING_ISSUE"],
        "source_validation_failure_count": code_counts["SOURCE_VALIDATION_FAILURE"],
        "withheld_claim_count": sum(1 for claim in claims if claim.claim_type is ClaimType.WITHHELD),
        "output_readability_checks": output_readability_checks,
        "structured_validation_issues": [issue.to_record() for issue in validation_issues],
    }


def summarize_claims(
    claims: tuple[ScientificClaim, ...],
    *,
    source_hypotheses_loaded: tuple[str, ...],
    source_hypotheses_missing: tuple[str, ...],
    graph_nodes_loaded: int,
    validation_passed: bool,
) -> dict[str, Any]:
    category_counts = Counter(claim.category.value for claim in claims)
    type_counts = Counter(claim.claim_type.value for claim in claims)
    status_counts = Counter(claim.claim_status.value for claim in claims)
    strength_counts = Counter(claim.evidence_strength.value for claim in claims)
    publication_counts = Counter(claim.publication_use.value for claim in claims)
    scores = [claim.evidence_score for claim in claims]
    return {
        "total_claims": len(claims),
        "count_by_category": dict(sorted(category_counts.items())),
        "count_by_claim_type": dict(sorted(type_counts.items())),
        "count_by_status": dict(sorted(status_counts.items())),
        "count_by_evidence_strength": dict(sorted(strength_counts.items())),
        "count_by_publication_use": dict(sorted(publication_counts.items())),
        "results_eligible_count": publication_counts[PublicationUse.RESULTS_ELIGIBLE.value],
        "discussion_eligible_count": publication_counts[PublicationUse.DISCUSSION_ELIGIBLE.value],
        "limitation_only_count": publication_counts[PublicationUse.LIMITATION_ONLY.value],
        "internal_review_only_count": publication_counts[PublicationUse.INTERNAL_REVIEW_ONLY.value],
        "withheld_count": type_counts[ClaimType.WITHHELD.value],
        "mean_evidence_score": None if not scores else round(mean(scores), 2),
        "minimum_evidence_score": None if not scores else min(scores),
        "maximum_evidence_score": None if not scores else max(scores),
        "source_hypotheses_loaded": list(sorted(source_hypotheses_loaded)),
        "source_hypotheses_missing": list(sorted(source_hypotheses_missing)),
        "graph_nodes_loaded": graph_nodes_loaded,
        "validation_passed": validation_passed,
    }


def _claims_document(
    claims: tuple[ScientificClaim, ...],
    *,
    validation_summary: dict[str, Any],
    schema_version: str,
    software_version: str,
    generated_at: str,
    hypotheses_dir: Path | str,
    reasoning_graph_dir: Path | str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "software_version": software_version,
        "generated_at": generated_at,
        "source_hypotheses_directory": str(hypotheses_dir),
        "source_reasoning_graph_directory": str(reasoning_graph_dir),
        "claims": [claim.to_record() for claim in claims],
        "validation_summary": validation_summary,
    }


def _claim_fieldnames() -> tuple[str, ...]:
    return (
        "claim_id",
        "category",
        "title",
        "claim_text",
        "claim_type",
        "claim_status",
        "evidence_strength",
        "publication_use",
        "supporting_hypothesis_ids",
        "competing_hypothesis_ids",
        "supporting_interpretation_ids",
        "supporting_observation_ids",
        "evidence_gap_ids",
        "validation_summary_ids",
        "reasoning_graph_node_ids",
        "assumptions",
        "limitations",
        "rationale",
        "evidence_score",
        "confidence_label",
        "language_policy_rule_ids",
        "reasoning_rule_ids",
        "created_at",
        "software_version",
        "source_hypothesis_schema_version",
        "source_graph_schema_version",
        "tags",
        "metadata",
    )


def _flatten_claim(claim: ScientificClaim) -> dict[str, Any]:
    record = claim.to_record()
    flattened = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_dependencies(path: Path, claims: tuple[ScientificClaim, ...]) -> None:
    fieldnames = (
        "claim_id",
        "claim_category",
        "dependency_type",
        "dependency_id",
        "hypothesis_id",
        "graph_node_id",
        "reasoning_rule_ids",
    )
    rows = []
    for claim in claims:
        rows.extend(_dependency_rows(claim, "supporting_hypothesis", claim.supporting_hypothesis_ids, hypothesis=True))
        rows.extend(_dependency_rows(claim, "competing_hypothesis", claim.competing_hypothesis_ids, hypothesis=True))
        rows.extend(_dependency_rows(claim, "supporting_interpretation", claim.supporting_interpretation_ids, graph=True))
        rows.extend(_dependency_rows(claim, "supporting_observation", claim.supporting_observation_ids, graph=True))
        rows.extend(_dependency_rows(claim, "evidence_gap", claim.evidence_gap_ids, graph=True))
        rows.extend(_dependency_rows(claim, "validation_summary", claim.validation_summary_ids, graph=True))
        rows.extend(_dependency_rows(claim, "reasoning_graph_node", claim.reasoning_graph_node_ids, graph=True))
    rows.sort(key=lambda row: (row["claim_id"], row["dependency_type"], row["dependency_id"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dependency_rows(
    claim: ScientificClaim,
    dependency_type: str,
    ids: tuple[str, ...],
    *,
    hypothesis: bool = False,
    graph: bool = False,
) -> list[dict[str, str]]:
    return [
        {
            "claim_id": claim.claim_id,
            "claim_category": claim.category.value,
            "dependency_type": dependency_type,
            "dependency_id": item_id,
            "hypothesis_id": item_id if hypothesis else "",
            "graph_node_id": item_id if graph else "",
            "reasoning_rule_ids": json.dumps(list(claim.reasoning_rule_ids), sort_keys=True),
        }
        for item_id in ids
    ]


def _write_evidence_scores(path: Path, claims: tuple[ScientificClaim, ...]) -> None:
    fieldnames = (
        "claim_id",
        "claim_category",
        "evidence_score",
        "evidence_strength",
        "confidence_label",
        "supporting_hypothesis_count",
        "competing_hypothesis_count",
        "supporting_interpretation_count",
        "supporting_observation_count",
        "evidence_gap_count",
        "graph_traceable",
    )
    rows = [
        {
            "claim_id": claim.claim_id,
            "claim_category": claim.category.value,
            "evidence_score": claim.evidence_score,
            "evidence_strength": claim.evidence_strength.value,
            "confidence_label": claim.confidence_label.value,
            "supporting_hypothesis_count": len(claim.supporting_hypothesis_ids),
            "competing_hypothesis_count": len(claim.competing_hypothesis_ids),
            "supporting_interpretation_count": len(claim.supporting_interpretation_ids),
            "supporting_observation_count": len(claim.supporting_observation_ids),
            "evidence_gap_count": len(claim.evidence_gap_ids),
            "graph_traceable": bool(claim.metadata.get("graph_traceable")),
        }
        for claim in claims
    ]
    _write_csv(path, rows, fieldnames=fieldnames)


def _write_publication_matrix(path: Path, claims: tuple[ScientificClaim, ...]) -> None:
    fieldnames = (
        "claim_id",
        "claim_category",
        "claim_type",
        "claim_status",
        "evidence_strength",
        "evidence_score",
        "publication_use",
        "results_eligible",
        "discussion_eligible",
        "limitation_only",
        "internal_review_only",
        "primary_limitation",
        "competing_hypothesis_count",
        "evidence_gap_count",
    )
    rows = [
        {
            "claim_id": claim.claim_id,
            "claim_category": claim.category.value,
            "claim_type": claim.claim_type.value,
            "claim_status": claim.claim_status.value,
            "evidence_strength": claim.evidence_strength.value,
            "evidence_score": claim.evidence_score,
            "publication_use": claim.publication_use.value,
            "results_eligible": claim.publication_use is PublicationUse.RESULTS_ELIGIBLE,
            "discussion_eligible": claim.publication_use is PublicationUse.DISCUSSION_ELIGIBLE,
            "limitation_only": claim.publication_use is PublicationUse.LIMITATION_ONLY,
            "internal_review_only": claim.publication_use is PublicationUse.INTERNAL_REVIEW_ONLY,
            "primary_limitation": claim.claim_type is ClaimType.LIMITATION,
            "competing_hypothesis_count": len(claim.competing_hypothesis_ids),
            "evidence_gap_count": len(claim.evidence_gap_ids),
        }
        for claim in claims
    ]
    _write_csv(path, rows, fieldnames=fieldnames)


def _markdown_report(claims: tuple[ScientificClaim, ...]) -> str:
    lines = ["# Scientific Claims", "", "This report contains evidence-bounded claims generated from validated BSIP hypotheses and reasoning graphs.", ""]
    grouped: dict[str, list[ScientificClaim]] = {}
    for claim in claims:
        grouped.setdefault(claim.category.value, []).append(claim)
    for category in sorted(grouped):
        lines.extend((f"## {category}", ""))
        for claim in sorted(grouped[category], key=lambda item: item.claim_id):
            marker = " **WITHHELD**" if claim.claim_type is ClaimType.WITHHELD else ""
            lines.extend(
                (
                    f"### {claim.claim_id}{marker}",
                    "",
                    f"**Claim ID:** {claim.claim_id}",
                    "",
                    f"**Title:** {claim.title}",
                    "",
                    f"**Claim text:** {claim.claim_text}",
                    "",
                    f"**Claim type:** {claim.claim_type.value}",
                    "",
                    f"**Claim status:** {claim.claim_status.value}",
                    "",
                    f"**Evidence strength:** {claim.evidence_strength.value}",
                    "",
                    f"**Evidence score:** {claim.evidence_score}",
                    "",
                    f"**Confidence label:** {claim.confidence_label.value}",
                    "",
                    f"**Publication use:** {claim.publication_use.value}",
                    "",
                    f"**Supporting hypothesis IDs:** {', '.join(claim.supporting_hypothesis_ids) or 'None'}",
                    "",
                    f"**Competing hypothesis IDs:** {', '.join(claim.competing_hypothesis_ids) or 'None'}",
                    "",
                    f"**Supporting interpretation IDs:** {', '.join(claim.supporting_interpretation_ids) or 'None'}",
                    "",
                    f"**Supporting observation IDs:** {', '.join(claim.supporting_observation_ids) or 'None'}",
                    "",
                    f"**Evidence-gap IDs:** {', '.join(claim.evidence_gap_ids) or 'None'}",
                    "",
                    f"**Validation-summary IDs:** {', '.join(claim.validation_summary_ids) or 'None'}",
                    "",
                    f"**Rationale:** {claim.rationale}",
                    "",
                    f"**Assumptions:** {json.dumps(list(claim.assumptions), sort_keys=True)}",
                    "",
                    f"**Limitations:** {json.dumps(list(claim.limitations), sort_keys=True)}",
                    "",
                    f"**Reasoning rule IDs:** {', '.join(claim.reasoning_rule_ids) or 'None'}",
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
