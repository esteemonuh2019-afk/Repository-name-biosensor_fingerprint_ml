"""Output writers for BSIP scientific hypotheses."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Hypothesis, HypothesisValidationIssue, json_ready


OUTPUT_FILENAMES: tuple[str, ...] = (
    "hypotheses.json",
    "hypotheses.csv",
    "hypotheses.md",
    "hypothesis_validation.json",
    "hypothesis_summary.json",
    "hypothesis_dependencies.csv",
    "hypothesis_competition_map.csv",
)


def write_hypothesis_outputs(
    *,
    project_root: Path | str,
    output_dir: Path | str,
    hypotheses: tuple[Hypothesis, ...],
    validation_issues: tuple[HypothesisValidationIssue, ...],
    schema_version: str,
    software_version: str,
    source_interpretation_dir: Path | str,
    generated_at: str,
    source_interpretations_loaded: tuple[str, ...],
    source_interpretations_missing: tuple[str, ...],
    overwrite: bool = False,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    directory = _resolve_output_directory(root, output_dir)
    _prepare_output_directory(root, directory, overwrite=overwrite)

    ordered = tuple(sorted(hypotheses, key=lambda hypothesis: hypothesis.hypothesis_id))
    validation_summary = summarize_validation(validation_issues, output_readability_checks={})
    summary = summarize_hypotheses(
        ordered,
        source_interpretations_loaded=source_interpretations_loaded,
        source_interpretations_missing=source_interpretations_missing,
        validation_passed=validation_summary["validation_passed"],
    )

    paths = {
        name: directory / name
        for name in OUTPUT_FILENAMES
    }
    _write_json(
        paths["hypotheses.json"],
        _hypotheses_document(
            ordered,
            validation_summary=validation_summary,
            schema_version=schema_version,
            software_version=software_version,
            source_interpretation_dir=source_interpretation_dir,
            generated_at=generated_at,
        ),
    )
    _write_csv(paths["hypotheses.csv"], [_flatten_hypothesis(item) for item in ordered])
    paths["hypotheses.md"].write_text(_markdown_report(ordered), encoding="utf-8")
    _write_json(paths["hypothesis_summary.json"], summary)
    _write_dependencies(paths["hypothesis_dependencies.csv"], ordered)
    _write_competition_map(paths["hypothesis_competition_map.csv"], ordered)
    _write_json(paths["hypothesis_validation.json"], validation_summary)

    readability = _readability_checks(paths)
    validation_summary = summarize_validation(validation_issues, output_readability_checks=readability)
    summary = summarize_hypotheses(
        ordered,
        source_interpretations_loaded=source_interpretations_loaded,
        source_interpretations_missing=source_interpretations_missing,
        validation_passed=validation_summary["validation_passed"],
    )
    _write_json(paths["hypothesis_validation.json"], validation_summary)
    _write_json(paths["hypothesis_summary.json"], summary)
    _write_json(
        paths["hypotheses.json"],
        _hypotheses_document(
            ordered,
            validation_summary=validation_summary,
            schema_version=schema_version,
            software_version=software_version,
            source_interpretation_dir=source_interpretation_dir,
            generated_at=generated_at,
        ),
    )
    return tuple(paths[name] for name in OUTPUT_FILENAMES)


def summarize_validation(
    validation_issues: tuple[HypothesisValidationIssue, ...],
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
        "missing_dependency_count": code_counts["MISSING_INTERPRETATION_DEPENDENCY"],
        "unsupported_hypothesis_count": code_counts["UNSUPPORTED_HYPOTHESIS"],
        "causal_overclaim_count": code_counts["CAUSAL_OVERCLAIM"],
        "recommendation_language_issue_count": code_counts["RECOMMENDATION_LANGUAGE"],
        "missing_falsifiability_count": code_counts["MISSING_FALSIFIABILITY"],
        "confidence_policy_issue_count": code_counts["CONFIDENCE_POLICY_ISSUE"],
        "competing_hypothesis_link_issue_count": code_counts["COMPETING_HYPOTHESIS_LINK_ISSUE"],
        "deterministic_ordering_issue_count": code_counts["DETERMINISTIC_ORDERING_ISSUE"],
        "output_readability_checks": output_readability_checks,
        "structured_validation_issues": [issue.to_record() for issue in validation_issues],
    }


def summarize_hypotheses(
    hypotheses: tuple[Hypothesis, ...],
    *,
    source_interpretations_loaded: tuple[str, ...],
    source_interpretations_missing: tuple[str, ...],
    validation_passed: bool,
) -> dict[str, Any]:
    category_counts = Counter(hypothesis.category.value for hypothesis in hypotheses)
    status_counts = Counter(hypothesis.status.value for hypothesis in hypotheses)
    confidence_counts = Counter(hypothesis.confidence.value for hypothesis in hypotheses)
    priority_counts = Counter(hypothesis.priority.value for hypothesis in hypotheses)
    return {
        "total_hypotheses": len(hypotheses),
        "count_by_category": dict(sorted(category_counts.items())),
        "count_by_status": dict(sorted(status_counts.items())),
        "count_by_confidence": dict(sorted(confidence_counts.items())),
        "count_by_priority": dict(sorted(priority_counts.items())),
        "plausible_count": status_counts["PLAUSIBLE"],
        "competing_count": status_counts["COMPETING"],
        "weakly_supported_count": status_counts["WEAKLY_SUPPORTED"],
        "conflicted_count": status_counts["CONFLICTED"],
        "insufficient_evidence_count": status_counts["INSUFFICIENT_EVIDENCE"],
        "source_interpretations_loaded": list(sorted(source_interpretations_loaded)),
        "source_interpretations_missing": list(sorted(source_interpretations_missing)),
        "validation_passed": validation_passed,
    }


def _hypotheses_document(
    hypotheses: tuple[Hypothesis, ...],
    *,
    validation_summary: dict[str, Any],
    schema_version: str,
    software_version: str,
    source_interpretation_dir: Path | str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "software_version": software_version,
        "source_interpretation_directory": str(source_interpretation_dir),
        "generated_at": generated_at,
        "hypotheses": [hypothesis.to_record() for hypothesis in hypotheses],
        "validation_summary": validation_summary,
    }


def _flatten_hypothesis(hypothesis: Hypothesis) -> dict[str, Any]:
    record = hypothesis.to_record()
    flattened = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "hypothesis_id",
        "category",
        "title",
        "statement",
        "status",
        "confidence",
        "supporting_interpretation_ids",
        "contradicting_interpretation_ids",
        "supporting_observation_ids",
        "assumptions",
        "alternative_hypothesis_ids",
        "evidence_gaps",
        "falsifiability_statement",
        "rationale",
        "reasoning_rule_ids",
        "priority_score",
        "priority",
        "created_at",
        "software_version",
        "source_interpretation_schema_version",
        "tags",
        "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_dependencies(path: Path, hypotheses: tuple[Hypothesis, ...]) -> None:
    fieldnames = (
        "hypothesis_id",
        "category",
        "dependency_type",
        "interpretation_id",
        "supporting_observation_ids",
        "reasoning_rule_ids",
    )
    rows = []
    for hypothesis in hypotheses:
        for interpretation_id in hypothesis.supporting_interpretation_ids:
            rows.append(_dependency_row(hypothesis, "supporting", interpretation_id))
        for interpretation_id in hypothesis.contradicting_interpretation_ids:
            rows.append(_dependency_row(hypothesis, "contradicting", interpretation_id))
    rows.sort(key=lambda row: (row["hypothesis_id"], row["dependency_type"], row["interpretation_id"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dependency_row(hypothesis: Hypothesis, dependency_type: str, interpretation_id: str) -> dict[str, str]:
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "category": hypothesis.category.value,
        "dependency_type": dependency_type,
        "interpretation_id": interpretation_id,
        "supporting_observation_ids": json.dumps(list(hypothesis.supporting_observation_ids), sort_keys=True),
        "reasoning_rule_ids": json.dumps(list(hypothesis.reasoning_rule_ids), sort_keys=True),
    }


def _write_competition_map(path: Path, hypotheses: tuple[Hypothesis, ...]) -> None:
    fieldnames = ("hypothesis_id", "alternative_hypothesis_id", "relationship", "reciprocal_link")
    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    rows = []
    for hypothesis in hypotheses:
        for alternative_id in hypothesis.alternative_hypothesis_ids:
            alternative = by_id.get(alternative_id)
            rows.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "alternative_hypothesis_id": alternative_id,
                    "relationship": "competing_or_alternative",
                    "reciprocal_link": str(
                        bool(alternative and hypothesis.hypothesis_id in alternative.alternative_hypothesis_ids)
                    ),
                }
            )
    rows.sort(key=lambda row: (row["hypothesis_id"], row["alternative_hypothesis_id"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(hypotheses: tuple[Hypothesis, ...]) -> str:
    lines = [
        "# Scientific Hypotheses",
        "",
        "This report contains explicit, testable hypotheses generated from validated Interpretation Engine outputs.",
        "",
    ]
    grouped: dict[str, list[Hypothesis]] = {}
    for hypothesis in hypotheses:
        grouped.setdefault(hypothesis.category.value, []).append(hypothesis)
    for category in sorted(grouped):
        lines.extend((f"## {category}", ""))
        for hypothesis in sorted(grouped[category], key=lambda item: item.hypothesis_id):
            record = hypothesis.to_record()
            lines.extend(
                (
                    f"### {hypothesis.hypothesis_id}",
                    "",
                    f"**Hypothesis ID:** {hypothesis.hypothesis_id}",
                    "",
                    f"**Title:** {hypothesis.title}",
                    "",
                    f"**Statement:** {hypothesis.statement}",
                    "",
                    f"**Status:** {hypothesis.status.value}",
                    "",
                    f"**Confidence:** {hypothesis.confidence.value}",
                    "",
                    f"**Priority score:** {hypothesis.priority_score}",
                    "",
                    f"**Supporting interpretation IDs:** {', '.join(record['supporting_interpretation_ids']) or 'None'}",
                    "",
                    f"**Contradicting interpretation IDs:** {', '.join(record['contradicting_interpretation_ids']) or 'None'}",
                    "",
                    f"**Supporting observation IDs:** {', '.join(record['supporting_observation_ids']) or 'None'}",
                    "",
                    f"**Alternative hypothesis IDs:** {', '.join(record['alternative_hypothesis_ids']) or 'None'}",
                    "",
                    f"**Rationale:** {hypothesis.rationale}",
                    "",
                    f"**Assumptions:** {json.dumps(list(hypothesis.assumptions), sort_keys=True)}",
                    "",
                    f"**Evidence gaps:** {json.dumps(list(hypothesis.evidence_gaps), sort_keys=True)}",
                    "",
                    f"**Falsifiability statement:** {hypothesis.falsifiability_statement or 'None'}",
                    "",
                    f"**Reasoning rule IDs:** {', '.join(record['reasoning_rule_ids']) or 'None'}",
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
            raise FileExistsError(
                f"Output directory is non-empty: {output_dir}. Use --overwrite to replace it."
            )
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
