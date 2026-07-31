"""Output writers for BSIP scientific interpretations."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Interpretation, InterpretationValidationIssue, json_ready


OUTPUT_FILENAMES: tuple[str, ...] = (
    "interpretations.json",
    "interpretations.csv",
    "interpretations.md",
    "interpretation_validation.json",
    "interpretation_summary.json",
    "interpretation_dependencies.csv",
)


def write_interpretation_outputs(
    *,
    project_root: Path | str,
    output_dir: Path | str,
    interpretations: tuple[Interpretation, ...],
    validation_issues: tuple[InterpretationValidationIssue, ...],
    schema_version: str,
    software_version: str,
    source_observation_dir: Path | str,
    generated_at: str,
    source_observations_loaded: tuple[str, ...],
    source_observations_missing: tuple[str, ...],
    overwrite: bool = False,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    directory = _resolve_output_directory(root, output_dir)
    _prepare_output_directory(root, directory, overwrite=overwrite)

    ordered = tuple(sorted(interpretations, key=lambda interpretation: interpretation.interpretation_id))
    validation_summary = summarize_validation(validation_issues, output_readability_checks={})
    summary = summarize_interpretations(
        ordered,
        source_observations_loaded=source_observations_loaded,
        source_observations_missing=source_observations_missing,
        validation_passed=validation_summary["validation_passed"],
    )

    paths = {
        "interpretations.json": directory / "interpretations.json",
        "interpretations.csv": directory / "interpretations.csv",
        "interpretations.md": directory / "interpretations.md",
        "interpretation_validation.json": directory / "interpretation_validation.json",
        "interpretation_summary.json": directory / "interpretation_summary.json",
        "interpretation_dependencies.csv": directory / "interpretation_dependencies.csv",
    }

    _write_json(
        paths["interpretations.json"],
        _interpretations_document(
            ordered,
            validation_summary=validation_summary,
            schema_version=schema_version,
            software_version=software_version,
            source_observation_dir=source_observation_dir,
            generated_at=generated_at,
        ),
    )
    _write_csv(paths["interpretations.csv"], [_flatten_interpretation(item) for item in ordered])
    paths["interpretations.md"].write_text(_markdown_report(ordered), encoding="utf-8")
    _write_json(paths["interpretation_summary.json"], summary)
    _write_dependencies(paths["interpretation_dependencies.csv"], ordered)
    _write_json(paths["interpretation_validation.json"], validation_summary)

    readability = _readability_checks(paths)
    validation_summary = summarize_validation(validation_issues, output_readability_checks=readability)
    summary = summarize_interpretations(
        ordered,
        source_observations_loaded=source_observations_loaded,
        source_observations_missing=source_observations_missing,
        validation_passed=validation_summary["validation_passed"],
    )
    _write_json(paths["interpretation_validation.json"], validation_summary)
    _write_json(paths["interpretation_summary.json"], summary)
    _write_json(
        paths["interpretations.json"],
        _interpretations_document(
            ordered,
            validation_summary=validation_summary,
            schema_version=schema_version,
            software_version=software_version,
            source_observation_dir=source_observation_dir,
            generated_at=generated_at,
        ),
    )

    return tuple(paths[name] for name in OUTPUT_FILENAMES)


def summarize_validation(
    validation_issues: tuple[InterpretationValidationIssue, ...],
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
        "missing_dependency_count": code_counts["NONEXISTENT_OBSERVATION_DEPENDENCY"]
        + code_counts["MISSING_SUPPORTING_OBSERVATION"],
        "unsupported_claim_count": (
            code_counts["FORBIDDEN_CAUSAL_LANGUAGE"]
            + code_counts["RECOMMENDATION_LANGUAGE"]
            + code_counts["HYPOTHESIS_LANGUAGE"]
            + code_counts["LITERATURE_COMPARISON_LANGUAGE"]
            + code_counts["BLIND_VALIDATION_OVERCLAIM"]
        ),
        "causal_language_issue_count": code_counts["FORBIDDEN_CAUSAL_LANGUAGE"],
        "recommendation_language_issue_count": code_counts["RECOMMENDATION_LANGUAGE"],
        "hypothesis_language_issue_count": code_counts["HYPOTHESIS_LANGUAGE"],
        "blind_validation_overclaim_count": code_counts["BLIND_VALIDATION_OVERCLAIM"],
        "confidence_policy_issue_count": code_counts["UNSUPPORTED_CONFIDENCE_ASSIGNMENT"],
        "structured_validation_issues": [issue.to_record() for issue in validation_issues],
        "output_readability_checks": output_readability_checks,
    }


def summarize_interpretations(
    interpretations: tuple[Interpretation, ...],
    *,
    source_observations_loaded: tuple[str, ...],
    source_observations_missing: tuple[str, ...],
    validation_passed: bool,
) -> dict[str, Any]:
    category_counts = Counter(interpretation.category.value for interpretation in interpretations)
    status_counts = Counter(interpretation.status.value for interpretation in interpretations)
    confidence_counts = Counter(interpretation.confidence.value for interpretation in interpretations)
    return {
        "total_interpretations": len(interpretations),
        "count_by_category": dict(sorted(category_counts.items())),
        "count_by_status": dict(sorted(status_counts.items())),
        "count_by_confidence": dict(sorted(confidence_counts.items())),
        "supported_interpretation_count": status_counts["SUPPORTED"],
        "partially_supported_count": status_counts["PARTIALLY_SUPPORTED"],
        "conflicted_count": status_counts["CONFLICTED"],
        "insufficient_evidence_count": status_counts["INSUFFICIENT_EVIDENCE"],
        "source_observations_loaded": list(sorted(source_observations_loaded)),
        "source_observations_missing": list(sorted(source_observations_missing)),
        "validation_passed": validation_passed,
    }


def _interpretations_document(
    interpretations: tuple[Interpretation, ...],
    *,
    validation_summary: dict[str, Any],
    schema_version: str,
    software_version: str,
    source_observation_dir: Path | str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "software_version": software_version,
        "source_observation_directory": str(source_observation_dir),
        "generated_at": generated_at,
        "interpretations": [interpretation.to_record() for interpretation in interpretations],
        "validation_summary": validation_summary,
    }


def _flatten_interpretation(interpretation: Interpretation) -> dict[str, Any]:
    record = interpretation.to_record()
    flattened = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "interpretation_id",
        "category",
        "title",
        "claim",
        "status",
        "confidence",
        "supporting_observation_ids",
        "contradicting_observation_ids",
        "assumptions",
        "limitations",
        "evidence_summary",
        "reasoning_rule_ids",
        "created_at",
        "software_version",
        "source_observation_schema_version",
        "tags",
        "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_dependencies(path: Path, interpretations: tuple[Interpretation, ...]) -> None:
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
    rows: list[dict[str, str]] = []
    for interpretation in interpretations:
        links_by_observation = {link.observation_id: link for link in interpretation.evidence_summary}
        for observation_id in interpretation.supporting_observation_ids:
            link = links_by_observation.get(observation_id)
            rows.append(_dependency_row(interpretation, "supporting", observation_id, link))
        for observation_id in interpretation.contradicting_observation_ids:
            link = links_by_observation.get(observation_id)
            rows.append(_dependency_row(interpretation, "contradicting", observation_id, link))
    rows.sort(key=lambda row: (row["interpretation_id"], row["dependency_type"], row["observation_id"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dependency_row(
    interpretation: Interpretation,
    dependency_type: str,
    observation_id: str,
    link,
) -> dict[str, str]:
    return {
        "interpretation_id": interpretation.interpretation_id,
        "category": interpretation.category.value,
        "dependency_type": dependency_type,
        "observation_id": observation_id,
        "evidence_direction": "" if link is None else link.direction.value,
        "metric_names": json.dumps([] if link is None else list(link.metric_names), sort_keys=True),
        "provenance_ids": json.dumps([] if link is None else list(link.provenance_ids), sort_keys=True),
        "source_files": json.dumps([] if link is None else list(link.source_files), sort_keys=True),
        "reasoning_rule_ids": json.dumps(list(interpretation.reasoning_rule_ids), sort_keys=True),
    }


def _markdown_report(interpretations: tuple[Interpretation, ...]) -> str:
    lines = [
        "# Scientific Interpretations",
        "",
        "This report contains conservative interpretation records generated from validated Observation Engine outputs.",
        "",
    ]
    grouped: dict[str, list[Interpretation]] = {}
    for interpretation in interpretations:
        grouped.setdefault(interpretation.category.value, []).append(interpretation)
    for category in sorted(grouped):
        lines.extend((f"## {category}", ""))
        for interpretation in sorted(grouped[category], key=lambda item: item.interpretation_id):
            record = interpretation.to_record()
            lines.extend(
                (
                    f"### {interpretation.interpretation_id}",
                    "",
                    f"**Title:** {interpretation.title}",
                    "",
                    f"**Claim:** {interpretation.claim}",
                    "",
                    f"**Status:** {interpretation.status.value}",
                    "",
                    f"**Confidence:** {interpretation.confidence.value}",
                    "",
                    f"**Supporting observation IDs:** {', '.join(record['supporting_observation_ids']) or 'None'}",
                    "",
                    f"**Contradicting observation IDs:** {', '.join(record['contradicting_observation_ids']) or 'None'}",
                    "",
                    "**Evidence summary:**",
                    "",
                )
            )
            for link in interpretation.evidence_summary:
                lines.append(
                    f"- {link.observation_id} ({link.direction.value}): {link.rationale}; "
                    f"metrics={', '.join(link.to_record()['metric_names']) or 'None'}"
                )
            if not interpretation.evidence_summary:
                lines.append("- None")
            lines.extend(
                (
                    "",
                    f"**Assumptions:** {json.dumps(list(interpretation.assumptions), sort_keys=True)}",
                    "",
                    f"**Limitations:** {json.dumps(list(interpretation.limitations), sort_keys=True)}",
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
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
