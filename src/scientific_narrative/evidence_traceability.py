"""Traceability index helpers for Stage 9B.2B."""

from __future__ import annotations

from src.scientific_narrative.scientific_summary import EvidenceInputRecord, SummaryRecord


def build_traceability_index(
    summaries: list[SummaryRecord],
    evidence_by_id: dict[str, EvidenceInputRecord],
) -> list[dict[str, str]]:
    """Build a row-per-summary-source-evidence traceability table."""

    rows: list[dict[str, str]] = []
    for summary in summaries:
        for evidence_id in summary.source_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                rows.append(
                    {
                        "summary_id": summary.summary_id,
                        "source_evidence_id": evidence_id,
                        "source_file": "",
                        "source_run": "",
                        "analysis_type": "",
                        "original_metric_name": "",
                        "original_metric_value": "",
                    }
                )
                continue
            rows.append(
                {
                    "summary_id": summary.summary_id,
                    "source_evidence_id": evidence.evidence_id,
                    "source_file": evidence.source_file,
                    "source_run": evidence.source_run,
                    "analysis_type": evidence.analysis_type,
                    "original_metric_name": evidence.metric_name,
                    "original_metric_value": "" if evidence.metric_value is None else str(evidence.metric_value),
                }
            )
    return rows


def traceability_coverage(summaries: list[SummaryRecord]) -> float:
    """Return fraction of non-missing summaries with at least one source evidence ID."""

    traceable = [summary for summary in summaries if summary.status != "MISSING"]
    if not traceable:
        return 0.0
    with_sources = sum(1 for summary in traceable if summary.source_evidence_ids)
    return with_sources / len(traceable)
