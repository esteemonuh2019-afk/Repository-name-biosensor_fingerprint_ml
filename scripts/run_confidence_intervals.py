"""Run bootstrap confidence interval analysis for LOEO performance metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.confidence_intervals import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_RANDOM_STATE,
    summarize_metric_confidence_intervals,
)


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PER_CHEMICAL_LOEO_PATH = TABLES_DIR / "per_chemical_loeo.csv"
MODEL_METRICS_PATH = TABLES_DIR / "model_metrics.json"
CONFIDENCE_INTERVALS_PATH = TABLES_DIR / "confidence_intervals.csv"
REPORT_PATH = REPORTS_DIR / "confidence_interval_report.md"
CI_METRICS: tuple[str, ...] = ("precision", "recall", "f1")


def run_confidence_intervals() -> dict[str, str]:
    """Compute CI artifacts from existing per-chemical LOEO outputs."""

    if not PER_CHEMICAL_LOEO_PATH.exists():
        raise FileNotFoundError(
            f"Per-chemical LOEO table not found: {PER_CHEMICAL_LOEO_PATH}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    per_chemical_df = pd.read_csv(PER_CHEMICAL_LOEO_PATH)
    missing_columns = sorted(set(CI_METRICS) - set(per_chemical_df.columns))
    if missing_columns:
        raise ValueError(f"Missing CI metric columns: {', '.join(missing_columns)}")

    confidence_intervals = summarize_metric_confidence_intervals(
        per_chemical_df.loc[:, CI_METRICS]
    )
    confidence_intervals.to_csv(CONFIDENCE_INTERVALS_PATH, index=False)

    REPORT_PATH.write_text(
        _build_report(
            confidence_intervals,
            per_chemical_df,
            _load_model_metrics(),
        ),
        encoding="utf-8",
    )

    return {
        "confidence_intervals": str(CONFIDENCE_INTERVALS_PATH),
        "report": str(REPORT_PATH),
    }


def _load_model_metrics() -> dict[str, Any] | None:
    if not MODEL_METRICS_PATH.exists():
        return None

    return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))


def _build_report(
    confidence_intervals: pd.DataFrame,
    per_chemical_df: pd.DataFrame,
    model_metrics: dict[str, Any] | None,
) -> str:
    lines = [
        "# Confidence Interval Analysis",
        "",
        "## Purpose",
        (
            "Bootstrap confidence intervals quantify statistical uncertainty in "
            "per-chemical LOEO precision, recall, and F1 estimates for SSDD/V&V "
            "evidence."
        ),
        "",
        "## Inputs",
        f"- Per-chemical LOEO table: `{_relative_path(PER_CHEMICAL_LOEO_PATH)}`",
        f"- Per-chemical rows analyzed: {len(per_chemical_df)}",
        f"- CI metrics: {', '.join(CI_METRICS)}",
        "",
        "## Method",
        "- Resampling unit: per-chemical LOEO metric row",
        f"- Bootstrap resamples: {DEFAULT_N_BOOTSTRAP}",
        f"- Confidence level: {DEFAULT_CONFIDENCE:.0%}",
        f"- Random seed: {DEFAULT_RANDOM_STATE}",
        "",
        "## Confidence Intervals",
        _format_ci_table(confidence_intervals),
        "",
    ]

    classification_metrics = (model_metrics or {}).get("classification", {})
    if classification_metrics:
        lines.extend(
            [
                "## Aggregate Model Metrics",
                (
                    "Aggregate metrics from `outputs/tables/model_metrics.json` are "
                    "shown for context; confidence intervals above are estimated from "
                    "per-chemical LOEO metric variability."
                ),
                "",
                _format_model_metrics_table(classification_metrics),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            (
                "These intervals summarize observed metric variability across "
                "chemicals. Wider bounds indicate less stable performance across "
                "the contaminant panel and should be interpreted alongside LOEO, "
                "leave-one-strain-out, and per-chemical analyses."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_ci_table(confidence_intervals: pd.DataFrame) -> str:
    rows = [
        "| Metric | Mean | CI Lower | CI Upper | Confidence | Bootstrap Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in confidence_intervals.iterrows():
        rows.append(
            "| {metric} | {mean:.4f} | {ci_lower:.4f} | {ci_upper:.4f} | "
            "{confidence:.2f} | {n_bootstrap} |".format(
                metric=row["metric"],
                mean=row["mean"],
                ci_lower=row["ci_lower"],
                ci_upper=row["ci_upper"],
                confidence=row["confidence"],
                n_bootstrap=int(row["n_bootstrap"]),
            )
        )
    return "\n".join(rows)


def _format_model_metrics_table(classification_metrics: dict[str, Any]) -> str:
    metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    rows = [
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric_name in metric_names:
        if metric_name in classification_metrics:
            rows.append(f"| {metric_name} | {classification_metrics[metric_name]:.4f} |")
    return "\n".join(rows)


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


if __name__ == "__main__":
    print(json.dumps(run_confidence_intervals(), indent=2))
