"""Run repeated random-seed robustness analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.repeated_runs import (
    DEFAULT_SEEDS,
    REPEATED_RUN_METRICS,
    create_repeated_run_boxplot,
    run_repeated_seed_evaluation,
    summarize_repeated_run_metrics,
)


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FEATURES_PATH = TABLES_DIR / "features.csv"
METRICS_PATH = TABLES_DIR / "repeated_run_metrics.csv"
SUMMARY_PATH = TABLES_DIR / "repeated_run_summary.csv"
REPORT_PATH = REPORTS_DIR / "repeated_run_analysis.md"
BOXPLOT_PATH = FIGURES_DIR / "repeated_run_boxplot.png"


def run_repeated_runs() -> dict[str, str]:
    """Generate repeated-run robustness evidence from the feature table."""

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(FEATURES_PATH)
    run_metrics = run_repeated_seed_evaluation(feature_df, DEFAULT_SEEDS)
    summary = summarize_repeated_run_metrics(run_metrics)
    boxplot_path = create_repeated_run_boxplot(run_metrics, BOXPLOT_PATH)

    run_metrics.to_csv(METRICS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    REPORT_PATH.write_text(
        _build_report(run_metrics, summary, boxplot_path),
        encoding="utf-8",
    )

    return {
        "metrics": str(METRICS_PATH),
        "summary": str(SUMMARY_PATH),
        "report": str(REPORT_PATH),
        "boxplot": str(boxplot_path),
    }


def _build_report(
    run_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    boxplot_path: Path,
) -> str:
    lines = [
        "# Repeated-Run Robustness Analysis",
        "",
        "## Purpose",
        (
            "Repeated random-seed evaluation quantifies model performance stability "
            "under randomized train/test partitioning and Random Forest initialization."
        ),
        "",
        "## Inputs",
        f"- Feature table: `{_relative_path(FEATURES_PATH)}`",
        f"- Runs: {len(run_metrics)}",
        f"- Seeds: {', '.join(str(seed) for seed in DEFAULT_SEEDS)}",
        f"- Metrics: {', '.join(REPEATED_RUN_METRICS)}",
        "",
        "## Method",
        "- Model: Random Forest classifier",
        "- Target: chemical identity",
        "- Split: stratified randomized train/test split when class counts allow",
        "- Test fraction: 0.20",
        "",
        "## Run-Level Metrics",
        _format_run_table(run_metrics),
        "",
        "## Summary Statistics",
        _format_summary_table(summary),
        "",
        "## Figure",
        f"- Boxplot: `{_relative_path(boxplot_path)}`",
        "",
        "## Interpretation",
        (
            "Low standard deviation and narrow min/max ranges indicate stable "
            "classification performance across random seeds. Larger ranges should "
            "be reviewed alongside LOEO, leave-one-strain-out, and confidence "
            "interval analyses before making claims about deployment robustness."
        ),
        "",
    ]
    return "\n".join(lines)


def _format_run_table(run_metrics: pd.DataFrame) -> str:
    rows = [
        "| Seed | Accuracy | Precision | Recall | F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in run_metrics.iterrows():
        rows.append(
            "| {seed} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                seed=int(row["seed"]),
                accuracy=row["accuracy"],
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
            )
        )
    return "\n".join(rows)


def _format_summary_table(summary: pd.DataFrame) -> str:
    rows = [
        "| Metric | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        rows.append(
            "| {metric} | {mean:.4f} | {std:.4f} | {min:.4f} | {max:.4f} |".format(
                metric=row["metric"],
                mean=row["mean"],
                std=row["std"],
                min=row["min"],
                max=row["max"],
            )
        )
    return "\n".join(rows)


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


if __name__ == "__main__":
    print(json.dumps(run_repeated_runs(), indent=2))
