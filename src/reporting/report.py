"""Markdown report generation for biosensor analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


REPORT_TITLE = "Whole-Cell Biosensor Fingerprint Analysis Report"


def generate_markdown_report(
    output_path: str | Path,
    sections: Mapping[str, str],
) -> Path:
    """Write a markdown report with the required biosensor report title."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {REPORT_TITLE}", ""]
    for title, text in sections.items():
        lines.extend([f"## {title}", "", str(text).strip(), ""])

    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination


def generate_validation_summary(metrics_dict: Mapping[str, Any]) -> str:
    """Convert model and scientific validation metrics into readable markdown."""

    lines: list[str] = []
    section_specs = (
        ("classification", "Classification Metrics"),
        ("regression", "Regression Metrics"),
        ("scientific_validation", "Scientific Validation Metrics"),
    )

    for key, title in section_specs:
        if key not in metrics_dict:
            continue

        lines.extend([f"## {title}", ""])
        section_metrics = metrics_dict[key]
        if isinstance(section_metrics, Mapping):
            lines.extend(_metric_lines(section_metrics))
        else:
            lines.append(f"- value: {section_metrics}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, Mapping):
            lines.append(f"- {metric_name}:")
            lines.extend(f"  - {nested_name}: {_format_metric_value(nested_value)}" for nested_name, nested_value in metric_value.items())
        else:
            lines.append(f"- {metric_name}: {_format_metric_value(metric_value)}")
    return lines


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
