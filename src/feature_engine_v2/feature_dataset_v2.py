"""Stage 8C Feature Engine V2 dataset container and output utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AdvancedFeatureDataset:
    """Engineered V2 features plus scientific metadata."""

    dataframe: pd.DataFrame
    feature_dictionary: pd.DataFrame
    feature_columns_by_family: dict[str, list[str]]
    metadata: dict[str, Any]
    summary: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write the Stage 8C feature dataset and metadata."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_paths = [
            target / "advanced_feature_dataset.csv",
            target / "advanced_feature_dictionary.csv",
            target / "advanced_feature_summary.json",
            target / "advanced_feature_report.md",
        ]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Advanced feature output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        self.dataframe.to_csv(output_paths[0], index=False, encoding="utf-8")
        self.feature_dictionary.to_csv(output_paths[1], index=False, encoding="utf-8")
        output_paths[2].write_text(
            json.dumps(self._summary_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output_paths[3].write_text(render_advanced_feature_report(self), encoding="utf-8")
        return output_paths

    def _summary_payload(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "feature_columns_by_family": self.feature_columns_by_family,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def render_advanced_feature_report(dataset: AdvancedFeatureDataset) -> str:
    """Render a Markdown report for Stage 8C Feature Engine V2."""

    lines = [
        "# Stage 8C Feature Engine V2 Report",
        "",
        "## Scope",
        "Feature Engine V2 is an isolated advanced feature-engineering layer. It does not replace the existing Stage 6B feature engine.",
        "",
        "## Summary",
    ]
    for key, value in dataset.summary.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Feature Families"])
    for family, columns in dataset.feature_columns_by_family.items():
        lines.append(f"- {family}: {len(columns)} features")

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in dataset.warnings) if dataset.warnings else lines.append("- None")
    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in dataset.errors) if dataset.errors else lines.append("- None")
    return "\n".join(lines) + "\n"
