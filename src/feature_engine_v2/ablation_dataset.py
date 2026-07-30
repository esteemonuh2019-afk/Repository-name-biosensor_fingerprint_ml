"""Stage 8C feature-family ablation result container."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureAblationResult:
    """Output bundle for Stage 8C feature-family ablation."""

    advanced_features: Any
    ablation_summary: pd.DataFrame
    classification_comparison: pd.DataFrame
    regression_r2_comparison: pd.DataFrame
    regression_rmse_comparison: pd.DataFrame
    regression_mae_comparison: pd.DataFrame
    runtime_comparison: pd.DataFrame
    feature_family_importance: pd.DataFrame
    feature_family_redundancy: pd.DataFrame
    metadata: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write Stage 8C ablation outputs, feature tables, and figures."""

        from src.feature_engine_v2.ablation_plots import write_ablation_plots

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        created = self.advanced_features.write_outputs(target, overwrite=overwrite)

        tables = {
            "feature_family_ablation_summary.csv": self.ablation_summary,
            "feature_family_vs_macro_f1.csv": self.classification_comparison,
            "feature_family_vs_r2.csv": self.regression_r2_comparison,
            "feature_family_vs_rmse.csv": self.regression_rmse_comparison,
            "feature_family_vs_mae.csv": self.regression_mae_comparison,
            "feature_family_runtime.csv": self.runtime_comparison,
            "feature_family_importance.csv": self.feature_family_importance,
            "feature_family_redundancy.csv": self.feature_family_redundancy,
        }
        for filename, dataframe in tables.items():
            path = target / filename
            _ensure_can_write(path, overwrite=overwrite)
            dataframe.to_csv(path, index=False, encoding="utf-8")
            created.append(path)

        summary_path = target / "stage_8c_summary.json"
        _ensure_can_write(summary_path, overwrite=overwrite)
        summary_path.write_text(
            json.dumps(_json_safe(self.to_summary_dict()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        created.append(summary_path)

        report_path = target / "stage_8c_feature_engineering_report.md"
        _ensure_can_write(report_path, overwrite=overwrite)
        report_path.write_text(render_stage_8c_report(self), encoding="utf-8")
        created.append(report_path)

        created.extend(write_ablation_plots(self, target, overwrite=overwrite))
        return created

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""

        return {
            "metadata": self.metadata,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "best_feature_family": self.metadata.get("best_feature_family"),
            "worst_feature_family": self.metadata.get("worst_feature_family"),
            "classification_improvement": self.metadata.get("best_classification_gain"),
            "regression_improvement": self.metadata.get("best_regression_r2_gain"),
            "runtime_increase_seconds": self.metadata.get("all_families_runtime_increase_seconds"),
        }


def render_stage_8c_report(result: FeatureAblationResult) -> str:
    """Render a concise Markdown report for Stage 8C."""

    metadata = result.metadata
    lines = [
        "# Stage 8C Advanced Temporal Feature Engineering Benchmark",
        "",
        "## Scope",
        "Feature Engine V2 is isolated from the existing feature engine. This ablation compares current core features with V2 feature families added independently and combined.",
        "",
        "## Summary",
        f"- Advanced feature rows: {metadata.get('advanced_feature_rows', 0)}",
        f"- New feature count: {metadata.get('new_feature_count', 0)}",
        f"- Feature families: {metadata.get('feature_family_count', 0)}",
        f"- Benchmark feature sets: {metadata.get('feature_set_count', 0)}",
        f"- Best feature family: {metadata.get('best_feature_family')}",
        f"- Worst feature family: {metadata.get('worst_feature_family')}",
        f"- Best classification gain: {metadata.get('best_classification_gain')}",
        f"- Best regression R2 gain: {metadata.get('best_regression_r2_gain')}",
        "",
        "## Method",
        f"- Classification models: {', '.join(metadata.get('classification_models', []))}",
        f"- Regression models: {', '.join(metadata.get('regression_models', []))}",
        f"- Splits: {metadata.get('n_splits')} folds x {metadata.get('n_repeats')} repeats",
        "- Benchmarks use sklearn pipelines from Stage 8A and Stage 8B.",
        "- No raw luminescence reader is used by Feature Engine V2.",
    ]
    if result.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines) + "\n"


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value
