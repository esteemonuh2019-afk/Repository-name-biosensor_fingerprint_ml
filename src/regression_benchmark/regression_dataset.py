"""Structured outputs for Stage 8B concentration regression benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RegressionBenchmarkResult:
    """Complete result bundle for a supervised regression benchmark."""

    summary: pd.DataFrame
    per_model_metrics: pd.DataFrame
    rankings: pd.DataFrame
    best_model_metrics: dict[str, Any]
    fold_metrics: pd.DataFrame
    prediction_vs_actual: pd.DataFrame
    residuals: pd.DataFrame
    feature_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    leave_one_strain_importance: pd.DataFrame
    metadata: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write all Stage 8B benchmark artifacts and figures."""

        from src.regression_benchmark.regression_plots import write_regression_plots

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_paths = [
            target / "regression_summary.csv",
            target / "best_regression_model.json",
            target / "per_model_metrics.csv",
            target / "fold_metrics.csv",
            target / "prediction_vs_actual.csv",
            target / "residuals.csv",
            target / "model_rankings.csv",
            target / "regression_report.md",
            target / "feature_importance.csv",
            target / "permutation_importance.csv",
            target / "leave_one_strain_importance.csv",
        ]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Regression output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        self.summary.to_csv(output_paths[0], index=False, encoding="utf-8")
        output_paths[1].write_text(
            json.dumps(_json_safe(self.best_model_metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.per_model_metrics.to_csv(output_paths[2], index=False, encoding="utf-8")
        self.fold_metrics.to_csv(output_paths[3], index=False, encoding="utf-8")
        self.prediction_vs_actual.to_csv(output_paths[4], index=False, encoding="utf-8")
        self.residuals.to_csv(output_paths[5], index=False, encoding="utf-8")
        self.rankings.to_csv(output_paths[6], index=False, encoding="utf-8")
        output_paths[7].write_text(render_regression_report(self), encoding="utf-8")
        self.feature_importance.to_csv(output_paths[8], index=False, encoding="utf-8")
        self.permutation_importance.to_csv(output_paths[9], index=False, encoding="utf-8")
        self.leave_one_strain_importance.to_csv(output_paths[10], index=False, encoding="utf-8")

        plot_paths = write_regression_plots(self, target, overwrite=overwrite)
        return [*output_paths, *plot_paths]


def render_regression_report(result: RegressionBenchmarkResult) -> str:
    """Render a concise Markdown report for the regression benchmark."""

    metadata = result.metadata
    best = result.best_model_metrics
    lines = [
        "# Stage 8B Concentration Regression Benchmark",
        "",
        "## Purpose",
        "",
        "This supervised benchmark compares regressors under identical data preparation, splitting, and metric rules. It does not perform blind prediction, concentration-specific deployment, classification, clustering, or PCA.",
        "",
        "## Dataset",
        "",
        f"- Sample count: {metadata.get('sample_count', 0)}",
        f"- Source rows: {metadata.get('source_row_count', 0)}",
        f"- Excluded rows: {metadata.get('excluded_row_count', 0)}",
        f"- Target column: {metadata.get('target_column', 'Concentration')}",
        f"- Target units: {metadata.get('target_units', 'ug/mL')}",
        f"- Concentration range: {metadata.get('concentration_min', 'NA')} to {metadata.get('concentration_max', 'NA')}",
        f"- Unique concentrations: {metadata.get('unique_concentration_count', 0)}",
        f"- Feature count: {metadata.get('feature_count', 0)}",
        "",
        "## Validation",
        "",
        f"- Strategy: {metadata.get('validation_strategy')}",
        f"- Effective splits: {metadata.get('effective_n_splits')}",
        f"- Repeats: {metadata.get('n_repeats')}",
        f"- Preprocessing: {metadata.get('preprocessing')}",
        "- Scaling is applied inside sklearn pipelines within each fold.",
        "- The original fingerprint table is copied before benchmarking and is not mutated.",
        "",
        "## Models",
        "",
    ]
    evaluated = metadata.get("models_evaluated", [])
    skipped = metadata.get("models_skipped", [])
    lines.extend(f"- Evaluated: {model}" for model in evaluated)
    if skipped:
        lines.extend(f"- Skipped optional: {model}" for model in skipped)
    else:
        lines.append("- Skipped optional: None")

    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"- Model: {best.get('model_name', 'NA')}",
            f"- R2 mean: {best.get('r2_mean', 'NA')}",
            f"- RMSE mean: {best.get('rmse_mean', 'NA')}",
            f"- MAE mean: {best.get('mae_mean', 'NA')}",
            "",
            "## Top Informative Features",
            "",
        ]
    )
    lines.extend(_top_feature_lines(result) or ["- No feature-importance values were available."])

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in result.warnings) if result.warnings else lines.append("- None")

    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in result.errors) if result.errors else lines.append("- None")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Concentration labels are parsed from validated fingerprint metadata; rows without finite numeric concentration are excluded and counted.",
            "- Metrics describe supervised concentration prediction on the current validated fingerprints only.",
            "- Permutation importance is an explanatory analysis of the fitted benchmark model, not an independent validation metric.",
            "- Leave-one-chemical-out regression is a research mode because all samples for a chemical are withheld together.",
        ]
    )
    return "\n".join(lines) + "\n"


def _top_feature_lines(result: RegressionBenchmarkResult) -> list[str]:
    if not result.permutation_importance.empty:
        table = result.permutation_importance.sort_values(
            ["importance_mean", "feature"],
            ascending=[False, True],
        ).head(10)
        return [
            f"- {row.feature}: {row.importance_mean:.6g}"
            for row in table.itertuples(index=False)
        ]
    if result.feature_importance.empty:
        return []
    table = (
        result.feature_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values(["importance", "feature"], ascending=[False, True])
        .head(10)
    )
    return [
        f"- {row.feature}: {row.importance:.6g}"
        for row in table.itertuples(index=False)
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
