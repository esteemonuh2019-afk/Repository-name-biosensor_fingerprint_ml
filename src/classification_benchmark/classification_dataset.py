"""Structured outputs for Stage 8A chemical classification benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ClassificationBenchmarkResult:
    """Complete result bundle for a supervised classification benchmark."""

    summary: pd.DataFrame
    rankings: pd.DataFrame
    best_model_metrics: dict[str, Any]
    confusion_matrix: pd.DataFrame
    per_class_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    leave_one_strain_importance: pd.DataFrame
    fold_metrics: pd.DataFrame
    metadata: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write all Stage 8A benchmark artifacts."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_paths = [
            target / "classification_summary.csv",
            target / "best_model_metrics.json",
            target / "confusion_matrix.csv",
            target / "per_class_metrics.csv",
            target / "feature_importance.csv",
            target / "permutation_importance.csv",
            target / "model_rankings.csv",
            target / "classification_report.md",
            target / "leave_one_strain_importance.csv",
            target / "fold_metrics.csv",
        ]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Classification output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        self.summary.to_csv(output_paths[0], index=False, encoding="utf-8")
        output_paths[1].write_text(
            json.dumps(_json_safe(self.best_model_metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.confusion_matrix.to_csv(output_paths[2], encoding="utf-8")
        self.per_class_metrics.to_csv(output_paths[3], index=False, encoding="utf-8")
        self.feature_importance.to_csv(output_paths[4], index=False, encoding="utf-8")
        self.permutation_importance.to_csv(output_paths[5], index=False, encoding="utf-8")
        self.rankings.to_csv(output_paths[6], index=False, encoding="utf-8")
        output_paths[7].write_text(render_classification_report(self), encoding="utf-8")
        self.leave_one_strain_importance.to_csv(output_paths[8], index=False, encoding="utf-8")
        self.fold_metrics.to_csv(output_paths[9], index=False, encoding="utf-8")
        return output_paths


def render_classification_report(result: ClassificationBenchmarkResult) -> str:
    """Render a concise Markdown report for the benchmark run."""

    metadata = result.metadata
    best = result.best_model_metrics
    lines = [
        "# Stage 8A Chemical Classification Benchmark",
        "",
        "## Purpose",
        "",
        "This supervised benchmark compares classifiers under identical data preparation, splitting, and metric rules. It does not perform blind prediction, concentration regression, clustering, or plotting.",
        "",
        "## Dataset",
        "",
        f"- Sample count: {metadata.get('sample_count', 0)}",
        f"- Class count: {metadata.get('class_count', 0)}",
        f"- Target column: {metadata.get('target_column', 'Chemical')}",
        f"- Feature count: {metadata.get('feature_count', 0)}",
        f"- Excluded rows: {metadata.get('excluded_row_count', 0)}",
        f"- Class imbalance ratio: {metadata.get('class_imbalance_ratio', 'NA')}",
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
            f"- Macro F1 mean: {best.get('f1_macro_mean', 'NA')}",
            f"- Balanced accuracy mean: {best.get('balanced_accuracy_mean', 'NA')}",
            f"- Accuracy mean: {best.get('accuracy_mean', 'NA')}",
            "",
            "## Top Informative Features",
            "",
        ]
    )
    top_features = _top_feature_lines(result)
    lines.extend(top_features or ["- No feature-importance values were available."])

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in result.warnings) if result.warnings else lines.append("- None")

    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in result.errors) if result.errors else lines.append("- None")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Reported cross-validation metrics estimate chemical-identification performance for the current validated fingerprint dataset only.",
            "- Permutation importance is an explanatory analysis of the fitted benchmark model, not an independent validation metric.",
            "- Leave-one-chemical-out validation is a research mode and intentionally tests labels absent from training folds.",
        ]
    )
    return "\n".join(lines) + "\n"


def _top_feature_lines(result: ClassificationBenchmarkResult) -> list[str]:
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
