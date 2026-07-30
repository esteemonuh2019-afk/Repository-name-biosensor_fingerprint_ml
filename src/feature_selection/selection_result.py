"""Stage 8D feature-selection result container and output utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_OUTPUT_FILENAMES: tuple[str, ...] = (
    "selected_features.csv",
    "feature_ranking.csv",
    "feature_selection_summary.csv",
    "classification_after_selection.csv",
    "regression_after_selection.csv",
    "performance_vs_feature_count.csv",
    "feature_selection_report.md",
)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Complete Stage 8D feature-selection benchmark result."""

    selected_features: pd.DataFrame
    feature_ranking: pd.DataFrame
    feature_selection_summary: pd.DataFrame
    classification_after_selection: pd.DataFrame
    regression_after_selection: pd.DataFrame
    performance_vs_feature_count: pd.DataFrame
    metadata: dict[str, Any]
    warnings: list[str]
    errors: list[str]

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write the required Stage 8D output tables, report, and figures."""

        from src.feature_selection.selection_plots import write_feature_selection_plots

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_paths = [target / filename for filename in REQUIRED_OUTPUT_FILENAMES]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Feature-selection output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        self.selected_features.to_csv(output_paths[0], index=False, encoding="utf-8")
        self.feature_ranking.to_csv(output_paths[1], index=False, encoding="utf-8")
        self.feature_selection_summary.to_csv(output_paths[2], index=False, encoding="utf-8")
        self.classification_after_selection.to_csv(output_paths[3], index=False, encoding="utf-8")
        self.regression_after_selection.to_csv(output_paths[4], index=False, encoding="utf-8")
        self.performance_vs_feature_count.to_csv(output_paths[5], index=False, encoding="utf-8")
        output_paths[6].write_text(render_feature_selection_report(self), encoding="utf-8")
        plot_paths = write_feature_selection_plots(self, target, overwrite=overwrite)
        return [*output_paths, *plot_paths]


def render_feature_selection_report(result: FeatureSelectionResult) -> str:
    """Render the Stage 8D Markdown report."""

    metadata = result.metadata
    class_rec = metadata.get("default_classification_feature_set", {})
    reg_rec = metadata.get("default_regression_feature_set", {})
    research_rec = metadata.get("research_feature_set", {})
    lines = [
        "# Stage 8D Automatic Feature Selection and Model Optimisation",
        "",
        "## Scope",
        "",
        "Feature selection is applied after Stage 6B and Stage 8C feature generation. Feature Engine V2 is not replaced or modified.",
        "",
        "## Reproducibility and Leakage Control",
        "",
        f"- Random state: {metadata.get('random_state')}",
        f"- Reduction levels: {', '.join(str(level) for level in metadata.get('reduction_levels', []))}",
        "- Classification and regression reruns use the existing Stage 8A/8B sklearn Pipelines.",
        "- Preprocessing is fitted inside each benchmark split, not on the full dataset before splitting.",
        f"- Boruta status: {metadata.get('boruta_status')}",
        "",
        "## Dataset",
        "",
        f"- Generated feature rows: {metadata.get('generated_feature_rows', 0)}",
        f"- Available feature count: {metadata.get('available_feature_count', 0)}",
        f"- Selector methods completed: {', '.join(metadata.get('selector_methods_completed', []))}",
        "",
        "## Recommended Feature Sets",
        "",
        f"- Default Classification Feature Set: {class_rec.get('selector_method')} at {class_rec.get('reduction_level_percent')}% ({class_rec.get('feature_count')} features), Macro F1 {class_rec.get('macro_f1_mean')}, balanced accuracy {class_rec.get('balanced_accuracy_mean')}",
        f"- Default Regression Feature Set: {reg_rec.get('selector_method')} at {reg_rec.get('reduction_level_percent')}% ({reg_rec.get('feature_count')} features), R2 {reg_rec.get('r2_mean')}, RMSE {reg_rec.get('rmse_mean')}, MAE {reg_rec.get('mae_mean')}",
        f"- Research Feature Set: {research_rec.get('feature_count')} features from the union of the default classification and regression recommendations.",
        "",
        "## Interpretation",
        "",
        "- The selected sets are benchmarked on the current validated feature table and should be treated as model-development recommendations, not blind-prediction performance estimates.",
        "- The smallest subset is recommended only when it maintains or improves the full-feature benchmark according to the task-specific decision rule.",
    ]
    if result.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines) + "\n"
