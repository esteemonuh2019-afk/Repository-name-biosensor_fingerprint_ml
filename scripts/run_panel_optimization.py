"""Run candidate strain-panel optimization on the real feature table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.panel_optimization import run_candidate_panels


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PANEL_OPTIMIZATION_PATH = TABLES_DIR / "panel_optimization.csv"
MACRO_F1_FIGURE_PATH = FIGURES_DIR / "panel_optimization_macro_f1.png"
ACCURACY_FIGURE_PATH = FIGURES_DIR / "panel_optimization_accuracy.png"


def run_panel_optimization() -> dict[str, str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(FEATURES_PATH)
    panel_results = run_candidate_panels(feature_df)
    panel_results.to_csv(PANEL_OPTIMIZATION_PATH, index=False)

    _plot_metric(
        panel_results,
        metric_column="macro_f1",
        output_path=MACRO_F1_FIGURE_PATH,
        title="Panel Optimization Macro F1",
        ylabel="Mean LOEO Macro F1",
    )
    _plot_metric(
        panel_results,
        metric_column="accuracy",
        output_path=ACCURACY_FIGURE_PATH,
        title="Panel Optimization Accuracy",
        ylabel="Mean LOEO Accuracy",
    )

    return {
        "panel_optimization_table": str(PANEL_OPTIMIZATION_PATH),
        "macro_f1_figure": str(MACRO_F1_FIGURE_PATH),
        "accuracy_figure": str(ACCURACY_FIGURE_PATH),
    }


def _plot_metric(
    panel_results: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plot_df = panel_results.dropna(subset=[metric_column])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(plot_df["panel_name"], plot_df[metric_column])
    ax.set_xlabel("Panel")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run_panel_optimization(), indent=2))
