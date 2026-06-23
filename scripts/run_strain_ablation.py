"""Run strain ablation analysis on the generated real biosensor feature table."""

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

from src.model_evaluation.strain_ablation import (
    run_leave_one_strain_out_loeo,
    run_single_strain_loeo,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SINGLE_STRAIN_PATH = TABLES_DIR / "single_strain_loeo.csv"
LEAVE_ONE_STRAIN_OUT_PATH = TABLES_DIR / "leave_one_strain_out_loeo.csv"
SINGLE_STRAIN_FIGURE_PATH = FIGURES_DIR / "single_strain_loeo.png"
LEAVE_ONE_STRAIN_OUT_FIGURE_PATH = FIGURES_DIR / "leave_one_strain_out_loeo.png"


def run_strain_ablation() -> dict[str, str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(FEATURES_PATH)
    single_strain_df = run_single_strain_loeo(feature_df)
    leave_one_strain_out_df = run_leave_one_strain_out_loeo(feature_df)

    single_strain_df.to_csv(SINGLE_STRAIN_PATH, index=False)
    leave_one_strain_out_df.to_csv(LEAVE_ONE_STRAIN_OUT_PATH, index=False)

    _plot_ablation_bar(
        single_strain_df,
        label_column="strain",
        output_path=SINGLE_STRAIN_FIGURE_PATH,
        title="Single-Strain LOEO Macro F1",
    )
    _plot_ablation_bar(
        leave_one_strain_out_df,
        label_column="removed_strain",
        output_path=LEAVE_ONE_STRAIN_OUT_FIGURE_PATH,
        title="Leave-One-Strain-Out LOEO Macro F1",
    )

    return {
        "single_strain_table": str(SINGLE_STRAIN_PATH),
        "leave_one_strain_out_table": str(LEAVE_ONE_STRAIN_OUT_PATH),
        "single_strain_figure": str(SINGLE_STRAIN_FIGURE_PATH),
        "leave_one_strain_out_figure": str(LEAVE_ONE_STRAIN_OUT_FIGURE_PATH),
    }


def _plot_ablation_bar(
    ablation_df: pd.DataFrame,
    label_column: str,
    output_path: Path,
    title: str,
) -> None:
    plot_df = ablation_df.dropna(subset=["macro_f1"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(plot_df[label_column].astype(str), plot_df["macro_f1"])
    ax.set_xlabel(label_column)
    ax.set_ylabel("Mean LOEO Macro F1")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run_strain_ablation(), indent=2))
