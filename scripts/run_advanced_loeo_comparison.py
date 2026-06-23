"""Run advanced-feature LOEO panel comparison on generated feature tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.advanced_loeo_comparison import (
    combine_original_and_advanced_features,
    plot_advanced_panel_macro_f1,
    run_advanced_panel_comparison,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
ADVANCED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features_advanced.csv"
ADVANCED_PANEL_RESULTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "advanced_panel_optimization.csv"
ADVANCED_PANEL_FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "advanced_panel_macro_f1.png"


def run_advanced_loeo_comparison() -> dict[str, str]:
    if not ADVANCED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Advanced feature table not found: {ADVANCED_FEATURES_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Original feature table not found: {FEATURES_PATH}")

    original_features = pd.read_csv(FEATURES_PATH)
    advanced_features = pd.read_csv(ADVANCED_FEATURES_PATH)
    combined_features = combine_original_and_advanced_features(original_features, advanced_features)

    ADVANCED_PANEL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADVANCED_PANEL_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    panel_results = run_advanced_panel_comparison(combined_features)
    panel_results.to_csv(ADVANCED_PANEL_RESULTS_PATH, index=False)
    plot_advanced_panel_macro_f1(panel_results, ADVANCED_PANEL_FIGURE_PATH)

    return {
        "advanced_panel_optimization": str(ADVANCED_PANEL_RESULTS_PATH),
        "advanced_panel_macro_f1": str(ADVANCED_PANEL_FIGURE_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(run_advanced_loeo_comparison(), indent=2))
