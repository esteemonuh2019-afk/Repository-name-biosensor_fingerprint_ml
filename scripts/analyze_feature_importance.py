"""Analyze feature importance and experiment effects from the real feature table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.feature_importance import (
    calculate_random_forest_feature_importance,
    generate_pca_by_chemical,
    generate_pca_by_experiment,
    plot_feature_importance,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def analyze_feature_importance() -> dict[str, str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(FEATURES_PATH)
    feature_importance_df = calculate_random_forest_feature_importance(feature_df)

    feature_importance_path = TABLES_DIR / "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)

    feature_importance_figure = plot_feature_importance(
        feature_importance_df,
        FIGURES_DIR / "feature_importance.png",
    )
    pca_by_chemical_figure = generate_pca_by_chemical(
        feature_df,
        FIGURES_DIR / "pca_by_chemical.png",
    )
    pca_by_experiment_figure = generate_pca_by_experiment(
        feature_df,
        FIGURES_DIR / "pca_by_experiment.png",
    )

    return {
        "feature_importance": str(feature_importance_path),
        "feature_importance_figure": str(feature_importance_figure),
        "pca_by_chemical": str(pca_by_chemical_figure),
        "pca_by_experiment": str(pca_by_experiment_figure),
    }


if __name__ == "__main__":
    print(json.dumps(analyze_feature_importance(), indent=2))
