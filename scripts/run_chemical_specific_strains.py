"""Run chemical-specific strain ranking on the advanced feature table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.chemical_specific_strains import (
    plot_chemical_specific_strain_heatmap,
    rank_strains_per_chemical,
)


ADVANCED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features_advanced.csv"
RANKING_PATH = PROJECT_ROOT / "outputs" / "tables" / "chemical_specific_strain_rankings.csv"
HEATMAP_PATH = PROJECT_ROOT / "outputs" / "figures" / "chemical_specific_strain_heatmap.png"


def run_chemical_specific_strains() -> dict[str, str]:
    if not ADVANCED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Advanced feature table not found: {ADVANCED_FEATURES_PATH}")

    feature_df = pd.read_csv(ADVANCED_FEATURES_PATH)
    rankings = rank_strains_per_chemical(feature_df)

    RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(RANKING_PATH, index=False)
    heatmap_path = plot_chemical_specific_strain_heatmap(rankings, HEATMAP_PATH)

    return {
        "chemical_specific_strain_rankings": str(RANKING_PATH),
        "chemical_specific_strain_heatmap": str(heatmap_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_chemical_specific_strains(), indent=2))
