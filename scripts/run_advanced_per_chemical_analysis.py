"""Run BL027 advanced per-chemical LOEO analysis on generated feature tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.advanced_loeo_comparison import combine_original_and_advanced_features
from src.model_evaluation.advanced_per_chemical_analysis import (
    generate_advanced_normalized_confusion_matrix,
    run_advanced_per_chemical_loeo,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
ADVANCED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features_advanced.csv"
PER_CHEMICAL_PATH = PROJECT_ROOT / "outputs" / "tables" / "advanced_per_chemical_loeo_BL027.csv"
CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "advanced_normalized_confusion_matrix_BL027.png"
)
BEST_PANEL_STRAINS = ("BL027",)


def run_advanced_per_chemical_analysis() -> dict[str, str]:
    if not ADVANCED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Advanced feature table not found: {ADVANCED_FEATURES_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Original feature table not found: {FEATURES_PATH}")

    original_features = pd.read_csv(FEATURES_PATH)
    advanced_features = pd.read_csv(ADVANCED_FEATURES_PATH)
    combined_features = combine_original_and_advanced_features(original_features, advanced_features)

    PER_CHEMICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    per_chemical_df = run_advanced_per_chemical_loeo(combined_features, BEST_PANEL_STRAINS)
    per_chemical_df.to_csv(PER_CHEMICAL_PATH, index=False)
    confusion_matrix_path = generate_advanced_normalized_confusion_matrix(
        combined_features,
        BEST_PANEL_STRAINS,
        CONFUSION_MATRIX_PATH,
    )

    return {
        "advanced_per_chemical_table": str(PER_CHEMICAL_PATH),
        "advanced_normalized_confusion_matrix": str(confusion_matrix_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_advanced_per_chemical_analysis(), indent=2))
