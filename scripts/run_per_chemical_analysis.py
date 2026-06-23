"""Run per-chemical LOEO analysis for the selected best strain panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.per_chemical_analysis import (
    generate_normalized_confusion_matrix,
    run_per_chemical_loeo,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PER_CHEMICAL_PATH = TABLES_DIR / "per_chemical_loeo.csv"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "normalized_confusion_matrix.png"
BEST_PANEL_STRAINS = ("BL027", "BL011")


def run_per_chemical_analysis() -> dict[str, str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(FEATURES_PATH)
    per_chemical_df = run_per_chemical_loeo(feature_df, BEST_PANEL_STRAINS)
    per_chemical_df.to_csv(PER_CHEMICAL_PATH, index=False)
    confusion_matrix_path = generate_normalized_confusion_matrix(
        feature_df,
        BEST_PANEL_STRAINS,
        CONFUSION_MATRIX_PATH,
    )

    return {
        "per_chemical_table": str(PER_CHEMICAL_PATH),
        "normalized_confusion_matrix": str(confusion_matrix_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_per_chemical_analysis(), indent=2))
