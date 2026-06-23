"""Generate advanced kinetic features from processed biosensor time-series data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.advanced_features import extract_advanced_features


PROCESSED_DATA_PATH = PROJECT_ROOT / "outputs" / "tables" / "processed_data.csv"
ADVANCED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features_advanced.csv"


def run_advanced_feature_generation() -> dict[str, Any]:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

    processed_data = pd.read_csv(PROCESSED_DATA_PATH)
    advanced_features = extract_advanced_features(processed_data)

    ADVANCED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    advanced_features.to_csv(ADVANCED_FEATURES_PATH, index=False)

    return {
        "processed_rows": len(processed_data),
        "advanced_feature_rows": len(advanced_features),
        "advanced_features_path": str(ADVANCED_FEATURES_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(run_advanced_feature_generation(), indent=2))
