"""Run specialist-strain ensemble LOEO classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.specialist_ensemble import (
    generate_specialist_confusion_matrix,
    run_specialist_ensemble_loeo,
)


ADVANCED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features_advanced.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "specialist_ensemble_metrics.json"
CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "specialist_ensemble_confusion_matrix.png"
)


def run_specialist_ensemble() -> dict[str, str]:
    if not ADVANCED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Advanced feature table not found: {ADVANCED_FEATURES_PATH}")

    feature_df = pd.read_csv(ADVANCED_FEATURES_PATH)
    loeo_result = run_specialist_ensemble_loeo(feature_df)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(_json_ready(loeo_result), indent=2) + "\n",
        encoding="utf-8",
    )
    confusion_matrix_path = generate_specialist_confusion_matrix(loeo_result, CONFUSION_MATRIX_PATH)

    return {
        "specialist_ensemble_metrics": str(METRICS_PATH),
        "specialist_ensemble_confusion_matrix": str(confusion_matrix_path),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    print(json.dumps(run_specialist_ensemble(), indent=2))
