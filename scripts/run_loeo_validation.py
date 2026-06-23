"""Run LOEO validation on the generated real biosensor feature table."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation.loeo_validation import (
    run_loeo_classification,
    run_loeo_regression,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
LOEO_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "loeo_metrics.json"


def run_loeo_validation() -> dict[str, Any]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    feature_df = pd.read_csv(FEATURES_PATH)
    results = {
        "classification": run_loeo_classification(feature_df),
        "regression": run_loeo_regression(feature_df),
    }

    LOEO_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOEO_METRICS_PATH.write_text(
        json.dumps(_json_ready(results), indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    print(json.dumps(_json_ready(run_loeo_validation()), indent=2))
