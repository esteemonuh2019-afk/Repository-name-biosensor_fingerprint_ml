"""Run LOEO validation with experiment-normalized feature columns."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.normalized_features import (
    add_experiment_zscore_features,
    add_strain_experiment_zscore_features,
    get_normalized_feature_columns,
)
from src.model_evaluation.loeo_validation import (
    run_loeo_classification,
    run_loeo_regression,
)


FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
NORMALIZED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features_normalized.csv"
NORMALIZED_LOEO_METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "loeo_metrics_normalized.json"
)


def run_normalized_loeo() -> dict[str, Any]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")

    feature_df = pd.read_csv(FEATURES_PATH)
    normalized_df = add_experiment_zscore_features(feature_df)
    normalized_df = add_strain_experiment_zscore_features(normalized_df)
    normalized_feature_columns = get_normalized_feature_columns()

    NORMALIZED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(NORMALIZED_FEATURES_PATH, index=False)

    results = {
        "feature_columns": normalized_feature_columns,
        "classification": run_loeo_classification(
            normalized_df,
            feature_columns=normalized_feature_columns,
        ),
        "regression": run_loeo_regression(
            normalized_df,
            feature_columns=normalized_feature_columns,
        ),
    }
    NORMALIZED_LOEO_METRICS_PATH.write_text(
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
    print(json.dumps(_json_ready(run_normalized_loeo()), indent=2))
