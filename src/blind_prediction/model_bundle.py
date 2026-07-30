"""Frozen, versioned model bundles for Stage 9A blind prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BLIND_PREDICTION_VERSION = "0.1.0"
PIPELINE_VERSION = "9A-blind-prediction-0.1.0"
MODEL_BUNDLE_FILENAME = "model_bundle.joblib"

TRAINING_OUTPUT_FILENAMES: tuple[str, ...] = (
    "model_bundle.joblib",
    "model_card.md",
    "model_metadata.json",
    "training_feature_manifest.csv",
    "classification_feature_set.csv",
    "regression_feature_set.csv",
    "training_distribution_summary.csv",
    "novelty_thresholds.json",
)


@dataclass(frozen=True)
class FeatureProfile:
    """Frozen classification and regression feature selections."""

    classification_features: list[str]
    regression_features: list[str]
    classification_profile: dict[str, Any] = field(default_factory=dict)
    regression_profile: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"


@dataclass(frozen=True)
class FrozenModelBundle:
    """Serializable Stage 9A model bundle.

    The bundle stores fitted estimators, fitted preprocessing transformers,
    selected feature manifests, training-only novelty thresholds, and model
    metadata. Blind samples are never appended to this object during prediction.
    """

    classifier_pipeline: Any
    global_regressor_pipeline: Any
    chemical_regressors: dict[str, Any]
    classification_features: list[str]
    regression_features: list[str]
    class_labels: list[str]
    concentration_units: str
    regression_strategy: str
    feature_engine_versions: dict[str, Any]
    canonical_schema_version: str
    model_metrics: dict[str, Any]
    training_data_summary: dict[str, Any]
    training_distribution: dict[str, Any]
    novelty_reference: dict[str, Any]
    novelty_thresholds: dict[str, Any]
    model_creation_timestamp: str
    software_version: str
    random_seeds: dict[str, Any]
    dependency_versions: dict[str, str | None]
    preprocessing: str
    classifier_model_id: str
    regressor_model_id: str
    time_window: dict[str, Any]
    required_strains: list[str]
    feature_family_map: dict[str, str]
    bundle_version: str = BLIND_PREDICTION_VERSION
    pipeline_version: str = PIPELINE_VERSION

    def save(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Save the bundle and required training artifacts."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_paths = [target / filename for filename in TRAINING_OUTPUT_FILENAMES]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Blind-prediction model output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        joblib.dump(self, target / MODEL_BUNDLE_FILENAME)
        metadata = self.metadata_dict()
        (target / "model_metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (target / "model_card.md").write_text(render_model_card(self), encoding="utf-8")
        self.training_feature_manifest().to_csv(
            target / "training_feature_manifest.csv",
            index=False,
            encoding="utf-8",
        )
        self.classification_feature_set().to_csv(
            target / "classification_feature_set.csv",
            index=False,
            encoding="utf-8",
        )
        self.regression_feature_set().to_csv(
            target / "regression_feature_set.csv",
            index=False,
            encoding="utf-8",
        )
        self.training_distribution_summary().to_csv(
            target / "training_distribution_summary.csv",
            index=False,
            encoding="utf-8",
        )
        (target / "novelty_thresholds.json").write_text(
            json.dumps(_json_safe(self.novelty_thresholds), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_paths

    def metadata_dict(self) -> dict[str, Any]:
        """Return metadata without embedded fitted estimator objects."""

        excluded = {"classifier_pipeline", "global_regressor_pipeline", "chemical_regressors"}
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in excluded
        }

    def classification_feature_set(self) -> pd.DataFrame:
        """Return the ordered classification feature manifest."""

        return pd.DataFrame(
            [
                {
                    "task": "classification",
                    "feature_rank": rank,
                    "feature_name": feature,
                    "feature_family": self.feature_family_map.get(feature, "unknown"),
                }
                for rank, feature in enumerate(self.classification_features, start=1)
            ]
        )

    def regression_feature_set(self) -> pd.DataFrame:
        """Return the ordered regression feature manifest."""

        return pd.DataFrame(
            [
                {
                    "task": "regression",
                    "feature_rank": rank,
                    "feature_name": feature,
                    "feature_family": self.feature_family_map.get(feature, "unknown"),
                }
                for rank, feature in enumerate(self.regression_features, start=1)
            ]
        )

    def training_feature_manifest(self) -> pd.DataFrame:
        """Return the union feature manifest used by the bundle."""

        rows: list[dict[str, Any]] = []
        all_features = sorted(set(self.classification_features) | set(self.regression_features))
        for feature in all_features:
            rows.append(
                {
                    "feature_name": feature,
                    "feature_family": self.feature_family_map.get(feature, "unknown"),
                    "used_for_classification": feature in self.classification_features,
                    "used_for_regression": feature in self.regression_features,
                    "classification_rank": _rank_or_none(self.classification_features, feature),
                    "regression_rank": _rank_or_none(self.regression_features, feature),
                }
            )
        return pd.DataFrame(rows)

    def training_distribution_summary(self) -> pd.DataFrame:
        """Return feature and label distribution summaries."""

        rows: list[dict[str, Any]] = []
        for feature, values in self.training_distribution.get("feature_summary", {}).items():
            rows.append({"summary_type": "feature", "name": feature, **values})
        for chemical, count in self.training_data_summary.get("class_counts", {}).items():
            rows.append({"summary_type": "class_count", "name": chemical, "count": count})
        for chemical, values in self.training_data_summary.get("concentration_ranges_by_chemical", {}).items():
            rows.append({"summary_type": "concentration_range", "name": chemical, **values})
        return pd.DataFrame(rows)


def load_model_bundle(model_dir: str | Path) -> FrozenModelBundle:
    """Load a frozen Stage 9A model bundle."""

    path = Path(model_dir) / MODEL_BUNDLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, FrozenModelBundle):
        raise TypeError("Loaded object is not a FrozenModelBundle.")
    return bundle


def load_feature_profile(feature_selection_dir: str | Path) -> FeatureProfile:
    """Load approved Stage 8D feature profiles from selected_features.csv."""

    source = Path(feature_selection_dir)
    selected_path = source / "selected_features.csv"
    summary_path = source / "feature_selection_summary.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)

    selected = pd.read_csv(selected_path)
    class_rows = _default_rows(selected, "default_classification_feature_set")
    reg_rows = _default_rows(selected, "default_regression_feature_set")
    if class_rows.empty:
        raise ValueError("No default classification feature set found in selected_features.csv.")
    if reg_rows.empty:
        raise ValueError("No default regression feature set found in selected_features.csv.")

    classification_features = _ordered_features(class_rows)
    regression_features = _ordered_features(reg_rows)
    class_profile: dict[str, Any] = {}
    reg_profile: dict[str, Any] = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        recommended = summary.loc[summary.get("recommended_default", pd.Series(False)).astype(str).eq("True")]
        for row in recommended.to_dict(orient="records"):
            if row.get("task") == "classification":
                class_profile = row
            elif row.get("task") == "regression":
                reg_profile = row

    return FeatureProfile(
        classification_features=classification_features,
        regression_features=regression_features,
        classification_profile=class_profile,
        regression_profile=reg_profile,
        source=str(source),
    )


def dependency_versions() -> dict[str, str | None]:
    """Return package versions needed to interpret the bundle."""

    packages = ("python", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost")
    versions: dict[str, str | None] = {}
    for package in packages:
        if package == "python":
            import sys

            versions[package] = sys.version.split()[0]
            continue
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def timestamp_utc() -> str:
    """Return an ISO timestamp in UTC."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def render_model_card(bundle: FrozenModelBundle) -> str:
    """Render a human-readable model card."""

    metrics = bundle.model_metrics
    lines = [
        "# Stage 9A Blind-Prediction Model Card",
        "",
        "## Intended Use",
        "",
        "This frozen model bundle predicts known chemical identity and concentration from biosensor feature fingerprints generated by the established canonical and feature pipelines.",
        "",
        "## Frozen Components",
        "",
        f"- Bundle version: {bundle.bundle_version}",
        f"- Pipeline version: {bundle.pipeline_version}",
        f"- Classifier: {bundle.classifier_model_id}",
        f"- Regressor: {bundle.regressor_model_id}",
        f"- Preprocessing: {bundle.preprocessing}",
        f"- Classification features: {len(bundle.classification_features)}",
        f"- Regression features: {len(bundle.regression_features)}",
        f"- Regression strategy: {bundle.regression_strategy}",
        f"- Class labels: {', '.join(bundle.class_labels)}",
        "",
        "## Training Summary",
        "",
        f"- Training rows for classification: {bundle.training_data_summary.get('classification_rows')}",
        f"- Training rows for regression: {bundle.training_data_summary.get('regression_rows')}",
        f"- Source files: {bundle.training_data_summary.get('source_file_count')}",
        f"- Strains observed: {', '.join(bundle.training_data_summary.get('strains', []))}",
        f"- Time window: {bundle.time_window.get('label')}",
        "",
        "## Model-Selection Evidence",
        "",
        f"- Classification Macro F1: {metrics.get('classification', {}).get('macro_f1_mean')}",
        f"- Classification balanced accuracy: {metrics.get('classification', {}).get('balanced_accuracy_mean')}",
        f"- Regression R2: {metrics.get('regression', {}).get('r2_mean')}",
        f"- Regression RMSE: {metrics.get('regression', {}).get('rmse_mean')}",
        f"- Regression MAE: {metrics.get('regression', {}).get('mae_mean')}",
        "",
        "## Leakage Controls",
        "",
        "- Blind samples are not used for feature selection, model fitting, preprocessing fitting, or novelty-threshold fitting.",
        "- The exact selected feature order is stored in the bundle.",
        "- Prediction refuses missing required features and ignores extra columns only with an explicit warning.",
        "",
        "## Limitations",
        "",
        "- Probabilities are model probabilities and must not be interpreted as certainty.",
        "- Chemical-specific concentration estimates are withheld when the predicted chemical lacks a valid chemical-specific regressor.",
        "- Out-of-distribution samples should be treated as unreliable even when a known class has the largest probability.",
    ]
    return "\n".join(lines) + "\n"


def _default_rows(selected: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in selected.columns:
        return pd.DataFrame()
    mask = selected[column].astype(str).str.casefold().eq("true")
    return selected.loc[mask].copy()


def _ordered_features(rows: pd.DataFrame) -> list[str]:
    ordered = rows.sort_values(["feature_rank_within_subset", "feature_name"], ascending=[True, True])
    return ordered["feature_name"].astype(str).tolist()


def _rank_or_none(features: list[str], feature: str) -> int | None:
    try:
        return features.index(feature) + 1
    except ValueError:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
