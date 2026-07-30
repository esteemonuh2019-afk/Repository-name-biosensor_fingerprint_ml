"""Result objects and output writers for Stage 9A blind prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BLIND_OUTPUT_FILENAMES: tuple[str, ...] = (
    "blind_prediction_summary.json",
    "chemical_probabilities.csv",
    "concentration_prediction.csv",
    "prediction_confidence.csv",
    "novelty_assessment.csv",
    "influential_features.csv",
    "influential_strains.csv",
    "blind_sample_qc_report.md",
    "blind_prediction_report.md",
)


@dataclass(frozen=True)
class BlindPredictionResult:
    """Structured Stage 9A blind-prediction result."""

    source_files: list[str]
    canonical_qc: dict[str, Any]
    feature_qc: dict[str, Any]
    fingerprint_qc: dict[str, Any]
    predicted_chemical: str | None
    chemical_probabilities: pd.DataFrame
    chemical_confidence: float | None
    predicted_concentration: float | None
    concentration_units: str
    concentration_interval: tuple[float | None, float | None]
    regression_confidence: float | None
    novelty_score: float | None
    novelty_status: str
    novelty_assessment: pd.DataFrame
    influential_features: pd.DataFrame
    influential_strains: pd.DataFrame
    concentration_prediction: pd.DataFrame
    prediction_confidence: pd.DataFrame
    warnings: list[str]
    errors: list[str]
    prediction_passed: bool
    model_versions: dict[str, Any]
    pipeline_version: str
    top_three_candidates: list[dict[str, Any]]
    prediction_margin: float | None
    probability_entropy: float | None
    nearest_training_examples: pd.DataFrame

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write all required Stage 9A blind-prediction outputs."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_paths = [target / filename for filename in BLIND_OUTPUT_FILENAMES]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Blind-prediction output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        (target / "blind_prediction_summary.json").write_text(
            json.dumps(_json_safe(self.summary_dict()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.chemical_probabilities.to_csv(target / "chemical_probabilities.csv", index=False, encoding="utf-8")
        self.concentration_prediction.to_csv(target / "concentration_prediction.csv", index=False, encoding="utf-8")
        self.prediction_confidence.to_csv(target / "prediction_confidence.csv", index=False, encoding="utf-8")
        self.novelty_assessment.to_csv(target / "novelty_assessment.csv", index=False, encoding="utf-8")
        self.influential_features.to_csv(target / "influential_features.csv", index=False, encoding="utf-8")
        self.influential_strains.to_csv(target / "influential_strains.csv", index=False, encoding="utf-8")
        (target / "blind_sample_qc_report.md").write_text(render_qc_report(self), encoding="utf-8")
        (target / "blind_prediction_report.md").write_text(render_prediction_report(self), encoding="utf-8")
        return output_paths

    def summary_dict(self) -> dict[str, Any]:
        """Return machine-readable prediction summary."""

        return {
            "source_files": self.source_files,
            "predicted_chemical": self.predicted_chemical,
            "chemical_confidence": self.chemical_confidence,
            "top_three_candidates": self.top_three_candidates,
            "prediction_margin": self.prediction_margin,
            "probability_entropy": self.probability_entropy,
            "predicted_concentration": self.predicted_concentration,
            "concentration_units": self.concentration_units,
            "concentration_interval": {
                "lower": self.concentration_interval[0],
                "upper": self.concentration_interval[1],
            },
            "regression_confidence": self.regression_confidence,
            "novelty_score": self.novelty_score,
            "novelty_status": self.novelty_status,
            "prediction_passed": self.prediction_passed,
            "canonical_qc": self.canonical_qc,
            "feature_qc": self.feature_qc,
            "fingerprint_qc": self.fingerprint_qc,
            "warnings": self.warnings,
            "errors": self.errors,
            "model_versions": self.model_versions,
            "pipeline_version": self.pipeline_version,
            "true_labels_included": False,
        }


def render_qc_report(result: BlindPredictionResult) -> str:
    """Render blind-sample QC report."""

    lines = [
        "# Blind Sample QC Report",
        "",
        "## Overall Status",
        f"- Prediction passed: {result.prediction_passed}",
        f"- Feature QC status: {result.feature_qc.get('status')}",
        f"- Novelty status: {result.novelty_status}",
        "",
        "## Canonical QC",
    ]
    for key, value in result.canonical_qc.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Feature / Fingerprint QC"])
    for key, value in result.feature_qc.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in result.warnings) if result.warnings else lines.append("- None")
    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in result.errors) if result.errors else lines.append("- None")
    return "\n".join(lines) + "\n"


def render_prediction_report(result: BlindPredictionResult) -> str:
    """Render human-readable blind-prediction report."""

    lines = [
        "# Blind Prediction Report",
        "",
        "## Prediction",
        f"- Predicted chemical: {result.predicted_chemical}",
        f"- Chemical confidence: {result.chemical_confidence}",
        f"- Prediction margin: {result.prediction_margin}",
        f"- Probability entropy: {result.probability_entropy}",
        f"- Predicted concentration: {result.predicted_concentration}",
        f"- Concentration units: {result.concentration_units}",
        f"- Concentration interval: {result.concentration_interval[0]} to {result.concentration_interval[1]}",
        f"- Regression confidence: {result.regression_confidence}",
        f"- Novelty status: {result.novelty_status}",
        f"- Novelty score: {result.novelty_score}",
        f"- Prediction passed: {result.prediction_passed}",
        "",
        "## Top Chemical Candidates",
    ]
    if result.top_three_candidates:
        lines.extend(
            f"- {row['chemical']}: {row['probability']}"
            for row in result.top_three_candidates
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Confidence Note",
            "Reported probabilities are uncalibrated model probabilities unless the bundle metadata explicitly states otherwise. They should not be interpreted as certainty.",
            "",
            "## Influential Features",
        ]
    )
    if result.influential_features.empty:
        lines.append("- None")
    else:
        for row in result.influential_features.head(10).itertuples(index=False):
            lines.append(f"- {row.feature_name}: {row.importance} ({row.direction})")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in result.warnings) if result.warnings else lines.append("- None")
    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in result.errors) if result.errors else lines.append("- None")
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
