"""Transparent confidence scoring for blind predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.blind_prediction.novelty_detection import normalized_entropy


@dataclass(frozen=True)
class ClassificationEvidence:
    """Ordered class-probability evidence."""

    predicted_label: str | None
    top_probability: float | None
    top_three: list[dict[str, Any]]
    probabilities: pd.DataFrame
    margin: float | None
    entropy: float | None


@dataclass(frozen=True)
class ConfidenceScore:
    """Composite confidence score and components."""

    numeric_confidence: float
    categorical_confidence: str
    components: dict[str, float]
    reasons: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Return component table for CSV output."""

        rows = [
            {
                "component": component,
                "score": score,
            }
            for component, score in self.components.items()
        ]
        rows.append(
            {
                "component": "composite_confidence",
                "score": self.numeric_confidence,
            }
        )
        rows.append(
            {
                "component": "categorical_confidence",
                "score": self.categorical_confidence,
            }
        )
        if self.reasons:
            rows.append({"component": "confidence_reasons", "score": ";".join(self.reasons)})
        return pd.DataFrame(rows)


def classify_probabilities(
    probabilities: np.ndarray,
    class_labels: list[str],
) -> ClassificationEvidence:
    """Summarize mean class probabilities across blind measurement units."""

    if probabilities.size == 0 or not class_labels:
        return ClassificationEvidence(
            predicted_label=None,
            top_probability=None,
            top_three=[],
            probabilities=pd.DataFrame(columns=["chemical", "probability", "rank"]),
            margin=None,
            entropy=None,
        )
    mean_probabilities = probabilities.mean(axis=0)
    total = float(np.sum(mean_probabilities))
    if total > 0:
        mean_probabilities = mean_probabilities / total
    order = np.argsort(mean_probabilities)[::-1]
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "chemical": class_labels[int(index)],
                "probability": float(mean_probabilities[int(index)]),
                "rank": int(rank),
            }
        )
    probabilities_table = pd.DataFrame(rows)
    top_three = probabilities_table.head(3).to_dict(orient="records")
    top_probability = float(mean_probabilities[int(order[0])])
    margin = _margin(mean_probabilities)
    entropy = _entropy(mean_probabilities)
    return ClassificationEvidence(
        predicted_label=str(class_labels[int(order[0])]),
        top_probability=top_probability,
        top_three=top_three,
        probabilities=probabilities_table,
        margin=margin,
        entropy=entropy,
    )


def calculate_confidence(
    *,
    classification: ClassificationEvidence,
    novelty_status: str,
    novelty_score: float | None,
    qc_status: str,
    valid_row_count: int,
    total_row_count: int,
    row_predicted_labels: list[str],
) -> ConfidenceScore:
    """Calculate deterministic composite confidence."""

    probability_score = float(classification.top_probability or 0.0)
    margin_score = float(classification.margin or 0.0)
    entropy_score = 1.0 - normalized_entropy(
        [float(row["probability"]) for row in classification.top_three]
    )
    if classification.probabilities is not None and not classification.probabilities.empty:
        entropy_score = 1.0 - normalized_entropy(classification.probabilities["probability"].to_numpy(dtype=float))
    replicate_score = _replicate_consistency(row_predicted_labels)
    feature_completeness = float(valid_row_count / total_row_count) if total_row_count else 0.0
    novelty_component = 1.0 - float(novelty_score if novelty_score is not None else 1.0)
    qc_component = {"PASS": 1.0, "PASS WITH WARNINGS": 0.75, "FAIL": 0.0}.get(qc_status, 0.0)
    components = {
        "classifier_probability": _clip(probability_score),
        "class_margin": _clip(margin_score),
        "entropy": _clip(entropy_score),
        "replicate_consistency": _clip(replicate_score),
        "feature_completeness": _clip(feature_completeness),
        "novelty": _clip(novelty_component),
        "qc": _clip(qc_component),
    }
    weights = {
        "classifier_probability": 0.25,
        "class_margin": 0.15,
        "entropy": 0.15,
        "replicate_consistency": 0.15,
        "feature_completeness": 0.10,
        "novelty": 0.10,
        "qc": 0.10,
    }
    numeric = float(sum(components[key] * weights[key] for key in weights))
    reasons: list[str] = []
    if qc_status == "FAIL":
        reasons.append("qc_failed")
    if novelty_status == "Out of Distribution":
        reasons.append("severe_novelty")
    if probability_score < 0.5:
        reasons.append("low_class_probability")
    if margin_score < 0.1:
        reasons.append("small_class_margin")
    category = _confidence_category(numeric)
    if qc_status == "FAIL" or novelty_status == "Out of Distribution":
        category = "Unreliable"
    return ConfidenceScore(
        numeric_confidence=_clip(numeric),
        categorical_confidence=category,
        components=components,
        reasons=reasons,
    )


def _margin(probabilities: np.ndarray) -> float:
    values = np.sort(np.asarray(probabilities, dtype=float))
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[-1])
    return float(values[-1] - values[-2])


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log2(values)))


def _replicate_consistency(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = pd.Series(labels, dtype="string").value_counts()
    return float(counts.max() / len(labels))


def _confidence_category(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.60:
        return "Moderate"
    if score >= 0.40:
        return "Low"
    return "Unreliable"


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
