"""Novelty and out-of-distribution checks for blind prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


NOVELTY_STATUSES: tuple[str, ...] = (
    "Within Training Distribution",
    "Borderline",
    "Out of Distribution",
    "Unable to Assess",
)


@dataclass(frozen=True)
class NoveltyAssessment:
    """Distance and confidence based novelty summary."""

    novelty_score: float | None
    novelty_status: str
    nearest_training_distance: float | None
    class_centroid_distance: float | None
    max_probability: float | None
    prediction_margin: float | None
    entropy: float | None
    thresholds: dict[str, Any]
    nearest_training_examples: pd.DataFrame
    reasons: list[str]

    def to_row(self) -> dict[str, Any]:
        """Return a flat row for CSV output."""

        return {
            "novelty_score": self.novelty_score,
            "novelty_status": self.novelty_status,
            "nearest_training_distance": self.nearest_training_distance,
            "class_centroid_distance": self.class_centroid_distance,
            "max_probability": self.max_probability,
            "prediction_margin": self.prediction_margin,
            "entropy": self.entropy,
            "reasons": ";".join(self.reasons),
        }


def fit_novelty_reference(
    *,
    classifier_pipeline: Any,
    X: pd.DataFrame,
    labels: pd.Series,
    class_labels: list[str],
    metadata: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit novelty reference distributions using training data only."""

    transformed = _transform_with_preprocessor(classifier_pipeline, X)
    label_values = labels.astype(str).reset_index(drop=True)
    vectors = np.asarray(transformed, dtype=float)
    nearest_distances = _leave_one_nearest_distances(vectors)
    centroids = _class_centroids(vectors, label_values)
    centroid_distances = _own_class_centroid_distances(vectors, label_values, centroids)
    probabilities = _predict_probabilities(classifier_pipeline, X, expected_columns=len(class_labels))
    max_probabilities = probabilities.max(axis=1) if len(probabilities) else np.array([], dtype=float)
    sorted_probabilities = np.sort(probabilities, axis=1) if len(probabilities) else np.empty((0, 0))
    margins = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        if sorted_probabilities.shape[1] >= 2
        else max_probabilities
    )
    entropies = np.array([probability_entropy(row) for row in probabilities], dtype=float)
    thresholds = {
        "nearest_distance_borderline": _quantile(nearest_distances, 0.95),
        "nearest_distance_ood": _quantile(nearest_distances, 0.99),
        "centroid_distance_borderline": _quantile(centroid_distances, 0.95),
        "centroid_distance_ood": _quantile(centroid_distances, 0.99),
        "max_probability_borderline": _quantile(max_probabilities, 0.05),
        "max_probability_ood": _quantile(max_probabilities, 0.01),
        "margin_borderline": _quantile(margins, 0.05),
        "margin_ood": _quantile(margins, 0.01),
        "entropy_borderline": _quantile(entropies, 0.95),
        "entropy_ood": _quantile(entropies, 0.99),
    }
    reference = {
        "training_vectors": vectors,
        "training_labels": label_values.tolist(),
        "class_centroids": {label: vector.tolist() for label, vector in centroids.items()},
        "training_metadata": metadata.reset_index(drop=True).to_dict(orient="records"),
    }
    return reference, thresholds


def assess_novelty(
    *,
    classifier_pipeline: Any,
    X: pd.DataFrame,
    predicted_label: str | None,
    probabilities: np.ndarray,
    novelty_reference: dict[str, Any],
    thresholds: dict[str, Any],
) -> NoveltyAssessment:
    """Assess blind-sample novelty using frozen training references."""

    if X.empty or probabilities.size == 0:
        return NoveltyAssessment(
            novelty_score=None,
            novelty_status="Unable to Assess",
            nearest_training_distance=None,
            class_centroid_distance=None,
            max_probability=None,
            prediction_margin=None,
            entropy=None,
            thresholds=thresholds,
            nearest_training_examples=pd.DataFrame(),
            reasons=["no_valid_prediction_rows"],
        )

    transformed = _transform_with_preprocessor(classifier_pipeline, X)
    blind_vectors = np.asarray(transformed, dtype=float)
    training_vectors = np.asarray(novelty_reference.get("training_vectors", []), dtype=float)
    if training_vectors.size == 0:
        return NoveltyAssessment(
            novelty_score=None,
            novelty_status="Unable to Assess",
            nearest_training_distance=None,
            class_centroid_distance=None,
            max_probability=float(np.max(probabilities)),
            prediction_margin=_probability_margin(probabilities.mean(axis=0)),
            entropy=probability_entropy(probabilities.mean(axis=0)),
            thresholds=thresholds,
            nearest_training_examples=pd.DataFrame(),
            reasons=["missing_training_reference_vectors"],
        )

    distances = _pairwise_distances(blind_vectors, training_vectors)
    row_nearest = distances.min(axis=1)
    nearest_distance = float(np.mean(row_nearest))
    nearest_examples = _nearest_examples(distances, novelty_reference)
    centroid_distance = _centroid_distance(blind_vectors, predicted_label, novelty_reference)
    mean_probabilities = probabilities.mean(axis=0)
    max_probability = float(np.max(mean_probabilities))
    margin = _probability_margin(mean_probabilities)
    entropy = probability_entropy(mean_probabilities)
    reasons: list[str] = []
    ood_flags = [
        _above(nearest_distance, thresholds.get("nearest_distance_ood"), "nearest_training_distance_ood", reasons),
        _above(centroid_distance, thresholds.get("centroid_distance_ood"), "class_centroid_distance_ood", reasons),
        _below(max_probability, thresholds.get("max_probability_ood"), "max_probability_ood", reasons),
        _below(margin, thresholds.get("margin_ood"), "prediction_margin_ood", reasons),
        _above(entropy, thresholds.get("entropy_ood"), "entropy_ood", reasons),
    ]
    borderline_flags = [
        _above(nearest_distance, thresholds.get("nearest_distance_borderline"), "nearest_training_distance_borderline", reasons),
        _above(centroid_distance, thresholds.get("centroid_distance_borderline"), "class_centroid_distance_borderline", reasons),
        _below(max_probability, thresholds.get("max_probability_borderline"), "max_probability_borderline", reasons),
        _below(margin, thresholds.get("margin_borderline"), "prediction_margin_borderline", reasons),
        _above(entropy, thresholds.get("entropy_borderline"), "entropy_borderline", reasons),
    ]
    if any(ood_flags):
        status = "Out of Distribution"
    elif any(borderline_flags):
        status = "Borderline"
    else:
        status = "Within Training Distribution"
        reasons.append("within_training_thresholds")
    novelty_score = _novelty_score(
        nearest_distance=nearest_distance,
        centroid_distance=centroid_distance,
        max_probability=max_probability,
        margin=margin,
        entropy=entropy,
        thresholds=thresholds,
    )
    return NoveltyAssessment(
        novelty_score=novelty_score,
        novelty_status=status,
        nearest_training_distance=nearest_distance,
        class_centroid_distance=centroid_distance,
        max_probability=max_probability,
        prediction_margin=margin,
        entropy=entropy,
        thresholds=thresholds,
        nearest_training_examples=nearest_examples,
        reasons=reasons,
    )


def probability_entropy(probabilities: np.ndarray | list[float]) -> float:
    """Return Shannon entropy for a probability vector."""

    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log2(values)))


def normalized_entropy(probabilities: np.ndarray | list[float]) -> float:
    """Return entropy scaled to 0..1 for a probability vector."""

    values = np.asarray(probabilities, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(probability_entropy(values) / np.log2(len(values)))


def _transform_with_preprocessor(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    preprocessor = pipeline.named_steps.get("preprocess") if hasattr(pipeline, "named_steps") else None
    if preprocessor is None or preprocessor == "passthrough":
        return X.to_numpy(dtype=float)
    return np.asarray(preprocessor.transform(X), dtype=float)


def _predict_probabilities(pipeline: Any, X: pd.DataFrame, *, expected_columns: int) -> np.ndarray:
    if not hasattr(pipeline, "predict_proba"):
        return np.zeros((len(X), expected_columns), dtype=float)
    probabilities = np.asarray(pipeline.predict_proba(X), dtype=float)
    if probabilities.shape[1] == expected_columns:
        return probabilities
    aligned = np.zeros((len(X), expected_columns), dtype=float)
    width = min(expected_columns, probabilities.shape[1])
    aligned[:, :width] = probabilities[:, :width]
    return aligned


def _leave_one_nearest_distances(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=float)
    distances = _pairwise_distances(vectors, vectors)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _class_centroids(vectors: np.ndarray, labels: pd.Series) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(labels.astype(str).unique().tolist()):
        mask = labels.astype(str).eq(label).to_numpy()
        if mask.any():
            centroids[label] = vectors[mask].mean(axis=0)
    return centroids


def _own_class_centroid_distances(
    vectors: np.ndarray,
    labels: pd.Series,
    centroids: dict[str, np.ndarray],
) -> np.ndarray:
    distances = []
    label_values = labels.astype(str).tolist()
    for vector, label in zip(vectors, label_values, strict=True):
        centroid = centroids.get(label)
        if centroid is None:
            continue
        distances.append(float(np.linalg.norm(vector - centroid)))
    return np.asarray(distances, dtype=float)


def _centroid_distance(
    blind_vectors: np.ndarray,
    predicted_label: str | None,
    novelty_reference: dict[str, Any],
) -> float | None:
    if predicted_label is None:
        return None
    centroids = novelty_reference.get("class_centroids", {})
    if predicted_label not in centroids:
        return None
    centroid = np.asarray(centroids[predicted_label], dtype=float)
    blind_mean = blind_vectors.mean(axis=0)
    return float(np.linalg.norm(blind_mean - centroid))


def _nearest_examples(distances: np.ndarray, novelty_reference: dict[str, Any]) -> pd.DataFrame:
    metadata = novelty_reference.get("training_metadata", [])
    if distances.size == 0 or not metadata:
        return pd.DataFrame()
    nearest_indices = np.argsort(distances.min(axis=0))[:5]
    rows = []
    for index in nearest_indices:
        row = dict(metadata[int(index)]) if int(index) < len(metadata) else {}
        row["nearest_distance"] = float(distances[:, int(index)].min())
        rows.append(row)
    return pd.DataFrame(rows)


def _probability_margin(probabilities: np.ndarray) -> float:
    values = np.sort(np.asarray(probabilities, dtype=float))
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[-1])
    return float(values[-1] - values[-2])


def _above(value: float | None, threshold: Any, reason: str, reasons: list[str]) -> bool:
    if value is None or threshold is None:
        return False
    result = float(value) > float(threshold)
    if result:
        reasons.append(reason)
    return result


def _below(value: float | None, threshold: Any, reason: str, reasons: list[str]) -> bool:
    if value is None or threshold is None:
        return False
    result = float(value) < float(threshold)
    if result:
        reasons.append(reason)
    return result


def _novelty_score(
    *,
    nearest_distance: float,
    centroid_distance: float | None,
    max_probability: float,
    margin: float,
    entropy: float,
    thresholds: dict[str, Any],
) -> float:
    components = [
        _ratio_high_bad(nearest_distance, thresholds.get("nearest_distance_borderline"), thresholds.get("nearest_distance_ood")),
        _ratio_high_bad(centroid_distance, thresholds.get("centroid_distance_borderline"), thresholds.get("centroid_distance_ood")),
        _ratio_low_bad(max_probability, thresholds.get("max_probability_borderline"), thresholds.get("max_probability_ood")),
        _ratio_low_bad(margin, thresholds.get("margin_borderline"), thresholds.get("margin_ood")),
        _ratio_high_bad(entropy, thresholds.get("entropy_borderline"), thresholds.get("entropy_ood")),
    ]
    finite = [value for value in components if value is not None]
    return float(np.clip(np.mean(finite), 0.0, 1.0)) if finite else 1.0


def _ratio_high_bad(value: float | None, borderline: Any, ood: Any) -> float | None:
    if value is None or borderline is None or ood is None:
        return None
    if float(ood) <= float(borderline):
        return 1.0 if float(value) > float(ood) else 0.0
    return float(np.clip((float(value) - float(borderline)) / (float(ood) - float(borderline)), 0.0, 1.0))


def _ratio_low_bad(value: float | None, borderline: Any, ood: Any) -> float | None:
    if value is None or borderline is None or ood is None:
        return None
    if float(borderline) <= float(ood):
        return 1.0 if float(value) < float(ood) else 0.0
    return float(np.clip((float(borderline) - float(value)) / (float(borderline) - float(ood)), 0.0, 1.0))


def _quantile(values: np.ndarray, q: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    return float(np.quantile(finite, q))
