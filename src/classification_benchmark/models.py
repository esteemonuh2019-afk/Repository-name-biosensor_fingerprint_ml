"""Model registry for Stage 8A chemical classification benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    """Definition of one benchmark classifier."""

    model_id: str
    display_name: str
    factory: Callable[[int], object] | None
    is_tree_model: bool = False
    optional: bool = False
    available: bool = True
    skip_reason: str | None = None


class ContiguousLabelClassifier(ClassifierMixin, BaseEstimator):
    """Adapt classifiers that require labels numbered from 0 for each fit."""

    def __init__(self, estimator_factory: Callable[[], object]):
        self.estimator_factory = estimator_factory

    def fit(self, X, y):  # noqa: ANN001 - sklearn estimator protocol.
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(y)
        self.estimator_ = self.estimator_factory()
        self.estimator_.fit(X, encoded_y)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):  # noqa: ANN001 - sklearn estimator protocol.
        encoded = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(np.asarray(encoded, dtype=int))

    def predict_proba(self, X):  # noqa: ANN001 - sklearn estimator protocol.
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


def available_model_specs(
    *,
    random_state: int = RANDOM_SEED,
    model_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[ModelSpec], list[ModelSpec]]:
    """Return available and skipped model specs in deterministic order."""

    del random_state  # The seed is passed to factories when estimators are built.
    requested = {model_id.strip() for model_id in model_ids or [] if model_id.strip()}
    specs = _required_specs() + _optional_specs()
    if requested:
        known = {spec.model_id for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown model ids: {', '.join(unknown)}")
        specs = [spec for spec in specs if spec.model_id in requested]

    available = [spec for spec in specs if spec.available]
    skipped = [spec for spec in specs if not spec.available]
    required_skipped = [spec for spec in skipped if not spec.optional]
    if required_skipped:
        formatted = ", ".join(spec.display_name for spec in required_skipped)
        raise RuntimeError(f"Required classifiers are unavailable: {formatted}")
    return available, skipped


def required_model_ids() -> tuple[str, ...]:
    """Return model ids that must be supported by Stage 8A."""

    return (
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "logistic_regression",
        "support_vector_machine",
        "knn",
    )


def _required_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_id="random_forest",
            display_name="Random Forest",
            factory=lambda seed: RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            is_tree_model=True,
        ),
        ModelSpec(
            model_id="extra_trees",
            display_name="Extra Trees",
            factory=lambda seed: ExtraTreesClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            is_tree_model=True,
        ),
        ModelSpec(
            model_id="gradient_boosting",
            display_name="Gradient Boosting",
            factory=lambda seed: GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.08,
                max_depth=3,
                random_state=seed,
            ),
            is_tree_model=True,
        ),
        ModelSpec(
            model_id="logistic_regression",
            display_name="Logistic Regression",
            factory=lambda seed: LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ModelSpec(
            model_id="support_vector_machine",
            display_name="Support Vector Machine",
            factory=lambda seed: SVC(
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ModelSpec(
            model_id="knn",
            display_name="k-Nearest Neighbours",
            factory=lambda seed: KNeighborsClassifier(n_neighbors=5),
        ),
    ]


def _optional_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    try:
        from xgboost import XGBClassifier
    except Exception as error:  # noqa: BLE001 - optional dependency may fail at import time.
        specs.append(
            ModelSpec(
                model_id="xgboost",
                display_name="XGBoost",
                factory=None,
                is_tree_model=True,
                optional=True,
                available=False,
                skip_reason=f"xgboost unavailable: {type(error).__name__}: {error}",
            )
        )
    else:
        specs.append(
            ModelSpec(
                model_id="xgboost",
                display_name="XGBoost",
                factory=lambda seed: ContiguousLabelClassifier(
                    estimator_factory=lambda: XGBClassifier(
                        n_estimators=80,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=seed,
                        n_jobs=1,
                        eval_metric="mlogloss",
                        tree_method="hist",
                    )
                ),
                is_tree_model=True,
                optional=True,
            )
        )

    try:
        from lightgbm import LGBMClassifier
    except Exception as error:  # noqa: BLE001 - optional dependency may fail at import time.
        specs.append(
            ModelSpec(
                model_id="lightgbm",
                display_name="LightGBM",
                factory=None,
                is_tree_model=True,
                optional=True,
                available=False,
                skip_reason=f"lightgbm unavailable: {type(error).__name__}: {error}",
            )
        )
    else:
        specs.append(
            ModelSpec(
                model_id="lightgbm",
                display_name="LightGBM",
                factory=lambda seed: ContiguousLabelClassifier(
                    estimator_factory=lambda: LGBMClassifier(
                        n_estimators=80,
                        learning_rate=0.08,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                        verbose=-1,
                    )
                ),
                is_tree_model=True,
                optional=True,
            )
        )
    return specs
