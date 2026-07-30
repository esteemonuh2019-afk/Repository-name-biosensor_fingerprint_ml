"""Model registry for Stage 8B concentration regression benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


RANDOM_SEED = 42


@dataclass(frozen=True)
class RegressionModelSpec:
    """Definition of one benchmark regressor."""

    model_id: str
    display_name: str
    factory: Callable[[int], object] | None
    is_tree_model: bool = False
    optional: bool = False
    available: bool = True
    skip_reason: str | None = None


def available_regression_model_specs(
    *,
    random_state: int = RANDOM_SEED,
    model_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[RegressionModelSpec], list[RegressionModelSpec]]:
    """Return available and skipped regression model specs in deterministic order."""

    del random_state
    requested = {model_id.strip() for model_id in model_ids or [] if model_id.strip()}
    specs = _required_specs() + _optional_specs()
    if requested:
        known = {spec.model_id for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown regression model ids: {', '.join(unknown)}")
        specs = [spec for spec in specs if spec.model_id in requested]

    available = [spec for spec in specs if spec.available]
    skipped = [spec for spec in specs if not spec.available]
    required_skipped = [spec for spec in skipped if not spec.optional]
    if required_skipped:
        formatted = ", ".join(spec.display_name for spec in required_skipped)
        raise RuntimeError(f"Required regressors are unavailable: {formatted}")
    return available, skipped


def required_regression_model_ids() -> tuple[str, ...]:
    """Return model ids that must be supported by Stage 8B."""

    return (
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "elastic_net",
        "ridge",
        "lasso",
        "support_vector_regression",
        "knn",
    )


def _required_specs() -> list[RegressionModelSpec]:
    return [
        RegressionModelSpec(
            model_id="random_forest",
            display_name="Random Forest Regressor",
            factory=lambda seed: RandomForestRegressor(
                n_estimators=100,
                random_state=seed,
                n_jobs=1,
            ),
            is_tree_model=True,
        ),
        RegressionModelSpec(
            model_id="extra_trees",
            display_name="Extra Trees Regressor",
            factory=lambda seed: ExtraTreesRegressor(
                n_estimators=100,
                random_state=seed,
                n_jobs=1,
            ),
            is_tree_model=True,
        ),
        RegressionModelSpec(
            model_id="gradient_boosting",
            display_name="Gradient Boosting Regressor",
            factory=lambda seed: GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                random_state=seed,
            ),
            is_tree_model=True,
        ),
        RegressionModelSpec(
            model_id="elastic_net",
            display_name="Elastic Net",
            factory=lambda seed: ElasticNet(
                alpha=0.05,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=seed,
            ),
        ),
        RegressionModelSpec(
            model_id="ridge",
            display_name="Ridge Regression",
            factory=lambda seed: Ridge(alpha=1.0, random_state=seed),
        ),
        RegressionModelSpec(
            model_id="lasso",
            display_name="Lasso Regression",
            factory=lambda seed: Lasso(
                alpha=0.05,
                max_iter=10000,
                random_state=seed,
            ),
        ),
        RegressionModelSpec(
            model_id="support_vector_regression",
            display_name="Support Vector Regression",
            factory=lambda seed: SVR(kernel="rbf", gamma="scale", C=10.0, epsilon=0.1),
        ),
        RegressionModelSpec(
            model_id="knn",
            display_name="kNN Regressor",
            factory=lambda seed: KNeighborsRegressor(n_neighbors=5),
        ),
    ]


def _optional_specs() -> list[RegressionModelSpec]:
    specs: list[RegressionModelSpec] = []
    try:
        from xgboost import XGBRegressor
    except Exception as error:  # noqa: BLE001 - optional dependency may fail at import time.
        specs.append(
            RegressionModelSpec(
                model_id="xgboost",
                display_name="XGBoost Regressor",
                factory=None,
                is_tree_model=True,
                optional=True,
                available=False,
                skip_reason=f"xgboost unavailable: {type(error).__name__}: {error}",
            )
        )
    else:
        specs.append(
            RegressionModelSpec(
                model_id="xgboost",
                display_name="XGBoost Regressor",
                factory=lambda seed: XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=seed,
                    n_jobs=1,
                    objective="reg:squarederror",
                    tree_method="hist",
                ),
                is_tree_model=True,
                optional=True,
            )
        )

    try:
        from lightgbm import LGBMRegressor
    except Exception as error:  # noqa: BLE001 - optional dependency may fail at import time.
        specs.append(
            RegressionModelSpec(
                model_id="lightgbm",
                display_name="LightGBM Regressor",
                factory=None,
                is_tree_model=True,
                optional=True,
                available=False,
                skip_reason=f"lightgbm unavailable: {type(error).__name__}: {error}",
            )
        )
    else:
        specs.append(
            RegressionModelSpec(
                model_id="lightgbm",
                display_name="LightGBM Regressor",
                factory=lambda seed: LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.08,
                    random_state=seed,
                    n_jobs=1,
                    verbose=-1,
                ),
                is_tree_model=True,
                optional=True,
            )
        )
    return specs
