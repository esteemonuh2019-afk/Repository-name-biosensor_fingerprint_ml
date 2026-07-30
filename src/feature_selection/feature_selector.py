"""Stage 8D automatic feature selection and benchmark reruns."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import RFE, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classification_benchmark import (
    BenchmarkConfig,
    prepare_classification_data,
    run_classification_benchmark,
)
from src.feature_engine import extract_features
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.feature_engine_v2 import FEATURE_FAMILIES, extract_advanced_features
from src.regression_benchmark import (
    RegressionBenchmarkConfig,
    prepare_regression_data,
    run_regression_benchmark,
)
from src.feature_selection.selection_result import FeatureSelectionResult


FEATURE_SELECTION_VERSION = "0.1.0"

REQUIRED_SELECTOR_METHODS: tuple[str, ...] = (
    "rfe",
    "sequential_forward",
    "sequential_backward",
    "permutation",
    "tree_importance",
)

REDUCTION_LEVELS: tuple[int, ...] = (100, 75, 50, 25, 10)

MERGE_KEY_COLUMNS: tuple[str, ...] = (
    "Experiment_ID",
    "Source_File",
    "Measurement_Unit_ID",
)


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Configuration for Stage 8D feature selection and benchmark reruns."""

    selector_methods: tuple[str, ...] = REQUIRED_SELECTOR_METHODS
    reduction_levels: tuple[int, ...] = REDUCTION_LEVELS
    classification_model_ids: tuple[str, ...] = ("extra_trees",)
    regression_model_ids: tuple[str, ...] = ("extra_trees",)
    preprocessing: str = "zscore"
    n_splits: int = 3
    n_repeats: int = 1
    random_state: int = 42
    benchmark_permutation_importance: bool = False
    selection_permutation_repeats: int = 3
    selection_tree_estimators: int = 100
    rfe_step: float = 0.25
    sequential_candidate_pool: int = 12
    max_sequential_greedy_steps: int = 16
    selection_cv_splits: int = 2
    include_boruta: bool = True
    performance_tolerance: float = 1e-12


def run_feature_selection(
    canonical_dataframe: pd.DataFrame,
    *,
    config: FeatureSelectionConfig | None = None,
) -> FeatureSelectionResult:
    """Generate features, select subsets, and rerun supervised benchmarks."""

    config = config or FeatureSelectionConfig()
    warnings: list[str] = []
    errors: list[str] = []
    started = time.perf_counter()
    generated = build_generated_feature_table(canonical_dataframe)
    feature_dataframe = generated["dataframe"]
    feature_names = generated["feature_names"]
    feature_family_map = generated["feature_family_map"]
    warnings.extend(generated["warnings"])
    errors.extend(generated["errors"])

    class_prepared = prepare_classification_data(
        feature_dataframe,
        feature_names=feature_names,
        validation_strategy="repeated_stratified_kfold",
        requested_n_splits=config.n_splits,
    )
    regression_prepared = prepare_regression_data(
        feature_dataframe,
        feature_names=feature_names,
    )
    warnings.extend(f"classification preparation: {warning}" for warning in class_prepared.warnings)
    warnings.extend(f"regression preparation: {warning}" for warning in regression_prepared.warnings)

    ranking_tables: list[pd.DataFrame] = []
    completed_methods: set[str] = set()
    for task, prepared in (("classification", class_prepared), ("regression", regression_prepared)):
        for method in _selector_methods(config):
            if method == "boruta":
                boruta_table, boruta_warnings = _boruta_ranking(task, prepared, feature_family_map, config)
                warnings.extend(boruta_warnings)
                if boruta_table.empty:
                    continue
                ranking_tables.append(boruta_table)
                completed_methods.add(method)
                continue
            ranking_tables.append(_rank_features(task, method, prepared, feature_family_map, config))
            completed_methods.add(method)

    feature_ranking = _annotate_reduction_membership(
        pd.concat(ranking_tables, ignore_index=True) if ranking_tables else pd.DataFrame(),
        available_feature_count=len(feature_names),
        reduction_levels=config.reduction_levels,
    )
    if feature_ranking.empty:
        raise ValueError("No feature-selection methods completed successfully.")

    selected_subsets = _selected_subsets(feature_ranking, config.reduction_levels)
    class_rows, class_warnings = _evaluate_classification_subsets(feature_dataframe, selected_subsets, config)
    reg_rows, reg_warnings = _evaluate_regression_subsets(feature_dataframe, selected_subsets, config)
    warnings.extend(class_warnings)
    warnings.extend(reg_warnings)
    classification_after_selection = pd.DataFrame(class_rows)
    regression_after_selection = pd.DataFrame(reg_rows)

    recommendations = _recommend_feature_sets(
        classification_after_selection,
        regression_after_selection,
        selected_subsets,
        tolerance=config.performance_tolerance,
    )
    selected_features = _selected_features_table(
        selected_subsets,
        recommendations=recommendations,
        feature_family_map=feature_family_map,
        available_feature_count=len(feature_names),
    )
    feature_selection_summary = _summary_table(
        classification_after_selection,
        regression_after_selection,
        recommendations=recommendations,
    )
    performance_vs_feature_count = _performance_vs_feature_count(
        classification_after_selection,
        regression_after_selection,
    )
    metadata = _metadata(
        generated=generated,
        config=config,
        completed_methods=completed_methods,
        recommendations=recommendations,
        runtime_seconds=time.perf_counter() - started,
        warnings=warnings,
    )
    return FeatureSelectionResult(
        selected_features=selected_features,
        feature_ranking=feature_ranking,
        feature_selection_summary=feature_selection_summary,
        classification_after_selection=classification_after_selection,
        regression_after_selection=regression_after_selection,
        performance_vs_feature_count=performance_vs_feature_count,
        metadata=metadata,
        warnings=warnings,
        errors=errors,
    )


def build_generated_feature_table(canonical_dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build the Stage 8D feature-selection input table after feature generation."""

    warnings: list[str] = []
    errors: list[str] = []
    advanced = extract_advanced_features(canonical_dataframe)
    current = extract_features(canonical_dataframe).dataframe.copy(deep=True)
    current = current.loc[
        ~current.get("QC_Status", pd.Series(dtype=str)).astype("string").eq("fail")
    ].reset_index(drop=True)
    warnings.extend(advanced.warnings)
    errors.extend(advanced.errors)
    dataframe = _merge_current_and_advanced(current, advanced.dataframe)

    family_columns = advanced.feature_columns_by_family
    feature_names = [column for column in CORE_FEATURE_COLUMNS if column in dataframe.columns]
    for family in FEATURE_FAMILIES:
        feature_names.extend(column for column in family_columns.get(family, []) if column in dataframe.columns)
    feature_names = list(dict.fromkeys(feature_names))
    feature_family_map = {feature: "core" for feature in CORE_FEATURE_COLUMNS}
    for family, columns in family_columns.items():
        feature_family_map.update({column: family for column in columns})

    return {
        "dataframe": dataframe,
        "feature_names": feature_names,
        "feature_family_map": feature_family_map,
        "warnings": warnings,
        "errors": errors,
        "current_feature_rows": int(len(current)),
        "advanced_feature_rows": int(len(advanced.dataframe)),
        "generated_feature_rows": int(len(dataframe)),
        "available_feature_count": int(len(feature_names)),
        "feature_engine_v2_replaced": False,
    }


def _merge_current_and_advanced(current: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MERGE_KEY_COLUMNS if column not in current.columns or column not in advanced.columns]
    if missing:
        raise ValueError(f"Missing merge keys for Stage 8D feature selection: {', '.join(missing)}")
    advanced_skip = {
        "Experiment_ID",
        "Measurement_Unit_ID",
        "Source_File",
        "Strain",
        "Chemical",
        "Concentration",
        "Replicate_ID",
        "Duration",
        "QC_Status",
        "Advanced_Feature_QC_Flags",
    }
    advanced_columns = [column for column in advanced.columns if column not in advanced_skip]
    return current.merge(
        advanced.loc[:, [*MERGE_KEY_COLUMNS, *advanced_columns]],
        on=list(MERGE_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)


def _selector_methods(config: FeatureSelectionConfig) -> tuple[str, ...]:
    configured = tuple(_canonical_method(method) for method in config.selector_methods)
    if config.include_boruta and "boruta" not in configured:
        return (*configured, "boruta")
    return configured


def _canonical_method(method: str) -> str:
    normalized = str(method).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "rfe": "rfe",
        "recursive_feature_elimination": "rfe",
        "sequential_forward": "sequential_forward",
        "sequential_forward_selection": "sequential_forward",
        "sfs": "sequential_forward",
        "sequential_backward": "sequential_backward",
        "sequential_backward_selection": "sequential_backward",
        "sbs": "sequential_backward",
        "permutation": "permutation",
        "permutation_based": "permutation",
        "permutation_based_feature_selection": "permutation",
        "tree_importance": "tree_importance",
        "tree_based_importance": "tree_importance",
        "tree_based_importance_selection": "tree_importance",
        "boruta": "boruta",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported feature-selection method: {method}")
    return aliases[normalized]


def _rank_features(
    task: str,
    method: str,
    prepared: Any,
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    X = prepared.X.copy(deep=True)
    y = prepared.y
    feature_names = list(prepared.feature_names)
    screening = _screening_scores(task, X, y)
    if method == "rfe":
        return _rfe_ranking(task, X, y, feature_names, screening, feature_family_map, config)
    if method == "sequential_forward":
        return _sequential_forward_ranking(task, X, y, feature_names, screening, feature_family_map, config)
    if method == "sequential_backward":
        return _sequential_backward_ranking(task, X, y, feature_names, screening, feature_family_map, config)
    if method == "permutation":
        return _permutation_ranking(task, X, y, feature_names, feature_family_map, config)
    if method == "tree_importance":
        return _tree_importance_ranking(task, X, y, feature_names, feature_family_map, config)
    raise ValueError(f"Unsupported feature-selection method: {method}")


def _rfe_ranking(
    task: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    screening: pd.Series,
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    estimator = _tree_estimator(task, config)
    target_count = _feature_count_for_level(len(feature_names), min(config.reduction_levels))
    selector = RFE(
        estimator=estimator,
        n_features_to_select=target_count,
        step=config.rfe_step,
    )
    selector.fit(X, y)
    ranks = pd.Series(selector.ranking_, index=feature_names, dtype=float)
    ordered = sorted(feature_names, key=lambda feature: (ranks[feature], -screening.get(feature, 0.0), feature))
    scores = {feature: float((len(feature_names) - ranks[feature] + 1.0) + screening.get(feature, 0.0) * 1e-9) for feature in feature_names}
    return _ranking_table(
        task=task,
        method="rfe",
        ordered_features=ordered,
        scores=scores,
        score_type="recursive_feature_elimination_rank",
        feature_family_map=feature_family_map,
    )


def _sequential_forward_ranking(
    task: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    screening: pd.Series,
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    screening_order = _screening_order(screening)
    cv = _selection_cv(task, y, config)
    if cv is None:
        return _ranking_table(
            task=task,
            method="sequential_forward",
            ordered_features=screening_order,
            scores=screening.to_dict(),
            score_type="screening_fallback_no_valid_cv",
            feature_family_map=feature_family_map,
        )

    selected: list[str] = []
    remaining = set(feature_names)
    scores = screening.to_dict()
    max_steps = min(config.max_sequential_greedy_steps, len(feature_names))
    for _ in range(max_steps):
        candidates = [feature for feature in screening_order if feature in remaining][: config.sequential_candidate_pool]
        if not candidates:
            break
        best_feature = candidates[0]
        best_score = -math.inf
        for candidate in candidates:
            columns = [*selected, candidate]
            score = _cv_score(task, X.loc[:, columns], y, cv=cv, config=config)
            if score > best_score or (math.isclose(score, best_score) and candidate < best_feature):
                best_feature = candidate
                best_score = score
        selected.append(best_feature)
        remaining.remove(best_feature)
        scores[best_feature] = float(best_score)
    ordered = [*selected, *[feature for feature in screening_order if feature in remaining]]
    return _ranking_table(
        task=task,
        method="sequential_forward",
        ordered_features=ordered,
        scores=scores,
        score_type="screened_sequential_forward_cv",
        feature_family_map=feature_family_map,
    )


def _sequential_backward_ranking(
    task: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    screening: pd.Series,
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    weak_to_strong = list(reversed(_screening_order(screening)))
    cv = _selection_cv(task, y, config)
    if cv is None:
        return _ranking_table(
            task=task,
            method="sequential_backward",
            ordered_features=list(reversed(weak_to_strong)),
            scores=screening.to_dict(),
            score_type="screening_fallback_no_valid_cv",
            feature_family_map=feature_family_map,
        )

    kept = set(feature_names)
    eliminated: list[str] = []
    scores = screening.to_dict()
    max_steps = min(config.max_sequential_greedy_steps, max(0, len(feature_names) - 1))
    for _ in range(max_steps):
        candidates = [feature for feature in weak_to_strong if feature in kept][: config.sequential_candidate_pool]
        if not candidates or len(kept) <= 1:
            break
        best_remove = candidates[0]
        best_score = -math.inf
        for candidate in candidates:
            columns = sorted(kept - {candidate})
            score = _cv_score(task, X.loc[:, columns], y, cv=cv, config=config)
            if score > best_score or (math.isclose(score, best_score) and candidate < best_remove):
                best_remove = candidate
                best_score = score
        kept.remove(best_remove)
        eliminated.append(best_remove)
        scores[best_remove] = float(best_score)
    elimination_order = [*eliminated, *[feature for feature in weak_to_strong if feature in kept]]
    ordered = list(reversed(elimination_order))
    return _ranking_table(
        task=task,
        method="sequential_backward",
        ordered_features=ordered,
        scores=scores,
        score_type="screened_sequential_backward_cv",
        feature_family_map=feature_family_map,
    )


def _permutation_ranking(
    task: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    estimator = Pipeline([("scale", StandardScaler()), ("model", _tree_estimator(task, config))])
    estimator.fit(X, y)
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring="f1_macro" if task == "classification" else "r2",
        n_repeats=max(1, int(config.selection_permutation_repeats)),
        random_state=config.random_state,
        n_jobs=1,
    )
    scores = {
        feature: float(score)
        for feature, score in zip(feature_names, result.importances_mean, strict=True)
    }
    ordered = sorted(feature_names, key=lambda feature: (-scores[feature], feature))
    return _ranking_table(
        task=task,
        method="permutation",
        ordered_features=ordered,
        scores=scores,
        score_type="permutation_importance",
        feature_family_map=feature_family_map,
    )


def _tree_importance_ranking(
    task: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    estimator = _tree_estimator(task, config)
    estimator.fit(X, y)
    importances = getattr(estimator, "feature_importances_", np.zeros(len(feature_names)))
    scores = {
        feature: float(score)
        for feature, score in zip(feature_names, importances, strict=True)
    }
    ordered = sorted(feature_names, key=lambda feature: (-scores[feature], feature))
    return _ranking_table(
        task=task,
        method="tree_importance",
        ordered_features=ordered,
        scores=scores,
        score_type="tree_feature_importance",
        feature_family_map=feature_family_map,
    )


def _boruta_ranking(
    task: str,
    prepared: Any,
    feature_family_map: dict[str, str],
    config: FeatureSelectionConfig,
) -> tuple[pd.DataFrame, list[str]]:
    try:
        from boruta import BorutaPy
    except Exception as error:  # noqa: BLE001 - optional dependency may fail at import time.
        return pd.DataFrame(), [
            f"Optional feature selector skipped: Boruta unavailable: {type(error).__name__}: {error}"
        ]

    X = prepared.X.copy(deep=True)
    y = prepared.y
    feature_names = list(prepared.feature_names)
    estimator = _tree_estimator(task, config)
    selector = BorutaPy(
        estimator,
        n_estimators="auto",
        random_state=config.random_state,
        max_iter=50,
        verbose=0,
    )
    selector.fit(X.to_numpy(dtype=float), np.asarray(y))
    ranks = pd.Series(selector.ranking_, index=feature_names, dtype=float)
    support = pd.Series(selector.support_, index=feature_names).astype(int)
    ordered = sorted(feature_names, key=lambda feature: (ranks[feature], -support[feature], feature))
    scores = {feature: float((len(feature_names) - ranks[feature] + 1.0) + support[feature]) for feature in feature_names}
    return (
        _ranking_table(
            task=task,
            method="boruta",
            ordered_features=ordered,
            scores=scores,
            score_type="boruta_rank",
            feature_family_map=feature_family_map,
        ),
        ["Optional feature selector completed: Boruta."],
    )


def _screening_scores(task: str, X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    values = X.apply(pd.to_numeric, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if task == "classification":
            scores, _ = f_classif(values, y)
        else:
            scores, _ = f_regression(values, y)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(scores, index=list(X.columns), dtype=float)


def _screening_order(screening: pd.Series) -> list[str]:
    return (
        screening.sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
        .sort_values([0, "feature"], ascending=[False, True])["feature"]
        .astype(str)
        .tolist()
    )


def _tree_estimator(task: str, config: FeatureSelectionConfig):
    if task == "classification":
        return ExtraTreesClassifier(
            n_estimators=config.selection_tree_estimators,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=1,
        )
    return ExtraTreesRegressor(
        n_estimators=config.selection_tree_estimators,
        random_state=config.random_state,
        n_jobs=1,
    )


def _selection_cv(task: str, y: np.ndarray, config: FeatureSelectionConfig):
    splits = max(2, int(config.selection_cv_splits))
    if task == "classification":
        counts = pd.Series(y).value_counts()
        effective = min(splits, int(counts.min())) if not counts.empty else 0
        if effective < 2:
            return None
        return StratifiedKFold(n_splits=effective, shuffle=True, random_state=config.random_state)
    effective = min(splits, len(y))
    if effective < 2:
        return None
    return KFold(n_splits=effective, shuffle=True, random_state=config.random_state)


def _cv_score(
    task: str,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cv: Any,
    config: FeatureSelectionConfig,
) -> float:
    if task == "classification":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)
        scoring = "f1_macro"
    else:
        model = Ridge(alpha=1.0)
        scoring = "r2"
    estimator = Pipeline([("scale", StandardScaler()), ("model", model)])
    try:
        scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv, n_jobs=1)
    except Exception:  # noqa: BLE001 - fallback keeps the selector robust on small/singular subsets.
        return -math.inf
    finite = scores[np.isfinite(scores)]
    return float(finite.mean()) if len(finite) else -math.inf


def _ranking_table(
    *,
    task: str,
    method: str,
    ordered_features: list[str],
    scores: dict[str, float],
    score_type: str,
    feature_family_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for rank, feature in enumerate(ordered_features, start=1):
        rows.append(
            {
                "task": task,
                "selector_method": method,
                "feature_name": feature,
                "feature_family": feature_family_map.get(feature, "unknown"),
                "rank": int(rank),
                "score": float(scores.get(feature, 0.0)),
                "score_type": score_type,
            }
        )
    return pd.DataFrame(rows)


def _annotate_reduction_membership(
    ranking: pd.DataFrame,
    *,
    available_feature_count: int,
    reduction_levels: Iterable[int],
) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    annotated = ranking.copy(deep=True)
    for level in sorted(set(int(level) for level in reduction_levels), reverse=True):
        count = _feature_count_for_level(available_feature_count, level)
        annotated[f"selected_at_{level}_percent"] = annotated["rank"].le(count)
    return annotated.sort_values(["task", "selector_method", "rank", "feature_name"]).reset_index(drop=True)


def _selected_subsets(
    ranking: pd.DataFrame,
    reduction_levels: Iterable[int],
) -> dict[tuple[str, str, int], list[str]]:
    subsets: dict[tuple[str, str, int], list[str]] = {}
    for (task, method), group in ranking.groupby(["task", "selector_method"], sort=True):
        ordered = group.sort_values(["rank", "feature_name"])["feature_name"].astype(str).tolist()
        for level in sorted(set(int(level) for level in reduction_levels), reverse=True):
            count = _feature_count_for_level(len(ordered), level)
            subsets[(str(task), str(method), level)] = ordered[:count]
    return subsets


def _evaluate_classification_subsets(
    dataframe: pd.DataFrame,
    subsets: dict[tuple[str, str, int], list[str]],
    config: FeatureSelectionConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for (task, method, level), features in sorted(subsets.items()):
        if task != "classification":
            continue
        started = time.perf_counter()
        result = run_classification_benchmark(
            dataframe,
            config=BenchmarkConfig(
                validation_strategy="repeated_stratified_kfold",
                preprocessing=config.preprocessing,
                n_splits=config.n_splits,
                n_repeats=config.n_repeats,
                random_state=config.random_state,
                model_ids=config.classification_model_ids,
                run_permutation_importance=config.benchmark_permutation_importance,
                run_leave_one_strain_importance=False,
            ),
            feature_names=features,
        )
        best = result.best_model_metrics
        warnings.extend(f"{method} {level}% classification benchmark: {warning}" for warning in result.warnings)
        rows.append(
            {
                "task": "classification",
                "selector_method": method,
                "reduction_level_percent": int(level),
                "feature_count": int(len(features)),
                "feature_subset_id": _subset_id("classification", method, level),
                "model_id": best.get("model_id"),
                "model_name": best.get("model_name"),
                "macro_f1_mean": best.get("f1_macro_mean"),
                "macro_f1_std": best.get("f1_macro_std"),
                "balanced_accuracy_mean": best.get("balanced_accuracy_mean"),
                "balanced_accuracy_std": best.get("balanced_accuracy_std"),
                "accuracy_mean": best.get("accuracy_mean"),
                "accuracy_std": best.get("accuracy_std"),
                "sample_count": result.metadata.get("sample_count"),
                "excluded_row_count": result.metadata.get("excluded_row_count"),
                "fold_count": result.metadata.get("fold_count"),
                "runtime_seconds": float(time.perf_counter() - started),
            }
        )
    return rows, warnings


def _evaluate_regression_subsets(
    dataframe: pd.DataFrame,
    subsets: dict[tuple[str, str, int], list[str]],
    config: FeatureSelectionConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for (task, method, level), features in sorted(subsets.items()):
        if task != "regression":
            continue
        started = time.perf_counter()
        result = run_regression_benchmark(
            dataframe,
            config=RegressionBenchmarkConfig(
                validation_strategy="repeated_kfold",
                preprocessing=config.preprocessing,
                n_splits=config.n_splits,
                n_repeats=config.n_repeats,
                random_state=config.random_state,
                model_ids=config.regression_model_ids,
                run_permutation_importance=config.benchmark_permutation_importance,
                run_leave_one_strain_importance=False,
            ),
            feature_names=features,
        )
        best = result.best_model_metrics
        warnings.extend(f"{method} {level}% regression benchmark: {warning}" for warning in result.warnings)
        rows.append(
            {
                "task": "regression",
                "selector_method": method,
                "reduction_level_percent": int(level),
                "feature_count": int(len(features)),
                "feature_subset_id": _subset_id("regression", method, level),
                "model_id": best.get("model_id"),
                "model_name": best.get("model_name"),
                "r2_mean": best.get("r2_mean"),
                "r2_std": best.get("r2_std"),
                "rmse_mean": best.get("rmse_mean"),
                "rmse_std": best.get("rmse_std"),
                "mae_mean": best.get("mae_mean"),
                "mae_std": best.get("mae_std"),
                "sample_count": result.metadata.get("sample_count"),
                "excluded_row_count": result.metadata.get("excluded_row_count"),
                "fold_count": result.metadata.get("fold_count"),
                "runtime_seconds": float(time.perf_counter() - started),
            }
        )
    return rows, warnings


def _recommend_feature_sets(
    classification: pd.DataFrame,
    regression: pd.DataFrame,
    subsets: dict[tuple[str, str, int], list[str]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    class_rec = _recommend_classification(classification, tolerance=tolerance)
    reg_rec = _recommend_regression(regression, tolerance=tolerance)
    class_features = set(subsets.get(("classification", class_rec["selector_method"], int(class_rec["reduction_level_percent"])), []))
    reg_features = set(subsets.get(("regression", reg_rec["selector_method"], int(reg_rec["reduction_level_percent"])), []))
    research_features = sorted(class_features | reg_features)
    return {
        "default_classification_feature_set": {**class_rec, "features": sorted(class_features)},
        "default_regression_feature_set": {**reg_rec, "features": sorted(reg_features)},
        "research_feature_set": {
            "selector_method": "recommended_union",
            "reduction_level_percent": None,
            "feature_count": int(len(research_features)),
            "features": research_features,
            "decision_rule": "union(default_classification_feature_set, default_regression_feature_set)",
        },
    }


def _recommend_classification(classification: pd.DataFrame, *, tolerance: float) -> dict[str, Any]:
    if classification.empty:
        raise ValueError("Cannot recommend a classification feature set from an empty table.")
    baseline = (
        classification.loc[classification["reduction_level_percent"].eq(100)]
        .sort_values(["macro_f1_mean", "balanced_accuracy_mean", "selector_method"], ascending=[False, False, True])
        .iloc[0]
    )
    eligible = classification.loc[
        (classification["macro_f1_mean"] >= float(baseline["macro_f1_mean"]) - tolerance)
        & (classification["balanced_accuracy_mean"] >= float(baseline["balanced_accuracy_mean"]) - tolerance)
    ].copy()
    if eligible.empty:
        chosen = classification.sort_values(
            ["macro_f1_mean", "balanced_accuracy_mean", "feature_count", "selector_method"],
            ascending=[False, False, True, True],
        ).iloc[0]
        decision = "best_macro_f1_when_no_reduced_subset_maintained_full_feature_performance"
    else:
        chosen = eligible.sort_values(
            ["feature_count", "macro_f1_mean", "balanced_accuracy_mean", "selector_method"],
            ascending=[True, False, False, True],
        ).iloc[0]
        decision = "smallest_subset_maintaining_or_improving_full_feature_macro_f1_and_balanced_accuracy"
    return {
        "selector_method": str(chosen["selector_method"]),
        "reduction_level_percent": int(chosen["reduction_level_percent"]),
        "feature_count": int(chosen["feature_count"]),
        "macro_f1_mean": float(chosen["macro_f1_mean"]),
        "balanced_accuracy_mean": float(chosen["balanced_accuracy_mean"]),
        "baseline_macro_f1_mean": float(baseline["macro_f1_mean"]),
        "baseline_balanced_accuracy_mean": float(baseline["balanced_accuracy_mean"]),
        "decision_rule": decision,
    }


def _recommend_regression(regression: pd.DataFrame, *, tolerance: float) -> dict[str, Any]:
    if regression.empty:
        raise ValueError("Cannot recommend a regression feature set from an empty table.")
    baseline = (
        regression.loc[regression["reduction_level_percent"].eq(100)]
        .sort_values(["r2_mean", "rmse_mean", "mae_mean", "selector_method"], ascending=[False, True, True, True])
        .iloc[0]
    )
    eligible = regression.loc[
        (regression["r2_mean"] >= float(baseline["r2_mean"]) - tolerance)
        & (regression["rmse_mean"] <= float(baseline["rmse_mean"]) + tolerance)
        & (regression["mae_mean"] <= float(baseline["mae_mean"]) + tolerance)
    ].copy()
    if eligible.empty:
        chosen = regression.sort_values(
            ["r2_mean", "rmse_mean", "mae_mean", "feature_count", "selector_method"],
            ascending=[False, True, True, True, True],
        ).iloc[0]
        decision = "best_r2_when_no_reduced_subset_maintained_full_feature_performance"
    else:
        chosen = eligible.sort_values(
            ["feature_count", "r2_mean", "rmse_mean", "mae_mean", "selector_method"],
            ascending=[True, False, True, True, True],
        ).iloc[0]
        decision = "smallest_subset_maintaining_or_improving_full_feature_r2_rmse_and_mae"
    return {
        "selector_method": str(chosen["selector_method"]),
        "reduction_level_percent": int(chosen["reduction_level_percent"]),
        "feature_count": int(chosen["feature_count"]),
        "r2_mean": float(chosen["r2_mean"]),
        "rmse_mean": float(chosen["rmse_mean"]),
        "mae_mean": float(chosen["mae_mean"]),
        "baseline_r2_mean": float(baseline["r2_mean"]),
        "baseline_rmse_mean": float(baseline["rmse_mean"]),
        "baseline_mae_mean": float(baseline["mae_mean"]),
        "decision_rule": decision,
    }


def _selected_features_table(
    subsets: dict[tuple[str, str, int], list[str]],
    *,
    recommendations: dict[str, Any],
    feature_family_map: dict[str, str],
    available_feature_count: int,
) -> pd.DataFrame:
    class_key = recommendations["default_classification_feature_set"]
    reg_key = recommendations["default_regression_feature_set"]
    research_features = set(recommendations["research_feature_set"]["features"])
    rows: list[dict[str, Any]] = []
    for (task, method, level), features in sorted(subsets.items()):
        is_default_class_subset = (
            task == "classification"
            and method == class_key["selector_method"]
            and level == class_key["reduction_level_percent"]
        )
        is_default_reg_subset = (
            task == "regression"
            and method == reg_key["selector_method"]
            and level == reg_key["reduction_level_percent"]
        )
        for rank, feature in enumerate(features, start=1):
            rows.append(
                {
                    "task": task,
                    "selector_method": method,
                    "reduction_level_percent": int(level),
                    "feature_subset_id": _subset_id(task, method, level),
                    "available_feature_count": int(available_feature_count),
                    "feature_count": int(len(features)),
                    "feature_rank_within_subset": int(rank),
                    "feature_name": feature,
                    "feature_family": feature_family_map.get(feature, "unknown"),
                    "default_classification_feature_set": bool(is_default_class_subset),
                    "default_regression_feature_set": bool(is_default_reg_subset),
                    "research_feature_set": bool(feature in research_features),
                }
            )
    for rank, feature in enumerate(sorted(research_features), start=1):
        rows.append(
            {
                "task": "research",
                "selector_method": "recommended_union",
                "reduction_level_percent": pd.NA,
                "feature_subset_id": "research__recommended_union",
                "available_feature_count": int(available_feature_count),
                "feature_count": int(len(research_features)),
                "feature_rank_within_subset": int(rank),
                "feature_name": feature,
                "feature_family": feature_family_map.get(feature, "unknown"),
                "default_classification_feature_set": False,
                "default_regression_feature_set": False,
                "research_feature_set": True,
            }
        )
    return pd.DataFrame(rows)


def _summary_table(
    classification: pd.DataFrame,
    regression: pd.DataFrame,
    *,
    recommendations: dict[str, Any],
) -> pd.DataFrame:
    class_table = classification.copy(deep=True)
    class_table["primary_metric"] = class_table["macro_f1_mean"]
    class_table["secondary_metric"] = class_table["balanced_accuracy_mean"]
    class_table["rmse_mean"] = pd.NA
    class_table["mae_mean"] = pd.NA
    reg_table = regression.copy(deep=True)
    reg_table["primary_metric"] = reg_table["r2_mean"]
    reg_table["secondary_metric"] = -reg_table["rmse_mean"]
    reg_table["macro_f1_mean"] = pd.NA
    reg_table["balanced_accuracy_mean"] = pd.NA
    combined = pd.concat([class_table, reg_table], ignore_index=True, sort=False)
    class_rec = recommendations["default_classification_feature_set"]
    reg_rec = recommendations["default_regression_feature_set"]
    combined["recommended_default"] = (
        combined["task"].eq("classification")
        & combined["selector_method"].eq(class_rec["selector_method"])
        & combined["reduction_level_percent"].eq(class_rec["reduction_level_percent"])
    ) | (
        combined["task"].eq("regression")
        & combined["selector_method"].eq(reg_rec["selector_method"])
        & combined["reduction_level_percent"].eq(reg_rec["reduction_level_percent"])
    )
    return combined.sort_values(["task", "feature_count", "selector_method"]).reset_index(drop=True)


def _performance_vs_feature_count(classification: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    class_rows = []
    for row in classification.itertuples(index=False):
        class_rows.append(
            {
                "task": "classification",
                "selector_method": row.selector_method,
                "reduction_level_percent": int(row.reduction_level_percent),
                "feature_count": int(row.feature_count),
                "primary_metric": float(row.macro_f1_mean),
                "primary_metric_name": "macro_f1",
                "secondary_metric": float(row.balanced_accuracy_mean),
                "secondary_metric_name": "balanced_accuracy",
            }
        )
    reg_rows = []
    for row in regression.itertuples(index=False):
        reg_rows.append(
            {
                "task": "regression",
                "selector_method": row.selector_method,
                "reduction_level_percent": int(row.reduction_level_percent),
                "feature_count": int(row.feature_count),
                "primary_metric": float(row.r2_mean),
                "primary_metric_name": "r2",
                "secondary_metric": float(row.rmse_mean),
                "secondary_metric_name": "rmse",
            }
        )
    return pd.DataFrame([*class_rows, *reg_rows]).sort_values(
        ["task", "selector_method", "feature_count"]
    ).reset_index(drop=True)


def _metadata(
    *,
    generated: dict[str, Any],
    config: FeatureSelectionConfig,
    completed_methods: set[str],
    recommendations: dict[str, Any],
    runtime_seconds: float,
    warnings: list[str],
) -> dict[str, Any]:
    boruta_warnings = [warning for warning in warnings if "Boruta" in warning]
    boruta_status = "not_requested"
    if config.include_boruta:
        boruta_status = "completed" if "Optional feature selector completed: Boruta." in boruta_warnings else "skipped_unavailable"
    return {
        "stage": "8D",
        "feature_selection_version": FEATURE_SELECTION_VERSION,
        "feature_selection_after_feature_generation": True,
        "feature_engine_v2_replaced": False,
        "raw_luminescence_used_by_selection": False,
        "classification_benchmark_rerun": True,
        "regression_benchmark_rerun": True,
        "uses_sklearn_pipelines": True,
        "full_dataset_scaled_before_splitting": False,
        "random_state": int(config.random_state),
        "reduction_levels": list(config.reduction_levels),
        "selector_methods_requested": list(_selector_methods(config)),
        "selector_methods_completed": sorted(completed_methods),
        "boruta_status": boruta_status,
        "generated_feature_rows": generated["generated_feature_rows"],
        "current_feature_rows": generated["current_feature_rows"],
        "advanced_feature_rows": generated["advanced_feature_rows"],
        "available_feature_count": generated["available_feature_count"],
        "classification_models": list(config.classification_model_ids),
        "regression_models": list(config.regression_model_ids),
        "preprocessing": config.preprocessing,
        "n_splits": int(config.n_splits),
        "n_repeats": int(config.n_repeats),
        "runtime_seconds": float(runtime_seconds),
        **recommendations,
    }


def _feature_count_for_level(feature_count: int, level: int) -> int:
    if feature_count <= 0:
        return 0
    if int(level) >= 100:
        return int(feature_count)
    return max(1, int(math.ceil(feature_count * int(level) / 100.0)))


def _subset_id(task: str, method: str, level: int) -> str:
    return f"{task}__{method}__{int(level)}pct"
