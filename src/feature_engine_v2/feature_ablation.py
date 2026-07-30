"""Stage 8C feature-family ablation benchmarking."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.classification_benchmark import BenchmarkConfig, run_classification_benchmark
from src.feature_engine import extract_features
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.feature_engine_v2.ablation_dataset import FeatureAblationResult
from src.feature_engine_v2.feature_definitions import FEATURE_FAMILIES
from src.feature_engine_v2.feature_extractor_v2 import extract_advanced_features
from src.regression_benchmark import RegressionBenchmarkConfig, run_regression_benchmark


DEFAULT_CLASSIFICATION_MODELS = ("extra_trees",)
DEFAULT_REGRESSION_MODELS = ("extra_trees",)


def run_feature_family_ablation(
    canonical_dataframe: pd.DataFrame,
    *,
    classification_models: tuple[str, ...] | list[str] | None = DEFAULT_CLASSIFICATION_MODELS,
    regression_models: tuple[str, ...] | list[str] | None = DEFAULT_REGRESSION_MODELS,
    n_splits: int = 3,
    n_repeats: int = 1,
    preprocessing: str = "zscore",
    random_state: int = 42,
    permutation_repeats: int = 2,
) -> FeatureAblationResult:
    """Extract V2 features and benchmark feature-family contributions."""

    warnings: list[str] = []
    errors: list[str] = []
    advanced = extract_advanced_features(canonical_dataframe)
    warnings.extend(advanced.warnings)
    errors.extend(advanced.errors)

    current = extract_features(canonical_dataframe).dataframe.copy(deep=True)
    current = current.loc[~current.get("QC_Status", pd.Series(dtype=str)).astype("string").eq("fail")].reset_index(drop=True)
    benchmark_dataframe = _merge_current_and_advanced(current, advanced.dataframe)
    feature_sets = _feature_sets(advanced.feature_columns_by_family)

    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    for feature_set, family, columns in feature_sets:
        started = time.perf_counter()
        feature_columns = [column for column in columns if column in benchmark_dataframe.columns]
        family_columns = [column for column in feature_columns if column not in CORE_FEATURE_COLUMNS]
        work = benchmark_dataframe.copy(deep=True)
        classification_result = run_classification_benchmark(
            work,
            config=BenchmarkConfig(
                validation_strategy="repeated_stratified_kfold",
                preprocessing=preprocessing,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                model_ids=tuple(classification_models) if classification_models is not None else None,
                permutation_repeats=permutation_repeats,
                run_leave_one_strain_importance=False,
            ),
            feature_names=feature_columns,
        )
        regression_result = run_regression_benchmark(
            work,
            config=RegressionBenchmarkConfig(
                validation_strategy="repeated_kfold",
                preprocessing=preprocessing,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                model_ids=tuple(regression_models) if regression_models is not None else None,
                permutation_repeats=permutation_repeats,
                run_leave_one_strain_importance=False,
            ),
            feature_names=feature_columns,
        )
        runtime = time.perf_counter() - started
        class_best = classification_result.best_model_metrics
        reg_best = regression_result.best_model_metrics
        rows.append(
            {
                "feature_set": feature_set,
                "feature_family": family,
                "total_feature_count": len(feature_columns),
                "new_feature_count": len(family_columns),
                "classification_macro_f1": class_best.get("f1_macro_mean"),
                "classification_accuracy": class_best.get("accuracy_mean"),
                "classification_sample_count": class_best.get("sample_count"),
                "classification_best_model": class_best.get("model_name"),
                "regression_r2": reg_best.get("r2_mean"),
                "regression_rmse": reg_best.get("rmse_mean"),
                "regression_mae": reg_best.get("mae_mean"),
                "regression_sample_count": reg_best.get("sample_count"),
                "regression_best_model": reg_best.get("model_name"),
                "total_runtime_seconds": float(runtime),
                "classification_runtime_seconds": float(classification_result.summary["fit_time_seconds_mean"].sum())
                if "fit_time_seconds_mean" in classification_result.summary.columns
                else None,
                "regression_runtime_seconds": float(regression_result.summary["fit_time_seconds_mean"].sum())
                if "fit_time_seconds_mean" in regression_result.summary.columns
                else None,
            }
        )
        importance_rows.extend(
            _importance_rows(feature_set, family, "classification", classification_result.permutation_importance)
        )
        importance_rows.extend(
            _importance_rows(feature_set, family, "regression", regression_result.permutation_importance)
        )
        redundancy_rows.append(_redundancy_row(feature_set, family, work, feature_columns))
        warnings.extend(f"{feature_set} classification warning: {warning}" for warning in classification_result.warnings)
        warnings.extend(f"{feature_set} regression warning: {warning}" for warning in regression_result.warnings)

    summary = _with_gains(pd.DataFrame(rows))
    metadata = _metadata(
        summary,
        advanced=advanced,
        classification_models=classification_models,
        regression_models=regression_models,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
    return FeatureAblationResult(
        advanced_features=advanced,
        ablation_summary=summary,
        classification_comparison=_classification_comparison(summary),
        regression_r2_comparison=_regression_comparison(summary, metric="r2"),
        regression_rmse_comparison=_regression_comparison(summary, metric="rmse"),
        regression_mae_comparison=_regression_comparison(summary, metric="mae"),
        runtime_comparison=summary.loc[:, ["feature_set", "feature_family", "total_runtime_seconds"]].copy(),
        feature_family_importance=pd.DataFrame(importance_rows),
        feature_family_redundancy=pd.DataFrame(redundancy_rows),
        metadata=metadata,
        warnings=warnings,
        errors=errors,
    )


def _merge_current_and_advanced(current: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    keys = ["Experiment_ID", "Source_File", "Measurement_Unit_ID"]
    missing = [column for column in keys if column not in current.columns or column not in advanced.columns]
    if missing:
        raise ValueError(f"Missing merge keys for Stage 8C ablation: {', '.join(missing)}")
    advanced_feature_columns = [
        column
        for column in advanced.columns
        if column not in {
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
    ]
    merged = current.merge(
        advanced.loc[:, [*keys, *advanced_feature_columns]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    return merged.reset_index(drop=True)


def _feature_sets(family_columns: dict[str, list[str]]) -> list[tuple[str, str, list[str]]]:
    base = list(CORE_FEATURE_COLUMNS)
    sets: list[tuple[str, str, list[str]]] = [("current_core_features", "current", base)]
    for family in FEATURE_FAMILIES:
        sets.append((family, family, [*base, *family_columns.get(family, [])]))
    all_columns = [*base]
    for family in FEATURE_FAMILIES:
        all_columns.extend(family_columns.get(family, []))
    sets.append(("all_v2_families", "all_v2_families", all_columns))
    return sets


def _with_gains(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    baseline = summary.loc[summary["feature_set"].eq("current_core_features")].iloc[0]
    summary = summary.copy(deep=True)
    summary["classification_macro_f1_gain"] = summary["classification_macro_f1"] - baseline["classification_macro_f1"]
    summary["regression_r2_gain"] = summary["regression_r2"] - baseline["regression_r2"]
    summary["regression_rmse_delta"] = summary["regression_rmse"] - baseline["regression_rmse"]
    summary["regression_mae_delta"] = summary["regression_mae"] - baseline["regression_mae"]
    summary["runtime_increase_seconds"] = summary["total_runtime_seconds"] - baseline["total_runtime_seconds"]
    return summary


def _classification_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.loc[
        :,
        [
            "feature_set",
            "feature_family",
            "classification_macro_f1",
            "classification_macro_f1_gain",
            "classification_accuracy",
            "classification_sample_count",
        ],
    ].rename(columns={"classification_macro_f1": "macro_f1_mean", "classification_macro_f1_gain": "macro_f1_gain"})


def _regression_comparison(summary: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    columns = ["feature_set", "feature_family", f"regression_{metric}", "regression_sample_count"]
    if metric == "r2":
        columns.insert(3, "regression_r2_gain")
        return summary.loc[:, columns].rename(columns={"regression_r2": "r2_mean", "regression_r2_gain": "r2_gain"})
    delta = f"regression_{metric}_delta"
    columns.insert(3, delta)
    return summary.loc[:, columns].rename(columns={f"regression_{metric}": f"{metric}_mean", delta: f"{metric}_delta"})


def _importance_rows(feature_set: str, family: str, task: str, importance: pd.DataFrame) -> list[dict[str, Any]]:
    if importance.empty:
        return []
    value_column = "importance_mean" if "importance_mean" in importance.columns else "importance"
    rows = []
    ordered = importance.sort_values([value_column, "feature"], ascending=[False, True])
    for rank, row in enumerate(ordered.itertuples(index=False), start=1):
        rows.append(
            {
                "feature_set": feature_set,
                "feature_family": family,
                "task": task,
                "feature": getattr(row, "feature"),
                "importance": float(getattr(row, value_column)),
                "importance_rank": rank,
                "importance_label": "most_informative" if rank <= 5 else "least_informative" if rank > max(0, len(ordered) - 5) else "intermediate",
            }
        )
    return rows


def _redundancy_row(feature_set: str, family: str, dataframe: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    values = dataframe.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if finite.shape[0] < 3 or finite.shape[1] < 2:
        return {
            "feature_set": feature_set,
            "feature_family": family,
            "feature_count": len(feature_columns),
            "mean_abs_correlation": None,
            "high_correlation_pair_count": 0,
            "redundancy": "insufficient_data",
        }
    correlation = finite.corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    high = int((upper >= 0.95).sum())
    mean_abs = float(upper.mean()) if len(upper) else 0.0
    return {
        "feature_set": feature_set,
        "feature_family": family,
        "feature_count": len(feature_columns),
        "mean_abs_correlation": mean_abs,
        "high_correlation_pair_count": high,
        "redundancy": "high" if high else "moderate" if mean_abs >= 0.75 else "low",
    }


def _metadata(
    summary: pd.DataFrame,
    *,
    advanced,
    classification_models: tuple[str, ...] | list[str] | None,
    regression_models: tuple[str, ...] | list[str] | None,
    n_splits: int,
    n_repeats: int,
) -> dict[str, Any]:
    family_rows = summary.loc[~summary["feature_family"].isin(["current", "all_v2_families"])].copy()
    best = family_rows.sort_values(["classification_macro_f1_gain", "regression_r2_gain"], ascending=[False, False]).head(1)
    worst = family_rows.sort_values(["classification_macro_f1_gain", "regression_r2_gain"], ascending=[True, True]).head(1)
    all_row = summary.loc[summary["feature_set"].eq("all_v2_families")].head(1)
    return {
        "stage": "8C",
        "feature_engine_v2_isolated": True,
        "existing_pipeline_unchanged": True,
        "advanced_feature_rows": int(advanced.summary["advanced_feature_rows"]),
        "new_feature_count": int(advanced.summary["advanced_feature_count"]),
        "feature_family_count": int(advanced.summary["feature_family_count"]),
        "feature_set_count": int(len(summary)),
        "classification_models": list(classification_models) if classification_models is not None else ["all_available"],
        "regression_models": list(regression_models) if regression_models is not None else ["all_available"],
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "best_feature_family": None if best.empty else str(best.iloc[0]["feature_family"]),
        "worst_feature_family": None if worst.empty else str(worst.iloc[0]["feature_family"]),
        "best_classification_gain": None if best.empty else float(best.iloc[0]["classification_macro_f1_gain"]),
        "best_regression_r2_gain": None if family_rows.empty else float(family_rows["regression_r2_gain"].max()),
        "all_families_runtime_increase_seconds": None if all_row.empty else float(all_row.iloc[0]["runtime_increase_seconds"]),
    }
