"""Stage 8B supervised concentration regression benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.fingerprint import FINGERPRINT_FEATURE_COLUMNS, FingerprintDataset
from src.regression_benchmark.models import (
    RegressionModelSpec,
    available_regression_model_specs,
)
from src.regression_benchmark.regression_dataset import RegressionBenchmarkResult


BENCHMARK_VERSION = "0.1.0"
DEFAULT_VALIDATION_STRATEGY = "repeated_kfold"
DEFAULT_PREPROCESSING = "zscore"
SUPPORTED_PREPROCESSING = ("none", "zscore", "robust", "minmax")
SUPPORTED_VALIDATION_STRATEGIES = (
    "repeated_kfold",
    "leave_one_strain_out",
    "leave_one_chemical_out",
)


@dataclass(frozen=True)
class RegressionBenchmarkConfig:
    """Configuration for Stage 8B benchmark comparisons."""

    validation_strategy: str = DEFAULT_VALIDATION_STRATEGY
    preprocessing: str = DEFAULT_PREPROCESSING
    n_splits: int = 5
    n_repeats: int = 2
    random_state: int = 42
    target_column: str = "Concentration"
    group_column: str = "Strain"
    chemical_group_column: str = "Chemical"
    model_ids: tuple[str, ...] | None = None
    permutation_repeats: int = 5
    run_permutation_importance: bool = True
    run_leave_one_strain_importance: bool = True


@dataclass(frozen=True)
class RegressionPreparedData:
    """Validated feature matrix and numeric concentration targets."""

    dataframe: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    target_column: str
    target_units: str
    metadata: dict[str, Any]
    warnings: list[str]


def run_regression_benchmark(
    fingerprint_input: FingerprintDataset | pd.DataFrame,
    *,
    config: RegressionBenchmarkConfig | None = None,
    feature_names: Iterable[str] | None = None,
) -> RegressionBenchmarkResult:
    """Compare concentration regressors on validated fingerprint features."""

    config = config or RegressionBenchmarkConfig()
    strategy = _canonical_validation_strategy(config.validation_strategy)
    preprocessing = _canonical_preprocessing(config.preprocessing)
    source_dataframe, input_warnings = _fingerprint_dataframe(fingerprint_input)
    prepared = prepare_regression_data(
        source_dataframe,
        feature_names=list(feature_names or FINGERPRINT_FEATURE_COLUMNS),
        target_column=config.target_column,
    )
    splits, split_metadata, split_warnings = make_validation_splits(
        prepared,
        validation_strategy=strategy,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
        group_column=config.group_column,
        chemical_group_column=config.chemical_group_column,
    )
    model_specs, skipped_specs = available_regression_model_specs(
        random_state=config.random_state,
        model_ids=config.model_ids,
    )

    fold_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    model_prediction_tables: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Pipeline] = {}
    all_warnings = [*input_warnings, *prepared.warnings, *split_warnings]
    for skipped in skipped_specs:
        if skipped.skip_reason:
            all_warnings.append(f"Optional regressor skipped: {skipped.display_name}: {skipped.skip_reason}")

    for spec in model_specs:
        try:
            fold_table, prediction_table, full_estimator = _evaluate_model(
                spec,
                prepared=prepared,
                splits=splits,
                preprocessing=preprocessing,
                random_state=config.random_state,
            )
        except Exception as error:  # noqa: BLE001 - optional libraries can fail at fit time.
            if spec.optional:
                all_warnings.append(
                    f"Optional regressor skipped after fit failure: {spec.display_name}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            raise
        fold_tables.append(fold_table)
        model_prediction_tables[spec.model_id] = prediction_table
        fitted_models[spec.model_id] = full_estimator
        summary_rows.append(
            _summary_row(
                spec=spec,
                fold_table=fold_table,
                full_estimator=full_estimator,
            )
        )

    if not summary_rows:
        raise ValueError("No regression models completed successfully.")

    fold_metrics = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    per_model_metrics = _per_model_metrics(summary)
    rankings = rank_regression_models(summary)
    best_model_id = str(rankings.iloc[0]["model_id"])
    best_predictions = model_prediction_tables[best_model_id].copy(deep=True)
    residuals = best_predictions.copy(deep=True)
    feature_importance = _feature_importance_table(
        model_specs=model_specs,
        fitted_models=fitted_models,
        feature_names=prepared.feature_names,
    )
    permutation_table = (
        _permutation_importance_table(
            best_model_id=best_model_id,
            best_model_name=str(rankings.iloc[0]["model_name"]),
            estimator=fitted_models[best_model_id],
            prepared=prepared,
            n_repeats=config.permutation_repeats,
            random_state=config.random_state,
        )
        if config.run_permutation_importance
        else pd.DataFrame()
    )
    strain_importance = (
        _leave_one_strain_importance_table(
            model_specs=model_specs,
            prepared=prepared,
            preprocessing=preprocessing,
            random_state=config.random_state,
            group_column=config.group_column,
        )
        if config.run_leave_one_strain_importance and config.group_column in prepared.dataframe.columns
        else pd.DataFrame()
    )

    completed_model_ids = set(summary["model_id"].astype(str))
    completed_specs = [spec for spec in model_specs if spec.model_id in completed_model_ids]
    best_metrics = _best_model_metrics(rankings.iloc[0].to_dict(), metadata=prepared.metadata)
    metadata = {
        "stage": "8B",
        "benchmark_version": BENCHMARK_VERSION,
        "input_contract": "validated fingerprint dataset",
        "raw_luminescence_used": False,
        "feature_validation_bypassed": False,
        "fingerprint_qc_bypassed": False,
        "classification_framework_modified": False,
        "blind_prediction_performed": False,
        "uses_sklearn_pipelines": True,
        "full_dataset_scaled_before_splitting": False,
        "validation_strategy": strategy,
        "preprocessing": preprocessing,
        "requested_n_splits": int(config.n_splits),
        "effective_n_splits": int(split_metadata.get("effective_n_splits", 0)),
        "n_repeats": int(split_metadata.get("n_repeats", config.n_repeats)),
        "random_state": int(config.random_state),
        "models_evaluated": [spec.display_name for spec in completed_specs],
        "model_ids_evaluated": [spec.model_id for spec in completed_specs],
        "models_skipped": [spec.display_name for spec in skipped_specs],
        "feature_names": prepared.feature_names,
        "feature_count": len(prepared.feature_names),
        **prepared.metadata,
        **split_metadata,
    }
    return RegressionBenchmarkResult(
        summary=summary,
        per_model_metrics=per_model_metrics,
        rankings=rankings,
        best_model_metrics=best_metrics,
        fold_metrics=fold_metrics,
        prediction_vs_actual=best_predictions,
        residuals=residuals,
        feature_importance=feature_importance,
        permutation_importance=permutation_table,
        leave_one_strain_importance=strain_importance,
        metadata=metadata,
        warnings=all_warnings,
        errors=[],
    )


def prepare_regression_data(
    dataframe: pd.DataFrame,
    *,
    feature_names: list[str],
    target_column: str = "Concentration",
) -> RegressionPreparedData:
    """Validate and copy fingerprint rows for supervised concentration regression."""

    source = dataframe.copy(deep=True)
    missing_features = [feature for feature in feature_names if feature not in source.columns]
    if missing_features:
        raise ValueError(f"Missing fingerprint feature columns: {', '.join(missing_features)}")
    if source.empty:
        raise ValueError("Fingerprint dataset is empty.")

    target_values, resolved_target_column, target_units, target_warnings = _resolve_concentration_target(
        source,
        target_column=target_column,
    )
    values = source.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    finite_feature_mask = values.apply(lambda column: np.isfinite(column.astype(float))).all(axis=1)
    target_numeric = pd.to_numeric(target_values, errors="coerce")
    finite_target_mask = target_numeric.map(_is_finite_number)
    nonnegative_target_mask = finite_target_mask & (target_numeric >= 0)
    usable_mask = finite_feature_mask & nonnegative_target_mask

    warnings = list(target_warnings)
    excluded_row_count = int((~usable_mask).sum())
    if excluded_row_count:
        warnings.append(
            "Rows excluded from regression because features or numeric concentration targets were unusable: "
            f"{excluded_row_count}."
        )
    negative_target_count = int((finite_target_mask & (target_numeric < 0)).sum())
    if negative_target_count:
        warnings.append(f"Negative concentration targets excluded: {negative_target_count}.")

    usable = source.loc[usable_mask].copy(deep=True).reset_index(drop=True)
    usable_features = values.loc[usable_mask].astype(float).reset_index(drop=True)
    for feature in feature_names:
        usable[feature] = usable_features[feature].to_numpy(dtype=float)
    usable["Concentration_Target_ug_mL"] = target_numeric.loc[usable_mask].astype(float).reset_index(drop=True)

    if len(usable) < 2:
        raise ValueError("Concentration regression requires at least two usable rows.")
    if usable["Concentration_Target_ug_mL"].nunique(dropna=True) < 2:
        raise ValueError("Concentration regression requires at least two unique target values.")

    target = usable["Concentration_Target_ug_mL"].to_numpy(dtype=float)
    concentration_counts = usable["Concentration_Target_ug_mL"].value_counts().sort_index()
    metadata = {
        "source_row_count": int(len(source)),
        "sample_count": int(len(usable)),
        "excluded_row_count": int(excluded_row_count),
        "target_column": resolved_target_column,
        "target_units": target_units,
        "concentration_min": float(np.min(target)),
        "concentration_max": float(np.max(target)),
        "concentration_median": float(np.median(target)),
        "unique_concentration_count": int(len(concentration_counts)),
        "concentration_counts": {str(key): int(value) for key, value in concentration_counts.items()},
        "zero_concentration_count": int(np.sum(target == 0)),
        "duplicated_measurement_unit_rows": _duplicated_rows(usable, "Measurement_Unit_ID"),
        "duplicate_fingerprint_rows": int(usable.loc[:, feature_names].duplicated(keep=False).sum()),
    }
    if metadata["zero_concentration_count"]:
        warnings.append(
            "Zero concentration targets retained for regression; MAPE excludes zero-valued actuals."
        )
    if metadata["duplicated_measurement_unit_rows"]:
        warnings.append(
            "Duplicated Measurement_Unit_ID rows retained for benchmark traceability: "
            f"{metadata['duplicated_measurement_unit_rows']}."
        )
    if metadata["duplicate_fingerprint_rows"]:
        warnings.append(
            "Duplicate fingerprint vectors retained for benchmark traceability: "
            f"{metadata['duplicate_fingerprint_rows']}."
        )

    return RegressionPreparedData(
        dataframe=usable,
        X=usable.loc[:, feature_names].copy(deep=True),
        y=target,
        feature_names=list(feature_names),
        target_column=resolved_target_column,
        target_units=target_units,
        metadata=metadata,
        warnings=warnings,
    )


def make_validation_splits(
    prepared: RegressionPreparedData,
    *,
    validation_strategy: str = DEFAULT_VALIDATION_STRATEGY,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    group_column: str = "Strain",
    chemical_group_column: str = "Chemical",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Create deterministic validation splits without preprocessing leakage."""

    strategy = _canonical_validation_strategy(validation_strategy)
    warnings: list[str] = []
    indices = np.arange(len(prepared.y))
    splits: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"n_repeats": int(n_repeats)}

    if strategy == "repeated_kfold":
        effective_n_splits = min(int(n_splits), int(len(prepared.y)))
        if effective_n_splits < 2:
            raise ValueError("Repeated K-fold regression requires at least 2 usable rows.")
        if effective_n_splits < n_splits:
            warnings.append(f"Effective fold count reduced from {n_splits} to {effective_n_splits}.")
        splitter = RepeatedKFold(
            n_splits=effective_n_splits,
            n_repeats=max(1, int(n_repeats)),
            random_state=random_state,
        )
        for fold_index, (train_index, test_index) in enumerate(splitter.split(prepared.X), start=1):
            splits.append(_split_record("fold", fold_index, train_index, test_index))
        metadata["effective_n_splits"] = int(effective_n_splits)
        metadata["fold_count"] = int(len(splits))
        return splits, metadata, warnings

    if strategy == "leave_one_strain_out":
        if group_column not in prepared.dataframe.columns:
            raise ValueError(f"Missing group column for leave-one-strain-out: {group_column}")
        groups = prepared.dataframe[group_column].astype("string").fillna("missing").to_numpy()
        _validate_group_count(groups, "leave-one-strain-out")
        splitter = LeaveOneGroupOut()
        for fold_index, (train_index, test_index) in enumerate(splitter.split(prepared.X, prepared.y, groups), start=1):
            group_value = str(groups[test_index][0])
            splits.append(_split_record(group_value, fold_index, train_index, test_index))
        metadata["effective_n_splits"] = int(len(splits))
        metadata["fold_count"] = int(len(splits))
        metadata["held_out_group_column"] = group_column
        return splits, metadata, warnings

    if strategy == "leave_one_chemical_out":
        if chemical_group_column not in prepared.dataframe.columns:
            raise ValueError(f"Missing group column for leave-one-chemical-out: {chemical_group_column}")
        groups = prepared.dataframe[chemical_group_column].astype("string").fillna("missing").to_numpy()
        _validate_group_count(groups, "leave-one-chemical-out")
        splitter = LeaveOneGroupOut()
        for fold_index, (train_index, test_index) in enumerate(splitter.split(prepared.X, prepared.y, groups), start=1):
            group_value = str(groups[test_index][0])
            splits.append(_split_record(group_value, fold_index, train_index, test_index))
        metadata["effective_n_splits"] = int(len(splits))
        metadata["fold_count"] = int(len(splits))
        metadata["held_out_group_column"] = chemical_group_column
        metadata["research_mode"] = "leave_one_chemical_out"
        warnings.append(
            "Leave-one-chemical-out regression is research mode: each chemical is withheld as a group."
        )
        return splits, metadata, warnings

    raise ValueError(f"Unsupported validation strategy: {validation_strategy}")


def rank_regression_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank models by R2, then RMSE, then MAE."""

    if summary.empty:
        raise ValueError("Cannot rank an empty regression summary.")
    ranked = summary.sort_values(
        ["r2_mean", "rmse_mean", "mae_mean", "model_name"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["selection_rule"] = "r2_mean; rmse_mean; mae_mean"
    return ranked


def _evaluate_model(
    spec: RegressionModelSpec,
    *,
    prepared: RegressionPreparedData,
    splits: list[dict[str, Any]],
    preprocessing: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    if spec.factory is None:
        raise RuntimeError(f"Regressor has no estimator factory: {spec.display_name}")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split in splits:
        pipeline = _pipeline(spec.factory(random_state), preprocessing)
        train_index = split["train_index"]
        test_index = split["test_index"]
        X_train = prepared.X.iloc[train_index]
        X_test = prepared.X.iloc[test_index]
        y_train = prepared.y[train_index]
        y_test = prepared.y[test_index]

        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_seconds = time.perf_counter() - started
        metrics = _regression_metrics(y_test, y_pred)
        fold_rows.append(
            {
                "model_id": spec.model_id,
                "model_name": spec.display_name,
                "fold": split["fold"],
                "held_out": split["held_out"],
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "fit_time_seconds": float(fit_seconds),
                "predict_time_seconds": float(predict_seconds),
                **metrics,
            }
        )
        prediction_rows.extend(
            _prediction_rows(
                prepared=prepared,
                split=split,
                model_id=spec.model_id,
                model_name=spec.display_name,
                y_true=y_test,
                y_pred=np.asarray(y_pred, dtype=float),
                test_index=test_index,
            )
        )

    full_estimator = _pipeline(spec.factory(random_state), preprocessing)
    full_estimator.fit(prepared.X, prepared.y)
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), full_estimator


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(y_true - y_pred)
    metrics: dict[str, float | int] = {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "explained_variance": (
            float(explained_variance_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
        ),
        "mape": float("nan"),
        "mape_valid_count": int(np.sum(y_true != 0)),
        "residual_mean": float(np.mean(y_true - y_pred)),
        "residual_std": float(np.std(y_true - y_pred, ddof=0)),
        "max_absolute_error": float(np.max(absolute_error)) if len(absolute_error) else float("nan"),
    }
    nonzero = y_true != 0
    if nonzero.any():
        metrics["mape"] = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
    return metrics


def _prediction_rows(
    *,
    prepared: RegressionPreparedData,
    split: dict[str, Any],
    model_id: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_index: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_columns = [
        "Fingerprint_ID",
        "Experiment_ID",
        "Measurement_Unit_ID",
        "Source_File",
        "Strain",
        "Chemical",
        "Concentration",
        "Replicate_ID",
        "Duration",
        "QC_Status",
    ]
    for local_index, source_index in enumerate(test_index):
        source_row = prepared.dataframe.iloc[int(source_index)]
        actual = float(y_true[local_index])
        predicted = float(y_pred[local_index])
        row = {
            "model_id": model_id,
            "model_name": model_name,
            "fold": split["fold"],
            "held_out": split["held_out"],
            "actual_concentration": actual,
            "predicted_concentration": predicted,
            "residual": actual - predicted,
            "absolute_error": abs(actual - predicted),
            "squared_error": (actual - predicted) ** 2,
        }
        for column in metadata_columns:
            if column in prepared.dataframe.columns:
                row[column] = source_row.get(column, pd.NA)
        rows.append(row)
    return rows


def _summary_row(
    *,
    spec: RegressionModelSpec,
    fold_table: pd.DataFrame,
    full_estimator: Pipeline,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": spec.model_id,
        "model_name": spec.display_name,
        "optional_model": bool(spec.optional),
        "fold_count": int(len(fold_table)),
        "parameter_count": _parameter_count(full_estimator),
    }
    metric_columns = [
        "r2",
        "rmse",
        "mae",
        "median_absolute_error",
        "explained_variance",
        "mape",
        "residual_mean",
        "residual_std",
        "max_absolute_error",
        "fit_time_seconds",
        "predict_time_seconds",
    ]
    for column in metric_columns:
        values = pd.to_numeric(fold_table[column], errors="coerce")
        row[f"{column}_mean"] = _finite_mean(values)
        row[f"{column}_std"] = _finite_std(values)
        row[f"{column}_ci95_low"] = _ci95(values)[0]
        row[f"{column}_ci95_high"] = _ci95(values)[1]
    row["mape_valid_count"] = int(pd.to_numeric(fold_table["mape_valid_count"], errors="coerce").fillna(0).sum())
    return row


def _per_model_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "r2",
        "rmse",
        "mae",
        "median_absolute_error",
        "explained_variance",
        "mape",
        "fit_time_seconds",
        "predict_time_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        for metric in metric_names:
            rows.append(
                {
                    "model_id": row["model_id"],
                    "model_name": row["model_name"],
                    "metric": metric,
                    "mean": row.get(f"{metric}_mean"),
                    "std": row.get(f"{metric}_std"),
                    "ci95_low": row.get(f"{metric}_ci95_low"),
                    "ci95_high": row.get(f"{metric}_ci95_high"),
                }
            )
    return pd.DataFrame(rows)


def _best_model_metrics(best_row: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_metric": "r2_mean",
        "tie_breakers": ["rmse_mean", "mae_mean"],
        "model_id": best_row.get("model_id"),
        "model_name": best_row.get("model_name"),
        "rank": int(best_row.get("rank", 1)),
        "sample_count": int(metadata.get("sample_count", 0)),
        "concentration_min": metadata.get("concentration_min"),
        "concentration_max": metadata.get("concentration_max"),
        "target_units": metadata.get("target_units"),
        "r2_mean": best_row.get("r2_mean"),
        "r2_std": best_row.get("r2_std"),
        "rmse_mean": best_row.get("rmse_mean"),
        "rmse_std": best_row.get("rmse_std"),
        "mae_mean": best_row.get("mae_mean"),
        "mae_std": best_row.get("mae_std"),
        "median_absolute_error_mean": best_row.get("median_absolute_error_mean"),
        "explained_variance_mean": best_row.get("explained_variance_mean"),
        "mape_mean": best_row.get("mape_mean"),
    }


def _feature_importance_table(
    *,
    model_specs: list[RegressionModelSpec],
    fitted_models: dict[str, Pipeline],
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        if not spec.is_tree_model or spec.model_id not in fitted_models:
            continue
        estimator = fitted_models[spec.model_id]
        importances = _model_feature_importances(estimator)
        if importances is None:
            continue
        for feature, importance in zip(feature_names, importances, strict=True):
            rows.append(
                {
                    "model_id": spec.model_id,
                    "model_name": spec.display_name,
                    "importance_type": "model_feature_importance",
                    "held_out_strain": pd.NA,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    return pd.DataFrame(rows)


def _permutation_importance_table(
    *,
    best_model_id: str,
    best_model_name: str,
    estimator: Pipeline,
    prepared: RegressionPreparedData,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator,
        prepared.X,
        prepared.y,
        scoring="r2",
        n_repeats=max(1, int(n_repeats)),
        random_state=random_state,
        n_jobs=1,
    )
    rows = []
    for feature, mean, std in zip(
        prepared.feature_names,
        result.importances_mean,
        result.importances_std,
        strict=True,
    ):
        rows.append(
            {
                "model_id": best_model_id,
                "model_name": best_model_name,
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
                "scoring": "r2",
                "n_repeats": int(max(1, n_repeats)),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_strain_importance_table(
    *,
    model_specs: list[RegressionModelSpec],
    prepared: RegressionPreparedData,
    preprocessing: str,
    random_state: int,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strains = prepared.dataframe[group_column].astype("string").fillna("missing")
    for spec in model_specs:
        if not spec.is_tree_model or spec.factory is None:
            continue
        for strain in sorted(strains.unique().tolist(), key=str):
            train_mask = strains.ne(strain).to_numpy()
            test_mask = ~train_mask
            if train_mask.sum() < 2 or test_mask.sum() < 1:
                continue
            estimator = _pipeline(spec.factory(random_state), preprocessing)
            estimator.fit(prepared.X.loc[train_mask], prepared.y[train_mask])
            importances = _model_feature_importances(estimator)
            if importances is None:
                continue
            predictions = estimator.predict(prepared.X.loc[test_mask])
            metrics = _regression_metrics(prepared.y[test_mask], np.asarray(predictions, dtype=float))
            for feature, importance in zip(prepared.feature_names, importances, strict=True):
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "model_name": spec.display_name,
                        "held_out_strain": str(strain),
                        "feature": feature,
                        "importance": float(importance),
                        "held_out_r2": metrics["r2"],
                        "held_out_rmse": metrics["rmse"],
                        "held_out_mae": metrics["mae"],
                    }
                )
    return pd.DataFrame(rows)


def _resolve_concentration_target(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.Series, str, str, list[str]]:
    warnings: list[str] = []
    if target_column in dataframe.columns:
        source = dataframe[target_column]
        resolved_column = target_column
    elif "Concentration_ug_mL" in dataframe.columns:
        source = dataframe["Concentration_ug_mL"]
        resolved_column = "Concentration_ug_mL"
        warnings.append(
            f"Requested target column {target_column} not found; using Concentration_ug_mL."
        )
    elif "Concentration" in dataframe.columns:
        source = dataframe["Concentration"]
        resolved_column = "Concentration"
        warnings.append(
            f"Requested target column {target_column} not found; using Concentration."
        )
    else:
        raise ValueError(f"Missing concentration target column: {target_column}")

    if pd.api.types.is_numeric_dtype(source):
        values = pd.to_numeric(source, errors="coerce")
        return values, resolved_column, "ug/mL", warnings

    parsed = source.map(_parse_concentration_to_ug_ml)
    numeric = pd.to_numeric(parsed, errors="coerce")
    explicit_unit_mask = source.map(_has_explicit_supported_unit)
    parsed_mask = numeric.notna()
    unitless_count = int((parsed_mask & ~explicit_unit_mask).sum())
    if unitless_count:
        warnings.append(
            "Numeric concentration labels without explicit units parsed using canonical ug/mL convention: "
            f"{unitless_count}."
        )
    missing_count = int(numeric.isna().sum())
    if missing_count:
        warnings.append(f"Rows with non-numeric or unsupported concentration labels: {missing_count}.")
    return numeric, resolved_column, "ug/mL", warnings


def _parse_concentration_to_ug_ml(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.casefold() == "control":
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    normalized = (
        text.casefold()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(" ", "")
    )
    if "ng/ml" in normalized or "ngm/l" in normalized:
        return number / 1000.0
    if "mg/ml" in normalized:
        return number * 1000.0
    if "mg/l" in normalized:
        return number
    if "g/l" in normalized:
        return number * 1000.0
    if "ug/ml" in normalized or "ugm/l" in normalized:
        return number
    if re.search(r"[a-z]", normalized):
        return None
    return number


def _has_explicit_supported_unit(value: Any) -> bool:
    if pd.isna(value):
        return False
    normalized = (
        str(value)
        .casefold()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(" ", "")
    )
    return any(unit in normalized for unit in ("ug/ml", "ugm/l", "ng/ml", "mg/ml", "mg/l", "g/l"))


def _pipeline(estimator: object, preprocessing: str) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(preprocessing)),
            ("model", estimator),
        ]
    )


def _preprocessor(preprocessing: str) -> object:
    method = _canonical_preprocessing(preprocessing)
    if method == "none":
        return "passthrough"
    if method == "zscore":
        return StandardScaler()
    if method == "robust":
        return RobustScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported preprocessing method: {preprocessing}")


def _model_feature_importances(pipeline: Pipeline) -> np.ndarray | None:
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    values = np.asarray(importances, dtype=float)
    if values.ndim != 1:
        return None
    return values


def _parameter_count(pipeline: Pipeline) -> int | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        return int(np.size(model.coef_) + np.size(getattr(model, "intercept_", [])))
    if hasattr(model, "estimators_"):
        estimators = np.asarray(model.estimators_, dtype=object).ravel()
        node_counts = [
            int(estimator.tree_.node_count)
            for estimator in estimators
            if hasattr(estimator, "tree_")
        ]
        return int(sum(node_counts)) if node_counts else int(len(estimators))
    if hasattr(model, "support_"):
        return int(np.size(model.support_))
    if hasattr(model, "_fit_X"):
        return int(np.size(model._fit_X))
    return None


def _fingerprint_dataframe(
    fingerprint_input: FingerprintDataset | pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if isinstance(fingerprint_input, FingerprintDataset):
        warnings.extend(f"Input fingerprint warning: {warning}" for warning in fingerprint_input.warnings)
        warnings.extend(f"Input fingerprint error retained as context: {error}" for error in fingerprint_input.errors)
        return fingerprint_input.dataframe.copy(deep=True), warnings
    if isinstance(fingerprint_input, pd.DataFrame):
        return fingerprint_input.copy(deep=True), warnings
    raise TypeError("Regression benchmark requires a FingerprintDataset or fingerprint DataFrame.")


def _split_record(
    held_out: str,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> dict[str, Any]:
    return {
        "held_out": held_out,
        "fold": int(fold),
        "train_index": np.asarray(train_index, dtype=int),
        "test_index": np.asarray(test_index, dtype=int),
    }


def _validate_group_count(groups: np.ndarray, label: str) -> None:
    if len(np.unique(groups)) < 2:
        raise ValueError(f"{label} regression requires at least two groups.")


def _finite_mean(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(finite.mean()) if len(finite) else float("nan")


def _finite_std(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(finite.std(ddof=0)) if len(finite) else float("nan")


def _ci95(values: pd.Series) -> tuple[float, float]:
    finite = _finite_values(values)
    if len(finite) < 2:
        mean = float(finite.mean()) if len(finite) else float("nan")
        return mean, mean
    mean = float(finite.mean())
    standard_error = float(finite.std(ddof=1) / math.sqrt(len(finite)))
    delta = 1.96 * standard_error
    return mean - delta, mean + delta


def _finite_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.loc[np.isfinite(numeric.astype(float))]


def _is_finite_number(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _duplicated_rows(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns or dataframe.empty:
        return 0
    return int(dataframe[column].astype("string").duplicated(keep=False).sum())


def _canonical_preprocessing(preprocessing: str) -> str:
    normalized = str(preprocessing).strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "none": "none",
        "no": "none",
        "z": "zscore",
        "zscore": "zscore",
        "standard": "zscore",
        "standardscaler": "zscore",
        "robust": "robust",
        "robustscaler": "robust",
        "minmax": "minmax",
        "minmaxscaler": "minmax",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported preprocessing method. Expected one of: none, zscore, robust, minmax."
        )
    return aliases[normalized]


def _canonical_validation_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().casefold().replace("-", "_")
    aliases = {
        "repeated_kfold": "repeated_kfold",
        "repeated_cv": "repeated_kfold",
        "kfold": "repeated_kfold",
        "leave_one_strain_out": "leave_one_strain_out",
        "loso": "leave_one_strain_out",
        "leave_one_chemical_out": "leave_one_chemical_out",
        "loco": "leave_one_chemical_out",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported validation strategy. Expected one of: "
            + ", ".join(SUPPORTED_VALIDATION_STRATEGIES)
            + "."
        )
    return aliases[normalized]
