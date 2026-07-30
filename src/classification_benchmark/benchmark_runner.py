"""Stage 8A supervised chemical classification benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from src.classification_benchmark.classification_dataset import ClassificationBenchmarkResult
from src.classification_benchmark.models import ModelSpec, available_model_specs
from src.fingerprint import FINGERPRINT_FEATURE_COLUMNS, FingerprintDataset


BENCHMARK_VERSION = "0.1.0"
DEFAULT_VALIDATION_STRATEGY = "repeated_stratified_kfold"
DEFAULT_PREPROCESSING = "zscore"
SUPPORTED_PREPROCESSING = ("none", "zscore", "robust", "minmax")
SUPPORTED_VALIDATION_STRATEGIES = (
    "train_test",
    "stratified_kfold",
    "repeated_stratified_kfold",
    "leave_one_strain_out",
    "leave_one_chemical_out",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for Stage 8A benchmark comparisons."""

    validation_strategy: str = DEFAULT_VALIDATION_STRATEGY
    preprocessing: str = DEFAULT_PREPROCESSING
    n_splits: int = 5
    n_repeats: int = 2
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "Chemical"
    group_column: str = "Strain"
    model_ids: tuple[str, ...] | None = None
    permutation_repeats: int = 5
    run_permutation_importance: bool = True
    run_leave_one_strain_importance: bool = True


@dataclass(frozen=True)
class ClassificationPreparedData:
    """Validated feature matrix and encoded labels for benchmarking."""

    dataframe: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    y_original: pd.Series
    label_encoder: LabelEncoder
    feature_names: list[str]
    target_column: str
    metadata: dict[str, Any]
    warnings: list[str]


def run_classification_benchmark(
    fingerprint_input: FingerprintDataset | pd.DataFrame,
    *,
    config: BenchmarkConfig | None = None,
    feature_names: Iterable[str] | None = None,
) -> ClassificationBenchmarkResult:
    """Compare chemical classifiers on validated fingerprint features."""

    config = config or BenchmarkConfig()
    strategy = _canonical_validation_strategy(config.validation_strategy)
    preprocessing = _canonical_preprocessing(config.preprocessing)
    source_dataframe, input_warnings = _fingerprint_dataframe(fingerprint_input)
    prepared = prepare_classification_data(
        source_dataframe,
        feature_names=list(feature_names or FINGERPRINT_FEATURE_COLUMNS),
        target_column=config.target_column,
        validation_strategy=strategy,
        requested_n_splits=config.n_splits,
    )
    splits, split_metadata, split_warnings = make_validation_splits(
        prepared,
        validation_strategy=strategy,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        test_size=config.test_size,
        random_state=config.random_state,
        group_column=config.group_column,
    )
    model_specs, skipped_specs = available_model_specs(
        random_state=config.random_state,
        model_ids=config.model_ids,
    )

    fold_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    model_prediction_cache: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    all_warnings = [*input_warnings, *prepared.warnings, *split_warnings]
    for skipped in skipped_specs:
        if skipped.skip_reason:
            all_warnings.append(f"Optional classifier skipped: {skipped.display_name}: {skipped.skip_reason}")

    for spec in model_specs:
        fold_table, prediction_cache, full_estimator = _evaluate_model(
            spec,
            prepared=prepared,
            splits=splits,
            preprocessing=preprocessing,
            random_state=config.random_state,
        )
        fold_tables.append(fold_table)
        model_prediction_cache[spec.model_id] = prediction_cache
        fitted_models[spec.model_id] = full_estimator
        summary_rows.append(
            _summary_row(
                spec=spec,
                fold_table=fold_table,
                full_estimator=full_estimator,
            )
        )

    fold_metrics = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    rankings = rank_models(summary)
    best_model_id = str(rankings.iloc[0]["model_id"])
    best_predictions = model_prediction_cache[best_model_id]
    class_names = prepared.label_encoder.classes_.astype(str).tolist()
    confusion = _confusion_dataframe(best_predictions, class_names)
    per_class = _per_class_metrics(best_predictions, class_names)
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

    best_metrics = _best_model_metrics(rankings.iloc[0].to_dict(), metadata=prepared.metadata)
    metadata = {
        "stage": "8A",
        "benchmark_version": BENCHMARK_VERSION,
        "input_contract": "validated fingerprint dataset",
        "raw_luminescence_used": False,
        "feature_validation_bypassed": False,
        "fingerprint_qc_bypassed": False,
        "uses_sklearn_pipelines": True,
        "full_dataset_scaled_before_splitting": False,
        "validation_strategy": strategy,
        "preprocessing": preprocessing,
        "requested_n_splits": int(config.n_splits),
        "effective_n_splits": int(split_metadata.get("effective_n_splits", 0)),
        "n_repeats": int(split_metadata.get("n_repeats", config.n_repeats)),
        "test_size": float(config.test_size),
        "random_state": int(config.random_state),
        "models_evaluated": [spec.display_name for spec in model_specs],
        "model_ids_evaluated": [spec.model_id for spec in model_specs],
        "models_skipped": [spec.display_name for spec in skipped_specs],
        "feature_names": prepared.feature_names,
        "feature_count": len(prepared.feature_names),
        **prepared.metadata,
        **split_metadata,
    }
    return ClassificationBenchmarkResult(
        summary=summary,
        rankings=rankings,
        best_model_metrics=best_metrics,
        confusion_matrix=confusion,
        per_class_metrics=per_class,
        feature_importance=feature_importance,
        permutation_importance=permutation_table,
        leave_one_strain_importance=strain_importance,
        fold_metrics=fold_metrics,
        metadata=metadata,
        warnings=all_warnings,
        errors=[],
    )


def prepare_classification_data(
    dataframe: pd.DataFrame,
    *,
    feature_names: list[str],
    target_column: str = "Chemical",
    validation_strategy: str = DEFAULT_VALIDATION_STRATEGY,
    requested_n_splits: int = 5,
) -> ClassificationPreparedData:
    """Validate and copy fingerprint rows for supervised classification."""

    source = dataframe.copy(deep=True)
    missing_features = [feature for feature in feature_names if feature not in source.columns]
    if missing_features:
        raise ValueError(f"Missing fingerprint feature columns: {', '.join(missing_features)}")
    if target_column not in source.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if source.empty:
        raise ValueError("Fingerprint dataset is empty.")

    values = source.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    finite_mask = values.apply(lambda column: np.isfinite(column.astype(float))).all(axis=1)
    target = source[target_column].astype("string").str.strip()
    target_mask = target.notna() & target.ne("")
    usable_mask = finite_mask & target_mask
    warnings: list[str] = []
    excluded_row_count = int((~usable_mask).sum())
    if excluded_row_count:
        warnings.append(f"Rows excluded from classification because labels or features were unusable: {excluded_row_count}.")

    usable = source.loc[usable_mask].copy(deep=True).reset_index(drop=True)
    usable_features = values.loc[usable_mask].astype(float).reset_index(drop=True)
    for feature in feature_names:
        usable[feature] = usable_features[feature].to_numpy(dtype=float)
    y_original = usable[target_column].astype("string")

    if validation_strategy != "leave_one_chemical_out":
        class_counts = y_original.value_counts()
        singleton_classes = class_counts.loc[class_counts < 2]
        if not singleton_classes.empty:
            singleton_names = sorted(singleton_classes.index.astype(str).tolist())
            warnings.append(
                "Classes with fewer than 2 observations excluded from stratified validation: "
                + ", ".join(singleton_names)
                + "."
            )
            keep_mask = ~y_original.isin(singleton_classes.index)
            usable = usable.loc[keep_mask].reset_index(drop=True)
            y_original = usable[target_column].astype("string")

    class_counts = y_original.value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError("Chemical classification requires at least two classes after QC filtering.")
    min_class_count = int(class_counts.min())
    if validation_strategy in {"stratified_kfold", "repeated_stratified_kfold"} and min_class_count < requested_n_splits:
        warnings.append(
            f"Requested {requested_n_splits} stratified folds but the smallest class has "
            f"{min_class_count} rows; the effective fold count will be reduced."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_original.astype(str))
    metadata = {
        "source_row_count": int(len(source)),
        "sample_count": int(len(usable)),
        "excluded_row_count": excluded_row_count + int(len(source.loc[usable_mask]) - len(usable)),
        "target_column": target_column,
        "class_count": int(len(class_counts)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "minimum_class_count": min_class_count,
        "maximum_class_count": int(class_counts.max()),
        "class_imbalance_ratio": float(class_counts.max() / min_class_count),
        "duplicated_measurement_unit_rows": _duplicated_rows(usable, "Measurement_Unit_ID"),
        "duplicate_fingerprint_rows": int(usable.loc[:, feature_names].duplicated(keep=False).sum()),
    }
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

    return ClassificationPreparedData(
        dataframe=usable,
        X=usable.loc[:, feature_names].copy(deep=True),
        y=y,
        y_original=y_original.reset_index(drop=True),
        label_encoder=label_encoder,
        feature_names=list(feature_names),
        target_column=target_column,
        metadata=metadata,
        warnings=warnings,
    )


def make_validation_splits(
    prepared: ClassificationPreparedData,
    *,
    validation_strategy: str = DEFAULT_VALIDATION_STRATEGY,
    n_splits: int = 5,
    n_repeats: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
    group_column: str = "Strain",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Create deterministic validation splits without preprocessing leakage."""

    strategy = _canonical_validation_strategy(validation_strategy)
    warnings: list[str] = []
    y = prepared.y
    indices = np.arange(len(y))
    splits: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"n_repeats": int(n_repeats)}

    if strategy == "train_test":
        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        splits.append(_split_record("train_test", 1, train_index, test_index))
        metadata["effective_n_splits"] = 1
        metadata["fold_count"] = 1
        return splits, metadata, warnings

    if strategy in {"stratified_kfold", "repeated_stratified_kfold"}:
        class_counts = pd.Series(y).value_counts()
        effective_n_splits = min(int(n_splits), int(class_counts.min()))
        if effective_n_splits < 2:
            raise ValueError("Stratified cross-validation requires at least 2 rows per class.")
        if effective_n_splits < n_splits:
            warnings.append(
                f"Effective stratified fold count reduced from {n_splits} to {effective_n_splits}."
            )
        if strategy == "stratified_kfold":
            splitter = StratifiedKFold(
                n_splits=effective_n_splits,
                shuffle=True,
                random_state=random_state,
            )
            for fold_index, (train_index, test_index) in enumerate(splitter.split(prepared.X, y), start=1):
                splits.append(_split_record("fold", fold_index, train_index, test_index))
            metadata["n_repeats"] = 1
        else:
            splitter = RepeatedStratifiedKFold(
                n_splits=effective_n_splits,
                n_repeats=max(1, int(n_repeats)),
                random_state=random_state,
            )
            for fold_index, (train_index, test_index) in enumerate(splitter.split(prepared.X, y), start=1):
                splits.append(_split_record("fold", fold_index, train_index, test_index))
        metadata["effective_n_splits"] = int(effective_n_splits)
        metadata["fold_count"] = int(len(splits))
        return splits, metadata, warnings

    if strategy == "leave_one_strain_out":
        if group_column not in prepared.dataframe.columns:
            raise ValueError(f"Missing group column for leave-one-strain-out: {group_column}")
        groups = prepared.dataframe[group_column].astype("string").fillna("missing").to_numpy()
        splitter = LeaveOneGroupOut()
        for fold_index, (train_index, test_index) in enumerate(splitter.split(prepared.X, y, groups), start=1):
            group_value = str(groups[test_index][0])
            splits.append(_split_record(group_value, fold_index, train_index, test_index))
        metadata["effective_n_splits"] = int(len(splits))
        metadata["fold_count"] = int(len(splits))
        metadata["held_out_group_column"] = group_column
        return splits, metadata, warnings

    if strategy == "leave_one_chemical_out":
        for fold_index, class_id in enumerate(sorted(np.unique(y).tolist()), start=1):
            test_index = indices[y == class_id]
            train_index = indices[y != class_id]
            if len(np.unique(y[train_index])) < 2:
                warnings.append("Leave-one-chemical-out stopped because the training fold had fewer than 2 classes.")
                continue
            class_name = str(prepared.label_encoder.inverse_transform([class_id])[0])
            splits.append(_split_record(class_name, fold_index, train_index, test_index))
        metadata["effective_n_splits"] = int(len(splits))
        metadata["fold_count"] = int(len(splits))
        metadata["research_mode"] = "leave_one_chemical_out"
        warnings.append(
            "Leave-one-chemical-out is research mode: the held-out chemical label is absent from each training fold."
        )
        return splits, metadata, warnings

    raise ValueError(f"Unsupported validation strategy: {validation_strategy}")


def rank_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank models by Macro F1, then balanced accuracy, then accuracy."""

    if summary.empty:
        raise ValueError("Cannot rank an empty classification summary.")
    ranked = summary.sort_values(
        ["f1_macro_mean", "balanced_accuracy_mean", "accuracy_mean", "model_name"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["selection_rule"] = "f1_macro_mean; balanced_accuracy_mean; accuracy_mean"
    return ranked


def _evaluate_model(
    spec: ModelSpec,
    *,
    prepared: ClassificationPreparedData,
    splits: list[dict[str, Any]],
    preprocessing: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any], Pipeline]:
    if spec.factory is None:
        raise RuntimeError(f"Classifier has no estimator factory: {spec.display_name}")

    fold_rows: list[dict[str, Any]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    all_proba: list[np.ndarray] = []
    labels = list(range(len(prepared.label_encoder.classes_)))
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
        y_proba = _predict_proba_aligned(pipeline, X_test, labels)
        metrics = _classification_metrics(y_test, y_pred, y_proba, labels)
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
        all_true.extend(int(value) for value in y_test)
        all_pred.extend(int(value) for value in y_pred)
        if y_proba is not None:
            all_proba.append(y_proba)

    full_estimator = _pipeline(spec.factory(random_state), preprocessing)
    full_estimator.fit(prepared.X, prepared.y)
    prediction_cache = {
        "y_true": np.asarray(all_true, dtype=int),
        "y_pred": np.asarray(all_pred, dtype=int),
        "y_proba": np.vstack(all_proba) if all_proba else None,
        "label_encoder": prepared.label_encoder,
    }
    return pd.DataFrame(fold_rows), prediction_cache, full_estimator


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    labels: list[int],
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_weighted": float("nan"),
        "log_loss": float("nan"),
    }
    if y_proba is None:
        return metrics
    try:
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(
                y_true,
                y_proba,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        metrics["roc_auc_ovr_weighted"] = float("nan")
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def _summary_row(
    *,
    spec: ModelSpec,
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
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_weighted",
        "log_loss",
        "fit_time_seconds",
        "predict_time_seconds",
    ]
    for column in metric_columns:
        values = pd.to_numeric(fold_table[column], errors="coerce")
        row[f"{column}_mean"] = _finite_mean(values)
        row[f"{column}_std"] = _finite_std(values)
        row[f"{column}_ci95_low"] = _ci95(values)[0]
        row[f"{column}_ci95_high"] = _ci95(values)[1]
    return row


def _best_model_metrics(best_row: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_metric": "f1_macro_mean",
        "tie_breakers": ["balanced_accuracy_mean", "accuracy_mean"],
        "model_id": best_row.get("model_id"),
        "model_name": best_row.get("model_name"),
        "rank": int(best_row.get("rank", 1)),
        "sample_count": int(metadata.get("sample_count", 0)),
        "class_count": int(metadata.get("class_count", 0)),
        "accuracy_mean": best_row.get("accuracy_mean"),
        "accuracy_std": best_row.get("accuracy_std"),
        "balanced_accuracy_mean": best_row.get("balanced_accuracy_mean"),
        "balanced_accuracy_std": best_row.get("balanced_accuracy_std"),
        "precision_macro_mean": best_row.get("precision_macro_mean"),
        "recall_macro_mean": best_row.get("recall_macro_mean"),
        "f1_macro_mean": best_row.get("f1_macro_mean"),
        "f1_macro_std": best_row.get("f1_macro_std"),
        "f1_weighted_mean": best_row.get("f1_weighted_mean"),
        "roc_auc_ovr_weighted_mean": best_row.get("roc_auc_ovr_weighted_mean"),
        "log_loss_mean": best_row.get("log_loss_mean"),
    }


def _feature_importance_table(
    *,
    model_specs: list[ModelSpec],
    fitted_models: dict[str, Pipeline],
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        if not spec.is_tree_model:
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
    prepared: ClassificationPreparedData,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator,
        prepared.X,
        prepared.y,
        scoring="f1_macro",
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
                "scoring": "f1_macro",
                "n_repeats": int(max(1, n_repeats)),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_strain_importance_table(
    *,
    model_specs: list[ModelSpec],
    prepared: ClassificationPreparedData,
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
            if len(np.unique(prepared.y[train_mask])) < 2 or not test_mask.any():
                continue
            estimator = _pipeline(spec.factory(random_state), preprocessing)
            estimator.fit(prepared.X.loc[train_mask], prepared.y[train_mask])
            importances = _model_feature_importances(estimator)
            if importances is None:
                continue
            predictions = estimator.predict(prepared.X.loc[test_mask])
            held_out_f1 = f1_score(
                prepared.y[test_mask],
                predictions,
                average="macro",
                zero_division=0,
            )
            for feature, importance in zip(prepared.feature_names, importances, strict=True):
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "model_name": spec.display_name,
                        "held_out_strain": str(strain),
                        "feature": feature,
                        "importance": float(importance),
                        "held_out_f1_macro": float(held_out_f1),
                    }
                )
    return pd.DataFrame(rows)


def _confusion_dataframe(prediction_cache: dict[str, Any], class_names: list[str]) -> pd.DataFrame:
    label_encoder: LabelEncoder = prediction_cache["label_encoder"]
    y_true = label_encoder.inverse_transform(prediction_cache["y_true"]).astype(str)
    y_pred = label_encoder.inverse_transform(prediction_cache["y_pred"]).astype(str)
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    return pd.DataFrame(
        matrix,
        index=[f"true:{label}" for label in class_names],
        columns=[f"predicted:{label}" for label in class_names],
    )


def _per_class_metrics(prediction_cache: dict[str, Any], class_names: list[str]) -> pd.DataFrame:
    label_encoder: LabelEncoder = prediction_cache["label_encoder"]
    y_true = label_encoder.inverse_transform(prediction_cache["y_true"]).astype(str)
    y_pred = label_encoder.inverse_transform(prediction_cache["y_pred"]).astype(str)
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in class_names:
        metrics = report.get(label, {})
        rows.append(
            {
                "chemical": label,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1-score", 0.0)),
                "support": int(metrics.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


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


def _predict_proba_aligned(
    pipeline: Pipeline,
    X: pd.DataFrame,
    labels: list[int],
) -> np.ndarray | None:
    model = pipeline.named_steps["model"]
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        probabilities = pipeline.predict_proba(X)
    except Exception:  # noqa: BLE001 - some classifiers intentionally lack probabilities.
        return None
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        return None
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    class_to_column = {int(label): index for index, label in enumerate(labels)}
    for source_column, class_id in enumerate(model_classes):
        target_column = class_to_column.get(int(class_id))
        if target_column is not None:
            aligned[:, target_column] = probabilities[:, source_column]
    row_sums = aligned.sum(axis=1)
    nonzero = row_sums > 0
    aligned[nonzero] = aligned[nonzero] / row_sums[nonzero, None]
    return aligned


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
    if hasattr(model, "n_support_"):
        return int(np.sum(model.n_support_))
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
    raise TypeError("Classification benchmark requires a FingerprintDataset or fingerprint DataFrame.")


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
        "train_test": "train_test",
        "stratified_train_test": "train_test",
        "stratified_kfold": "stratified_kfold",
        "kfold": "stratified_kfold",
        "repeated_stratified_kfold": "repeated_stratified_kfold",
        "repeated_kfold": "repeated_stratified_kfold",
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
