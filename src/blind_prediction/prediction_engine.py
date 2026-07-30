"""Stage 9A blind-prediction training, prediction, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.classification_benchmark import available_model_specs, prepare_classification_data
from src.data_schema.canonical_schema import CANONICAL_SCHEMA_VERSION
from src.feature_engine.feature_extractor import FEATURE_ENGINE_VERSION
from src.feature_engine_v2 import FEATURE_ENGINE_V2_VERSION
from src.feature_selection import build_generated_feature_table
from src.regression_benchmark import available_regression_model_specs, prepare_regression_data
from src.quality_control.canonical_qc import audit_canonical_dataframe
from src.blind_prediction.confidence_scoring import (
    calculate_confidence,
    classify_probabilities,
)
from src.blind_prediction.model_bundle import (
    FrozenModelBundle,
    FeatureProfile,
    dependency_versions,
    timestamp_utc,
)
from src.blind_prediction.novelty_detection import assess_novelty, fit_novelty_reference
from src.blind_prediction.prediction_qc import (
    PredictionQCResult,
    enforce_feature_order,
    evaluate_prediction_qc,
)
from src.blind_prediction.prediction_report import BlindPredictionResult


@dataclass(frozen=True)
class BlindTrainingConfig:
    """Training configuration for a frozen blind-prediction bundle."""

    classifier_model_id: str = "extra_trees"
    regressor_model_id: str = "extra_trees"
    preprocessing: str = "zscore"
    random_state: int = 42
    concentration_units: str = "ug/mL"
    min_chemical_specific_rows: int = 6
    min_chemical_specific_concentrations: int = 2
    required_strains: tuple[str, ...] = ()
    minimum_measurement_units: int = 1


def train_blind_prediction_bundle(
    canonical_dataframe: pd.DataFrame,
    *,
    feature_profile: FeatureProfile,
    config: BlindTrainingConfig | None = None,
) -> FrozenModelBundle:
    """Train a frozen model bundle from training canonical data only."""

    config = config or BlindTrainingConfig()
    canonical = canonical_dataframe.copy(deep=True)
    generated = build_generated_feature_table(canonical)
    feature_dataframe = generated["dataframe"]
    feature_family_map = generated["feature_family_map"]
    class_features = list(feature_profile.classification_features)
    reg_features = list(feature_profile.regression_features)

    class_prepared = prepare_classification_data(
        feature_dataframe,
        feature_names=class_features,
        validation_strategy="repeated_stratified_kfold",
        requested_n_splits=3,
    )
    reg_prepared = prepare_regression_data(
        feature_dataframe,
        feature_names=reg_features,
    )

    classifier_spec = _model_spec(config.classifier_model_id, config.random_state)
    regressor_spec = _regression_spec(config.regressor_model_id, config.random_state)
    classifier_pipeline = _pipeline(classifier_spec.factory(config.random_state), config.preprocessing)
    classifier_pipeline.fit(class_prepared.X, class_prepared.y)
    global_regressor = _pipeline(regressor_spec.factory(config.random_state), config.preprocessing)
    global_regressor.fit(reg_prepared.X, reg_prepared.y)

    chemical_regressors, chemical_uncertainty = _fit_chemical_specific_regressors(
        reg_prepared,
        regressor_spec=regressor_spec,
        preprocessing=config.preprocessing,
        random_state=config.random_state,
        min_rows=config.min_chemical_specific_rows,
        min_concentrations=config.min_chemical_specific_concentrations,
    )
    global_uncertainty = _residual_uncertainty(global_regressor, reg_prepared.X, reg_prepared.y)
    novelty_reference, novelty_thresholds = fit_novelty_reference(
        classifier_pipeline=classifier_pipeline,
        X=class_prepared.X,
        labels=class_prepared.y_original,
        class_labels=class_prepared.label_encoder.classes_.astype(str).tolist(),
        metadata=_training_reference_metadata(class_prepared.dataframe),
    )
    training_distribution = _training_distribution(
        feature_dataframe,
        feature_names=sorted(set(class_features) | set(reg_features)),
    )
    concentration_ranges = _concentration_ranges_by_chemical(reg_prepared.dataframe)
    training_summary = {
        "source_rows": int(len(canonical)),
        "generated_feature_rows": int(len(feature_dataframe)),
        "classification_rows": int(len(class_prepared.X)),
        "regression_rows": int(len(reg_prepared.X)),
        "source_file_count": _nunique(feature_dataframe, "Source_File"),
        "source_files": _strings(feature_dataframe, "Source_File"),
        "strains": _strings(feature_dataframe, "Strain"),
        "chemicals": class_prepared.label_encoder.classes_.astype(str).tolist(),
        "class_counts": class_prepared.metadata.get("class_counts", {}),
        "concentration_min": reg_prepared.metadata.get("concentration_min"),
        "concentration_max": reg_prepared.metadata.get("concentration_max"),
        "concentration_ranges_by_chemical": concentration_ranges,
        "chemical_specific_regressors": sorted(chemical_regressors),
        "chemical_specific_unavailable": sorted(set(class_prepared.label_encoder.classes_.astype(str)) - set(chemical_regressors)),
        "global_regression_uncertainty": global_uncertainty,
        "chemical_regression_uncertainty": chemical_uncertainty,
        "minimum_measurement_units": int(config.minimum_measurement_units),
    }
    return FrozenModelBundle(
        classifier_pipeline=classifier_pipeline,
        global_regressor_pipeline=global_regressor,
        chemical_regressors=chemical_regressors,
        classification_features=class_features,
        regression_features=reg_features,
        class_labels=class_prepared.label_encoder.classes_.astype(str).tolist(),
        concentration_units=config.concentration_units,
        regression_strategy="chemical_specific_with_global_comparison",
        feature_engine_versions={
            "feature_engine_v1": FEATURE_ENGINE_VERSION,
            "feature_engine_v2": FEATURE_ENGINE_V2_VERSION,
        },
        canonical_schema_version=CANONICAL_SCHEMA_VERSION,
        model_metrics={
            "classification": feature_profile.classification_profile,
            "regression": feature_profile.regression_profile,
            "calibration": "uncalibrated_model_probabilities",
        },
        training_data_summary=training_summary,
        training_distribution=training_distribution,
        novelty_reference=novelty_reference,
        novelty_thresholds=novelty_thresholds,
        model_creation_timestamp=timestamp_utc(),
        software_version="biosensor_fingerprint_ml-stage-9A",
        random_seeds={"model_random_state": int(config.random_state)},
        dependency_versions=dependency_versions(),
        preprocessing=config.preprocessing,
        classifier_model_id=config.classifier_model_id,
        regressor_model_id=config.regressor_model_id,
        time_window=_time_window(canonical, class_features + reg_features),
        required_strains=list(config.required_strains),
        feature_family_map=feature_family_map,
    )


def predict_blind_sample(
    canonical_dataframe: pd.DataFrame,
    *,
    bundle: FrozenModelBundle,
    source_files: list[str] | None = None,
) -> BlindPredictionResult:
    """Predict a blind sample without retraining or modifying the bundle."""

    canonical = canonical_dataframe.copy(deep=True)
    canonical_qc = audit_canonical_dataframe(canonical)
    warnings = [f"canonical_qc: {warning}" for warning in canonical_qc.warnings]
    errors = [f"canonical_qc: {error}" for error in canonical_qc.errors]
    generated = build_generated_feature_table(canonical)
    feature_dataframe = generated["dataframe"].copy(deep=True)
    warnings.extend(f"feature_generation: {warning}" for warning in generated["warnings"])
    errors.extend(f"feature_generation: {error}" for error in generated["errors"])
    prediction_qc = evaluate_prediction_qc(
        canonical_qc=canonical_qc,
        canonical_dataframe=canonical,
        feature_dataframe=feature_dataframe,
        bundle=bundle,
        minimum_measurement_units=int(bundle.training_data_summary.get("minimum_measurement_units", 1)),
    )
    warnings.extend(prediction_qc.warnings)
    errors.extend(prediction_qc.errors)

    classification = _classification_prediction(feature_dataframe, prediction_qc, bundle)
    probabilities = classification["probabilities"]
    evidence = classify_probabilities(probabilities, bundle.class_labels)
    novelty = assess_novelty(
        classifier_pipeline=bundle.classifier_pipeline,
        X=classification["X"],
        predicted_label=evidence.predicted_label,
        probabilities=probabilities,
        novelty_reference=bundle.novelty_reference,
        thresholds=bundle.novelty_thresholds,
    )
    if novelty.novelty_status == "Out of Distribution":
        prediction_qc = prediction_qc.with_severe_novelty("Severe novelty detected: Out of Distribution.")
        errors.append("Severe novelty detected: Out of Distribution.")

    concentration = _concentration_prediction(feature_dataframe, prediction_qc, bundle, evidence.predicted_label)
    confidence = calculate_confidence(
        classification=evidence,
        novelty_status=novelty.novelty_status,
        novelty_score=novelty.novelty_score,
        qc_status=prediction_qc.status,
        valid_row_count=int(prediction_qc.valid_classification_mask.sum()),
        total_row_count=int(len(feature_dataframe)),
        row_predicted_labels=classification["row_labels"],
    )
    influential_features = _influential_features(feature_dataframe, bundle)
    influential_strains = _influential_strains(
        feature_dataframe,
        probabilities=probabilities,
        class_labels=bundle.class_labels,
        predicted_label=evidence.predicted_label,
        valid_mask=prediction_qc.valid_classification_mask,
    )
    warning_errors = list(dict.fromkeys(errors))
    status_passed = prediction_qc.status != "FAIL" and novelty.novelty_status != "Out of Distribution"
    return BlindPredictionResult(
        source_files=source_files or _strings(canonical, "Source_File"),
        canonical_qc=canonical_qc.to_summary_dict(),
        feature_qc=prediction_qc.summary,
        fingerprint_qc=prediction_qc.summary,
        predicted_chemical=evidence.predicted_label,
        chemical_probabilities=evidence.probabilities,
        chemical_confidence=evidence.top_probability,
        predicted_concentration=concentration["predicted_concentration"],
        concentration_units=bundle.concentration_units,
        concentration_interval=concentration["concentration_interval"],
        regression_confidence=concentration["regression_confidence"],
        novelty_score=novelty.novelty_score,
        novelty_status=novelty.novelty_status,
        novelty_assessment=pd.DataFrame([novelty.to_row()]),
        influential_features=influential_features,
        influential_strains=influential_strains,
        concentration_prediction=concentration["table"],
        prediction_confidence=confidence.to_dataframe(),
        warnings=list(dict.fromkeys(warnings)),
        errors=warning_errors,
        prediction_passed=bool(status_passed),
        model_versions={
            "bundle_version": bundle.bundle_version,
            "pipeline_version": bundle.pipeline_version,
            "feature_engine_versions": bundle.feature_engine_versions,
            "canonical_schema_version": bundle.canonical_schema_version,
            "software_version": bundle.software_version,
        },
        pipeline_version=bundle.pipeline_version,
        top_three_candidates=evidence.top_three,
        prediction_margin=evidence.margin,
        probability_entropy=evidence.entropy,
        nearest_training_examples=novelty.nearest_training_examples,
    )


def evaluate_blind_predictions(
    prediction_dir: str | Path,
    truth_file: str | Path,
) -> dict[str, Any]:
    """Evaluate saved blind predictions after truth is supplied separately."""

    prediction_path = Path(prediction_dir) / "blind_prediction_summary.json"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    truth_path = Path(truth_file)
    if not truth_path.exists():
        raise FileNotFoundError(truth_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    truth = _read_truth(truth_path)
    true_chemical = truth.get("true_chemical")
    true_concentration = _safe_float(truth.get("true_concentration"))
    predicted_concentration = _safe_float(prediction.get("predicted_concentration"))
    interval = prediction.get("concentration_interval", {})
    lower = _safe_float(interval.get("lower"))
    upper = _safe_float(interval.get("upper"))
    absolute_error = (
        None
        if true_concentration is None or predicted_concentration is None
        else abs(predicted_concentration - true_concentration)
    )
    percentage_error = (
        None
        if absolute_error is None or true_concentration in {None, 0.0}
        else abs(absolute_error / true_concentration) * 100.0
    )
    in_interval = (
        None
        if true_concentration is None or lower is None or upper is None
        else bool(lower <= true_concentration <= upper)
    )
    return {
        "chemical_prediction_correct": (
            None if true_chemical is None else prediction.get("predicted_chemical") == str(true_chemical)
        ),
        "predicted_chemical": prediction.get("predicted_chemical"),
        "true_chemical": true_chemical,
        "predicted_concentration": predicted_concentration,
        "true_concentration": true_concentration,
        "concentration_absolute_error": absolute_error,
        "concentration_percentage_error": percentage_error,
        "true_value_within_prediction_interval": in_interval,
        "chemical_confidence": prediction.get("chemical_confidence"),
        "regression_confidence": prediction.get("regression_confidence"),
        "novelty_status": prediction.get("novelty_status"),
        "prediction_passed": prediction.get("prediction_passed"),
        "truth_file_read_by_prediction_command": False,
    }


def run_simulated_blind_test(
    canonical_dataframe: pd.DataFrame,
    *,
    feature_profile: FeatureProfile,
    group_column: str = "Source_File",
    holdout_group: str | None = None,
    config: BlindTrainingConfig | None = None,
) -> dict[str, Any]:
    """Hold out one group before training and predict it as a blind sample."""

    canonical = canonical_dataframe.copy(deep=True)
    if group_column not in canonical.columns:
        raise ValueError(f"Missing simulated blind group column: {group_column}")
    groups = sorted(canonical[group_column].dropna().astype(str).unique().tolist())
    if len(groups) < 2 and holdout_group is None:
        raise ValueError("Simulated blind testing requires at least two holdout groups.")
    selected_group = holdout_group or groups[0]
    holdout_mask = canonical[group_column].astype(str).eq(str(selected_group))
    if not holdout_mask.any():
        raise ValueError(f"Holdout group not found: {selected_group}")
    training = canonical.loc[~holdout_mask].copy(deep=True)
    blind = canonical.loc[holdout_mask].copy(deep=True)
    bundle = train_blind_prediction_bundle(training, feature_profile=feature_profile, config=config)
    result = predict_blind_sample(blind, bundle=bundle)
    truth = _truth_from_canonical(blind)
    evaluation = _evaluate_result_against_truth(result, truth)
    return {
        "holdout_group_column": group_column,
        "holdout_group": selected_group,
        "training_rows": int(len(training)),
        "blind_rows": int(len(blind)),
        "training_measurement_units": _nunique(training, "Measurement_Unit_ID"),
        "blind_measurement_units": _nunique(blind, "Measurement_Unit_ID"),
        "group_leakage_prevented": True,
        "prediction": result,
        "truth": truth,
        "evaluation": evaluation,
    }


def _classification_prediction(
    feature_dataframe: pd.DataFrame,
    prediction_qc: PredictionQCResult,
    bundle: FrozenModelBundle,
) -> dict[str, Any]:
    usable = feature_dataframe.loc[prediction_qc.valid_classification_mask].copy(deep=True)
    if usable.empty:
        return {
            "X": pd.DataFrame(columns=bundle.classification_features),
            "probabilities": np.empty((0, len(bundle.class_labels))),
            "row_labels": [],
        }
    X, _ = enforce_feature_order(usable, bundle.classification_features)
    probabilities = bundle.classifier_pipeline.predict_proba(X)
    row_indices = np.argmax(probabilities, axis=1)
    row_labels = [bundle.class_labels[int(index)] for index in row_indices]
    return {"X": X, "probabilities": np.asarray(probabilities, dtype=float), "row_labels": row_labels}


def _concentration_prediction(
    feature_dataframe: pd.DataFrame,
    prediction_qc: PredictionQCResult,
    bundle: FrozenModelBundle,
    predicted_chemical: str | None,
) -> dict[str, Any]:
    usable = feature_dataframe.loc[prediction_qc.valid_regression_mask].copy(deep=True)
    rows: list[dict[str, Any]] = []
    if usable.empty or predicted_chemical is None:
        row = _withheld_concentration_row(bundle, predicted_chemical, "no_valid_regression_rows")
        return _concentration_payload(row)
    X, _ = enforce_feature_order(usable, bundle.regression_features)
    global_predictions = np.asarray(bundle.global_regressor_pipeline.predict(X), dtype=float)
    if predicted_chemical not in bundle.chemical_regressors:
        row = _withheld_concentration_row(bundle, predicted_chemical, "chemical_specific_regressor_unavailable")
        row["global_comparison_prediction"] = float(np.median(global_predictions))
        row["global_comparison_only"] = True
        return _concentration_payload(row)

    regressor = bundle.chemical_regressors[predicted_chemical]
    predictions = np.asarray(regressor.predict(X), dtype=float)
    predicted = float(np.median(predictions))
    uncertainty = _uncertainty_for(bundle, predicted_chemical)
    lower = max(0.0, predicted - 1.96 * uncertainty) if uncertainty is not None else float(np.quantile(predictions, 0.1))
    upper = predicted + 1.96 * uncertainty if uncertainty is not None else float(np.quantile(predictions, 0.9))
    range_status, nearest_range = concentration_range_status(predicted, predicted_chemical, bundle)
    replicate_spread = float(np.std(predictions, ddof=0)) if len(predictions) else 0.0
    confidence = float(1.0 / (1.0 + replicate_spread / (abs(predicted) + 1e-9)))
    row = {
        "predicted_chemical": predicted_chemical,
        "regression_strategy": bundle.regression_strategy,
        "regressor_used": "chemical_specific",
        "predicted_concentration": predicted,
        "concentration_units": bundle.concentration_units,
        "interval_lower": float(lower),
        "interval_upper": float(upper),
        "prediction_interval_method": "training_residual_uncertainty",
        "regression_confidence": confidence,
        "nearest_trained_concentration_range": nearest_range,
        "interpolation_status": range_status,
        "row_prediction_count": int(len(predictions)),
        "row_prediction_mean": float(np.mean(predictions)),
        "row_prediction_std": replicate_spread,
        "global_comparison_prediction": float(np.median(global_predictions)),
        "global_comparison_only": False,
        "withheld_reason": "",
    }
    rows.append(row)
    return _concentration_payload(row)


def concentration_range_status(
    predicted_concentration: float,
    predicted_chemical: str,
    bundle: FrozenModelBundle,
) -> tuple[str, str]:
    """Classify concentration as interpolation or extrapolation."""

    ranges = bundle.training_data_summary.get("concentration_ranges_by_chemical", {})
    selected_range = ranges.get(predicted_chemical)
    if selected_range is None:
        selected_range = {
            "min": bundle.training_data_summary.get("concentration_min"),
            "max": bundle.training_data_summary.get("concentration_max"),
        }
    minimum = _safe_float(selected_range.get("min"))
    maximum = _safe_float(selected_range.get("max"))
    if minimum is None or maximum is None:
        return "Unable to Assess", "training range unavailable"
    label = f"{minimum:g} to {maximum:g} {bundle.concentration_units}"
    if predicted_concentration < minimum or predicted_concentration > maximum:
        return "Extrapolation", label
    return "Interpolation", label


def _concentration_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_concentration": row.get("predicted_concentration"),
        "concentration_interval": (row.get("interval_lower"), row.get("interval_upper")),
        "regression_confidence": row.get("regression_confidence"),
        "table": pd.DataFrame([row]),
    }


def _withheld_concentration_row(bundle: FrozenModelBundle, predicted_chemical: str | None, reason: str) -> dict[str, Any]:
    return {
        "predicted_chemical": predicted_chemical,
        "regression_strategy": bundle.regression_strategy,
        "regressor_used": "withheld",
        "predicted_concentration": None,
        "concentration_units": bundle.concentration_units,
        "interval_lower": None,
        "interval_upper": None,
        "prediction_interval_method": "withheld",
        "regression_confidence": 0.0,
        "nearest_trained_concentration_range": "",
        "interpolation_status": "Unable to Assess",
        "row_prediction_count": 0,
        "row_prediction_mean": None,
        "row_prediction_std": None,
        "global_comparison_prediction": None,
        "global_comparison_only": False,
        "withheld_reason": reason,
    }


def _fit_chemical_specific_regressors(
    prepared: Any,
    *,
    regressor_spec: Any,
    preprocessing: str,
    random_state: int,
    min_rows: int,
    min_concentrations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    regressors: dict[str, Any] = {}
    uncertainty: dict[str, Any] = {}
    if "Chemical" not in prepared.dataframe.columns:
        return regressors, uncertainty
    for chemical, group in prepared.dataframe.groupby("Chemical", dropna=True, sort=True):
        indices = group.index.to_numpy(dtype=int)
        if len(indices) < int(min_rows):
            continue
        target = prepared.y[indices]
        if len(np.unique(target)) < int(min_concentrations):
            continue
        X = prepared.X.iloc[indices].copy(deep=True)
        pipeline = _pipeline(regressor_spec.factory(random_state), preprocessing)
        pipeline.fit(X, target)
        regressors[str(chemical)] = pipeline
        uncertainty[str(chemical)] = _residual_uncertainty(pipeline, X, target)
    return regressors, uncertainty


def _residual_uncertainty(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
    predictions = np.asarray(pipeline.predict(X), dtype=float)
    residuals = np.asarray(y, dtype=float) - predictions
    if len(residuals) < 2:
        return float(np.abs(residuals).mean()) if len(residuals) else 0.0
    return float(np.std(residuals, ddof=1))


def _uncertainty_for(bundle: FrozenModelBundle, chemical: str) -> float | None:
    chemical_uncertainty = bundle.training_data_summary.get("chemical_regression_uncertainty", {})
    value = chemical_uncertainty.get(chemical)
    if value is not None:
        return float(value)
    value = bundle.training_data_summary.get("global_regression_uncertainty")
    return None if value is None else float(value)


def _pipeline(estimator: Any, preprocessing: str) -> Pipeline:
    return Pipeline([("preprocess", _preprocessor(preprocessing)), ("model", estimator)])


def _preprocessor(preprocessing: str) -> Any:
    normalized = str(preprocessing).strip().casefold().replace("-", "").replace("_", "")
    if normalized in {"none", "no"}:
        return "passthrough"
    if normalized in {"z", "zscore", "standard"}:
        return StandardScaler()
    if normalized in {"robust", "robustscaler"}:
        return RobustScaler()
    if normalized in {"minmax", "minmaxscaler"}:
        return MinMaxScaler()
    raise ValueError("Unsupported preprocessing method for blind prediction.")


def _model_spec(model_id: str, random_state: int) -> Any:
    available, _ = available_model_specs(random_state=random_state, model_ids=(model_id,))
    if not available:
        raise ValueError(f"Classifier unavailable: {model_id}")
    return available[0]


def _regression_spec(model_id: str, random_state: int) -> Any:
    available, _ = available_regression_model_specs(random_state=random_state, model_ids=(model_id,))
    if not available:
        raise ValueError(f"Regressor unavailable: {model_id}")
    return available[0]


def _influential_features(feature_dataframe: pd.DataFrame, bundle: FrozenModelBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_importance_rows("classification", bundle.classifier_pipeline, bundle.classification_features, feature_dataframe, bundle))
    rows.extend(_importance_rows("regression", bundle.global_regressor_pipeline, bundle.regression_features, feature_dataframe, bundle))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)


def _importance_rows(
    task: str,
    pipeline: Pipeline,
    feature_names: list[str],
    feature_dataframe: pd.DataFrame,
    bundle: FrozenModelBundle,
) -> list[dict[str, Any]]:
    model = pipeline.named_steps.get("model")
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    means = bundle.training_distribution.get("feature_summary", {})
    rows = []
    values = feature_dataframe.loc[:, [feature for feature in feature_names if feature in feature_dataframe.columns]].apply(pd.to_numeric, errors="coerce")
    for feature, importance in zip(feature_names, importances, strict=True):
        blind_mean = _safe_float(values[feature].mean()) if feature in values.columns else None
        training_mean = _safe_float(means.get(feature, {}).get("mean")) if feature in means else None
        direction = "not_available"
        if blind_mean is not None and training_mean is not None:
            direction = "higher_than_training_mean" if blind_mean > training_mean else "lower_than_training_mean" if blind_mean < training_mean else "similar_to_training_mean"
        rows.append(
            {
                "task": task,
                "feature_name": feature,
                "feature_family": bundle.feature_family_map.get(feature, "unknown"),
                "importance": float(importance),
                "direction": direction,
                "blind_mean": blind_mean,
                "training_mean": training_mean,
            }
        )
    return rows


def _influential_strains(
    feature_dataframe: pd.DataFrame,
    *,
    probabilities: np.ndarray,
    class_labels: list[str],
    predicted_label: str | None,
    valid_mask: pd.Series,
) -> pd.DataFrame:
    if "Strain" not in feature_dataframe.columns or predicted_label is None or probabilities.size == 0:
        return pd.DataFrame(columns=["strain", "row_count", "predicted_chemical_probability"])
    usable = feature_dataframe.loc[valid_mask].reset_index(drop=True)
    if usable.empty:
        return pd.DataFrame(columns=["strain", "row_count", "predicted_chemical_probability"])
    label_index = class_labels.index(predicted_label)
    usable = usable.copy()
    usable["predicted_chemical_probability"] = probabilities[:, label_index]
    return (
        usable.groupby("Strain", dropna=False, sort=True)
        .agg(
            row_count=("predicted_chemical_probability", "size"),
            predicted_chemical_probability=("predicted_chemical_probability", "mean"),
        )
        .reset_index()
        .rename(columns={"Strain": "strain"})
    )


def _training_distribution(dataframe: pd.DataFrame, *, feature_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for feature in feature_names:
        if feature not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe[feature], errors="coerce")
        finite = values[np.isfinite(values.astype(float))]
        summary[feature] = {
            "mean": _safe_float(finite.mean()),
            "std": _safe_float(finite.std(ddof=0)),
            "min": _safe_float(finite.min()),
            "max": _safe_float(finite.max()),
            "finite_count": int(len(finite)),
        }
    return {"feature_summary": summary}


def _training_reference_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Experiment_ID",
        "Measurement_Unit_ID",
        "Source_File",
        "Strain",
        "Chemical",
        "Concentration",
        "Replicate_ID",
    ]
    return dataframe.loc[:, [column for column in columns if column in dataframe.columns]].copy(deep=True)


def _concentration_ranges_by_chemical(dataframe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "Chemical" not in dataframe.columns or "Concentration_Target_ug_mL" not in dataframe.columns:
        return {}
    ranges = {}
    for chemical, group in dataframe.groupby("Chemical", dropna=True, sort=True):
        values = pd.to_numeric(group["Concentration_Target_ug_mL"], errors="coerce").dropna()
        if values.empty:
            continue
        ranges[str(chemical)] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "unique_count": int(values.nunique()),
            "row_count": int(len(values)),
        }
    return ranges


def _time_window(canonical: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    times = pd.to_numeric(canonical.get("Time_Hours", pd.Series(dtype=float)), errors="coerce").dropna()
    minimum = float(times.min()) if len(times) else None
    maximum = float(times.max()) if len(times) else None
    required = 0.0
    if any(feature.startswith("window_12_24h") for feature in selected_features):
        required = 24.0
    elif any(feature.startswith("window_6_12h") for feature in selected_features):
        required = 12.0
    elif any(feature.startswith("window_2_6h") for feature in selected_features):
        required = 6.0
    elif any(feature.startswith("window_0_2h") for feature in selected_features):
        required = 2.0
    if maximum is None:
        label = "Unable to Assess"
    elif maximum >= 23.99:
        label = "common 0-24 h window"
    elif maximum >= 11.99:
        label = "common 0-12 h window"
    else:
        label = f"explicit 0-{maximum:g} h window"
    return {
        "label": label,
        "min_time_hours": minimum,
        "max_time_hours": maximum,
        "required_max_time_hours": required,
    }


def _truth_from_canonical(canonical: pd.DataFrame) -> dict[str, Any]:
    chemicals = canonical.get("Chemical_Name_Original", pd.Series(dtype=str)).dropna().astype(str)
    concentrations = pd.to_numeric(canonical.get("Concentration_ug_mL", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "true_chemical": None if chemicals.empty else str(chemicals.mode().iloc[0]),
        "true_concentration": None if concentrations.empty else float(concentrations.median()),
    }


def _evaluate_result_against_truth(result: BlindPredictionResult, truth: dict[str, Any]) -> dict[str, Any]:
    true_concentration = _safe_float(truth.get("true_concentration"))
    predicted_concentration = _safe_float(result.predicted_concentration)
    absolute_error = (
        None
        if true_concentration is None or predicted_concentration is None
        else abs(predicted_concentration - true_concentration)
    )
    return {
        "chemical_prediction_correct": (
            None if truth.get("true_chemical") is None else result.predicted_chemical == truth["true_chemical"]
        ),
        "concentration_absolute_error": absolute_error,
        "concentration_percentage_error": (
            None
            if absolute_error is None or true_concentration in {None, 0.0}
            else abs(absolute_error / true_concentration) * 100.0
        ),
        "novelty_status": result.novelty_status,
        "prediction_passed": result.prediction_passed,
    }


def _read_truth(path: Path) -> dict[str, Any]:
    if path.suffix.casefold() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError("Truth file is empty.")
    return table.iloc[0].to_dict()


def _strings(dataframe: pd.DataFrame, column: str) -> list[str]:
    if column not in dataframe.columns:
        return []
    return sorted(dataframe[column].dropna().astype(str).unique().tolist())


def _nunique(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(dataframe[column].dropna().astype("string").nunique())


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        if not math.isfinite(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
