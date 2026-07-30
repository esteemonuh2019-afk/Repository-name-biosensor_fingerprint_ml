"""QC gates for Stage 9A blind prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


QC_STATUSES: tuple[str, ...] = ("PASS", "PASS WITH WARNINGS", "FAIL")


@dataclass(frozen=True)
class PredictionQCResult:
    """Prediction gate status and detailed issue table."""

    status: str
    warnings: list[str]
    errors: list[str]
    summary: dict[str, Any]
    gate_table: pd.DataFrame
    valid_classification_mask: pd.Series
    valid_regression_mask: pd.Series

    @property
    def passed(self) -> bool:
        """Return whether prediction may be treated as ordinary."""

        return self.status in {"PASS", "PASS WITH WARNINGS"}

    def with_severe_novelty(self, reason: str) -> "PredictionQCResult":
        """Return a failed QC result after severe novelty is detected."""

        rows = [*self.gate_table.to_dict(orient="records")]
        rows.append({"gate": "severe_novelty", "status": "FAIL", "detail": reason})
        return PredictionQCResult(
            status="FAIL",
            warnings=list(self.warnings),
            errors=[*self.errors, reason],
            summary={**self.summary, "severe_novelty": True},
            gate_table=pd.DataFrame(rows),
            valid_classification_mask=self.valid_classification_mask.copy(),
            valid_regression_mask=self.valid_regression_mask.copy(),
        )


def evaluate_prediction_qc(
    *,
    canonical_qc: Any,
    canonical_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    bundle: Any,
    minimum_measurement_units: int = 1,
) -> PredictionQCResult:
    """Evaluate strict blind-prediction QC gates without mutating inputs."""

    warnings: list[str] = []
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    features = feature_dataframe.copy(deep=True)
    total_feature_rows = int(len(features))

    _gate(
        rows,
        "canonical_qc",
        "FAIL" if getattr(canonical_qc, "errors", []) else "PASS WITH WARNINGS" if getattr(canonical_qc, "warnings", []) else "PASS",
        "; ".join(getattr(canonical_qc, "errors", []) or getattr(canonical_qc, "warnings", []) or ["canonical QC passed"]),
    )
    if getattr(canonical_qc, "errors", []):
        errors.extend(f"canonical_qc: {error}" for error in canonical_qc.errors)
    elif getattr(canonical_qc, "warnings", []):
        warnings.extend(f"canonical_qc: {warning}" for warning in canonical_qc.warnings)

    missing_classification = [feature for feature in bundle.classification_features if feature not in features.columns]
    missing_regression = [feature for feature in bundle.regression_features if feature not in features.columns]
    missing_required = sorted(set(missing_classification + missing_regression))
    if missing_required:
        detail = "Missing required model features: " + ", ".join(missing_required)
        errors.append(detail)
        _gate(rows, "missing_required_features", "FAIL", detail)
    else:
        _gate(rows, "missing_required_features", "PASS", "all selected features present")

    model_features = set(bundle.classification_features) | set(bundle.regression_features)
    candidate_feature_columns = [
        column
        for column in features.columns
        if column not in _metadata_columns() and pd.api.types.is_numeric_dtype(pd.to_numeric(features[column], errors="coerce"))
    ]
    extra_features = sorted(set(candidate_feature_columns) - model_features)
    if extra_features:
        warnings.append("Extra feature columns ignored: " + ", ".join(extra_features[:20]))
        _gate(rows, "extra_features", "PASS WITH WARNINGS", f"{len(extra_features)} extra feature columns ignored")
    else:
        _gate(rows, "extra_features", "PASS", "no extra feature columns")

    class_mask = _finite_mask(features, bundle.classification_features)
    reg_mask = _finite_mask(features, bundle.regression_features)
    valid_class_count = int(class_mask.sum())
    valid_reg_count = int(reg_mask.sum())
    if valid_class_count == 0:
        errors.append("No rows contain finite values for all classification features.")
        _gate(rows, "classification_feature_values", "FAIL", "no finite classification feature rows")
    elif valid_class_count < total_feature_rows:
        warnings.append(
            f"Rows excluded from blind classification because selected features were non-finite: {total_feature_rows - valid_class_count}."
        )
        _gate(rows, "classification_feature_values", "PASS WITH WARNINGS", f"{valid_class_count} usable rows")
    else:
        _gate(rows, "classification_feature_values", "PASS", f"{valid_class_count} usable rows")

    if valid_reg_count == 0:
        errors.append("No rows contain finite values for all regression features.")
        _gate(rows, "regression_feature_values", "FAIL", "no finite regression feature rows")
    elif valid_reg_count < total_feature_rows:
        warnings.append(
            f"Rows excluded from blind regression because selected features were non-finite: {total_feature_rows - valid_reg_count}."
        )
        _gate(rows, "regression_feature_values", "PASS WITH WARNINGS", f"{valid_reg_count} usable rows")
    else:
        _gate(rows, "regression_feature_values", "PASS", f"{valid_reg_count} usable rows")

    measurement_units = _unique_count(features, "Measurement_Unit_ID")
    if measurement_units < int(minimum_measurement_units):
        detail = f"Insufficient measurement units: {measurement_units}; required {minimum_measurement_units}."
        errors.append(detail)
        _gate(rows, "measurement_unit_count", "FAIL", detail)
    else:
        _gate(rows, "measurement_unit_count", "PASS", f"{measurement_units} measurement units")

    required_strains = list(getattr(bundle, "required_strains", []) or [])
    observed_strains = set(_strings(features, "Strain"))
    missing_strains = sorted(set(required_strains) - observed_strains)
    if missing_strains:
        detail = "Missing required strains: " + ", ".join(missing_strains)
        errors.append(detail)
        _gate(rows, "required_strains", "FAIL", detail)
    else:
        _gate(rows, "required_strains", "PASS", "required strains present or not configured")

    time_status, time_detail = _time_window_gate(canonical_dataframe, bundle.time_window)
    _gate(rows, "time_window_compatibility", time_status, time_detail)
    if time_status == "FAIL":
        errors.append(time_detail)
    elif time_status == "PASS WITH WARNINGS":
        warnings.append(time_detail)

    failed_feature_rows = _failed_feature_rows(features)
    if failed_feature_rows:
        warnings.append(f"Feature rows with QC_Status=fail retained as context but excluded: {failed_feature_rows}.")
        _gate(rows, "feature_qc_status", "PASS WITH WARNINGS", f"{failed_feature_rows} failed feature rows")
    else:
        _gate(rows, "feature_qc_status", "PASS", "no failed feature rows")

    status = "FAIL" if errors else "PASS WITH WARNINGS" if warnings else "PASS"
    summary = {
        "status": status,
        "feature_rows": total_feature_rows,
        "valid_classification_rows": valid_class_count,
        "valid_regression_rows": valid_reg_count,
        "measurement_unit_count": measurement_units,
        "missing_required_feature_count": len(missing_required),
        "extra_feature_count": len(extra_features),
        "required_strain_count": len(required_strains),
        "missing_required_strain_count": len(missing_strains),
    }
    return PredictionQCResult(
        status=status,
        warnings=warnings,
        errors=errors,
        summary=summary,
        gate_table=pd.DataFrame(rows),
        valid_classification_mask=class_mask,
        valid_regression_mask=reg_mask,
    )


def enforce_feature_order(
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return selected features in exact model order and explicit warnings for extras."""

    missing = [feature for feature in feature_names if feature not in dataframe.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))
    extra = sorted(
        column
        for column in dataframe.columns
        if column not in feature_names and column not in _metadata_columns()
    )
    warnings = [f"Extra columns ignored during feature alignment: {', '.join(extra[:20])}."] if extra else []
    values = dataframe.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").astype(float)
    return values, warnings


def _finite_mask(dataframe: pd.DataFrame, feature_names: list[str]) -> pd.Series:
    if dataframe.empty or any(feature not in dataframe.columns for feature in feature_names):
        return pd.Series(False, index=dataframe.index)
    values = dataframe.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    return values.apply(lambda column: np.isfinite(column.astype(float))).all(axis=1)


def _time_window_gate(canonical_dataframe: pd.DataFrame, time_window: dict[str, Any]) -> tuple[str, str]:
    if canonical_dataframe.empty or "Time_Hours" not in canonical_dataframe.columns:
        return "FAIL", "Blind sample has no Time_Hours values for time-window compatibility."
    observed = pd.to_numeric(canonical_dataframe["Time_Hours"], errors="coerce")
    max_observed = observed.max(skipna=True)
    if pd.isna(max_observed):
        return "FAIL", "Blind sample has no finite Time_Hours values."
    required_max = time_window.get("required_max_time_hours")
    if required_max is not None and float(max_observed) + 1e-9 < float(required_max):
        return (
            "FAIL",
            f"Blind sample maximum time window {float(max_observed):.6g} h is shorter than required {float(required_max):.6g} h.",
        )
    training_max = time_window.get("max_time_hours")
    if training_max is not None and float(max_observed) > float(training_max) * 1.25:
        return (
            "PASS WITH WARNINGS",
            f"Blind sample maximum time {float(max_observed):.6g} h exceeds training maximum {float(training_max):.6g} h.",
        )
    return "PASS", f"Blind sample maximum time {float(max_observed):.6g} h is compatible."


def _gate(rows: list[dict[str, Any]], gate: str, status: str, detail: str) -> None:
    rows.append({"gate": gate, "status": status, "detail": detail})


def _metadata_columns() -> set[str]:
    return {
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
        "Feature_QC_Flags",
        "Start_Time",
        "End_Time",
        "Input_Row_Count",
        "Valid_Observation_Count",
        "Missing_Observation_Count",
        "Duplicate_Timestamp_Count",
        "Duplicate_Timestamp_Group_Count",
        "Source_QC_Statuses",
        "Source_QC_Flags",
        "Advanced_Feature_QC_Flags",
    }


def _unique_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(dataframe[column].dropna().astype("string").nunique())


def _strings(dataframe: pd.DataFrame, column: str) -> list[str]:
    if column not in dataframe.columns:
        return []
    return sorted(dataframe[column].dropna().astype(str).unique().tolist())


def _failed_feature_rows(dataframe: pd.DataFrame) -> int:
    if "QC_Status" not in dataframe.columns:
        return 0
    return int(dataframe["QC_Status"].astype("string").eq("fail").sum())
