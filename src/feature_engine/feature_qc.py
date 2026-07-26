"""Quality checks for canonical feature datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import pandas as pd


FEATURE_QC_FLAG_COLUMN = "Feature_QC_Flags"
FEATURE_QC_STATUS_COLUMN = "QC_Status"


@dataclass(frozen=True)
class FeatureQCResult:
    """Feature-level QC summary and per-series issue table."""

    passed: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]
    issue_table: pd.DataFrame = field(repr=False)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable QC summary."""

        return {
            **self.summary,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def evaluate_feature_qc(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
) -> FeatureQCResult:
    """Audit a feature DataFrame without changing feature values."""

    feature_columns = [column for column in feature_columns if column in dataframe.columns]
    errors: list[str] = []
    warnings: list[str] = []

    row_count = int(len(dataframe))
    missing_feature_value_count = _missing_feature_value_count(dataframe, feature_columns)
    infinite_feature_value_count = _infinite_feature_value_count(dataframe, feature_columns)
    impossible_time_to_peak_count = _impossible_time_to_peak_count(dataframe)
    zero_baseline_count = _zero_baseline_count(dataframe)
    negative_time_count = _negative_time_count(dataframe)
    empty_series_count = _flag_count(dataframe, "empty_series")
    duplicate_measurement_unit_row_count = _duplicate_measurement_unit_row_count(dataframe)
    duplicated_measurement_unit_count = _duplicated_measurement_unit_count(dataframe)
    failed_feature_rows = _status_count(dataframe, "fail")
    warning_feature_rows = _status_count(dataframe, "warning")

    if missing_feature_value_count:
        warnings.append(f"Missing feature values detected: {missing_feature_value_count}.")
    if infinite_feature_value_count:
        errors.append(f"Infinite feature values detected: {infinite_feature_value_count}.")
    if impossible_time_to_peak_count:
        errors.append(
            "Impossible time-to-peak values detected: "
            f"{impossible_time_to_peak_count} feature rows."
        )
    if zero_baseline_count:
        warnings.append(
            "Zero baseline values prevent fold-change calculation: "
            f"{zero_baseline_count} feature rows."
        )
    if negative_time_count:
        errors.append(f"Negative time values detected: {negative_time_count} feature rows.")
    if empty_series_count:
        errors.append(f"Empty measurement series detected: {empty_series_count}.")
    if duplicate_measurement_unit_row_count:
        warnings.append(
            "Feature rows contain duplicated Measurement_Unit_ID values across "
            f"{duplicated_measurement_unit_count} identifiers and "
            f"{duplicate_measurement_unit_row_count} rows."
        )
    if failed_feature_rows:
        errors.append(f"Feature QC failed for {failed_feature_rows} rows.")
    if warning_feature_rows:
        warnings.append(f"Feature QC warnings present for {warning_feature_rows} rows.")

    issue_table = _issue_table(dataframe)
    summary = {
        "feature_row_count": row_count,
        "feature_column_count": len(feature_columns),
        "missing_feature_value_count": missing_feature_value_count,
        "infinite_feature_value_count": infinite_feature_value_count,
        "impossible_time_to_peak_count": impossible_time_to_peak_count,
        "zero_baseline_count": zero_baseline_count,
        "negative_time_count": negative_time_count,
        "empty_series_count": empty_series_count,
        "duplicate_measurement_unit_row_count": duplicate_measurement_unit_row_count,
        "duplicated_measurement_unit_count": duplicated_measurement_unit_count,
        "failed_feature_rows": failed_feature_rows,
        "warning_feature_rows": warning_feature_rows,
    }
    return FeatureQCResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        summary=summary,
        issue_table=issue_table,
    )


def render_feature_qc_report(qc_result: FeatureQCResult, summary: dict[str, Any]) -> str:
    """Render a Markdown feature QC report."""

    lines = [
        "# Stage 6B Feature QC Report",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        if isinstance(value, (list, dict)):
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Feature QC Counts"])
    for key, value in qc_result.summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Warnings"])
    if qc_result.warnings:
        lines.extend(f"- {warning}" for warning in qc_result.warnings)
    else:
        lines.append("- None")

    lines.extend(["", "## Errors"])
    if qc_result.errors:
        lines.extend(f"- {error}" for error in qc_result.errors)
    else:
        lines.append("- None")

    lines.extend(["", "## Notes"])
    lines.append("- Duplicate timestamps are flagged and are not averaged.")
    lines.append("- Fold-change features are null when baseline is zero or non-positive.")
    lines.append("- Time-dependent features are null for unresolved duplicate time intervals.")
    return "\n".join(lines) + "\n"


def _missing_feature_value_count(dataframe: pd.DataFrame, feature_columns: list[str]) -> int:
    if dataframe.empty or not feature_columns:
        return 0
    return int(dataframe.loc[:, feature_columns].isna().sum().sum())


def _infinite_feature_value_count(dataframe: pd.DataFrame, feature_columns: list[str]) -> int:
    if dataframe.empty or not feature_columns:
        return 0

    count = 0
    for column in feature_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        count += sum(1 for value in values.dropna() if math.isinf(float(value)))
    return int(count)


def _impossible_time_to_peak_count(dataframe: pd.DataFrame) -> int:
    if "time_to_peak" not in dataframe.columns:
        return 0
    time_to_peak = pd.to_numeric(dataframe["time_to_peak"], errors="coerce")

    if {"Start_Time", "End_Time"} <= set(dataframe.columns):
        start_time = pd.to_numeric(dataframe["Start_Time"], errors="coerce")
        end_time = pd.to_numeric(dataframe["End_Time"], errors="coerce")
        mask = time_to_peak.notna() & start_time.notna() & end_time.notna() & (
            (time_to_peak < start_time) | (time_to_peak > end_time)
        )
        return int(mask.sum())

    if "Duration" not in dataframe.columns:
        return 0
    duration = pd.to_numeric(dataframe["Duration"], errors="coerce")
    mask = time_to_peak.notna() & duration.notna() & (time_to_peak > duration)
    return int(mask.sum())


def _zero_baseline_count(dataframe: pd.DataFrame) -> int:
    if "baseline" not in dataframe.columns:
        return 0
    baseline = pd.to_numeric(dataframe["baseline"], errors="coerce")
    return int(baseline.eq(0).sum())


def _negative_time_count(dataframe: pd.DataFrame) -> int:
    count = 0
    if "Duration" in dataframe.columns:
        duration = pd.to_numeric(dataframe["Duration"], errors="coerce")
        count += int((duration < 0).sum())
    count += _flag_count(dataframe, "negative_time_values")
    return int(count)


def _duplicate_measurement_unit_row_count(dataframe: pd.DataFrame) -> int:
    if "Measurement_Unit_ID" not in dataframe.columns or dataframe.empty:
        return 0
    mask = dataframe["Measurement_Unit_ID"].astype("string").duplicated(keep=False)
    return int(mask.sum())


def _duplicated_measurement_unit_count(dataframe: pd.DataFrame) -> int:
    if "Measurement_Unit_ID" not in dataframe.columns or dataframe.empty:
        return 0
    duplicated = dataframe.loc[
        dataframe["Measurement_Unit_ID"].astype("string").duplicated(keep=False),
        "Measurement_Unit_ID",
    ]
    return int(duplicated.astype("string").nunique())


def _status_count(dataframe: pd.DataFrame, status: str) -> int:
    if FEATURE_QC_STATUS_COLUMN not in dataframe.columns:
        return 0
    return int(dataframe[FEATURE_QC_STATUS_COLUMN].astype("string").eq(status).sum())


def _flag_count(dataframe: pd.DataFrame, flag: str) -> int:
    if FEATURE_QC_FLAG_COLUMN not in dataframe.columns or dataframe.empty:
        return 0
    flags = dataframe[FEATURE_QC_FLAG_COLUMN].astype("string").fillna("")
    mask = flags.map(lambda value: flag in _split_flags(value)).astype(bool)
    return int(mask.sum())


def _issue_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "Experiment_ID",
            "Source_File",
            "Measurement_Unit_ID",
            FEATURE_QC_STATUS_COLUMN,
            FEATURE_QC_FLAG_COLUMN,
        )
        if column in dataframe.columns
    ]
    if not columns or dataframe.empty or FEATURE_QC_FLAG_COLUMN not in dataframe.columns:
        return pd.DataFrame(columns=columns)

    flags = dataframe[FEATURE_QC_FLAG_COLUMN].astype("string").fillna("")
    issue_mask = flags.str.strip().ne("")
    return dataframe.loc[issue_mask, columns].reset_index(drop=True)


def _split_flags(value: str) -> set[str]:
    return {part.strip() for part in value.split(";") if part.strip()}
