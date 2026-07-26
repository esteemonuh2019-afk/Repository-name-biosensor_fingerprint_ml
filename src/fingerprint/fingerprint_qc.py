"""Quality checks for Stage 7A fingerprint datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import pandas as pd


REQUIRED_METADATA_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class FingerprintQCResult:
    """Fingerprint-level QC summary and issue table."""

    passed: bool
    warnings: list[str]
    errors: list[str]
    summary: dict[str, Any]
    issue_table: pd.DataFrame = field(repr=False)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable QC summary."""

        return {
            **self.summary,
            "passed": self.passed,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def evaluate_fingerprint_qc(
    dataframe: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    expected_feature_names: Iterable[str],
    source_dataframe: pd.DataFrame | None = None,
    excluded_dataframe: pd.DataFrame | None = None,
    source_feature_order: Iterable[str] | None = None,
) -> FingerprintQCResult:
    """Evaluate fingerprint structure without modifying fingerprint values."""

    feature_names = list(feature_names)
    expected_feature_names = list(expected_feature_names)
    source = source_dataframe if source_dataframe is not None else dataframe
    excluded = excluded_dataframe if excluded_dataframe is not None else pd.DataFrame()

    warnings: list[str] = []
    errors: list[str] = []

    missing_expected_features = [feature for feature in expected_feature_names if feature not in feature_names]
    unexpected_features = [feature for feature in feature_names if feature not in expected_feature_names]
    unexpected_order = feature_names != expected_feature_names
    source_order = list(source_feature_order or [])
    unexpected_source_order = bool(source_order) and source_order != expected_feature_names

    duplicate_fingerprint_row_count = _duplicate_fingerprint_row_count(dataframe, feature_names)
    duplicate_fingerprint_group_count = _duplicate_fingerprint_group_count(dataframe, feature_names)
    duplicated_measurement_unit_row_count = _duplicated_measurement_unit_row_count(dataframe)
    duplicated_measurement_unit_count = _duplicated_measurement_unit_count(dataframe)
    missing_metadata_cell_count = _missing_metadata_cell_count(dataframe)
    missing_feature_cell_count = _missing_feature_cell_count(dataframe, feature_names)
    source_missing_feature_cell_count = _missing_feature_cell_count(source, expected_feature_names)
    nonfinite_fingerprint_value_count = _nonfinite_feature_value_count(dataframe, feature_names)
    source_nonfinite_feature_value_count = _nonfinite_feature_value_count(source, expected_feature_names)
    excluded_rows = int(len(excluded))

    if missing_expected_features:
        errors.append(
            "Missing expected fingerprint features: "
            + ", ".join(missing_expected_features)
            + "."
        )
    if unexpected_features:
        warnings.append(
            "Unexpected fingerprint feature names: " + ", ".join(unexpected_features) + "."
        )
    if unexpected_order:
        warnings.append("Fingerprint feature order differs from the expected core feature order.")
    if unexpected_source_order:
        warnings.append("Validated feature input order differs from the expected core feature order.")
    if duplicate_fingerprint_row_count:
        warnings.append(
            "Duplicate fingerprint vectors detected across "
            f"{duplicate_fingerprint_group_count} groups and "
            f"{duplicate_fingerprint_row_count} rows."
        )
    if duplicated_measurement_unit_row_count:
        warnings.append(
            "Duplicated Measurement_Unit_ID values detected across "
            f"{duplicated_measurement_unit_count} identifiers and "
            f"{duplicated_measurement_unit_row_count} fingerprint rows."
        )
    if missing_metadata_cell_count:
        warnings.append(f"Missing fingerprint metadata cells detected: {missing_metadata_cell_count}.")
    if missing_feature_cell_count:
        warnings.append(f"Missing fingerprint feature cells detected: {missing_feature_cell_count}.")
    if nonfinite_fingerprint_value_count:
        errors.append(f"Non-finite fingerprint values detected: {nonfinite_fingerprint_value_count}.")
    if excluded_rows:
        warnings.append(f"Feature rows excluded from fingerprint matrix: {excluded_rows}.")
    if source_missing_feature_cell_count:
        warnings.append(
            "Validated feature input contained missing feature cells: "
            f"{source_missing_feature_cell_count}."
        )
    if source_nonfinite_feature_value_count:
        warnings.append(
            "Validated feature input contained non-finite feature cells: "
            f"{source_nonfinite_feature_value_count}."
        )

    issue_table = _issue_table(dataframe, excluded)
    summary = {
        "fingerprint_rows": int(len(dataframe)),
        "fingerprint_feature_count": int(len(feature_names)),
        "expected_feature_count": int(len(expected_feature_names)),
        "missing_expected_feature_count": int(len(missing_expected_features)),
        "unexpected_feature_name_count": int(len(unexpected_features)),
        "unexpected_feature_order": bool(unexpected_order),
        "unexpected_source_feature_order": bool(unexpected_source_order),
        "duplicate_fingerprint_row_count": duplicate_fingerprint_row_count,
        "duplicate_fingerprint_group_count": duplicate_fingerprint_group_count,
        "duplicated_measurement_unit_row_count": duplicated_measurement_unit_row_count,
        "duplicated_measurement_unit_count": duplicated_measurement_unit_count,
        "missing_metadata_cell_count": missing_metadata_cell_count,
        "missing_feature_cell_count": missing_feature_cell_count,
        "source_missing_feature_cell_count": source_missing_feature_cell_count,
        "nonfinite_fingerprint_value_count": nonfinite_fingerprint_value_count,
        "source_nonfinite_feature_value_count": source_nonfinite_feature_value_count,
        "excluded_rows": excluded_rows,
    }
    return FingerprintQCResult(
        passed=not errors,
        warnings=warnings,
        errors=errors,
        summary=summary,
        issue_table=issue_table,
    )


def render_fingerprint_qc_report(summary: dict[str, Any], qc_result: FingerprintQCResult) -> str:
    """Render a Markdown report for fingerprint QC."""

    lines = [
        "# Stage 7A Fingerprint QC Report",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## QC Counts"])
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

    lines.extend(
        [
            "",
            "## Notes",
            "- Fingerprints contain metadata and validated core feature values only.",
            "- Raw luminescence time series are not included.",
            "- Rows with failed feature QC or non-finite core features are excluded from the fingerprint matrix and counted in QC.",
            "- Normalised fingerprints are written separately from the original feature-scale fingerprints.",
        ]
    )
    return "\n".join(lines) + "\n"


def _duplicate_fingerprint_row_count(dataframe: pd.DataFrame, feature_names: list[str]) -> int:
    if dataframe.empty or any(feature not in dataframe.columns for feature in feature_names):
        return 0
    mask = dataframe.loc[:, feature_names].duplicated(keep=False)
    return int(mask.sum())


def _duplicate_fingerprint_group_count(dataframe: pd.DataFrame, feature_names: list[str]) -> int:
    if dataframe.empty or any(feature not in dataframe.columns for feature in feature_names):
        return 0
    duplicated = dataframe.loc[dataframe.loc[:, feature_names].duplicated(keep=False), feature_names]
    if duplicated.empty:
        return 0
    return int(len(duplicated.drop_duplicates()))


def _duplicated_measurement_unit_row_count(dataframe: pd.DataFrame) -> int:
    if dataframe.empty or "Measurement_Unit_ID" not in dataframe.columns:
        return 0
    return int(dataframe["Measurement_Unit_ID"].astype("string").duplicated(keep=False).sum())


def _duplicated_measurement_unit_count(dataframe: pd.DataFrame) -> int:
    if dataframe.empty or "Measurement_Unit_ID" not in dataframe.columns:
        return 0
    duplicated = dataframe.loc[
        dataframe["Measurement_Unit_ID"].astype("string").duplicated(keep=False),
        "Measurement_Unit_ID",
    ]
    return int(duplicated.astype("string").nunique())


def _missing_metadata_cell_count(dataframe: pd.DataFrame) -> int:
    if dataframe.empty:
        return 0
    columns = [column for column in REQUIRED_METADATA_COLUMNS if column in dataframe.columns]
    return int(dataframe.loc[:, columns].isna().sum().sum()) if columns else 0


def _missing_feature_cell_count(dataframe: pd.DataFrame, feature_names: list[str]) -> int:
    columns = [feature for feature in feature_names if feature in dataframe.columns]
    if dataframe.empty or not columns:
        return 0
    return int(dataframe.loc[:, columns].isna().sum().sum())


def _nonfinite_feature_value_count(dataframe: pd.DataFrame, feature_names: list[str]) -> int:
    columns = [feature for feature in feature_names if feature in dataframe.columns]
    if dataframe.empty or not columns:
        return 0
    count = 0
    for feature in columns:
        numeric = pd.to_numeric(dataframe[feature], errors="coerce")
        count += int(numeric.isna().sum())
        count += sum(
            1
            for value in numeric.dropna()
            if not math.isfinite(float(value))
        )
    return int(count)


def _issue_table(dataframe: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in excluded.iterrows():
        rows.append(
            {
                "Fingerprint_ID": row.get("Fingerprint_ID", pd.NA),
                "Measurement_Unit_ID": row.get("Measurement_Unit_ID", pd.NA),
                "Source_File": row.get("Source_File", pd.NA),
                "issue": row.get("Fingerprint_Exclusion_Reason", "excluded_feature_row"),
            }
        )

    if not dataframe.empty and "Measurement_Unit_ID" in dataframe.columns:
        duplicate_mask = dataframe["Measurement_Unit_ID"].astype("string").duplicated(keep=False)
        for _, row in dataframe.loc[duplicate_mask].iterrows():
            rows.append(
                {
                    "Fingerprint_ID": row.get("Fingerprint_ID", pd.NA),
                    "Measurement_Unit_ID": row.get("Measurement_Unit_ID", pd.NA),
                    "Source_File": row.get("Source_File", pd.NA),
                    "issue": "duplicated_measurement_unit_id",
                }
            )
    return pd.DataFrame(rows)
