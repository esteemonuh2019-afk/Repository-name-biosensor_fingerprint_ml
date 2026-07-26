"""Quality-control audit for canonical biosensor datasets.

The checks in this module are intentionally read-only. They classify duplicate
rows and suspicious measurement keys, but they do not drop, impute, rename, or
standardise source-derived values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from src.data_schema.canonical_schema import (
    CANONICAL_COLUMNS,
    MEASUREMENT_KEY_COLUMNS,
    REQUIRED_FIELDS,
    SERIES_GROUPING_KEY_COLUMNS,
    coerce_canonical_dtypes,
)


LEGACY_SCHEMA_LOGICAL_KEY = [
    "Experiment_ID",
    "Plate_ID",
    "Strain_Original",
    "Chemical_Name_Original",
    "Concentration_Label",
    "Replicate_ID",
    "Well_ID",
    "Time_Minutes",
]

SOURCE_ROW_KEY = ["Source_File", "Source_Row_ID"]

MEASUREMENT_KEY = list(MEASUREMENT_KEY_COLUMNS)
SERIES_GROUPING_KEY = list(SERIES_GROUPING_KEY_COLUMNS)
SOURCE_AWARE_LEGACY_LOGICAL_KEY = ["Source_File", *LEGACY_SCHEMA_LOGICAL_KEY]


@dataclass(frozen=True)
class CanonicalQCResult:
    """Result object returned by :func:`audit_canonical_dataframe`."""

    row_count: int
    measurement_unit_count: int
    synthetic_measurement_unit_count: int
    unresolved_measurement_unit_count: int
    exact_duplicate_count: int
    source_row_id_duplicate_count: int
    logical_duplicate_count: int
    duplicate_group_count: int
    legacy_logical_duplicate_count: int
    legacy_duplicate_group_count: int
    source_aware_logical_duplicate_count: int
    source_aware_duplicate_group_count: int
    identical_value_duplicate_count: int
    conflicting_value_duplicate_count: int
    ambiguous_measurement_identity_count: int
    ambiguous_measurement_identity_group_count: int
    separate_replicate_measurement_count: int
    separate_replicate_measurement_group_count: int
    ambiguous_replicate_count: int
    ambiguous_replicate_group_count: int
    missing_required_value_counts: dict[str, int]
    missing_identifier_counts: dict[str, int]
    invalid_numeric_counts: dict[str, int]
    negative_luminescence_count: int
    infinite_luminescence_count: int
    non_monotonic_time_group_count: int
    duplicate_timepoint_group_count: int
    time_series_issue_count: int
    strains_detected: list[str]
    chemicals_detected: list[str]
    concentrations_detected: list[str]
    source_files: list[str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    qc_passed: bool = False

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary without embedded DataFrames."""

        return {
            "row_count": self.row_count,
            "measurement_unit_count": self.measurement_unit_count,
            "synthetic_measurement_unit_count": self.synthetic_measurement_unit_count,
            "unresolved_measurement_unit_count": self.unresolved_measurement_unit_count,
            "exact_duplicate_count": self.exact_duplicate_count,
            "source_row_id_duplicate_count": self.source_row_id_duplicate_count,
            "logical_duplicate_count": self.logical_duplicate_count,
            "duplicate_group_count": self.duplicate_group_count,
            "legacy_logical_duplicate_count": self.legacy_logical_duplicate_count,
            "legacy_duplicate_group_count": self.legacy_duplicate_group_count,
            "source_aware_logical_duplicate_count": (
                self.source_aware_logical_duplicate_count
            ),
            "source_aware_duplicate_group_count": (
                self.source_aware_duplicate_group_count
            ),
            "identical_value_duplicate_count": self.identical_value_duplicate_count,
            "conflicting_value_duplicate_count": self.conflicting_value_duplicate_count,
            "ambiguous_measurement_identity_count": self.ambiguous_measurement_identity_count,
            "ambiguous_measurement_identity_group_count": (
                self.ambiguous_measurement_identity_group_count
            ),
            "separate_replicate_measurement_count": self.separate_replicate_measurement_count,
            "separate_replicate_measurement_group_count": (
                self.separate_replicate_measurement_group_count
            ),
            "ambiguous_replicate_count": self.ambiguous_replicate_count,
            "ambiguous_replicate_group_count": self.ambiguous_replicate_group_count,
            "missing_required_value_counts": self.missing_required_value_counts,
            "missing_identifier_counts": self.missing_identifier_counts,
            "invalid_numeric_counts": self.invalid_numeric_counts,
            "negative_luminescence_count": self.negative_luminescence_count,
            "infinite_luminescence_count": self.infinite_luminescence_count,
            "non_monotonic_time_group_count": self.non_monotonic_time_group_count,
            "duplicate_timepoint_group_count": self.duplicate_timepoint_group_count,
            "time_series_issue_count": self.time_series_issue_count,
            "strains_detected": self.strains_detected,
            "chemicals_detected": self.chemicals_detected,
            "concentrations_detected": self.concentrations_detected,
            "source_files": self.source_files,
            "warnings": self.warnings,
            "errors": self.errors,
            "qc_passed": self.qc_passed,
        }


def audit_canonical_dataframe(dataframe: pd.DataFrame) -> CanonicalQCResult:
    """Audit a canonical biosensor DataFrame without mutating the input."""

    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [col for col in CANONICAL_COLUMNS if col not in dataframe.columns]
    unexpected_columns = [col for col in dataframe.columns if col not in CANONICAL_COLUMNS]

    if missing_columns:
        errors.append(
            "Canonical dataframe is missing required schema columns: "
            + ", ".join(missing_columns)
        )
    if unexpected_columns:
        warnings.append(
            "Canonical dataframe contains unexpected columns: "
            + ", ".join(unexpected_columns)
        )

    if dataframe.empty:
        errors.append("Canonical dataframe is empty.")

    df = _coerce_without_mutation(dataframe)
    row_count = int(len(df))

    missing_required, missing_required_errors, missing_required_warnings = (
        _missing_required_counts(df)
    )
    errors.extend(missing_required_errors)
    warnings.extend(missing_required_warnings)

    missing_identifier_counts = {
        col: _missing_count(df, col)
        for col in [
            "Experiment_ID",
            "Plate_ID",
            "Well_ID",
            "Replicate_ID",
            "Source_Row_ID",
            "Measurement_Unit_ID",
        ]
        if col in df.columns
    }
    measurement_unit_count = _unique_count(df, "Measurement_Unit_ID")
    synthetic_measurement_unit_count = _flagged_unit_count(
        df,
        "measurement_unit_id_synthetic",
    )
    unresolved_measurement_unit_count = _missing_count(df, "Measurement_Unit_ID")

    source_files = _unique_strings(df, "Source_File")
    strains = _unique_strings(df, "Strain_Original")
    chemicals = _unique_strings(df, "Chemical_Name_Original")
    concentrations = _unique_strings(df, "Concentration_Label")

    exact_duplicate_count = _exact_duplicate_count(df)
    if exact_duplicate_count:
        errors.append(
            f"Found {exact_duplicate_count} rows duplicated across all canonical columns."
        )

    source_row_id_duplicate_count, source_row_groups = _duplicate_groups(
        df,
        SOURCE_ROW_KEY,
        include_missing=False,
    )
    if source_row_id_duplicate_count:
        warnings.append(
            f"Found {source_row_id_duplicate_count} rows sharing Source_File and Source_Row_ID."
        )

    legacy_logical_duplicate_count, legacy_logical_groups = _duplicate_groups(
        df,
        LEGACY_SCHEMA_LOGICAL_KEY,
        include_missing=True,
    )
    if legacy_logical_duplicate_count:
        warnings.append(
            "The legacy Stage 5A logical key reports "
            f"{legacy_logical_duplicate_count} duplicate rows in "
            f"{len(legacy_logical_groups)} groups."
        )

    logical_duplicate_count, logical_groups = _duplicate_groups(
        df,
        MEASUREMENT_KEY,
        include_missing=False,
    )
    if logical_duplicate_count:
        warnings.append(
            "The corrected measurement key reports "
            f"{logical_duplicate_count} duplicate rows in "
            f"{len(logical_groups)} groups."
        )

    (
        identical_value_duplicate_count,
        conflicting_value_duplicate_count,
        duplicate_value_groups,
        conflicting_duplicate_rows,
    ) = _classify_duplicate_values(df, MEASUREMENT_KEY)

    if conflicting_value_duplicate_count:
        errors.append(
            "Duplicate measurement-key groups contain conflicting Luminescence_Raw values."
        )

    if identical_value_duplicate_count:
        warnings.append(
            "Duplicate measurement-key groups contain identical Luminescence_Raw values."
        )

    source_aware_logical_duplicate_count, source_aware_groups = _duplicate_groups(
        df,
        SOURCE_AWARE_LEGACY_LOGICAL_KEY,
        include_missing=True,
    )
    if source_aware_logical_duplicate_count:
        warnings.append(
            "Source-aware legacy logical key reports "
            f"{source_aware_logical_duplicate_count} duplicate rows in "
            f"{len(source_aware_groups)} groups."
        )

    (
        separate_replicate_measurement_count,
        separate_replicate_measurement_group_count,
        separate_replicate_measurements,
    ) = _separate_measurements_resolved_by_unit(df)
    if separate_replicate_measurement_count:
        warnings.append(
            "Legacy duplicate rows are separated by Measurement_Unit_ID: "
            f"{separate_replicate_measurement_count} rows."
        )

    (
        ambiguous_measurement_identity_count,
        ambiguous_measurement_identity_group_count,
        ambiguous_measurement_identities,
    ) = _ambiguous_measurement_identities(df)
    if ambiguous_measurement_identity_count:
        warnings.append(
            "Measurement-unit identity remains ambiguous for "
            f"{ambiguous_measurement_identity_count} rows."
        )

    ambiguous_replicate_group_count = _ambiguous_duplicate_group_count(
        df,
        LEGACY_SCHEMA_LOGICAL_KEY,
    )
    ambiguous_replicate_count = _ambiguous_duplicate_row_count(
        df,
        LEGACY_SCHEMA_LOGICAL_KEY,
    )
    if ambiguous_replicate_count:
        warnings.append(
            "Duplicate measurement groups cannot be fully resolved because "
            "Replicate_ID, Plate_ID, or Well_ID is missing in at least one row."
        )

    invalid_numeric_counts = _invalid_numeric_counts(df)
    for label, count in invalid_numeric_counts.items():
        if count:
            if label == "negative_luminescence":
                warnings.append(
                    f"Found {count} negative Luminescence_Raw values; values were retained."
                )
            else:
                errors.append(f"Found {count} rows with {label.replace('_', ' ')}.")

    negative_luminescence_count = invalid_numeric_counts["negative_luminescence"]
    infinite_luminescence_count = invalid_numeric_counts["infinite_luminescence"]

    non_monotonic_count, non_monotonic_table = _non_monotonic_time_groups(df)
    if non_monotonic_count:
        errors.append(
            f"Found {non_monotonic_count} measurement series with non-monotonic time."
        )

    duplicate_timepoint_group_count, duplicate_timepoint_table = _duplicate_timepoints(df)
    if duplicate_timepoint_group_count:
        warnings.append(
            f"Found {duplicate_timepoint_group_count} duplicate time-point groups."
        )

    if missing_identifier_counts.get("Plate_ID", 0):
        warnings.append("Plate_ID is missing for one or more rows.")
    if missing_identifier_counts.get("Well_ID", 0):
        warnings.append("Well_ID is missing for one or more rows.")
    if missing_identifier_counts.get("Replicate_ID", 0):
        warnings.append("Replicate_ID is missing for one or more rows.")

    missing_values_table = _missing_values_table(df)
    source_file_summary = _source_file_summary(df)
    time_series_issues = _combine_time_issue_tables(
        non_monotonic_table,
        duplicate_timepoint_table,
    )

    error_like_numeric_keys = [
        "negative_concentration",
        "negative_time",
        "time_hours_minutes_mismatch",
        "infinite_luminescence",
    ]
    numeric_errors = any(invalid_numeric_counts[key] for key in error_like_numeric_keys)
    qc_passed = not errors and not numeric_errors and row_count > 0

    return CanonicalQCResult(
        row_count=row_count,
        measurement_unit_count=measurement_unit_count,
        synthetic_measurement_unit_count=synthetic_measurement_unit_count,
        unresolved_measurement_unit_count=unresolved_measurement_unit_count,
        exact_duplicate_count=exact_duplicate_count,
        source_row_id_duplicate_count=source_row_id_duplicate_count,
        logical_duplicate_count=logical_duplicate_count,
        duplicate_group_count=len(logical_groups),
        legacy_logical_duplicate_count=legacy_logical_duplicate_count,
        legacy_duplicate_group_count=len(legacy_logical_groups),
        source_aware_logical_duplicate_count=source_aware_logical_duplicate_count,
        source_aware_duplicate_group_count=len(source_aware_groups),
        identical_value_duplicate_count=identical_value_duplicate_count,
        conflicting_value_duplicate_count=conflicting_value_duplicate_count,
        ambiguous_measurement_identity_count=ambiguous_measurement_identity_count,
        ambiguous_measurement_identity_group_count=ambiguous_measurement_identity_group_count,
        separate_replicate_measurement_count=separate_replicate_measurement_count,
        separate_replicate_measurement_group_count=separate_replicate_measurement_group_count,
        ambiguous_replicate_count=ambiguous_replicate_count,
        ambiguous_replicate_group_count=ambiguous_replicate_group_count,
        missing_required_value_counts=missing_required,
        missing_identifier_counts=missing_identifier_counts,
        invalid_numeric_counts=invalid_numeric_counts,
        negative_luminescence_count=negative_luminescence_count,
        infinite_luminescence_count=infinite_luminescence_count,
        non_monotonic_time_group_count=non_monotonic_count,
        duplicate_timepoint_group_count=duplicate_timepoint_group_count,
        time_series_issue_count=non_monotonic_count + duplicate_timepoint_group_count,
        strains_detected=strains,
        chemicals_detected=chemicals,
        concentrations_detected=concentrations,
        source_files=source_files,
        warnings=warnings,
        errors=errors,
        summary_tables={
            "logical_duplicate_groups": logical_groups,
            "legacy_logical_duplicate_groups": legacy_logical_groups,
            "source_aware_duplicate_groups": source_aware_groups,
            "duplicate_value_groups": duplicate_value_groups,
            "conflicting_duplicate_rows": conflicting_duplicate_rows,
            "ambiguous_measurement_identities": ambiguous_measurement_identities,
            "separate_replicate_measurements": separate_replicate_measurements,
            "source_row_id_duplicate_groups": source_row_groups,
            "missing_values": missing_values_table,
            "source_file_summary": source_file_summary,
            "time_series_issues": time_series_issues,
        },
        qc_passed=qc_passed,
    )


def write_qc_outputs(
    result: CanonicalQCResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write optional QC audit artifacts to a new output directory."""

    target = Path(output_dir)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {target}. "
                "Use --overwrite to replace QC outputs."
            )
        if target.is_file():
            raise FileExistsError(f"Output path is a file: {target}")
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []

    summary_path = target / "qc_summary.json"
    summary_path.write_text(
        json.dumps(result.to_summary_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    created.append(summary_path)

    table_map = {
        "logical_duplicate_groups": "logical_duplicate_groups.csv",
        "legacy_logical_duplicate_groups": "legacy_logical_duplicate_groups.csv",
        "source_aware_duplicate_groups": "source_aware_duplicate_groups.csv",
        "duplicate_value_groups": "duplicate_value_groups.csv",
        "conflicting_duplicate_rows": "conflicting_duplicate_rows.csv",
        "ambiguous_measurement_identities": "ambiguous_measurement_identities.csv",
        "separate_replicate_measurements": "separate_replicate_measurements.csv",
        "source_row_id_duplicate_groups": "source_row_id_duplicate_groups.csv",
        "missing_values": "missing_values.csv",
        "source_file_summary": "source_file_summary.csv",
        "time_series_issues": "time_series_issues.csv",
    }
    for table_name, filename in table_map.items():
        table = result.summary_tables.get(table_name, pd.DataFrame())
        path = target / filename
        table.to_csv(path, index=False)
        created.append(path)

    report_path = target / "canonical_qc_report.md"
    report_path.write_text(_render_markdown_report(result), encoding="utf-8")
    created.append(report_path)

    return created


def _coerce_without_mutation(dataframe: pd.DataFrame) -> pd.DataFrame:
    if all(col in dataframe.columns for col in CANONICAL_COLUMNS):
        return coerce_canonical_dtypes(dataframe.copy(deep=True))
    return dataframe.copy(deep=True)


def _missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        mask = mask | series.astype("string").str.strip().eq("")
    return mask.fillna(False)


def _missing_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(_missing_mask(df[column]).sum())


def _unique_strings(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    values = df.loc[~_missing_mask(df[column]), column].astype(str).unique().tolist()
    return sorted(values)


def _unique_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df.loc[~_missing_mask(df[column]), column].astype("string").nunique())


def _missing_required_counts(df: pd.DataFrame) -> tuple[dict[str, int], list[str], list[str]]:
    counts = {
        col: _missing_count(df, col)
        for col in REQUIRED_FIELDS
        if col in df.columns and _missing_count(df, col) > 0
    }
    if not counts:
        return counts, [], []

    errors: list[str] = []
    warnings: list[str] = []
    valid_scope = pd.Series(True, index=df.index)
    if "Record_Valid" in df.columns:
        valid_scope = df["Record_Valid"].astype("boolean").ne(False).fillna(True)

    blocking_counts: dict[str, int] = {}
    retained_invalid_counts: dict[str, int] = {}
    for column, count in counts.items():
        if column in {"QC_Status", "Record_Valid"}:
            blocking_counts[column] = count
            continue
        missing_valid = int((_missing_mask(df[column]) & valid_scope).sum())
        missing_invalid = count - missing_valid
        if missing_valid:
            blocking_counts[column] = missing_valid
        if missing_invalid:
            retained_invalid_counts[column] = missing_invalid

    if blocking_counts:
        formatted = ", ".join(f"{column}={count}" for column, count in sorted(blocking_counts.items()))
        errors.append(f"Required canonical fields are missing in valid records: {formatted}.")
    if retained_invalid_counts:
        formatted = ", ".join(
            f"{column}={count}" for column, count in sorted(retained_invalid_counts.items())
        )
        warnings.append(f"Required canonical fields are missing only in invalid retained records: {formatted}.")

    return counts, errors, warnings


def _flagged_unit_count(df: pd.DataFrame, flag: str) -> int:
    if "Measurement_Unit_ID" not in df.columns or "QC_Flags" not in df.columns:
        return 0
    mask = _qc_flag_mask(df, flag) & ~_missing_mask(df["Measurement_Unit_ID"])
    return int(df.loc[mask, "Measurement_Unit_ID"].astype("string").nunique())


def _qc_flag_mask(df: pd.DataFrame, flag: str) -> pd.Series:
    if "QC_Flags" not in df.columns:
        return pd.Series(False, index=df.index)

    flags = df["QC_Flags"].astype("string")
    escaped = flags.fillna("").str.split(";")
    return escaped.map(
        lambda values: any(str(value).strip() == flag for value in values)
    ).astype(bool)


def _available_key(df: pd.DataFrame, key: list[str]) -> list[str]:
    return [col for col in key if col in df.columns]


def _usable_key_mask(df: pd.DataFrame, key: list[str], include_missing: bool) -> pd.Series:
    if include_missing:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for col in key:
        mask = mask & ~_missing_mask(df[col])
    return mask


def _duplicate_groups(
    df: pd.DataFrame,
    key: list[str],
    *,
    include_missing: bool,
) -> tuple[int, pd.DataFrame]:
    available = _available_key(df, key)
    if len(available) != len(key) or not available or df.empty:
        return 0, pd.DataFrame(columns=available + ["row_count"])

    eligible = df.loc[_usable_key_mask(df, available, include_missing), available]
    if eligible.empty:
        return 0, pd.DataFrame(columns=available + ["row_count"])

    counts = (
        eligible.groupby(available, dropna=False, sort=True)
        .size()
        .reset_index(name="row_count")
    )
    duplicate_groups = counts.loc[counts["row_count"] > 1].reset_index(drop=True)
    duplicate_count = int(duplicate_groups["row_count"].sum()) if not duplicate_groups.empty else 0
    return duplicate_count, duplicate_groups


def _duplicate_mask(
    df: pd.DataFrame,
    key: list[str],
    *,
    include_missing: bool,
) -> pd.Series:
    available = _available_key(df, key)
    if len(available) != len(key) or not available or df.empty:
        return pd.Series(False, index=df.index)

    eligible_mask = _usable_key_mask(df, available, include_missing)
    result = pd.Series(False, index=df.index)
    result.loc[eligible_mask] = df.loc[eligible_mask, available].duplicated(keep=False)
    return result


def _separate_measurements_resolved_by_unit(df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    legacy_duplicate_mask = _duplicate_mask(
        df,
        LEGACY_SCHEMA_LOGICAL_KEY,
        include_missing=True,
    )
    corrected_duplicate_mask = _duplicate_mask(
        df,
        MEASUREMENT_KEY,
        include_missing=False,
    )
    separate_mask = legacy_duplicate_mask & ~corrected_duplicate_mask
    if "Measurement_Unit_ID" in df.columns:
        separate_mask = separate_mask & ~_missing_mask(df["Measurement_Unit_ID"])

    if not separate_mask.any():
        return 0, 0, pd.DataFrame(columns=[*LEGACY_SCHEMA_LOGICAL_KEY, "row_count"])

    rows: list[dict[str, Any]] = []
    grouped = df.loc[separate_mask].groupby(
        _available_key(df, LEGACY_SCHEMA_LOGICAL_KEY),
        dropna=False,
        sort=True,
    )
    for group_key, group in grouped:
        row = _key_tuple_to_dict(_available_key(df, LEGACY_SCHEMA_LOGICAL_KEY), group_key)
        row.update(
            {
                "row_count": int(len(group)),
                "measurement_unit_count": _safe_nunique(group, "Measurement_Unit_ID"),
            }
        )
        rows.append(row)

    return int(separate_mask.sum()), len(rows), pd.DataFrame(rows)


def _ambiguous_measurement_identities(df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    if df.empty:
        return 0, 0, pd.DataFrame(columns=["row_count"])

    mask = pd.Series(False, index=df.index)
    if "Measurement_Unit_ID" in df.columns:
        mask = mask | _missing_mask(df["Measurement_Unit_ID"])
    mask = mask | _qc_flag_mask(df, "measurement_unit_identity_ambiguous")

    if {"Replicate_ID", "Plate_ID", "Well_ID"} <= set(df.columns):
        no_physical_identifier = _missing_mask(df["Plate_ID"]) & _missing_mask(df["Well_ID"])
        mask = mask | (_missing_mask(df["Replicate_ID"]) & no_physical_identifier)

    if not mask.any():
        return 0, 0, pd.DataFrame(columns=["row_count"])

    group_columns = [
        column
        for column in (
            "Source_File",
            "Experiment_ID",
            "Measurement_Unit_ID",
            "Strain_Original",
            "Chemical_Name_Original",
            "Concentration_Label",
            "Replicate_ID",
        )
        if column in df.columns
    ]
    if not group_columns:
        return int(mask.sum()), 1, pd.DataFrame({"row_count": [int(mask.sum())]})

    groups = (
        df.loc[mask, group_columns]
        .groupby(group_columns, dropna=False, sort=True)
        .size()
        .reset_index(name="row_count")
    )
    return int(mask.sum()), int(len(groups)), groups


def _exact_duplicate_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    comparable_columns = [col for col in CANONICAL_COLUMNS if col in df.columns]
    if not comparable_columns:
        return 0
    return int(df.loc[:, comparable_columns].duplicated(keep=False).sum())


def _safe_nunique(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df.loc[~_missing_mask(df[column]), column].astype("string").nunique())


def _classify_duplicate_values(
    df: pd.DataFrame,
    key: list[str],
) -> tuple[int, int, pd.DataFrame, pd.DataFrame]:
    available = _available_key(df, key)
    if len(available) != len(key) or "Luminescence_Raw" not in df.columns or df.empty:
        empty_groups = pd.DataFrame(columns=available + ["row_count", "value_status"])
        return 0, 0, empty_groups, df.iloc[0:0].copy()

    duplicate_rows = df.loc[_duplicate_mask(df, available, include_missing=False)]
    if duplicate_rows.empty:
        empty_groups = pd.DataFrame(columns=available + ["row_count", "value_status"])
        return 0, 0, empty_groups, duplicate_rows.copy()

    group_rows: list[dict[str, Any]] = []
    identical_count = 0
    conflicting_count = 0
    conflicting_indices: list[Any] = []

    for group_key, group in duplicate_rows.groupby(available, dropna=False, sort=True):
        values = group["Luminescence_Raw"].astype("Float64")
        value_count = int(values.nunique(dropna=False))
        status = "identical" if value_count <= 1 else "conflicting"
        row_count = int(len(group))
        if status == "identical":
            identical_count += row_count
        else:
            conflicting_count += row_count
            conflicting_indices.extend(group.index.tolist())

        row = _key_tuple_to_dict(available, group_key)
        row.update(
            {
                "row_count": row_count,
                "value_status": status,
                "distinct_luminescence_values": value_count,
                "min_luminescence": _safe_float(values.min(skipna=True)),
                "max_luminescence": _safe_float(values.max(skipna=True)),
            }
        )
        group_rows.append(row)

    duplicate_value_groups = pd.DataFrame(group_rows)
    conflicting_rows = df.loc[conflicting_indices].copy() if conflicting_indices else df.iloc[0:0].copy()
    return identical_count, conflicting_count, duplicate_value_groups, conflicting_rows


def _ambiguous_duplicate_group_count(df: pd.DataFrame, key: list[str]) -> int:
    return len(_ambiguous_duplicate_groups(df, key))


def _ambiguous_duplicate_row_count(df: pd.DataFrame, key: list[str]) -> int:
    groups = _ambiguous_duplicate_groups(df, key)
    if not groups:
        return 0
    return int(sum(len(group) for group in groups))


def _ambiguous_duplicate_groups(df: pd.DataFrame, key: list[str]) -> list[pd.DataFrame]:
    available = _available_key(df, key)
    identifier_columns = [col for col in ["Replicate_ID", "Plate_ID", "Well_ID"] if col in df.columns]
    if len(available) != len(key) or not identifier_columns or df.empty:
        return []

    duplicate_rows = df.loc[df[available].duplicated(keep=False)]
    groups: list[pd.DataFrame] = []
    for _, group in duplicate_rows.groupby(available, dropna=False, sort=True):
        if any(_missing_mask(group[col]).any() for col in identifier_columns):
            groups.append(group)
    return groups


def _invalid_numeric_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {
        "negative_concentration": 0,
        "negative_time": 0,
        "time_hours_minutes_mismatch": 0,
        "negative_luminescence": 0,
        "infinite_luminescence": 0,
    }

    if "Concentration_ug_mL" in df.columns:
        concentration = pd.to_numeric(df["Concentration_ug_mL"], errors="coerce")
        counts["negative_concentration"] = int((concentration < 0).sum())

    if "Time_Minutes" in df.columns:
        time_minutes = pd.to_numeric(df["Time_Minutes"], errors="coerce")
        counts["negative_time"] = int((time_minutes < 0).sum())
    else:
        time_minutes = pd.Series(dtype="Float64")

    if "Time_Hours" in df.columns and "Time_Minutes" in df.columns:
        time_hours = pd.to_numeric(df["Time_Hours"], errors="coerce")
        mismatch = (
            time_hours.notna()
            & time_minutes.notna()
            & ~np.isclose(
                time_hours.to_numpy(dtype=float, na_value=np.nan) * 60.0,
                time_minutes.to_numpy(dtype=float, na_value=np.nan),
            )
        )
        counts["time_hours_minutes_mismatch"] = int(mismatch.sum())

    if "Luminescence_Raw" in df.columns:
        luminescence = pd.to_numeric(df["Luminescence_Raw"], errors="coerce")
        counts["negative_luminescence"] = int((luminescence < 0).sum())
        counts["infinite_luminescence"] = int(
            np.isinf(luminescence.to_numpy(dtype=float, na_value=np.nan)).sum()
        )

    return counts


def _series_key(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in SERIES_GROUPING_KEY
        if col in df.columns and _missing_count(df, col) < len(df)
    ]


def _non_monotonic_time_groups(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    if df.empty or "Time_Minutes" not in df.columns:
        return 0, pd.DataFrame(columns=["issue"])

    key = _series_key(df)
    if not key:
        return 0, pd.DataFrame(columns=["issue"])

    rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby(key, dropna=False, sort=True):
        ordered = group
        if "Source_Row_ID" in ordered.columns:
            ordered = ordered.sort_values("Source_Row_ID", kind="mergesort")
        times = pd.to_numeric(ordered["Time_Minutes"], errors="coerce").dropna()
        if len(times) > 1 and bool((times.diff().dropna() < 0).any()):
            row = _key_tuple_to_dict(key, group_key)
            row.update(
                {
                    "issue": "non_monotonic_time",
                    "row_count": int(len(group)),
                    "min_time_minutes": _safe_float(times.min()),
                    "max_time_minutes": _safe_float(times.max()),
                }
            )
            rows.append(row)

    return len(rows), pd.DataFrame(rows)


def _duplicate_timepoints(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    if df.empty or "Time_Minutes" not in df.columns:
        return 0, pd.DataFrame(columns=["issue"])

    key = _series_key(df) + ["Time_Minutes"]
    duplicate_count, groups = _duplicate_groups(df, key, include_missing=True)
    if duplicate_count == 0:
        return 0, pd.DataFrame(columns=key + ["row_count", "issue"])

    groups = groups.copy()
    groups["issue"] = "duplicate_timepoint"
    return len(groups), groups


def _missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"column": col, "missing_count": _missing_count(df, col)}
        for col in df.columns
        if _missing_count(df, col) > 0
    ]
    return pd.DataFrame(rows, columns=["column", "missing_count"])


def _source_file_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Source_File" not in df.columns or df.empty:
        return pd.DataFrame(columns=["Source_File", "row_count"])
    return (
        df.groupby("Source_File", dropna=False, sort=True)
        .size()
        .reset_index(name="row_count")
    )


def _combine_time_issue_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    populated = [table for table in tables if not table.empty]
    if not populated:
        return pd.DataFrame(columns=["issue"])
    return pd.concat(populated, ignore_index=True, sort=False)


def _key_tuple_to_dict(key: list[str], group_key: Any) -> dict[str, Any]:
    if len(key) == 1:
        values = (group_key,)
    else:
        values = group_key
    return dict(zip(key, values))


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _render_markdown_report(result: CanonicalQCResult) -> str:
    summary = result.to_summary_dict()
    lines = [
        "# Stage 5B Canonical Dataset QC Audit",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        if isinstance(value, (list, dict)):
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Missing Identifiers"])
    for key, value in result.missing_identifier_counts.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Warnings"])
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- None")

    lines.extend(["", "## Errors"])
    if result.errors:
        lines.extend(f"- {error}" for error in result.errors)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Duplicate Key Note",
            "The corrected canonical measurement key is:",
            "",
            "```text",
            ", ".join(MEASUREMENT_KEY),
            "```",
            "",
            "The time-series grouping key is:",
            "",
            "```text",
            ", ".join(SERIES_GROUPING_KEY),
            "```",
            "",
            "The legacy Stage 5A duplicate key is retained as a comparison metric only.",
        ]
    )
    return "\n".join(lines) + "\n"
