"""Canonical long-format schema for biosensor measurement rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


CANONICAL_SCHEMA_VERSION = "1.1.0"
TIME_CONSISTENCY_TOLERANCE = 1e-6

SOURCE_TYPE_VALUES: tuple[str, ...] = ("csv", "xlsx")
DATA_SOURCE_VALUES: tuple[str, ...] = ("24_hour_csv", "12_hour_excel", "unknown")
CONTROL_STATUS_VALUES: tuple[str, ...] = ("treatment", "control", "unknown")
CONTROL_TYPE_VALUES: tuple[str, ...] = (
    "untreated",
    "solvent",
    "blank",
    "zero_concentration",
    "shared_control",
    "unknown",
)
REPLICATE_TYPE_VALUES: tuple[str, ...] = ("technical", "biological", "unspecified")
QC_STATUS_VALUES: tuple[str, ...] = ("pass", "warning", "fail", "not_evaluated")
ANALYSIS_WINDOW_VALUES: tuple[str, ...] = (
    "unassigned",
    "0-12h_Common",
    "0-12h_Early",
    "12-24h_Late",
    "0-24h_Full",
    "0-12h_Full",
)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "Experiment_ID",
    "Plate_ID",
    "Source_File",
    "Source_Path",
    "Source_Type",
    "Worksheet",
    "Data_Source",
    "Time_Series_Duration_Hours",
    "Analysis_Window",
    "Import_Timestamp",
    "Source_Row_ID",
    "Measurement_Unit_ID",
    "Strain_Original",
    "Strain_Standardized",
    "Chemical_Name_Original",
    "Chemical_Name_Standardized",
    "Concentration_Label",
    "Concentration_ug_mL",
    "Control_Status",
    "Control_Type",
    "Replicate_ID",
    "Replicate_Type",
    "Well_ID",
    "Time_Original",
    "Time_Unit_Original",
    "Time_Minutes",
    "Time_Hours",
    "Timepoint_Index",
    "Luminescence_Raw",
    "Luminescence_Normalized",
    "Normalization_Method",
    "QC_Status",
    "QC_Flags",
    "Record_Valid",
    "Notes",
)

MEASUREMENT_KEY_COLUMNS: tuple[str, ...] = (
    "Experiment_ID",
    "Source_File",
    "Measurement_Unit_ID",
    "Time_Minutes",
)

SERIES_GROUPING_KEY_COLUMNS: tuple[str, ...] = (
    "Experiment_ID",
    "Source_File",
    "Measurement_Unit_ID",
)

REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "Experiment_ID",
        "Source_File",
        "Source_Type",
        "Strain_Original",
        "Chemical_Name_Original",
        "Concentration_Label",
        "Measurement_Unit_ID",
        "Time_Minutes",
        "Time_Hours",
        "Luminescence_Raw",
        "QC_Status",
        "Record_Valid",
    }
)

OPTIONAL_NULLABLE_FIELDS: frozenset[str] = frozenset(
    set(CANONICAL_COLUMNS) - set(REQUIRED_FIELDS)
)

CANONICAL_DTYPES: dict[str, str] = {
    "Experiment_ID": "string",
    "Plate_ID": "string",
    "Source_File": "string",
    "Source_Path": "string",
    "Source_Type": "string",
    "Worksheet": "string",
    "Data_Source": "string",
    "Time_Series_Duration_Hours": "Float64",
    "Analysis_Window": "string",
    "Import_Timestamp": "datetime64[ns, UTC]",
    "Source_Row_ID": "Int64",
    "Measurement_Unit_ID": "string",
    "Strain_Original": "string",
    "Strain_Standardized": "string",
    "Chemical_Name_Original": "string",
    "Chemical_Name_Standardized": "string",
    "Concentration_Label": "string",
    "Concentration_ug_mL": "Float64",
    "Control_Status": "string",
    "Control_Type": "string",
    "Replicate_ID": "string",
    "Replicate_Type": "string",
    "Well_ID": "string",
    "Time_Original": "string",
    "Time_Unit_Original": "string",
    "Time_Minutes": "Float64",
    "Time_Hours": "Float64",
    "Timepoint_Index": "Int64",
    "Luminescence_Raw": "Float64",
    "Luminescence_Normalized": "Float64",
    "Normalization_Method": "string",
    "QC_Status": "string",
    "QC_Flags": "string",
    "Record_Valid": "boolean",
    "Notes": "string",
}

CONTROLLED_VALUES: dict[str, tuple[Any, ...]] = {
    "Source_Type": SOURCE_TYPE_VALUES,
    "Data_Source": DATA_SOURCE_VALUES,
    "Control_Status": CONTROL_STATUS_VALUES,
    "Control_Type": CONTROL_TYPE_VALUES,
    "Replicate_Type": REPLICATE_TYPE_VALUES,
    "QC_Status": QC_STATUS_VALUES,
    "Analysis_Window": ANALYSIS_WINDOW_VALUES,
    "Record_Valid": (True, False),
}


@dataclass(frozen=True)
class CanonicalFieldDefinition:
    """Formal field-level schema metadata."""

    name: str
    description: str
    required: bool
    dtype: str
    allowed_missingness: str
    allowed_values: tuple[Any, ...] | None
    scientific_meaning: str
    validation_rules: tuple[str, ...]
    example_value: Any


@dataclass(frozen=True)
class CanonicalSchemaValidationResult:
    """Structured result for non-mutating canonical schema validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    missing_columns: list[str]
    unexpected_columns: list[str]
    invalid_values: dict[str, list[Any]]
    row_problem_counts: dict[str, int]


FIELD_DEFINITIONS: tuple[CanonicalFieldDefinition, ...] = (
    CanonicalFieldDefinition(
        "Experiment_ID",
        "Identifier for the source experiment or run.",
        True,
        CANONICAL_DTYPES["Experiment_ID"],
        "No missing values allowed for canonical measurement rows.",
        None,
        "Separates independent experimental runs before replicate or time analysis.",
        ("Must be present.", "Must not be missing."),
        "EXP-001",
    ),
    CanonicalFieldDefinition(
        "Plate_ID",
        "Optional identifier for a physical plate.",
        False,
        CANONICAL_DTYPES["Plate_ID"],
        "May be null when the source does not provide plate identifiers.",
        None,
        "Supports tracing measurements to a plate when plate metadata are available.",
        ("Do not invent missing plate identifiers.",),
        "Plate-01",
    ),
    CanonicalFieldDefinition(
        "Source_File",
        "Original filename that supplied the row.",
        True,
        CANONICAL_DTYPES["Source_File"],
        "No missing values allowed.",
        None,
        "Provides provenance back to the raw CSV or workbook.",
        ("Must be present.", "Must preserve the original filename."),
        "BL027ab.csv",
    ),
    CanonicalFieldDefinition(
        "Source_Path",
        "Absolute or recorded path to the raw source file.",
        False,
        CANONICAL_DTYPES["Source_Path"],
        "May be null in portable/exported datasets.",
        None,
        "Supports auditability without requiring path stability.",
        ("Should not be used as the only biological identifier.",),
        r"C:\\data\\BL011.csv",
    ),
    CanonicalFieldDefinition(
        "Source_Type",
        "Raw file extension family.",
        True,
        CANONICAL_DTYPES["Source_Type"],
        "No missing values allowed.",
        SOURCE_TYPE_VALUES,
        "Distinguishes CSV-derived rows from Excel-derived rows.",
        ("Must be one of the controlled Source_Type values.",),
        "csv",
    ),
    CanonicalFieldDefinition(
        "Worksheet",
        "Original worksheet name for Excel-derived rows.",
        False,
        CANONICAL_DTYPES["Worksheet"],
        "May be null for CSV-derived rows.",
        None,
        "Preserves workbook sheet provenance.",
        ("Must preserve original worksheet names when available.",),
        "Sheet1",
    ),
    CanonicalFieldDefinition(
        "Data_Source",
        "Dataset-duration/source bucket assigned by importer metadata.",
        False,
        CANONICAL_DTYPES["Data_Source"],
        "May be unknown when source duration has not been established.",
        DATA_SOURCE_VALUES,
        "Separates 24-hour CSV, 12-hour Excel, and unknown source families.",
        ("Must be controlled when provided.",),
        "24_hour_csv",
    ),
    CanonicalFieldDefinition(
        "Time_Series_Duration_Hours",
        "Measured duration represented by the source series.",
        False,
        CANONICAL_DTYPES["Time_Series_Duration_Hours"],
        "May be null until derived from measured data.",
        None,
        "Describes the observed duration without extrapolating time points.",
        ("Must be non-negative when present.", "Should be derived from measured data."),
        24.0,
    ),
    CanonicalFieldDefinition(
        "Analysis_Window",
        "Analysis window assigned to the row.",
        False,
        CANONICAL_DTYPES["Analysis_Window"],
        "May be null before window assignment, though importers should prefer unassigned.",
        ANALYSIS_WINDOW_VALUES,
        "Documents whether the row belongs to a common, early, late, or full time window.",
        ("Must be controlled when provided.", "Should initially be unassigned during import."),
        "unassigned",
    ),
    CanonicalFieldDefinition(
        "Import_Timestamp",
        "UTC timestamp when the canonical row was imported.",
        False,
        CANONICAL_DTYPES["Import_Timestamp"],
        "May be null in synthetic tests or pre-import drafts.",
        None,
        "Provides reproducibility and audit context.",
        ("Must be timezone-aware UTC when present.",),
        "2026-07-25T12:00:00Z",
    ),
    CanonicalFieldDefinition(
        "Source_Row_ID",
        "Row number or row-like identifier from the source table.",
        False,
        CANONICAL_DTYPES["Source_Row_ID"],
        "May be null when source-row provenance is unavailable.",
        None,
        "Supports fallback identity when plate or well identifiers are missing.",
        ("Must be non-negative when present.",),
        42,
    ),
    CanonicalFieldDefinition(
        "Measurement_Unit_ID",
        "Deterministic identifier for one experimental unit measured over time.",
        True,
        CANONICAL_DTYPES["Measurement_Unit_ID"],
        "No missing values allowed for valid measurement rows.",
        None,
        "Distinguishes wells, measurement channels, replicate blocks, or source-position-derived units before applying the time point.",
        (
            "Must be deterministic.",
            "Must not be based on luminescence, current date, or random UUIDs.",
            "Synthetic identifiers must be flagged in QC_Flags or Notes.",
        ),
        "unit_r0002377",
    ),
    CanonicalFieldDefinition(
        "Strain_Original",
        "Strain label exactly as represented by the source or filename.",
        True,
        CANONICAL_DTYPES["Strain_Original"],
        "No missing values allowed.",
        None,
        "Preserves source biological identity before approved standardization.",
        ("Must not be automatically standardized.", "BL027ab must remain BL027ab unless mapped later."),
        "BL027ab",
    ),
    CanonicalFieldDefinition(
        "Strain_Standardized",
        "Approved standardized strain label.",
        False,
        CANONICAL_DTYPES["Strain_Standardized"],
        "May be null until an explicit mapping is approved.",
        None,
        "Separates source labels from later canonical strain names.",
        ("Must not overwrite Strain_Original.",),
        "BL027",
    ),
    CanonicalFieldDefinition(
        "Chemical_Name_Original",
        "Chemical label exactly as written in the source data.",
        True,
        CANONICAL_DTYPES["Chemical_Name_Original"],
        "No missing values allowed.",
        None,
        "Preserves source treatment identity before approved standardization.",
        ("Must preserve labels such as Lambda Cyclotherin exactly.",),
        "Lambda Cyclotherin",
    ),
    CanonicalFieldDefinition(
        "Chemical_Name_Standardized",
        "Approved standardized chemical label.",
        False,
        CANONICAL_DTYPES["Chemical_Name_Standardized"],
        "May be null until an explicit mapping is approved.",
        None,
        "Stores verified mappings without overwriting original chemical names.",
        ("Must remain null or unchanged until a verified mapping is applied.",),
        pd.NA,
    ),
    CanonicalFieldDefinition(
        "Concentration_Label",
        "Concentration label exactly as written in the source data.",
        True,
        CANONICAL_DTYPES["Concentration_Label"],
        "No missing values allowed.",
        None,
        "Preserves source dose/control labels before numeric parsing.",
        ("Must not be silently rewritten.",),
        "Control",
    ),
    CanonicalFieldDefinition(
        "Concentration_ug_mL",
        "Parsed numeric concentration in micrograms per milliliter.",
        False,
        CANONICAL_DTYPES["Concentration_ug_mL"],
        "May be null for controls or unknown units.",
        None,
        "Numeric dose used for downstream dose-response analysis after units are known.",
        ("Negative values fail validation.", "Unknown units must be flagged before conversion."),
        5.0,
    ),
    CanonicalFieldDefinition(
        "Control_Status",
        "Whether the row is treatment, control, or unknown.",
        False,
        CANONICAL_DTYPES["Control_Status"],
        "May be unknown when control interpretation is unresolved.",
        CONTROL_STATUS_VALUES,
        "Separates treatment rows from control rows without relying only on concentration.",
        ("Must be controlled when provided.",),
        "treatment",
    ),
    CanonicalFieldDefinition(
        "Control_Type",
        "Specific type of control condition.",
        False,
        CANONICAL_DTYPES["Control_Type"],
        "May be null when not applicable or unavailable.",
        CONTROL_TYPE_VALUES,
        "Distinguishes untreated, solvent, blank, zero concentration, and shared controls.",
        ("Zero concentration must not automatically imply every control type.",),
        "shared_control",
    ),
    CanonicalFieldDefinition(
        "Replicate_ID",
        "Replicate label from source data.",
        False,
        CANONICAL_DTYPES["Replicate_ID"],
        "May be null when the source does not provide enough replicate evidence.",
        None,
        "Identifies repeated measurements within a condition, but is not globally unique and does not define measurement identity by itself.",
        (
            "Preserve explicit source labels.",
            "Generated values must come from clear source structure and be flagged.",
            "Do not label time points as replicates.",
        ),
        "1",
    ),
    CanonicalFieldDefinition(
        "Replicate_Type",
        "Replicate category.",
        False,
        CANONICAL_DTYPES["Replicate_Type"],
        "May be null or unspecified when unavailable.",
        REPLICATE_TYPE_VALUES,
        "Distinguishes technical and biological replication where known.",
        ("Must be controlled when provided.",),
        "unspecified",
    ),
    CanonicalFieldDefinition(
        "Well_ID",
        "Optional plate well identifier.",
        False,
        CANONICAL_DTYPES["Well_ID"],
        "May be null when not available in source files.",
        None,
        "Links a measurement to a physical well when available.",
        ("Do not invent missing well identifiers.",),
        "A01",
    ),
    CanonicalFieldDefinition(
        "Time_Original",
        "Original source time value before numeric conversion.",
        False,
        CANONICAL_DTYPES["Time_Original"],
        "May be null if source already provides clean numeric minutes.",
        None,
        "Preserves source time labels for audit and troubleshooting.",
        ("Must not be extrapolated.",),
        "120",
    ),
    CanonicalFieldDefinition(
        "Time_Unit_Original",
        "Original time unit from the source.",
        False,
        CANONICAL_DTYPES["Time_Unit_Original"],
        "May be null if units are implicit or unavailable.",
        None,
        "Records whether source time was minutes, hours, or another unit.",
        ("Unknown units should be flagged, not guessed.",),
        "min",
    ),
    CanonicalFieldDefinition(
        "Time_Minutes",
        "Measurement time in minutes.",
        True,
        CANONICAL_DTYPES["Time_Minutes"],
        "No missing values allowed.",
        None,
        "Primary time axis for long-format biosensor curves.",
        ("Must be numerically consistent with Time_Hours.", "Negative values fail validation."),
        120.0,
    ),
    CanonicalFieldDefinition(
        "Time_Hours",
        "Measurement time in hours.",
        True,
        CANONICAL_DTYPES["Time_Hours"],
        "No missing values allowed.",
        None,
        "Hour-scale time axis used for windowing and reporting.",
        ("Must be numerically consistent with Time_Minutes.", "Negative values fail validation."),
        2.0,
    ),
    CanonicalFieldDefinition(
        "Timepoint_Index",
        "Index of the time point within an independent curve.",
        False,
        CANONICAL_DTYPES["Timepoint_Index"],
        "May be null until assigned by an importer.",
        None,
        "Supports deterministic ordering and fallback record identity.",
        ("Should increase within each independent curve.",),
        24,
    ),
    CanonicalFieldDefinition(
        "Luminescence_Raw",
        "Raw luminescence measurement from the source file.",
        True,
        CANONICAL_DTYPES["Luminescence_Raw"],
        "No missing values allowed for valid measurement rows.",
        None,
        "Unmodified signal used as the basis for normalization and feature extraction.",
        ("Must be numeric or missing.", "Infinite values fail validation.", "Negative values are retained but warned."),
        12345.0,
    ),
    CanonicalFieldDefinition(
        "Luminescence_Normalized",
        "Normalized luminescence value produced by a later normalization step.",
        False,
        CANONICAL_DTYPES["Luminescence_Normalized"],
        "May be null until normalization is performed.",
        None,
        "Stores derived signal values without overwriting the raw measurement.",
        ("Must remain null until normalization.", "Must not overwrite Luminescence_Raw."),
        pd.NA,
    ),
    CanonicalFieldDefinition(
        "Normalization_Method",
        "Name of the normalization method used for Luminescence_Normalized.",
        False,
        CANONICAL_DTYPES["Normalization_Method"],
        "May be null when no normalization has been applied.",
        None,
        "Documents how normalized signal values were produced.",
        ("Required only when normalized values are generated by later stages.",),
        "baseline_ratio",
    ),
    CanonicalFieldDefinition(
        "QC_Status",
        "Quality-control status for the measurement row.",
        True,
        CANONICAL_DTYPES["QC_Status"],
        "No missing values allowed.",
        QC_STATUS_VALUES,
        "Records whether a row passed, warned, failed, or has not been evaluated.",
        ("Must be controlled.",),
        "not_evaluated",
    ),
    CanonicalFieldDefinition(
        "QC_Flags",
        "Machine-readable quality-control flags.",
        False,
        CANONICAL_DTYPES["QC_Flags"],
        "May be null when no flags are assigned.",
        None,
        "Retains non-fatal issues without deleting records.",
        ("Must not be used to silently remove rows.",),
        "negative_luminescence",
    ),
    CanonicalFieldDefinition(
        "Record_Valid",
        "Boolean validity flag for the measurement row.",
        True,
        CANONICAL_DTYPES["Record_Valid"],
        "No missing values allowed.",
        (True, False),
        "Indicates whether a row is eligible for downstream analysis.",
        ("Must be boolean.",),
        True,
    ),
    CanonicalFieldDefinition(
        "Notes",
        "Optional free-text notes.",
        False,
        CANONICAL_DTYPES["Notes"],
        "May be null.",
        None,
        "Human-readable context for unresolved import or QC issues.",
        ("Should not encode required machine-readable QC state.",),
        "Source label preserved pending verification.",
    ),
)

FIELD_DEFINITION_BY_NAME: dict[str, CanonicalFieldDefinition] = {
    definition.name: definition for definition in FIELD_DEFINITIONS
}


def create_empty_canonical_dataframe() -> pd.DataFrame:
    """Return an empty canonical DataFrame with deterministic columns and dtypes."""

    dataframe = pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in CANONICAL_DTYPES.items()})
    return dataframe.loc[:, list(CANONICAL_COLUMNS)]


def coerce_canonical_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with known canonical columns coerced to documented dtypes."""

    coerced = dataframe.copy(deep=True)
    for column in CANONICAL_COLUMNS:
        if column not in coerced.columns:
            continue
        dtype = CANONICAL_DTYPES[column]
        coerced[column] = _coerce_series(coerced[column], dtype)

    ordered_columns = [column for column in CANONICAL_COLUMNS if column in coerced.columns]
    unexpected_columns = [column for column in coerced.columns if column not in CANONICAL_COLUMNS]
    return coerced.loc[:, [*ordered_columns, *unexpected_columns]]


def validate_canonical_schema(dataframe: pd.DataFrame) -> CanonicalSchemaValidationResult:
    """Validate the canonical schema without mutating input data or deleting rows."""

    errors: list[str] = []
    warnings: list[str] = []
    invalid_values: dict[str, list[Any]] = {}
    row_problem_counts: dict[str, int] = {}

    missing_columns = [column for column in CANONICAL_COLUMNS if column not in dataframe.columns]
    unexpected_columns = [column for column in dataframe.columns if column not in CANONICAL_COLUMNS]

    if missing_columns:
        errors.append(f"Missing canonical columns: {', '.join(missing_columns)}")
    if unexpected_columns:
        errors.append(f"Unexpected canonical columns: {', '.join(map(str, unexpected_columns))}")

    working = coerce_canonical_dtypes(dataframe)
    _validate_required_values(working, errors, warnings, row_problem_counts)
    _validate_controlled_values(working, errors, invalid_values, row_problem_counts)
    _validate_numeric_rules(working, errors, warnings, row_problem_counts)
    _validate_time_rules(working, errors, warnings, row_problem_counts)
    _validate_logical_records(working, warnings, row_problem_counts)
    _validate_name_preservation(working, warnings, row_problem_counts)

    valid = not errors
    return CanonicalSchemaValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        invalid_values=invalid_values,
        row_problem_counts=row_problem_counts,
    )


def _coerce_series(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "datetime64[ns, UTC]":
        return pd.to_datetime(series, errors="coerce", utc=True)
    if dtype == "boolean":
        if is_bool_dtype(series):
            return series.astype("boolean")
        return series.map(_coerce_bool_value).astype("boolean")
    if dtype in {"Float64", "Int64"}:
        return pd.to_numeric(series, errors="coerce").astype(dtype)
    if dtype == "string":
        return series.astype("string")
    return series.astype(dtype)


def _coerce_bool_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return pd.NA


def _validate_required_values(
    dataframe: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    valid_scope = pd.Series(True, index=dataframe.index)
    if "Record_Valid" in dataframe.columns:
        record_valid = _coerce_series(dataframe["Record_Valid"], "boolean")
        valid_scope = record_valid.ne(False).fillna(True)

    for field in sorted(REQUIRED_FIELDS):
        if field not in dataframe.columns:
            continue
        missing_count = int(dataframe[field].isna().sum())
        if not missing_count:
            continue

        _add_problem(row_problem_counts, f"missing_required_{field}", missing_count)
        if field in {"QC_Status", "Record_Valid"}:
            errors.append(f"Required field {field} has {missing_count} missing values.")
            continue

        missing_valid_count = int((dataframe[field].isna() & valid_scope).sum())
        missing_invalid_count = missing_count - missing_valid_count
        if missing_valid_count:
            errors.append(
                f"Required field {field} has {missing_valid_count} missing values in valid records."
            )
        if missing_invalid_count:
            warnings.append(
                f"Required field {field} is missing in {missing_invalid_count} invalid records."
            )


def _validate_controlled_values(
    dataframe: pd.DataFrame,
    errors: list[str],
    invalid_values: dict[str, list[Any]],
    row_problem_counts: dict[str, int],
) -> None:
    for field, allowed_values in CONTROLLED_VALUES.items():
        if field not in dataframe.columns:
            continue
        series = dataframe[field].dropna()
        if field == "Record_Valid":
            if not is_bool_dtype(dataframe[field]):
                invalid_mask = ~series.isin(list(allowed_values))
            else:
                invalid_mask = pd.Series(False, index=series.index)
        else:
            invalid_mask = ~series.isin(list(allowed_values))

        if not invalid_mask.any():
            continue
        values = _unique_values(series.loc[invalid_mask])
        invalid_values[field] = values
        _add_problem(row_problem_counts, f"invalid_{field}", int(invalid_mask.sum()))
        errors.append(
            f"Field {field} contains values outside the controlled vocabulary: {values}"
        )


def _validate_numeric_rules(
    dataframe: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    if "Concentration_ug_mL" in dataframe.columns:
        concentration = dataframe["Concentration_ug_mL"].dropna()
        negative_count = int((concentration < 0).sum())
        if negative_count:
            _add_problem(row_problem_counts, "negative_concentration", negative_count)
            errors.append(f"Negative concentrations detected: {negative_count} rows.")

    if "Luminescence_Raw" in dataframe.columns:
        luminescence = dataframe["Luminescence_Raw"].dropna()
        infinite_count = _count_infinite(luminescence)
        if infinite_count:
            _add_problem(row_problem_counts, "infinite_luminescence_raw", infinite_count)
            errors.append(f"Infinite raw luminescence values detected: {infinite_count} rows.")
        negative_count = int((luminescence < 0).sum())
        if negative_count:
            _add_problem(row_problem_counts, "negative_luminescence_raw", negative_count)
            warnings.append(f"Negative raw luminescence values retained: {negative_count} rows.")

    if "Luminescence_Normalized" in dataframe.columns:
        normalized = dataframe["Luminescence_Normalized"].dropna()
        infinite_count = _count_infinite(normalized)
        if infinite_count:
            _add_problem(row_problem_counts, "infinite_luminescence_normalized", infinite_count)
            errors.append(
                f"Infinite normalized luminescence values detected: {infinite_count} rows."
            )


def _validate_time_rules(
    dataframe: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    for field in ("Time_Minutes", "Time_Hours", "Time_Series_Duration_Hours"):
        if field not in dataframe.columns:
            continue
        series = dataframe[field].dropna()
        negative_count = int((series < 0).sum())
        if negative_count:
            _add_problem(row_problem_counts, f"negative_{field}", negative_count)
            errors.append(f"Negative {field} values detected: {negative_count} rows.")

    if {"Time_Minutes", "Time_Hours"} <= set(dataframe.columns):
        comparable = dataframe[["Time_Minutes", "Time_Hours"]].dropna()
        inconsistent_mask = (
            (comparable["Time_Minutes"] / 60.0 - comparable["Time_Hours"]).abs()
            > TIME_CONSISTENCY_TOLERANCE
        )
        inconsistent_count = int(inconsistent_mask.sum())
        if inconsistent_count:
            _add_problem(row_problem_counts, "time_minutes_hours_inconsistent", inconsistent_count)
            errors.append(
                "Time_Minutes and Time_Hours are inconsistent beyond "
                f"{TIME_CONSISTENCY_TOLERANCE}: {inconsistent_count} rows."
            )

    if "Analysis_Window" in dataframe.columns:
        non_unassigned = dataframe["Analysis_Window"].dropna()
        if not non_unassigned.empty:
            count = int((non_unassigned != "unassigned").sum())
            if count:
                _add_problem(row_problem_counts, "analysis_window_already_assigned", count)
                warnings.append(
                    "Analysis_Window values other than 'unassigned' are present: "
                    f"{count} rows."
                )

    _warn_nonmonotonic_timepoint_index(dataframe, warnings, row_problem_counts)
    _warn_duplicate_time_points(dataframe, warnings, row_problem_counts)


def _validate_logical_records(
    dataframe: pd.DataFrame,
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    primary_key = list(MEASUREMENT_KEY_COLUMNS)
    fallback_key = ["Source_File", "Source_Row_ID", "Measurement_Unit_ID", "Timepoint_Index"]

    _warn_duplicate_key(
        dataframe,
        key_columns=primary_key,
        problem_name="duplicate_logical_records",
        warning_prefix="Duplicate logical measurement records detected",
        warnings=warnings,
        row_problem_counts=row_problem_counts,
    )
    _warn_duplicate_key(
        dataframe,
        key_columns=fallback_key,
        problem_name="duplicate_fallback_records",
        warning_prefix="Duplicate fallback measurement records detected",
        warnings=warnings,
        row_problem_counts=row_problem_counts,
    )


def _validate_name_preservation(
    dataframe: pd.DataFrame,
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    if {"Strain_Original", "Strain_Standardized"} <= set(dataframe.columns):
        changed = dataframe[["Strain_Original", "Strain_Standardized"]].dropna()
        changed_count = int((changed["Strain_Original"] != changed["Strain_Standardized"]).sum())
        if changed_count:
            _add_problem(row_problem_counts, "strain_standardized_differs_from_original", changed_count)
            warnings.append(
                "Strain_Standardized differs from Strain_Original in "
                f"{changed_count} rows; confirm mapping approval."
            )

    if {"Chemical_Name_Original", "Chemical_Name_Standardized"} <= set(dataframe.columns):
        changed = dataframe[["Chemical_Name_Original", "Chemical_Name_Standardized"]].dropna()
        changed_count = int(
            (changed["Chemical_Name_Original"] != changed["Chemical_Name_Standardized"]).sum()
        )
        if changed_count:
            _add_problem(
                row_problem_counts,
                "chemical_standardized_differs_from_original",
                changed_count,
            )
            warnings.append(
                "Chemical_Name_Standardized differs from Chemical_Name_Original in "
                f"{changed_count} rows; confirm mapping approval."
            )


def _warn_nonmonotonic_timepoint_index(
    dataframe: pd.DataFrame,
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    group_columns = [
        column
        for column in SERIES_GROUPING_KEY_COLUMNS
        if column in dataframe.columns
    ]
    if "Timepoint_Index" not in dataframe.columns or not group_columns:
        return

    problem_groups = 0
    for _, group in dataframe.dropna(subset=["Timepoint_Index"]).groupby(group_columns, dropna=False):
        if not group["Timepoint_Index"].is_monotonic_increasing:
            problem_groups += 1

    if problem_groups:
        _add_problem(row_problem_counts, "nonmonotonic_timepoint_index_groups", problem_groups)
        warnings.append(
            "Timepoint_Index is not increasing within "
            f"{problem_groups} independent curves."
        )


def _warn_duplicate_time_points(
    dataframe: pd.DataFrame,
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    key_columns = [
        column
        for column in (*SERIES_GROUPING_KEY_COLUMNS, "Time_Minutes")
        if column in dataframe.columns
    ]
    _warn_duplicate_key(
        dataframe,
        key_columns=key_columns,
        problem_name="duplicate_time_points",
        warning_prefix="Duplicate time points detected",
        warnings=warnings,
        row_problem_counts=row_problem_counts,
    )


def _warn_duplicate_key(
    dataframe: pd.DataFrame,
    key_columns: list[str],
    problem_name: str,
    warning_prefix: str,
    warnings: list[str],
    row_problem_counts: dict[str, int],
) -> None:
    if not key_columns or not set(key_columns) <= set(dataframe.columns):
        return

    key_dataframe = dataframe[key_columns]
    if key_dataframe.empty:
        return

    duplicated_mask = key_dataframe.duplicated(keep=False)
    duplicate_count = int(duplicated_mask.sum())
    if duplicate_count:
        _add_problem(row_problem_counts, problem_name, duplicate_count)
        warnings.append(f"{warning_prefix}: {duplicate_count} rows.")


def _count_infinite(series: pd.Series) -> int:
    if not is_numeric_dtype(series):
        return 0
    return sum(1 for value in series if isinstance(value, int | float) and math.isinf(float(value)))


def _unique_values(series: pd.Series) -> list[Any]:
    values = []
    for value in series.drop_duplicates().tolist():
        if pd.isna(value):
            continue
        values.append(value.item() if hasattr(value, "item") else value)
    return sorted(values, key=str)


def _add_problem(row_problem_counts: dict[str, int], key: str, count: int) -> None:
    row_problem_counts[key] = row_problem_counts.get(key, 0) + int(count)
