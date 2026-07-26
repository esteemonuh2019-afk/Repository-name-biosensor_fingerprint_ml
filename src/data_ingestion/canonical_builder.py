"""Build canonical biosensor measurement tables from reader results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data_ingestion.csv_reader import CsvReadResult
from src.data_ingestion.excel_reader import ExcelReadResult
from src.data_schema.canonical_schema import (
    CANONICAL_COLUMNS,
    CanonicalSchemaValidationResult,
    coerce_canonical_dtypes,
    validate_canonical_schema,
)


@dataclass(frozen=True)
class CanonicalBuildResult:
    """Canonical table plus build and schema-validation metadata."""

    dataframe: pd.DataFrame
    schema_valid: bool
    errors: list[str]
    warnings: list[str]
    source_files: list[str]
    row_count: int
    valid_record_count: int
    warning_record_count: int
    invalid_record_count: int
    schema_validation: CanonicalSchemaValidationResult


@dataclass(frozen=True)
class MeasurementUnitAssignment:
    """Per-row measurement-unit identity assignments and QC masks."""

    unit_ids: pd.Series
    synthetic_mask: pd.Series
    source_column_mask: pd.Series
    ambiguous_mask: pd.Series
    warning_mask: pd.Series


_SOURCE_ROW_ID_COLUMN = "__canonical_source_row_id"
_MEASUREMENT_VALUE_COLUMN = "__canonical_measurement_value"
_MEASUREMENT_COLUMN_INDEX_COLUMN = "__canonical_measurement_column_index"
_MEASUREMENT_COLUMN_LABEL_COLUMN = "__canonical_measurement_column_label"
_MULTIPLE_MEASUREMENT_COLUMNS_COLUMN = "__canonical_multiple_measurement_columns"

_PLATE_COLUMN_NAMES = ("Plate_ID", "plate_id", "plate", "Plate")
_WELL_COLUMN_NAMES = ("Well_ID", "well_id", "well", "Well")
_SAMPLE_COLUMN_NAMES = ("sample", "sample_id", "Sample", "Sample_ID")


def build_canonical_from_csv(
    read_result: CsvReadResult,
    experiment_id: str | None = None,
) -> CanonicalBuildResult:
    """Map one CSV reader result into the canonical biosensor schema."""

    dataframe, warnings = _map_reader_result(
        read_result=read_result,
        source_type="csv",
        data_source="24_hour_csv",
        source_file=read_result.source_file,
        absolute_path=read_result.absolute_path,
        worksheet=pd.NA,
        inferred_strain=read_result.strain_label_original,
        reader_warnings=read_result.warnings,
        experiment_id=experiment_id,
    )
    return _build_result(dataframe, warnings, [read_result.source_file])


def build_canonical_from_excel(
    read_result: ExcelReadResult,
    experiment_id: str | None = None,
) -> CanonicalBuildResult:
    """Map one Excel reader result into the canonical biosensor schema."""

    dataframe, warnings = _map_reader_result(
        read_result=read_result,
        source_type="xlsx",
        data_source="12_hour_excel",
        source_file=read_result.filename,
        absolute_path=read_result.absolute_path,
        worksheet=read_result.active_worksheet,
        inferred_strain=read_result.inferred_strain,
        reader_warnings=read_result.warnings,
        experiment_id=experiment_id,
    )
    return _build_result(dataframe, warnings, [read_result.filename])


def build_canonical_dataset(
    read_results: Iterable[CsvReadResult | ExcelReadResult],
    experiment_id: str | None = None,
) -> CanonicalBuildResult:
    """Map and combine CSV and Excel reader results into one canonical table."""

    dataframes: list[pd.DataFrame] = []
    warnings: list[str] = []
    source_files: list[str] = []

    for read_result in read_results:
        if isinstance(read_result, CsvReadResult):
            result = build_canonical_from_csv(read_result, experiment_id=experiment_id)
        elif isinstance(read_result, ExcelReadResult):
            result = build_canonical_from_excel(read_result, experiment_id=experiment_id)
        else:
            raise TypeError(f"Unsupported reader result type: {type(read_result)!r}")

        dataframes.append(result.dataframe)
        warnings.extend(result.warnings)
        source_files.extend(result.source_files)

    if dataframes:
        combined = pd.concat(dataframes, ignore_index=True)
        combined = coerce_canonical_dtypes(combined)
    else:
        combined = coerce_canonical_dtypes(pd.DataFrame(columns=list(CANONICAL_COLUMNS)))

    return _build_result(combined, warnings, source_files)


def save_canonical_dataset(
    build_result: CanonicalBuildResult,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    source_folder: str | Path | None = None,
) -> Path:
    """Save a canonical dataset CSV only when explicitly requested."""

    path = Path(output_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")

    if source_folder is not None:
        source_root = Path(source_folder).expanduser().resolve()
        resolved_parent = path.parent.resolve()
        if resolved_parent == source_root or source_root in resolved_parent.parents:
            raise ValueError("Refusing to save canonical output inside the raw source folder.")

    path.parent.mkdir(parents=True, exist_ok=True)
    build_result.dataframe.loc[:, list(CANONICAL_COLUMNS)].to_csv(
        path,
        index=False,
        encoding="utf-8",
    )
    return path


def _map_reader_result(
    *,
    read_result: CsvReadResult | ExcelReadResult,
    source_type: str,
    data_source: str,
    source_file: str,
    absolute_path: str,
    worksheet: Any,
    inferred_strain: str | None,
    reader_warnings: list[str],
    experiment_id: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    source_raw = read_result.dataframe.copy(deep=True)
    raw = _expand_measurement_columns(source_raw)
    row_count = len(raw)
    warnings = [f"{source_file}: reader warning: {warning}" for warning in reader_warnings]

    source_experiment = _series_by_column_name(raw, "Experiment")
    source_experiment_labels = _format_label_series(source_experiment, row_count)
    experiment_ids = source_experiment_labels.map(
        lambda value: _deterministic_experiment_id(
            source_type=source_type,
            source_file=source_file,
            source_experiment=value,
            experiment_id=experiment_id,
        )
    )

    strain_source = _strain_series(raw, inferred_strain, row_count)
    chemical_original = _format_label_series(_series_by_column_name(raw, "antibiotic"), row_count)
    concentration_label = _format_label_series(_series_by_column_name(raw, "concentration"), row_count)
    replicate_id = _format_label_series(_series_by_column_name(raw, "replicate"), row_count)
    sample_id = _format_label_series(_series_by_any_column_name(raw, _SAMPLE_COLUMN_NAMES), row_count)
    time_original = _format_label_series(_series_by_column_name(raw, "time_min"), row_count)
    time_minutes = _numeric_series(_series_by_column_name(raw, "time_min"), row_count)
    time_hours = time_minutes / 60.0
    luminescence_raw = _numeric_series(raw[_MEASUREMENT_VALUE_COLUMN], row_count)
    concentration_numeric = _parse_concentration_series(concentration_label)

    source_row_id = pd.to_numeric(raw[_SOURCE_ROW_ID_COLUMN], errors="coerce").astype("Int64")
    base = pd.DataFrame(
        {
            "Experiment_ID": experiment_ids,
            "Plate_ID": _format_label_series(_series_by_any_column_name(raw, _PLATE_COLUMN_NAMES), row_count),
            "Source_File": source_file,
            "Source_Path": absolute_path,
            "Source_Type": source_type,
            "Worksheet": worksheet,
            "Data_Source": data_source,
            "Time_Series_Duration_Hours": _duration_hours(time_minutes),
            "Analysis_Window": "unassigned",
            "Import_Timestamp": pd.NaT,
            "Source_Row_ID": source_row_id,
            "Measurement_Unit_ID": pd.NA,
            "Strain_Original": strain_source,
            "Strain_Standardized": pd.NA,
            "Chemical_Name_Original": chemical_original,
            "Chemical_Name_Standardized": pd.NA,
            "Concentration_Label": concentration_label,
            "Concentration_ug_mL": concentration_numeric,
            "Control_Status": _control_status(concentration_label),
            "Control_Type": _control_type(concentration_label),
            "Replicate_ID": replicate_id,
            "Replicate_Type": "unspecified",
            "Well_ID": _format_label_series(_series_by_any_column_name(raw, _WELL_COLUMN_NAMES), row_count),
            "Time_Original": time_original,
            "Time_Unit_Original": _time_unit(time_original),
            "Time_Minutes": time_minutes,
            "Time_Hours": time_hours,
            "Timepoint_Index": pd.Series(pd.NA, index=raw.index, dtype="Int64"),
            "Luminescence_Raw": luminescence_raw,
            "Luminescence_Normalized": pd.NA,
            "Normalization_Method": pd.NA,
            "QC_Status": "pass",
            "QC_Flags": pd.NA,
            "Record_Valid": True,
            "Notes": pd.NA,
        }
    )

    flags = pd.Series("", index=base.index, dtype="string")
    notes = pd.Series("", index=base.index, dtype="string")
    invalid_mask = pd.Series(False, index=base.index)
    warning_mask = pd.Series(False, index=base.index)

    warning_mask = _resolve_missing_replicates_from_source_position(
        base,
        raw,
        flags=flags,
        notes=notes,
        warning_mask=warning_mask,
    )
    unit_info = _assign_measurement_unit_ids(base, raw, sample_id)
    base["Measurement_Unit_ID"] = unit_info.unit_ids
    warning_mask = warning_mask | unit_info.warning_mask
    _append_flag(flags, unit_info.synthetic_mask, "measurement_unit_id_synthetic")
    _append_note(
        notes,
        unit_info.synthetic_mask,
        "Measurement_Unit_ID generated deterministically from source position.",
    )
    _append_flag(flags, unit_info.source_column_mask, "measurement_unit_id_uses_source_column")
    _append_note(
        notes,
        unit_info.source_column_mask,
        "Measurement_Unit_ID includes source measurement-column position.",
    )
    _append_flag(flags, unit_info.ambiguous_mask, "measurement_unit_identity_ambiguous")
    _append_note(
        notes,
        unit_info.ambiguous_mask,
        "Measurement unit identity remains ambiguous because source identifiers are incomplete.",
    )
    base["Timepoint_Index"] = _timepoint_index(base)

    invalid_mask = _flag_missing_required(
        base,
        flags=flags,
        notes=notes,
        invalid_mask=invalid_mask,
    )
    warning_mask = _flag_missing_replicate_id(
        base,
        flags=flags,
        notes=notes,
        warning_mask=warning_mask,
    )
    invalid_mask = _flag_negative_concentration(
        base,
        flags=flags,
        notes=notes,
        invalid_mask=invalid_mask,
    )
    invalid_mask = _flag_infinite_luminescence(
        base,
        flags=flags,
        notes=notes,
        invalid_mask=invalid_mask,
    )
    warning_mask = _flag_negative_luminescence(
        base,
        flags=flags,
        notes=notes,
        warning_mask=warning_mask,
    )
    warning_mask = _flag_unverified_concentration_units(
        base,
        flags=flags,
        notes=notes,
        warning_mask=warning_mask,
    )
    warning_mask = _flag_strain_source_mismatch(
        raw,
        inferred_strain=inferred_strain,
        flags=flags,
        notes=notes,
        warning_mask=warning_mask,
    )

    base.loc[invalid_mask, "Record_Valid"] = False
    base.loc[invalid_mask, "QC_Status"] = "fail"
    base.loc[~invalid_mask & warning_mask, "QC_Status"] = "warning"
    base["QC_Flags"] = flags.mask(flags == "", pd.NA)
    base["Notes"] = notes.mask(notes == "", pd.NA)

    warnings.extend(_source_level_warnings(source_file, base))
    canonical = coerce_canonical_dtypes(base.loc[:, list(CANONICAL_COLUMNS)])
    return canonical, warnings


def _build_result(
    dataframe: pd.DataFrame,
    build_warnings: list[str],
    source_files: list[str],
) -> CanonicalBuildResult:
    canonical = coerce_canonical_dtypes(dataframe.loc[:, list(CANONICAL_COLUMNS)])
    schema_validation = validate_canonical_schema(canonical)
    warnings = [*build_warnings, *schema_validation.warnings]
    errors = list(schema_validation.errors)

    valid_mask = canonical["Record_Valid"].fillna(False)
    warning_mask = canonical["QC_Status"].eq("warning")
    invalid_mask = ~valid_mask

    return CanonicalBuildResult(
        dataframe=canonical,
        schema_valid=schema_validation.valid,
        errors=errors,
        warnings=warnings,
        source_files=list(source_files),
        row_count=len(canonical),
        valid_record_count=int(valid_mask.sum()),
        warning_record_count=int(warning_mask.sum()),
        invalid_record_count=int(invalid_mask.sum()),
        schema_validation=schema_validation,
    )


def _expand_measurement_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per source luminescence value while preserving row position."""

    source_row_ids = pd.Series(range(1, len(raw) + 1), index=raw.index, dtype="Int64")
    measurement_columns = _column_positions_by_name(raw, "luminescence")

    if not measurement_columns:
        expanded = raw.copy(deep=True)
        expanded[_SOURCE_ROW_ID_COLUMN] = source_row_ids.to_numpy()
        expanded[_MEASUREMENT_VALUE_COLUMN] = pd.NA
        expanded[_MEASUREMENT_COLUMN_INDEX_COLUMN] = pd.NA
        expanded[_MEASUREMENT_COLUMN_LABEL_COLUMN] = pd.NA
        expanded[_MULTIPLE_MEASUREMENT_COLUMNS_COLUMN] = False
        return expanded.reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    has_multiple_measurement_columns = len(measurement_columns) > 1
    for column_position in measurement_columns:
        frame = raw.copy(deep=True)
        frame[_SOURCE_ROW_ID_COLUMN] = source_row_ids.to_numpy()
        frame[_MEASUREMENT_VALUE_COLUMN] = raw.iloc[:, column_position].to_numpy()
        frame[_MEASUREMENT_COLUMN_INDEX_COLUMN] = column_position + 1
        frame[_MEASUREMENT_COLUMN_LABEL_COLUMN] = _format_label_value(raw.columns[column_position])
        frame[_MULTIPLE_MEASUREMENT_COLUMNS_COLUMN] = has_multiple_measurement_columns
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def _column_positions_by_name(dataframe: pd.DataFrame, column_name: str) -> list[int]:
    return [
        index
        for index, existing_column in enumerate(dataframe.columns)
        if existing_column == column_name
    ]


def _series_by_column_name(dataframe: pd.DataFrame, column_name: str) -> pd.Series | None:
    for index, existing_column in enumerate(dataframe.columns):
        if existing_column == column_name:
            return dataframe.iloc[:, index].copy()
    return None


def _series_by_any_column_name(
    dataframe: pd.DataFrame,
    column_names: tuple[str, ...],
) -> pd.Series | None:
    for column_name in column_names:
        series = _series_by_column_name(dataframe, column_name)
        if series is not None:
            return series
    return None


def _strain_series(
    raw: pd.DataFrame,
    inferred_strain: str | None,
    row_count: int,
) -> pd.Series:
    if inferred_strain is not None:
        return pd.Series([inferred_strain] * row_count)

    source = _series_by_column_name(raw, "bacteria_id")
    if source is None and len(raw.columns) > 0:
        source = raw.iloc[:, 0]
    return _format_label_series(source, row_count)


def _format_label_series(series: pd.Series | None, row_count: int) -> pd.Series:
    if series is None:
        return pd.Series([pd.NA] * row_count, dtype="string")
    return series.map(_format_label_value).astype("string")


def _format_label_value(value: Any) -> str | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    stripped_value = str(value).strip()
    if stripped_value == "":
        return pd.NA
    return stripped_value


def _numeric_series(series: pd.Series | None, row_count: int) -> pd.Series:
    if series is None:
        return pd.Series([pd.NA] * row_count, dtype="Float64")
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _parse_concentration_series(concentration_label: pd.Series) -> pd.Series:
    parsed = concentration_label.map(_parse_concentration_label)
    return pd.to_numeric(parsed, errors="coerce").astype("Float64")


def _parse_concentration_label(value: Any) -> float | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.casefold() == "control":
        return pd.NA
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return pd.NA
    return float(match.group(0))


def _control_status(concentration_label: pd.Series) -> pd.Series:
    status = concentration_label.map(
        lambda value: "control"
        if not pd.isna(value) and str(value).strip().casefold() == "control"
        else "treatment"
        if not pd.isna(value)
        else "unknown"
    )
    return status.astype("string")


def _control_type(concentration_label: pd.Series) -> pd.Series:
    control_type = concentration_label.map(
        lambda value: "unknown"
        if not pd.isna(value) and str(value).strip().casefold() == "control"
        else pd.NA
    )
    return control_type.astype("string")


def _time_unit(time_original: pd.Series) -> pd.Series:
    return time_original.map(lambda value: "min" if not pd.isna(value) else pd.NA).astype("string")


def _duration_hours(time_minutes: pd.Series) -> float | pd._libs.missing.NAType:
    numeric_time = time_minutes.dropna()
    if numeric_time.empty:
        return pd.NA
    return float((numeric_time.max() - numeric_time.min()) / 60.0)


def _timepoint_index(dataframe: pd.DataFrame) -> pd.Series:
    group_columns = [
        "Experiment_ID",
        "Source_File",
        "Measurement_Unit_ID",
    ]
    return (
        dataframe.groupby(group_columns, dropna=False)
        .cumcount()
        .astype("Int64")
    )


def _resolve_missing_replicates_from_source_position(
    dataframe: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    warning_mask: pd.Series,
) -> pd.Series:
    missing_indices = dataframe.index[dataframe["Replicate_ID"].isna()]
    if missing_indices.empty:
        return warning_mask

    context = [
        "Source_File",
        "Worksheet",
        "Experiment_ID",
        "Strain_Original",
        "Chemical_Name_Original",
        "Concentration_Label",
    ]
    working = dataframe.loc[:, [*context, "Source_Row_ID", "Replicate_ID", "Time_Minutes"]].copy()
    working[_MEASUREMENT_COLUMN_INDEX_COLUMN] = raw[_MEASUREMENT_COLUMN_INDEX_COLUMN].values

    sort_columns = [_MEASUREMENT_COLUMN_INDEX_COLUMN, "Source_Row_ID"]
    source_context = ["Source_File", "Worksheet", _MEASUREMENT_COLUMN_INDEX_COLUMN]
    fill_values: dict[int, Any] = {}

    for _, source_group in working.groupby(source_context, dropna=False, sort=False):
        ordered = source_group.sort_values(sort_columns, kind="mergesort")
        ordered_indices = ordered.index.tolist()
        positions = {index: offset for offset, index in enumerate(ordered_indices)}

        for index in missing_indices.intersection(ordered.index):
            position = positions[index]
            if position == 0 or position == len(ordered_indices) - 1:
                continue

            previous_row = working.loc[ordered_indices[position - 1]]
            current_row = working.loc[index]
            next_row = working.loc[ordered_indices[position + 1]]

            previous_replicate = previous_row["Replicate_ID"]
            next_replicate = next_row["Replicate_ID"]
            if pd.isna(previous_replicate) or pd.isna(next_replicate):
                continue
            if str(previous_replicate) != str(next_replicate):
                continue
            if not _rows_share_context(previous_row, current_row, next_row, context):
                continue
            if not _time_is_between_neighbors(previous_row, current_row, next_row):
                continue

            fill_values[index] = previous_replicate

    if not fill_values:
        return warning_mask

    fill_indices = list(fill_values)
    dataframe.loc[fill_indices, "Replicate_ID"] = [fill_values[index] for index in fill_indices]
    fill_mask = dataframe.index.isin(fill_indices)
    _append_flag(flags, fill_mask, "replicate_id_inferred_from_source_position")
    _append_note(
        notes,
        fill_mask,
        "Replicate_ID inferred from adjacent source rows with matching context.",
    )
    return warning_mask | pd.Series(fill_mask, index=dataframe.index)


def _rows_share_context(
    previous_row: pd.Series,
    current_row: pd.Series,
    next_row: pd.Series,
    context: list[str],
) -> bool:
    for column in context:
        current_value = current_row[column]
        if pd.isna(current_value):
            if pd.isna(previous_row[column]) and pd.isna(next_row[column]):
                continue
            return False
        if not _same_label(previous_row[column], current_value):
            return False
        if not _same_label(next_row[column], current_value):
            return False
    return True


def _same_label(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return str(left) == str(right)


def _time_is_between_neighbors(
    previous_row: pd.Series,
    current_row: pd.Series,
    next_row: pd.Series,
) -> bool:
    previous_time = previous_row["Time_Minutes"]
    current_time = current_row["Time_Minutes"]
    next_time = next_row["Time_Minutes"]
    if pd.isna(previous_time) or pd.isna(current_time) or pd.isna(next_time):
        return False
    return float(previous_time) <= float(current_time) <= float(next_time)


def _assign_measurement_unit_ids(
    dataframe: pd.DataFrame,
    raw: pd.DataFrame,
    sample_id: pd.Series,
) -> MeasurementUnitAssignment:
    unit_ids = pd.Series(pd.NA, index=dataframe.index, dtype="string")
    synthetic_mask = pd.Series(False, index=dataframe.index)
    source_column_mask = (
        raw[_MULTIPLE_MEASUREMENT_COLUMNS_COLUMN]
        .fillna(False)
        .astype(bool)
        .reset_index(drop=True)
    )
    ambiguous_mask = pd.Series(False, index=dataframe.index)
    warning_mask = pd.Series(False, index=dataframe.index)

    plate_well_mask = dataframe["Plate_ID"].notna() & dataframe["Well_ID"].notna()
    if plate_well_mask.any():
        unit_ids.loc[plate_well_mask] = _physical_unit_ids(
            dataframe.loc[plate_well_mask],
            raw.loc[plate_well_mask],
            include_plate=True,
        )

    well_only_mask = unit_ids.isna() & dataframe["Well_ID"].notna()
    if well_only_mask.any():
        unit_ids.loc[well_only_mask] = _physical_unit_ids(
            dataframe.loc[well_only_mask],
            raw.loc[well_only_mask],
            include_plate=False,
        )

    remaining_mask = unit_ids.isna()
    if remaining_mask.any():
        _assign_source_position_unit_ids(
            dataframe.loc[remaining_mask],
            raw.loc[remaining_mask],
            unit_ids,
            synthetic_mask,
        )

    has_physical_identifier = dataframe["Plate_ID"].notna() | dataframe["Well_ID"].notna()
    has_sample_identifier = sample_id.notna()
    unresolved_replicate = (
        dataframe["Replicate_ID"].isna()
        & ~has_physical_identifier
        & ~has_sample_identifier
        & ~source_column_mask
    )
    ambiguous_mask = ambiguous_mask | unit_ids.isna() | unresolved_replicate
    warning_mask = warning_mask | synthetic_mask | source_column_mask | ambiguous_mask

    return MeasurementUnitAssignment(
        unit_ids=unit_ids,
        synthetic_mask=synthetic_mask,
        source_column_mask=source_column_mask,
        ambiguous_mask=ambiguous_mask,
        warning_mask=warning_mask,
    )


def _assign_source_position_unit_ids(
    dataframe: pd.DataFrame,
    raw: pd.DataFrame,
    unit_ids: pd.Series,
    synthetic_mask: pd.Series,
) -> None:
    if dataframe.empty:
        return

    working = dataframe[
        [
            "Source_File",
            "Worksheet",
            "Experiment_ID",
            "Strain_Original",
            "Chemical_Name_Original",
            "Concentration_Label",
            "Replicate_ID",
            "Source_Row_ID",
            "Time_Minutes",
        ]
    ].copy()
    working[_MEASUREMENT_COLUMN_INDEX_COLUMN] = raw[_MEASUREMENT_COLUMN_INDEX_COLUMN].values
    working[_MEASUREMENT_COLUMN_LABEL_COLUMN] = raw[_MEASUREMENT_COLUMN_LABEL_COLUMN].values

    group_columns = [
        "Source_File",
        "Worksheet",
        "Experiment_ID",
        "Strain_Original",
        "Chemical_Name_Original",
        "Concentration_Label",
        "Replicate_ID",
        _MEASUREMENT_COLUMN_INDEX_COLUMN,
    ]

    ordered = working.sort_values([*group_columns, "Source_Row_ID"], kind="mergesort")
    grouped = ordered.groupby(group_columns, dropna=False, sort=False)
    time_minutes = pd.to_numeric(ordered["Time_Minutes"], errors="coerce")
    previous_time = grouped["Time_Minutes"].shift()
    previous_time = pd.to_numeric(previous_time, errors="coerce")
    first_in_group = grouped.cumcount().eq(0)
    series_reset = first_in_group | (
        time_minutes.notna()
        & previous_time.notna()
        & time_minutes.lt(previous_time)
    )
    start_rows = ordered["Source_Row_ID"].where(series_reset)
    start_rows = start_rows.groupby([ordered[column] for column in group_columns], dropna=False, sort=False).ffill()
    start_rows = pd.to_numeric(start_rows, errors="coerce").astype("Int64")

    assigned_ids = [
        _source_position_unit_id(
            _safe_int(start_row_id, fallback=int(index) + 1),
            source_column_index,
            source_column_label,
        )
        for index, start_row_id, source_column_index, source_column_label in zip(
            ordered.index,
            start_rows,
            ordered[_MEASUREMENT_COLUMN_INDEX_COLUMN],
            ordered[_MEASUREMENT_COLUMN_LABEL_COLUMN],
        )
    ]
    unit_ids.loc[ordered.index] = assigned_ids
    synthetic_mask.loc[ordered.index] = True


def _physical_unit_ids(
    dataframe: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    include_plate: bool,
) -> list[str]:
    ids: list[str] = []
    for (_, row), (_, raw_row) in zip(dataframe.iterrows(), raw.iterrows()):
        source_column_suffix = _source_column_suffix(raw_row)
        well_id = _safe_identifier(str(row["Well_ID"]))
        if include_plate:
            plate_id = _safe_identifier(str(row["Plate_ID"]))
            ids.append(f"plate_{plate_id}__well_{well_id}{source_column_suffix}")
        else:
            ids.append(f"well_{well_id}{source_column_suffix}")
    return ids


def _source_position_unit_id(
    start_row_id: int,
    source_column_index: Any,
    source_column_label: Any,
) -> str:
    return f"unit_r{start_row_id:06d}{_source_column_suffix_from_values(source_column_index, source_column_label)}"


def _source_column_suffix(row: pd.Series) -> str:
    if not bool(row.get(_MULTIPLE_MEASUREMENT_COLUMNS_COLUMN, False)):
        return ""
    return _source_column_suffix_from_values(
        row.get(_MEASUREMENT_COLUMN_INDEX_COLUMN, pd.NA),
        row.get(_MEASUREMENT_COLUMN_LABEL_COLUMN, pd.NA),
    )


def _source_column_suffix_from_values(source_column_index: Any, source_column_label: Any) -> str:
    if pd.isna(source_column_index):
        return ""
    label = "unlabeled" if pd.isna(source_column_label) else _safe_identifier(str(source_column_label))
    return f"__col{int(source_column_index):03d}_{label}"


def _safe_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, *, fallback: int) -> int:
    if pd.isna(value):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _deterministic_experiment_id(
    *,
    source_type: str,
    source_file: str,
    source_experiment: Any,
    experiment_id: str | None,
) -> str:
    base = experiment_id if experiment_id is not None else f"{source_type}_{Path(source_file).stem}"
    if pd.isna(source_experiment):
        return _safe_identifier(base)
    return _safe_identifier(f"{base}_experiment_{source_experiment}")


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _flag_missing_required(
    dataframe: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    invalid_mask: pd.Series,
) -> pd.Series:
    required_fields = (
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
    )
    for field in required_fields:
        missing = dataframe[field].isna()
        _append_flag(flags, missing, f"missing_{field}")
        _append_note(notes, missing, f"Missing required field {field}.")
        invalid_mask = invalid_mask | missing
    return invalid_mask


def _flag_missing_replicate_id(
    dataframe: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    warning_mask: pd.Series,
) -> pd.Series:
    mask = dataframe["Replicate_ID"].isna()
    _append_flag(flags, mask, "missing_Replicate_ID")
    _append_note(notes, mask, "Replicate_ID unavailable from source structure.")
    return warning_mask | mask


def _flag_negative_concentration(
    dataframe: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    invalid_mask: pd.Series,
) -> pd.Series:
    mask = dataframe["Concentration_ug_mL"].notna() & (dataframe["Concentration_ug_mL"] < 0)
    _append_flag(flags, mask, "negative_concentration")
    _append_note(notes, mask, "Negative concentration is invalid.")
    return invalid_mask | mask


def _flag_infinite_luminescence(
    dataframe: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    invalid_mask: pd.Series,
) -> pd.Series:
    mask = dataframe["Luminescence_Raw"].map(_is_infinite)
    _append_flag(flags, mask, "infinite_luminescence_raw")
    _append_note(notes, mask, "Infinite raw luminescence is invalid.")
    return invalid_mask | mask


def _flag_negative_luminescence(
    dataframe: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    warning_mask: pd.Series,
) -> pd.Series:
    mask = dataframe["Luminescence_Raw"].notna() & (dataframe["Luminescence_Raw"] < 0)
    _append_flag(flags, mask, "negative_luminescence_raw")
    _append_note(notes, mask, "Negative raw luminescence retained for review.")
    return warning_mask | mask


def _flag_unverified_concentration_units(
    dataframe: pd.DataFrame,
    *,
    flags: pd.Series,
    notes: pd.Series,
    warning_mask: pd.Series,
) -> pd.Series:
    mask = dataframe["Concentration_ug_mL"].notna()
    _append_flag(flags, mask, "concentration_units_unverified")
    _append_note(notes, mask, "Numeric concentration parsed; source units require verification.")
    return warning_mask | mask


def _flag_strain_source_mismatch(
    raw: pd.DataFrame,
    *,
    inferred_strain: str | None,
    flags: pd.Series,
    notes: pd.Series,
    warning_mask: pd.Series,
) -> pd.Series:
    if inferred_strain is None:
        return warning_mask

    source_strain = _series_by_column_name(raw, "bacteria_id")
    if source_strain is None and len(raw.columns) > 0 and raw.columns[0] == inferred_strain:
        source_strain = raw.iloc[:, 0]
    if source_strain is None:
        return warning_mask

    source_labels = _format_label_series(source_strain, len(raw))
    mask = source_labels.notna() & source_labels.ne(inferred_strain)
    _append_flag(flags, mask, "strain_source_value_differs_from_filename")
    _append_note(
        notes,
        mask,
        "Source strain value differs from the filename-inferred strain label.",
    )
    return warning_mask | mask


def _source_level_warnings(source_file: str, dataframe: pd.DataFrame) -> list[str]:
    warnings = []
    if dataframe["Plate_ID"].isna().all():
        warnings.append(f"{source_file}: Plate_ID unavailable; left null.")
    if dataframe["Well_ID"].isna().all():
        warnings.append(f"{source_file}: Well_ID unavailable; left null.")
    if dataframe["Concentration_ug_mL"].notna().any():
        warnings.append(
            f"{source_file}: numeric concentrations parsed as ug/mL pending unit verification."
        )
    return warnings


def _append_flag(flags: pd.Series, mask: pd.Series, flag: str) -> None:
    if not mask.any():
        return
    flags.loc[mask] = flags.loc[mask].map(lambda value: _join_text(value, flag))


def _append_note(notes: pd.Series, mask: pd.Series, note: str) -> None:
    if not mask.any():
        return
    notes.loc[mask] = notes.loc[mask].map(lambda value: _join_text(value, note))


def _join_text(existing: Any, addition: str) -> str:
    if pd.isna(existing) or existing == "":
        return addition
    return f"{existing}; {addition}"


def _is_infinite(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return bool(pd.notna(value) and value in (float("inf"), float("-inf")))
    except TypeError:
        return False
