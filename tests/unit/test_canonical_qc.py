from pathlib import Path

import pandas as pd
import pytest

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.quality_control.canonical_qc import (
    audit_canonical_dataframe,
    write_qc_outputs,
)


def canonical_dataframe(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame([_canonical_row(**row) for row in rows], columns=list(CANONICAL_COLUMNS))


def _canonical_row(**overrides) -> dict:
    row = {
        "Experiment_ID": "csv_BL011_experiment_1",
        "Plate_ID": "Plate-1",
        "Source_File": "BL011.csv",
        "Source_Path": r"C:\raw\BL011.csv",
        "Source_Type": "csv",
        "Worksheet": pd.NA,
        "Data_Source": "24_hour_csv",
        "Time_Series_Duration_Hours": 24.0,
        "Analysis_Window": "unassigned",
        "Import_Timestamp": pd.NaT,
        "Source_Row_ID": 1,
        "Measurement_Unit_ID": "unit_r000001",
        "Strain_Original": "BL011",
        "Strain_Standardized": pd.NA,
        "Chemical_Name_Original": "Lambda Cyclotherin",
        "Chemical_Name_Standardized": pd.NA,
        "Concentration_Label": "5",
        "Concentration_ug_mL": 5.0,
        "Control_Status": "treatment",
        "Control_Type": "unknown",
        "Replicate_ID": "1",
        "Replicate_Type": "unspecified",
        "Well_ID": "A01",
        "Time_Original": "0",
        "Time_Unit_Original": "min",
        "Time_Minutes": 0.0,
        "Time_Hours": 0.0,
        "Timepoint_Index": 0,
        "Luminescence_Raw": 100.0,
        "Luminescence_Normalized": pd.NA,
        "Normalization_Method": pd.NA,
        "QC_Status": "pass",
        "QC_Flags": pd.NA,
        "Record_Valid": True,
        "Notes": pd.NA,
    }
    row.update(overrides)
    return row


def test_clean_canonical_data_passes_qc() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Time_Minutes": 0.0, "Time_Hours": 0.0, "Timepoint_Index": 0},
        {
            "Source_Row_ID": 2,
            "Time_Minutes": 60.0,
            "Time_Hours": 1.0,
            "Time_Original": "60",
            "Timepoint_Index": 1,
            "Luminescence_Raw": 120.0,
        },
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.qc_passed is True
    assert result.row_count == 2
    assert result.exact_duplicate_count == 0
    assert result.logical_duplicate_count == 0


def test_exact_duplicate_rows_are_counted() -> None:
    row = _canonical_row()
    dataframe = pd.DataFrame([row, row], columns=list(CANONICAL_COLUMNS))

    result = audit_canonical_dataframe(dataframe)

    assert result.exact_duplicate_count == 2
    assert result.qc_passed is False


def test_duplicate_source_row_ids_are_counted_within_source_file() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 7, "Time_Minutes": 0.0, "Time_Hours": 0.0},
        {"Source_Row_ID": 7, "Time_Minutes": 60.0, "Time_Hours": 1.0, "Time_Original": "60"},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.source_row_id_duplicate_count == 2
    assert result.qc_passed is True


def test_same_source_row_id_in_different_files_is_not_a_source_row_duplicate() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Source_File": "BL011.csv"},
        {"Source_Row_ID": 1, "Source_File": "BL029.csv", "Time_Minutes": 60.0, "Time_Hours": 1.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.source_row_id_duplicate_count == 0


def test_logical_duplicate_key_omits_source_file() -> None:
    dataframe = canonical_dataframe(
        {"Source_File": "BL011.csv", "Source_Row_ID": 1},
        {"Source_File": "BL011_copy.csv", "Source_Row_ID": 1, "Luminescence_Raw": 101.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.legacy_logical_duplicate_count == 2
    assert result.logical_duplicate_count == 0
    assert result.source_aware_logical_duplicate_count == 0
    assert result.conflicting_value_duplicate_count == 0


def test_conflicting_logical_duplicates_are_classified() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Luminescence_Raw": 100.0},
        {"Source_Row_ID": 2, "Luminescence_Raw": 101.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.logical_duplicate_count == 2
    assert result.duplicate_group_count == 1
    assert result.conflicting_value_duplicate_count == 2
    assert result.identical_value_duplicate_count == 0


def test_identical_value_logical_duplicates_are_classified() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Luminescence_Raw": 100.0},
        {"Source_Row_ID": 2, "Luminescence_Raw": 100.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.identical_value_duplicate_count == 2
    assert result.conflicting_value_duplicate_count == 0


def test_different_wells_are_not_logical_duplicates() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Well_ID": "A01", "Measurement_Unit_ID": "well_A01"},
        {"Source_Row_ID": 2, "Well_ID": "A02", "Measurement_Unit_ID": "well_A02", "Luminescence_Raw": 101.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.logical_duplicate_count == 0


def test_different_replicates_are_not_logical_duplicates() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Replicate_ID": "1", "Measurement_Unit_ID": "unit_r000001"},
        {"Source_Row_ID": 2, "Replicate_ID": "2", "Measurement_Unit_ID": "unit_r000002", "Luminescence_Raw": 101.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.logical_duplicate_count == 0


def test_missing_replicate_in_duplicate_group_is_ambiguous() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Replicate_ID": pd.NA, "Plate_ID": pd.NA, "Well_ID": pd.NA},
        {
            "Source_Row_ID": 2,
            "Replicate_ID": pd.NA,
            "Plate_ID": pd.NA,
            "Well_ID": pd.NA,
            "Luminescence_Raw": 101.0,
        },
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.ambiguous_replicate_count == 2
    assert result.ambiguous_replicate_group_count == 1
    assert result.ambiguous_measurement_identity_count == 2


def test_missing_well_id_is_reported_but_not_fatal_without_duplicates() -> None:
    dataframe = canonical_dataframe({"Well_ID": pd.NA})

    result = audit_canonical_dataframe(dataframe)

    assert result.missing_identifier_counts["Well_ID"] == 1
    assert result.qc_passed is True


def test_missing_plate_id_is_reported_but_not_fatal_without_duplicates() -> None:
    dataframe = canonical_dataframe({"Plate_ID": pd.NA})

    result = audit_canonical_dataframe(dataframe)

    assert result.missing_identifier_counts["Plate_ID"] == 1
    assert result.qc_passed is True


def test_negative_luminescence_is_retained_as_warning() -> None:
    dataframe = canonical_dataframe({"Luminescence_Raw": -1.0})

    result = audit_canonical_dataframe(dataframe)

    assert result.negative_luminescence_count == 1
    assert result.invalid_numeric_counts["negative_luminescence"] == 1
    assert result.qc_passed is True


def test_infinite_luminescence_fails_qc() -> None:
    dataframe = canonical_dataframe({"Luminescence_Raw": float("inf")})

    result = audit_canonical_dataframe(dataframe)

    assert result.infinite_luminescence_count == 1
    assert result.qc_passed is False


def test_negative_concentration_fails_qc() -> None:
    dataframe = canonical_dataframe({"Concentration_ug_mL": -5.0})

    result = audit_canonical_dataframe(dataframe)

    assert result.invalid_numeric_counts["negative_concentration"] == 1
    assert result.qc_passed is False


def test_negative_time_fails_qc() -> None:
    dataframe = canonical_dataframe({"Time_Minutes": -1.0, "Time_Hours": -1.0 / 60.0})

    result = audit_canonical_dataframe(dataframe)

    assert result.invalid_numeric_counts["negative_time"] == 1
    assert result.qc_passed is False


def test_time_minutes_hours_mismatch_fails_qc() -> None:
    dataframe = canonical_dataframe({"Time_Minutes": 60.0, "Time_Hours": 2.0})

    result = audit_canonical_dataframe(dataframe)

    assert result.invalid_numeric_counts["time_hours_minutes_mismatch"] == 1
    assert result.qc_passed is False


def test_non_monotonic_time_series_is_counted() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Time_Minutes": 60.0, "Time_Hours": 1.0, "Timepoint_Index": 1},
        {"Source_Row_ID": 2, "Time_Minutes": 0.0, "Time_Hours": 0.0, "Timepoint_Index": 0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.non_monotonic_time_group_count == 1
    assert result.qc_passed is False


def test_duplicate_timepoint_group_is_counted() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Time_Minutes": 60.0, "Time_Hours": 1.0, "Timepoint_Index": 1},
        {
            "Source_Row_ID": 2,
            "Time_Minutes": 60.0,
            "Time_Hours": 1.0,
            "Timepoint_Index": 2,
            "Luminescence_Raw": 101.0,
        },
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.duplicate_timepoint_group_count == 1


def test_audit_does_not_mutate_input_dataframe() -> None:
    dataframe = canonical_dataframe({"Luminescence_Raw": "100.0"})
    before = dataframe.copy(deep=True)

    audit_canonical_dataframe(dataframe)

    pd.testing.assert_frame_equal(dataframe, before)


def test_original_labels_are_preserved_in_summary() -> None:
    dataframe = canonical_dataframe(
        {
            "Source_File": "BL027ab.csv",
            "Strain_Original": "BL027ab",
            "Chemical_Name_Original": "Lambda Cyclotherin",
        }
    )

    result = audit_canonical_dataframe(dataframe)

    assert result.strains_detected == ["BL027ab"]
    assert result.chemicals_detected == ["Lambda Cyclotherin"]


def test_empty_dataframe_is_reported_clearly() -> None:
    dataframe = pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    result = audit_canonical_dataframe(dataframe)

    assert result.row_count == 0
    assert result.qc_passed is False
    assert "Canonical dataframe is empty." in result.errors


def test_missing_schema_column_is_error() -> None:
    dataframe = canonical_dataframe({}).drop(columns=["Luminescence_Raw"])

    result = audit_canonical_dataframe(dataframe)

    assert result.qc_passed is False
    assert any("missing required schema columns" in error for error in result.errors)


def test_unexpected_column_is_warning() -> None:
    dataframe = canonical_dataframe({})
    dataframe["Unexpected"] = "value"

    result = audit_canonical_dataframe(dataframe)

    assert any("unexpected columns" in warning for warning in result.warnings)


def test_summary_tables_are_returned() -> None:
    dataframe = canonical_dataframe(
        {"Source_Row_ID": 1, "Luminescence_Raw": 100.0},
        {"Source_Row_ID": 2, "Luminescence_Raw": 101.0},
    )

    result = audit_canonical_dataframe(dataframe)

    assert set(result.summary_tables) >= {
        "logical_duplicate_groups",
        "source_file_summary",
        "missing_values",
        "time_series_issues",
    }
    assert not result.summary_tables["logical_duplicate_groups"].empty


def test_write_qc_outputs_creates_expected_files(tmp_path: Path) -> None:
    result = audit_canonical_dataframe(canonical_dataframe({}))
    output_dir = tmp_path / "qc_outputs"

    created = write_qc_outputs(result, output_dir)

    assert output_dir.exists()
    assert output_dir / "qc_summary.json" in created
    assert output_dir / "canonical_qc_report.md" in created


def test_write_qc_outputs_refuses_existing_directory(tmp_path: Path) -> None:
    result = audit_canonical_dataframe(canonical_dataframe({}))
    output_dir = tmp_path / "qc_outputs"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        write_qc_outputs(result, output_dir)
