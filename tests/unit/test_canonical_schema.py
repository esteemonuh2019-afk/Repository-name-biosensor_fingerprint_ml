import math

import pandas as pd

from src.data_schema.canonical_schema import (
    CANONICAL_COLUMNS,
    CANONICAL_DTYPES,
    REQUIRED_FIELDS,
    coerce_canonical_dtypes,
    create_empty_canonical_dataframe,
    validate_canonical_schema,
)


def valid_record(**overrides):
    record = {
        "Experiment_ID": "EXP-001",
        "Plate_ID": pd.NA,
        "Source_File": "BL027ab.csv",
        "Source_Path": pd.NA,
        "Source_Type": "csv",
        "Worksheet": pd.NA,
        "Data_Source": "24_hour_csv",
        "Time_Series_Duration_Hours": 24.0,
        "Analysis_Window": "unassigned",
        "Import_Timestamp": pd.Timestamp("2026-07-25T12:00:00Z"),
        "Source_Row_ID": 1,
        "Measurement_Unit_ID": "unit_r000001",
        "Strain_Original": "BL027ab",
        "Strain_Standardized": pd.NA,
        "Chemical_Name_Original": "Lambda Cyclotherin",
        "Chemical_Name_Standardized": pd.NA,
        "Concentration_Label": "5",
        "Concentration_ug_mL": 5.0,
        "Control_Status": "treatment",
        "Control_Type": pd.NA,
        "Replicate_ID": "1",
        "Replicate_Type": pd.NA,
        "Well_ID": pd.NA,
        "Time_Original": "120",
        "Time_Unit_Original": "min",
        "Time_Minutes": 120.0,
        "Time_Hours": 2.0,
        "Timepoint_Index": 24,
        "Luminescence_Raw": 12345.0,
        "Luminescence_Normalized": pd.NA,
        "Normalization_Method": pd.NA,
        "QC_Status": "not_evaluated",
        "QC_Flags": pd.NA,
        "Record_Valid": True,
        "Notes": pd.NA,
    }
    record.update(overrides)
    return record


def valid_dataframe(**overrides) -> pd.DataFrame:
    return pd.DataFrame([valid_record(**overrides)], columns=list(CANONICAL_COLUMNS))


def test_empty_canonical_dataframe_has_all_fields_in_order() -> None:
    dataframe = create_empty_canonical_dataframe()

    assert list(dataframe.columns) == list(CANONICAL_COLUMNS)
    assert dataframe.empty


def test_required_columns_are_detected_when_missing() -> None:
    dataframe = valid_dataframe().drop(columns=["Experiment_ID"])

    result = validate_canonical_schema(dataframe)

    assert result.valid is False
    assert result.missing_columns == ["Experiment_ID"]


def test_optional_columns_may_contain_null_values() -> None:
    dataframe = valid_dataframe()

    result = validate_canonical_schema(dataframe)

    assert result.valid is True


def test_data_types_are_coerced_correctly() -> None:
    dataframe = valid_dataframe(
        Time_Minutes="120",
        Time_Hours="2",
        Luminescence_Raw="12345",
        Source_Row_ID="1",
        Record_Valid="true",
    )

    coerced = coerce_canonical_dtypes(dataframe)

    assert str(coerced["Time_Minutes"].dtype) == CANONICAL_DTYPES["Time_Minutes"]
    assert str(coerced["Time_Hours"].dtype) == CANONICAL_DTYPES["Time_Hours"]
    assert str(coerced["Luminescence_Raw"].dtype) == CANONICAL_DTYPES["Luminescence_Raw"]
    assert str(coerced["Source_Row_ID"].dtype) == CANONICAL_DTYPES["Source_Row_ID"]
    assert str(coerced["Record_Valid"].dtype) == CANONICAL_DTYPES["Record_Valid"]


def test_unknown_source_type_is_flagged() -> None:
    result = validate_canonical_schema(valid_dataframe(Source_Type="json"))

    assert result.valid is False
    assert result.invalid_values["Source_Type"] == ["json"]


def test_unknown_qc_status_is_flagged() -> None:
    result = validate_canonical_schema(valid_dataframe(QC_Status="maybe"))

    assert result.valid is False
    assert result.invalid_values["QC_Status"] == ["maybe"]


def test_negative_concentration_fails_validation() -> None:
    result = validate_canonical_schema(valid_dataframe(Concentration_ug_mL=-1.0))

    assert result.valid is False
    assert result.row_problem_counts["negative_concentration"] == 1


def test_infinite_luminescence_fails_validation() -> None:
    result = validate_canonical_schema(valid_dataframe(Luminescence_Raw=math.inf))

    assert result.valid is False
    assert result.row_problem_counts["infinite_luminescence_raw"] == 1


def test_negative_luminescence_is_retained_but_warned() -> None:
    dataframe = valid_dataframe(Luminescence_Raw=-10.0)

    result = validate_canonical_schema(dataframe)

    assert result.valid is True
    assert dataframe.loc[0, "Luminescence_Raw"] == -10.0
    assert result.row_problem_counts["negative_luminescence_raw"] == 1


def test_time_minutes_and_hours_inconsistency_is_detected() -> None:
    result = validate_canonical_schema(valid_dataframe(Time_Minutes=120.0, Time_Hours=3.0))

    assert result.valid is False
    assert result.row_problem_counts["time_minutes_hours_inconsistent"] == 1


def test_original_labels_are_preserved_by_dtype_coercion() -> None:
    dataframe = valid_dataframe(
        Strain_Original="BL027ab",
        Chemical_Name_Original="Lambda Cyclotherin",
    )

    coerced = coerce_canonical_dtypes(dataframe)

    assert coerced.loc[0, "Strain_Original"] == "BL027ab"
    assert coerced.loc[0, "Chemical_Name_Original"] == "Lambda Cyclotherin"


def test_lambda_cyclotherin_is_not_changed() -> None:
    coerced = coerce_canonical_dtypes(valid_dataframe(Chemical_Name_Original="Lambda Cyclotherin"))

    assert coerced.loc[0, "Chemical_Name_Original"] == "Lambda Cyclotherin"


def test_bl027ab_is_not_changed() -> None:
    coerced = coerce_canonical_dtypes(valid_dataframe(Strain_Original="BL027ab"))

    assert coerced.loc[0, "Strain_Original"] == "BL027ab"


def test_normalized_luminescence_may_be_null() -> None:
    result = validate_canonical_schema(valid_dataframe(Luminescence_Normalized=pd.NA))

    assert result.valid is True


def test_duplicate_logical_records_are_flagged() -> None:
    dataframe = pd.DataFrame(
        [
            valid_record(Source_Row_ID=1, Timepoint_Index=1),
            valid_record(Source_Row_ID=2, Timepoint_Index=2),
        ],
        columns=list(CANONICAL_COLUMNS),
    )

    result = validate_canonical_schema(dataframe)

    assert result.valid is True
    assert result.row_problem_counts["duplicate_logical_records"] == 2


def test_unexpected_columns_are_reported() -> None:
    dataframe = valid_dataframe()
    dataframe["Unexpected"] = "value"

    result = validate_canonical_schema(dataframe)

    assert result.valid is False
    assert result.unexpected_columns == ["Unexpected"]


def test_validation_does_not_delete_rows() -> None:
    dataframe = pd.DataFrame(
        [valid_record(Source_Row_ID=1), valid_record(Source_Row_ID=2)],
        columns=list(CANONICAL_COLUMNS),
    )

    validate_canonical_schema(dataframe)

    assert len(dataframe) == 2


def test_validation_does_not_mutate_original_dataframe() -> None:
    dataframe = valid_dataframe(Time_Minutes="120")
    before = dataframe.copy(deep=True)

    validate_canonical_schema(dataframe)

    pd.testing.assert_frame_equal(dataframe, before)


def test_canonical_column_order_is_deterministic_after_coercion() -> None:
    dataframe = valid_dataframe()
    shuffled = dataframe[list(reversed(CANONICAL_COLUMNS))]

    coerced = coerce_canonical_dtypes(shuffled)

    assert list(coerced.columns) == list(CANONICAL_COLUMNS)


def test_valid_synthetic_record_passes() -> None:
    result = validate_canonical_schema(valid_dataframe())

    assert result.valid is True
    assert result.errors == []
