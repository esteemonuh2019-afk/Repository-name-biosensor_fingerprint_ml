import math

import pandas as pd
import pytest

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_engine import extract_features
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.feature_engine.feature_qc import evaluate_feature_qc


def test_core_features_match_known_synthetic_curve() -> None:
    result = extract_features(_canonical_dataframe([(0, 10), (5, 20), (10, 15)]))

    row = result.dataframe.iloc[0]
    assert row["baseline"] == 10.0
    assert row["peak"] == 20.0
    assert row["minimum"] == 10.0
    assert row["endpoint"] == 15.0
    assert row["dynamic_range"] == 10.0
    assert row["time_to_peak"] == 5.0
    assert row["auc"] == 162.5
    assert row["initial_slope"] == 2.0
    assert row["maximum_slope"] == 2.0
    assert row["fold_change"] == 1.0
    assert row["log2_fold_change"] == pytest.approx(math.log2(1.5))
    assert row["Duration"] == 10.0
    assert row["QC_Status"] == "pass"
    assert result.summary["core_feature_count"] == 11


def test_metadata_is_retained_in_feature_vector() -> None:
    result = extract_features(
        _canonical_dataframe(
            [(0, 100), (10, 110)],
            experiment_id="EXP-9",
            source_file="BL032.csv",
            measurement_unit_id="unit-9",
            strain="BL032",
            chemical="Trimethoprim",
            concentration="5",
            replicate="3",
        )
    )

    row = result.dataframe.iloc[0]
    assert row["Experiment_ID"] == "EXP-9"
    assert row["Measurement_Unit_ID"] == "unit-9"
    assert row["Source_File"] == "BL032.csv"
    assert row["Strain"] == "BL032"
    assert row["Chemical"] == "Trimethoprim"
    assert row["Concentration"] == "5"
    assert row["Replicate_ID"] == "3"


def test_extracts_one_feature_row_per_canonical_series() -> None:
    dataframe = pd.concat(
        [
            _canonical_dataframe([(0, 10), (5, 15)], measurement_unit_id="unit-1"),
            _canonical_dataframe([(0, 20), (5, 25)], measurement_unit_id="unit-2"),
        ],
        ignore_index=True,
    )

    result = extract_features(dataframe)

    assert len(result.dataframe) == 2
    assert set(result.dataframe["Measurement_Unit_ID"]) == {"unit-1", "unit-2"}


def test_zero_baseline_flags_fold_change_without_dividing() -> None:
    result = extract_features(_canonical_dataframe([(0, 0), (5, 10), (10, 20)]))

    row = result.dataframe.iloc[0]
    assert pd.isna(row["fold_change"])
    assert pd.isna(row["log2_fold_change"])
    assert "zero_baseline_for_fold_change" in row["Feature_QC_Flags"]
    assert "zero_baseline_for_log2_fold_change" in row["Feature_QC_Flags"]
    assert result.qc.summary["zero_baseline_count"] == 1


def test_conflicting_duplicate_timestamps_are_flagged_and_not_averaged() -> None:
    result = extract_features(_canonical_dataframe([(0, 10), (5, 20), (5, 30), (10, 15)]))

    row = result.dataframe.iloc[0]
    assert row["peak"] == 30.0
    assert row["time_to_peak"] == 5.0
    assert pd.isna(row["auc"])
    assert pd.isna(row["initial_slope"])
    assert pd.isna(row["maximum_slope"])
    assert row["Duplicate_Timestamp_Count"] == 2
    assert row["Duplicate_Timestamp_Group_Count"] == 1
    assert row["QC_Status"] == "fail"
    assert "conflicting_duplicate_timestamps" in row["Feature_QC_Flags"]
    assert "duplicate_timestamps_prevent_auc" in row["Feature_QC_Flags"]


def test_identical_duplicate_timestamps_are_flagged_without_averaging() -> None:
    result = extract_features(_canonical_dataframe([(0, 10), (5, 20), (5, 20), (10, 15)]))

    row = result.dataframe.iloc[0]
    assert row["Duplicate_Timestamp_Count"] == 2
    assert row["Duplicate_Timestamp_Group_Count"] == 1
    assert pd.isna(row["auc"])
    assert row["QC_Status"] == "warning"
    assert "duplicate_timestamps" in row["Feature_QC_Flags"]


def test_missing_and_infinite_observations_are_counted_not_silently_corrected() -> None:
    result = extract_features(_canonical_dataframe([(0, 10), (5, float("inf")), (10, None)]))

    row = result.dataframe.iloc[0]
    assert row["Input_Row_Count"] == 3
    assert row["Valid_Observation_Count"] == 1
    assert row["Missing_Observation_Count"] == 2
    assert pd.isna(row["auc"])
    assert "missing_time_or_signal_rows" in row["Feature_QC_Flags"]
    assert "insufficient_distinct_timepoints_for_auc" in row["Feature_QC_Flags"]


def test_missing_time_points_are_flagged_and_remaining_valid_points_are_used() -> None:
    result = extract_features(_canonical_dataframe([(None, 10), (5, 20), (10, 30)]))

    row = result.dataframe.iloc[0]
    assert row["Input_Row_Count"] == 3
    assert row["Valid_Observation_Count"] == 2
    assert row["Missing_Observation_Count"] == 1
    assert row["baseline"] == 20.0
    assert row["endpoint"] == 30.0
    assert row["auc"] == 125.0
    assert "missing_time_or_signal_rows" in row["Feature_QC_Flags"]


def test_negative_time_values_are_flagged() -> None:
    result = extract_features(_canonical_dataframe([(-5, 10), (0, 20)]))

    row = result.dataframe.iloc[0]
    assert row["QC_Status"] == "fail"
    assert "negative_time_values" in row["Feature_QC_Flags"]
    assert result.qc.summary["negative_time_count"] == 1


def test_empty_canonical_dataframe_returns_empty_feature_dataset() -> None:
    result = extract_features(pd.DataFrame(columns=list(CANONICAL_COLUMNS)))

    assert result.dataframe.empty
    assert result.summary["feature_rows"] == 0
    assert result.summary["core_feature_count"] == len(CORE_FEATURE_COLUMNS)


def test_feature_qc_detects_impossible_time_to_peak() -> None:
    dataframe = pd.DataFrame(
        {
            "Experiment_ID": ["EXP-1"],
            "Source_File": ["BL011.csv"],
            "Measurement_Unit_ID": ["unit-1"],
            "QC_Status": ["pass"],
            "Feature_QC_Flags": [""],
            "Start_Time": [5.0],
            "End_Time": [10.0],
            "Duration": [5.0],
            "time_to_peak": [0.0],
            "baseline": [1.0],
            "fold_change": [1.0],
        }
    )

    qc = evaluate_feature_qc(dataframe, feature_columns=["baseline", "time_to_peak", "fold_change"])

    assert qc.summary["impossible_time_to_peak_count"] == 1
    assert qc.passed is False


def test_feature_qc_detects_duplicate_measurement_unit_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "Experiment_ID": ["EXP-1", "EXP-2"],
            "Source_File": ["BL011.csv", "BL011.csv"],
            "Measurement_Unit_ID": ["unit-1", "unit-1"],
            "QC_Status": ["pass", "pass"],
            "Feature_QC_Flags": ["", ""],
            "Start_Time": [0.0, 0.0],
            "End_Time": [10.0, 10.0],
            "Duration": [10.0, 10.0],
            "time_to_peak": [5.0, 5.0],
            "baseline": [1.0, 1.0],
            "fold_change": [1.0, 1.0],
        }
    )

    qc = evaluate_feature_qc(dataframe, feature_columns=["baseline", "time_to_peak", "fold_change"])

    assert qc.summary["duplicated_measurement_unit_count"] == 1
    assert qc.summary["duplicate_measurement_unit_row_count"] == 2
    assert qc.passed is True


def test_extract_features_is_deterministic_for_same_input() -> None:
    dataframe = pd.concat(
        [
            _canonical_dataframe([(0, 10), (5, 20), (10, 15)], measurement_unit_id="unit-1"),
            _canonical_dataframe([(0, 8), (5, 12), (10, 16)], measurement_unit_id="unit-2"),
        ],
        ignore_index=True,
    )

    first = extract_features(dataframe)
    second = extract_features(dataframe)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.summary == second.summary
    assert first.metadata == second.metadata


def test_extract_features_does_not_mutate_input_dataframe() -> None:
    dataframe = _canonical_dataframe([(0, 10), (5, 20), (10, 15)])
    before = dataframe.copy(deep=True)

    extract_features(dataframe)

    pd.testing.assert_frame_equal(dataframe, before)


def test_one_feature_row_is_created_per_valid_measurement_unit_id() -> None:
    dataframe = pd.concat(
        [
            _canonical_dataframe([(0, 10), (5, 20)], measurement_unit_id="unit-a"),
            _canonical_dataframe([(0, 30), (5, 45)], measurement_unit_id="unit-b"),
            _canonical_dataframe([(0, 5), (5, 7)], measurement_unit_id="unit-c"),
        ],
        ignore_index=True,
    )

    result = extract_features(dataframe)

    assert len(result.dataframe) == 3
    assert result.dataframe["Measurement_Unit_ID"].is_unique
    assert set(result.dataframe["Measurement_Unit_ID"]) == {"unit-a", "unit-b", "unit-c"}


def _canonical_dataframe(
    points: list[tuple[float | None, float | None]],
    *,
    experiment_id: str = "EXP-1",
    source_file: str = "BL011.csv",
    measurement_unit_id: str = "unit-1",
    strain: str = "BL011",
    chemical: str = "Diazinon",
    concentration: str = "5",
    replicate: str = "1",
) -> pd.DataFrame:
    rows = []
    for index, (time_minutes, luminescence) in enumerate(points, start=1):
        rows.append(
            {
                "Experiment_ID": experiment_id,
                "Plate_ID": pd.NA,
                "Source_File": source_file,
                "Source_Path": pd.NA,
                "Source_Type": "csv",
                "Worksheet": pd.NA,
                "Data_Source": "24_hour_csv",
                "Time_Series_Duration_Hours": 24.0,
                "Analysis_Window": "unassigned",
                "Import_Timestamp": pd.NaT,
                "Source_Row_ID": index,
                "Measurement_Unit_ID": measurement_unit_id,
                "Strain_Original": strain,
                "Strain_Standardized": pd.NA,
                "Chemical_Name_Original": chemical,
                "Chemical_Name_Standardized": pd.NA,
                "Concentration_Label": concentration,
                "Concentration_ug_mL": 5.0,
                "Control_Status": "treatment",
                "Control_Type": pd.NA,
                "Replicate_ID": replicate,
                "Replicate_Type": "unspecified",
                "Well_ID": pd.NA,
                "Time_Original": pd.NA if time_minutes is None else str(time_minutes),
                "Time_Unit_Original": "min",
                "Time_Minutes": time_minutes,
                "Time_Hours": pd.NA if time_minutes is None else time_minutes / 60.0,
                "Timepoint_Index": index - 1,
                "Luminescence_Raw": luminescence,
                "Luminescence_Normalized": pd.NA,
                "Normalization_Method": pd.NA,
                "QC_Status": "pass",
                "QC_Flags": pd.NA,
                "Record_Valid": True,
                "Notes": pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
