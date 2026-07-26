import math

import pandas as pd
import pytest

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_engine import FeatureDataset, extract_features


def test_canonical_dataframe_to_feature_dataset_to_feature_qc() -> None:
    canonical_dataframe = pd.concat(
        [
            _canonical_dataframe(
                [(0, 10), (5, 20), (10, 15)],
                measurement_unit_id="unit-clean",
            ),
            _canonical_dataframe(
                [(0, 0), (5, 5), (10, 10)],
                measurement_unit_id="unit-zero-baseline",
            ),
        ],
        ignore_index=True,
    )

    feature_dataset = extract_features(canonical_dataframe)

    assert isinstance(feature_dataset, FeatureDataset)
    assert len(feature_dataset.dataframe) == 2
    assert feature_dataset.summary["input_canonical_rows"] == 6
    assert feature_dataset.summary["feature_rows"] == 2
    assert feature_dataset.qc.summary["feature_row_count"] == 2
    assert feature_dataset.qc.summary["zero_baseline_count"] == 1
    assert feature_dataset.qc.summary["failed_feature_rows"] == 0
    assert feature_dataset.qc.summary["warning_feature_rows"] == 1
    assert feature_dataset.qc.passed is True

    clean = feature_dataset.dataframe.set_index("Measurement_Unit_ID").loc["unit-clean"]
    assert clean["baseline"] == 10.0
    assert clean["peak"] == 20.0
    assert clean["minimum"] == 10.0
    assert clean["endpoint"] == 15.0
    assert clean["dynamic_range"] == 10.0
    assert clean["time_to_peak"] == 5.0
    assert clean["auc"] == 162.5
    assert clean["initial_slope"] == 2.0
    assert clean["maximum_slope"] == 2.0
    assert clean["fold_change"] == 1.0
    assert clean["log2_fold_change"] == pytest.approx(math.log2(1.5))


def _canonical_dataframe(
    points: list[tuple[float | None, float | None]],
    *,
    measurement_unit_id: str,
) -> pd.DataFrame:
    rows = []
    for index, (time_minutes, luminescence) in enumerate(points, start=1):
        rows.append(
            {
                "Experiment_ID": "EXP-1",
                "Plate_ID": pd.NA,
                "Source_File": "synthetic.csv",
                "Source_Path": pd.NA,
                "Source_Type": "csv",
                "Worksheet": pd.NA,
                "Data_Source": "24_hour_csv",
                "Time_Series_Duration_Hours": 1.0,
                "Analysis_Window": "unassigned",
                "Import_Timestamp": pd.NaT,
                "Source_Row_ID": index,
                "Measurement_Unit_ID": measurement_unit_id,
                "Strain_Original": "BL011",
                "Strain_Standardized": pd.NA,
                "Chemical_Name_Original": "Diazinon",
                "Chemical_Name_Standardized": pd.NA,
                "Concentration_Label": "5",
                "Concentration_ug_mL": 5.0,
                "Control_Status": "treatment",
                "Control_Type": pd.NA,
                "Replicate_ID": "1",
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

