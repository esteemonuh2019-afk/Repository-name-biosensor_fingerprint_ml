import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_engine import FeatureDataset, extract_features
from src.feature_validation import FeatureValidationResult, validate_features


def test_canonical_to_feature_dataset_to_feature_validation_result() -> None:
    canonical_dataframe = pd.concat(
        [
            _canonical_dataframe(
                [(0, 10), (5, 20), (10, 15)],
                measurement_unit_id="unit-clean-1",
                replicate_id="1",
            ),
            _canonical_dataframe(
                [(0, 12), (5, 21), (10, 16)],
                measurement_unit_id="unit-clean-2",
                replicate_id="2",
            ),
            _canonical_dataframe(
                [(0, 0), (5, 5), (10, 10)],
                measurement_unit_id="unit-zero-baseline",
                replicate_id="3",
            ),
        ],
        ignore_index=True,
    )

    feature_dataset = extract_features(canonical_dataframe)
    validation = validate_features(feature_dataset)

    assert isinstance(feature_dataset, FeatureDataset)
    assert isinstance(validation, FeatureValidationResult)
    assert feature_dataset.qc.summary["feature_row_count"] == 3
    assert feature_dataset.qc.summary["zero_baseline_count"] == 1
    assert validation.metadata["feature_rows"] == 3
    assert validation.metadata["valid_feature_rows"] == 3
    assert validation.metadata["feature_columns_assessed"] == 11
    assert validation.metadata["selection_is_supervised"] is False
    assert validation.metadata["biological_reproducibility_claimed"] is False

    missing = validation.missing_value_summary.set_index("feature")
    assert missing.loc["fold_change", "missing_count"] == 1
    assert missing.loc["log2_fold_change", "missing_count"] == 1
    assert (
        validation.range_validation_summary["violation"]
        .astype(str)
        .eq("undefined_fold_change_from_zero_baseline")
        .any()
    )
    assert not validation.feature_statistics.empty
    assert not validation.correlation_summary["pearson"].empty
    assert not validation.correlation_summary["spearman"].empty
    assert not validation.replicate_reproducibility_summary.empty
    assert set(validation.feature_recommendations["feature"]) == set(
        validation.metadata["feature_columns"]
    )


def _canonical_dataframe(
    points: list[tuple[float | None, float | None]],
    *,
    measurement_unit_id: str,
    replicate_id: str,
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
                "Data_Source": "synthetic_unit_test",
                "Time_Series_Duration_Hours": 1.0,
                "Analysis_Window": "unassigned",
                "Import_Timestamp": pd.NaT,
                "Source_Row_ID": index,
                "Measurement_Unit_ID": measurement_unit_id,
                "Strain_Original": "BL011",
                "Strain_Standardized": pd.NA,
                "Chemical_Name_Original": "Diazinon",
                "Chemical_Name_Standardized": pd.NA,
                "Concentration_Label": "5 ug/mL",
                "Concentration_ug_mL": 5.0,
                "Control_Status": "treatment",
                "Control_Type": pd.NA,
                "Replicate_ID": replicate_id,
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
