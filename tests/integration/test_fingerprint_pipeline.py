import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_engine import extract_features
from src.feature_validation import validate_features
from src.fingerprint import FingerprintDataset, build_fingerprint_dataset


def test_canonical_to_validated_features_to_fingerprint_dataset() -> None:
    canonical_dataframe = pd.concat(
        [
            _canonical_dataframe(
                [(0, 10), (5, 20), (10, 15)],
                measurement_unit_id="unit-1",
                replicate_id="1",
            ),
            _canonical_dataframe(
                [(0, 12), (5, 21), (10, 16)],
                measurement_unit_id="unit-2",
                replicate_id="2",
                concentration="10 ug/mL",
            ),
            _canonical_dataframe(
                [(0, 8), (5, 14), (10, 13)],
                measurement_unit_id="unit-3",
                replicate_id="3",
                strain="BL032",
            ),
        ],
        ignore_index=True,
    )

    feature_dataset = extract_features(canonical_dataframe)
    validation_result = validate_features(feature_dataset)
    fingerprint_dataset = build_fingerprint_dataset(validation_result)

    assert isinstance(fingerprint_dataset, FingerprintDataset)
    assert fingerprint_dataset.metadata["input_contract"] == "FeatureValidationResult"
    assert fingerprint_dataset.metadata["feature_validation_bypassed"] is False
    assert fingerprint_dataset.summary["feature_rows"] == 3
    assert fingerprint_dataset.summary["fingerprint_rows"] == 3
    assert fingerprint_dataset.summary["consensus_fingerprint_rows"] == 3
    assert fingerprint_dataset.summary["excluded_rows"] == 0
    assert fingerprint_dataset.summary["distance_matrix_rows"] == 3
    assert fingerprint_dataset.feature_names == list(validation_result.metadata["feature_columns"])
    assert "Luminescence_Raw" not in fingerprint_dataset.dataframe.columns
    assert "Time_Minutes" not in fingerprint_dataset.dataframe.columns
    assert list(fingerprint_dataset.dataframe["Measurement_Unit_ID"]) == [
        "unit-1",
        "unit-2",
        "unit-3",
    ]
    assert fingerprint_dataset.normalized_dataframe.shape == fingerprint_dataset.dataframe.shape
    assert len(fingerprint_dataset.consensus_dataframe) == 3
    assert "distance_matrix_euclidean.csv" not in {
        output["filename"]
        for output in fingerprint_dataset.distance_output_plan(distance_mode="consensus")
    }
    assert "consensus_distance_matrix_euclidean.csv" in {
        output["filename"]
        for output in fingerprint_dataset.distance_output_plan(distance_mode="consensus")
    }


def _canonical_dataframe(
    points: list[tuple[float | None, float | None]],
    *,
    measurement_unit_id: str,
    replicate_id: str,
    strain: str = "BL011",
    concentration: str = "5 ug/mL",
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
                "Data_Source": "synthetic_integration_test",
                "Time_Series_Duration_Hours": 1.0,
                "Analysis_Window": "unassigned",
                "Import_Timestamp": pd.NaT,
                "Source_Row_ID": index,
                "Measurement_Unit_ID": measurement_unit_id,
                "Strain_Original": strain,
                "Strain_Standardized": pd.NA,
                "Chemical_Name_Original": "Diazinon",
                "Chemical_Name_Standardized": pd.NA,
                "Concentration_Label": concentration,
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
