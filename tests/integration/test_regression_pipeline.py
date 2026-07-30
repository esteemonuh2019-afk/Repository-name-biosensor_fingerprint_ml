import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_engine import extract_features
from src.feature_validation import validate_features
from src.fingerprint import FingerprintDataset, build_fingerprint_dataset
from src.regression_benchmark import RegressionBenchmarkConfig, RegressionBenchmarkResult, run_regression_benchmark


def test_canonical_to_fingerprint_to_regression_benchmark() -> None:
    canonical_dataframe = _canonical_dataset()

    feature_dataset = extract_features(canonical_dataframe)
    validation_result = validate_features(feature_dataset)
    fingerprint_dataset = build_fingerprint_dataset(validation_result)
    benchmark_result = run_regression_benchmark(
        fingerprint_dataset,
        config=RegressionBenchmarkConfig(
            validation_strategy="repeated_kfold",
            n_splits=2,
            n_repeats=1,
            model_ids=("random_forest", "ridge"),
            permutation_repeats=2,
            run_leave_one_strain_importance=False,
        ),
    )

    assert isinstance(fingerprint_dataset, FingerprintDataset)
    assert isinstance(benchmark_result, RegressionBenchmarkResult)
    assert benchmark_result.metadata["sample_count"] == 64
    assert benchmark_result.metadata["unique_concentration_count"] == 4
    assert benchmark_result.metadata["input_contract"] == "validated fingerprint dataset"
    assert benchmark_result.metadata["raw_luminescence_used"] is False
    assert benchmark_result.metadata["feature_validation_bypassed"] is False
    assert benchmark_result.metadata["full_dataset_scaled_before_splitting"] is False
    assert benchmark_result.metadata["classification_framework_modified"] is False
    assert set(benchmark_result.summary["model_id"]) == {"random_forest", "ridge"}
    assert not benchmark_result.prediction_vs_actual.empty
    assert not benchmark_result.residuals.empty
    assert "Luminescence_Raw" not in fingerprint_dataset.dataframe.columns
    assert "Time_Minutes" not in fingerprint_dataset.dataframe.columns


def _canonical_dataset() -> pd.DataFrame:
    frames = []
    for chemical_index, chemical in enumerate(["Chem-A", "Chem-B"]):
        for strain in ["BL011", "BL032"]:
            for concentration in [0.5, 5.0, 50.0, 500.0]:
                for replicate in range(1, 5):
                    frames.append(
                        _canonical_dataframe(
                            chemical=chemical,
                            chemical_index=chemical_index,
                            strain=strain,
                            concentration=concentration,
                            replicate=replicate,
                        )
                    )
    return pd.concat(frames, ignore_index=True)


def _canonical_dataframe(
    *,
    chemical: str,
    chemical_index: int,
    strain: str,
    concentration: float,
    replicate: int,
) -> pd.DataFrame:
    baseline = 10.0 + concentration * 0.03 + chemical_index * 2.0 + replicate * 0.05
    points = [
        (0.0, baseline),
        (5.0, baseline + concentration * 0.08 + chemical_index),
        (10.0, baseline + concentration * 0.04 + chemical_index),
    ]
    rows = []
    measurement_unit_id = f"{chemical}-{strain}-{concentration}-{replicate}"
    for source_row_id, (time_minutes, luminescence) in enumerate(points, start=1):
        rows.append(
            {
                "Experiment_ID": "EXP-1",
                "Plate_ID": pd.NA,
                "Source_File": "synthetic.csv",
                "Source_Path": pd.NA,
                "Source_Type": "csv",
                "Worksheet": pd.NA,
                "Data_Source": "synthetic_regression_integration_test",
                "Time_Series_Duration_Hours": 1.0,
                "Analysis_Window": "unassigned",
                "Import_Timestamp": pd.NaT,
                "Source_Row_ID": source_row_id,
                "Measurement_Unit_ID": measurement_unit_id,
                "Strain_Original": strain,
                "Strain_Standardized": pd.NA,
                "Chemical_Name_Original": chemical,
                "Chemical_Name_Standardized": pd.NA,
                "Concentration_Label": f"{concentration:g}",
                "Concentration_ug_mL": concentration,
                "Control_Status": "treatment",
                "Control_Type": pd.NA,
                "Replicate_ID": str(replicate),
                "Replicate_Type": "unspecified",
                "Well_ID": pd.NA,
                "Time_Original": str(time_minutes),
                "Time_Unit_Original": "min",
                "Time_Minutes": time_minutes,
                "Time_Hours": time_minutes / 60.0,
                "Timepoint_Index": source_row_id - 1,
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
