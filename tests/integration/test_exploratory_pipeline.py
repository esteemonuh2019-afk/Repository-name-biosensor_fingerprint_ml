import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.exploratory_analysis import ExploratoryAnalysisResult, run_exploratory_analysis
from src.feature_engine import extract_features
from src.feature_validation import validate_features
from src.fingerprint import build_fingerprint_dataset


def test_canonical_to_fingerprint_to_exploratory_analysis() -> None:
    canonical = pd.concat(
        [
            _canonical_dataframe([(0, 10), (5, 20), (10, 15)], "unit-1", "BL011", "Diazinon", "2 ug/mL", "1"),
            _canonical_dataframe([(0, 12), (5, 25), (10, 18)], "unit-2", "BL011", "Diazinon", "10 ug/mL", "2"),
            _canonical_dataframe([(0, 8), (5, 12), (10, 10)], "unit-3", "BL032", "DEET", "2 ug/mL", "1"),
            _canonical_dataframe([(0, 20), (5, 34), (10, 28)], "unit-4", "BL032", "Glyphosate", "5 ug/mL", "1"),
        ],
        ignore_index=True,
    )

    feature_dataset = extract_features(canonical)
    validation = validate_features(feature_dataset)
    fingerprints = build_fingerprint_dataset(validation)
    result = run_exploratory_analysis(
        fingerprints.dataframe,
        fingerprints.consensus_dataframe,
    )

    assert isinstance(result, ExploratoryAnalysisResult)
    assert result.analysis_passed is True
    assert len(result.pca_scores) == len(fingerprints.consensus_dataframe)
    assert not result.pca_loadings.empty
    assert not result.explained_variance.empty
    assert not result.clustering_results["cluster_assignments"].empty
    assert not result.heatmap_tables["chemical_similarity_heatmap_table"].empty
    assert len(result.concentration_trajectories) >= 1
    assert len(result.replicate_to_consensus_distances) == len(fingerprints.dataframe)
    assert result.metadata["supervised_machine_learning_performed"] is False


def _canonical_dataframe(
    points: list[tuple[float, float]],
    measurement_unit_id: str,
    strain: str,
    chemical: str,
    concentration: str,
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
                "Data_Source": "synthetic_stage_7b",
                "Time_Series_Duration_Hours": 1.0,
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
                "Replicate_ID": replicate_id,
                "Replicate_Type": "unspecified",
                "Well_ID": pd.NA,
                "Time_Original": str(time_minutes),
                "Time_Unit_Original": "min",
                "Time_Minutes": time_minutes,
                "Time_Hours": time_minutes / 60.0,
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
