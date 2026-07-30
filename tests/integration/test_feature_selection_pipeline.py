import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_selection import FeatureSelectionConfig, FeatureSelectionResult, run_feature_selection


def test_canonical_to_feature_generation_to_feature_selection_pipeline() -> None:
    canonical = _canonical_dataset()

    result = run_feature_selection(
        canonical,
        config=FeatureSelectionConfig(
            selector_methods=("tree_importance", "permutation"),
            reduction_levels=(100, 50, 10),
            classification_model_ids=("knn",),
            regression_model_ids=("knn",),
            n_splits=2,
            n_repeats=1,
            selection_permutation_repeats=1,
            selection_tree_estimators=20,
            include_boruta=False,
        ),
    )

    assert isinstance(result, FeatureSelectionResult)
    assert result.metadata["classification_benchmark_rerun"] is True
    assert result.metadata["regression_benchmark_rerun"] is True
    assert result.metadata["generated_feature_rows"] == 24
    assert result.metadata["available_feature_count"] == 89
    assert set(result.performance_vs_feature_count["task"]) == {"classification", "regression"}
    assert result.metadata["default_classification_feature_set"]["feature_count"] >= 1
    assert result.metadata["default_regression_feature_set"]["feature_count"] >= 1
    assert result.errors == []


def _canonical_dataset() -> pd.DataFrame:
    frames = []
    for chemical_index, chemical in enumerate(["Chem-A", "Chem-B", "Chem-C"]):
        for strain_index, strain in enumerate(["BL011", "BL032"]):
            for concentration in [5.0, 50.0]:
                for replicate in range(1, 3):
                    frames.append(
                        _canonical_dataframe(
                            chemical=chemical,
                            chemical_index=chemical_index,
                            strain=strain,
                            strain_index=strain_index,
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
    strain_index: int,
    concentration: float,
    replicate: int,
) -> pd.DataFrame:
    base = 9.0 + chemical_index * 3.5 + strain_index * 1.2 + replicate * 0.1
    scale = concentration / 50.0
    points = [
        (0.0, base),
        (60.0, base + 1.5 * scale),
        (120.0, base + 4.0 * scale),
        (360.0, base + 7.0 * scale),
        (720.0, base + 3.0 * scale),
        (1440.0, base + 0.8 * scale),
    ]
    rows = []
    measurement_unit_id = f"{chemical}-{strain}-{concentration:g}-{replicate}"
    for source_row_id, (time_minutes, luminescence) in enumerate(points, start=1):
        rows.append(
            {
                "Experiment_ID": "EXP-1",
                "Plate_ID": pd.NA,
                "Source_File": "synthetic.csv",
                "Source_Path": pd.NA,
                "Source_Type": "csv",
                "Worksheet": pd.NA,
                "Data_Source": "synthetic_stage_8d_integration_test",
                "Time_Series_Duration_Hours": 24.0,
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
