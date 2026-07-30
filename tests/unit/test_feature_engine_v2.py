import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_engine_v2 import (
    FEATURE_FAMILIES,
    AdvancedFeatureDataset,
    extract_advanced_features,
    feature_columns_by_family,
    feature_dictionary,
    run_feature_family_ablation,
)


def test_feature_dictionary_tracks_family_definitions_units_and_dependencies() -> None:
    definitions = feature_dictionary()
    grouped = feature_columns_by_family()

    assert len(definitions) == 78
    assert set(grouped) == set(FEATURE_FAMILIES)
    assert all(definition.feature_name for definition in definitions)
    assert all(definition.feature_family in FEATURE_FAMILIES for definition in definitions)
    assert all(definition.mathematical_definition for definition in definitions)
    assert all(definition.units for definition in definitions)
    assert all(definition.dependencies for definition in definitions)
    assert len(grouped["window_features"]) == 32


def test_extract_advanced_features_generates_all_families_without_mutating_input() -> None:
    canonical = _canonical_dataset()
    before = canonical.copy(deep=True)

    result = extract_advanced_features(canonical)

    assert isinstance(result, AdvancedFeatureDataset)
    pd.testing.assert_frame_equal(canonical, before)
    assert result.summary["existing_feature_engine_replaced"] is False
    assert result.summary["advanced_feature_count"] == 78
    assert result.summary["advanced_feature_rows"] == 24
    for family, columns in result.feature_columns_by_family.items():
        assert columns
        for column in columns:
            assert column in result.dataframe.columns
    row = result.dataframe.iloc[0]
    assert row["temporal_time_to_peak"] >= 0
    assert row["window_0_2h_mean"] > 0
    assert row["shape_signal_energy"] > 0
    assert row["frequency_spectral_energy"] >= 0
    assert pd.notna(row["strain_interaction_mean"])
    assert row["baseline_noise"] >= 0


def test_feature_family_ablation_runs_all_feature_sets(tmp_path) -> None:
    result = run_feature_family_ablation(
        _canonical_dataset(),
        classification_models=("knn",),
        regression_models=("knn",),
        n_splits=2,
        n_repeats=1,
        permutation_repeats=1,
    )

    assert result.metadata["feature_engine_v2_isolated"] is True
    assert result.metadata["existing_pipeline_unchanged"] is True
    assert result.metadata["new_feature_count"] == 78
    assert len(result.ablation_summary) == 10
    assert set(result.ablation_summary["feature_set"]) >= {"current_core_features", "all_v2_families"}
    assert len(result.classification_comparison) == 10
    assert len(result.regression_r2_comparison) == 10
    assert not result.feature_family_redundancy.empty

    paths = result.write_outputs(tmp_path)
    names = {path.name for path in paths}
    assert "advanced_feature_dataset.csv" in names
    assert "advanced_feature_dictionary.csv" in names
    assert "feature_family_vs_macro_f1.csv" in names
    assert "feature_family_vs_r2.csv" in names
    assert "feature_family_vs_rmse.csv" in names
    assert "feature_family_vs_mae.csv" in names
    assert "feature_family_runtime.csv" in names
    assert "feature_family_comparison.png" in names
    assert "runtime_comparison.pdf" in names


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
    base = 10.0 + chemical_index * 4.0 + strain_index * 1.5 + replicate * 0.1
    scale = concentration / 50.0
    points = [
        (0.0, base),
        (60.0, base + 2.0 * scale),
        (120.0, base + 5.0 * scale),
        (360.0, base + 8.0 * scale),
        (720.0, base + 4.0 * scale),
        (1440.0, base + 1.0 * scale),
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
                "Data_Source": "synthetic_stage_8c_test",
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
