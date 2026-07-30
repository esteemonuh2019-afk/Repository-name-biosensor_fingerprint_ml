import pandas as pd

from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_selection import (
    REDUCTION_LEVELS,
    REQUIRED_SELECTOR_METHODS,
    FeatureSelectionConfig,
    FeatureSelectionResult,
    build_generated_feature_table,
    run_feature_selection,
)
from src.feature_selection.selection_result import REQUIRED_OUTPUT_FILENAMES


def test_feature_selection_runs_required_methods_levels_and_writes_required_outputs(tmp_path) -> None:
    canonical = _canonical_dataset()
    before = canonical.copy(deep=True)

    result = run_feature_selection(canonical, config=_fast_config())

    assert isinstance(result, FeatureSelectionResult)
    pd.testing.assert_frame_equal(canonical, before)
    assert set(result.metadata["selector_methods_completed"]) == set(REQUIRED_SELECTOR_METHODS)
    assert result.metadata["feature_selection_after_feature_generation"] is True
    assert result.metadata["feature_engine_v2_replaced"] is False
    assert result.metadata["uses_sklearn_pipelines"] is True
    assert result.metadata["full_dataset_scaled_before_splitting"] is False
    assert set(result.classification_after_selection["reduction_level_percent"]) == set(REDUCTION_LEVELS)
    assert set(result.regression_after_selection["reduction_level_percent"]) == set(REDUCTION_LEVELS)
    assert not result.selected_features.empty
    assert not result.feature_ranking.empty
    assert not result.performance_vs_feature_count.empty

    paths = result.write_outputs(tmp_path)
    names = {path.name for path in paths}
    assert set(REQUIRED_OUTPUT_FILENAMES).issubset(names)
    assert "performance_vs_feature_count.png" in names
    assert "performance_vs_feature_count.pdf" in names
    assert "feature_importance.png" in names
    assert "feature_ranking.pdf" in names
    csv_md_names = {name for name in names if name.endswith((".csv", ".md"))}
    assert csv_md_names == set(REQUIRED_OUTPUT_FILENAMES)


def test_feature_selection_recommendations_are_present_and_reproducible() -> None:
    config = FeatureSelectionConfig(
        selector_methods=("tree_importance", "permutation"),
        reduction_levels=(100, 50, 10),
        classification_model_ids=("knn",),
        regression_model_ids=("knn",),
        n_splits=2,
        n_repeats=1,
        selection_permutation_repeats=1,
        selection_tree_estimators=20,
        include_boruta=False,
    )

    first = run_feature_selection(_canonical_dataset(), config=config)
    second = run_feature_selection(_canonical_dataset(), config=config)

    first_class = first.metadata["default_classification_feature_set"]
    first_reg = first.metadata["default_regression_feature_set"]
    assert first_class["feature_count"] >= 1
    assert first_reg["feature_count"] >= 1
    assert first.metadata["research_feature_set"]["feature_count"] >= max(
        first_class["feature_count"],
        first_reg["feature_count"],
    )
    pd.testing.assert_frame_equal(first.feature_ranking, second.feature_ranking)
    pd.testing.assert_frame_equal(
        first.classification_after_selection.drop(columns=["runtime_seconds"]).sort_index(axis=1),
        second.classification_after_selection.drop(columns=["runtime_seconds"]).sort_index(axis=1),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        first.regression_after_selection.drop(columns=["runtime_seconds"]).sort_index(axis=1),
        second.regression_after_selection.drop(columns=["runtime_seconds"]).sort_index(axis=1),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_generated_feature_table_combines_core_and_v2_features_after_generation() -> None:
    generated = build_generated_feature_table(_canonical_dataset())

    assert generated["feature_engine_v2_replaced"] is False
    assert generated["generated_feature_rows"] == 24
    assert generated["available_feature_count"] == 89
    assert "baseline" in generated["feature_names"]
    assert "temporal_time_to_peak" in generated["feature_names"]
    assert "window_0_2h_auc" in generated["feature_names"]


def _fast_config() -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        selector_methods=REQUIRED_SELECTOR_METHODS,
        reduction_levels=REDUCTION_LEVELS,
        classification_model_ids=("knn",),
        regression_model_ids=("knn",),
        n_splits=2,
        n_repeats=1,
        selection_permutation_repeats=1,
        selection_tree_estimators=20,
        max_sequential_greedy_steps=4,
        sequential_candidate_pool=5,
        selection_cv_splits=2,
        include_boruta=False,
    )


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
                "Data_Source": "synthetic_stage_8d_test",
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
