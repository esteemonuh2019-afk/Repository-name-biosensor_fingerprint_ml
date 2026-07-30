import pandas as pd

from src.blind_prediction import (
    BlindTrainingConfig,
    FeatureProfile,
    predict_blind_sample,
    run_simulated_blind_test,
    train_blind_prediction_bundle,
)
from src.blind_prediction.prediction_report import BLIND_OUTPUT_FILENAMES
from src.data_schema.canonical_schema import CANONICAL_COLUMNS


FEATURES = [
    "baseline",
    "peak",
    "endpoint",
    "dynamic_range",
    "auc",
    "maximum_slope",
    "log2_fold_change",
    "temporal_peak_width",
    "shape_signal_energy",
    "window_0_2h_mean",
    "window_12_24h_mean",
]


def test_training_to_blind_prediction_pipeline_with_synthetic_data(tmp_path) -> None:
    training = _canonical_dataset(source_files=("train-a.csv", "train-b.csv"))
    blind = _canonical_dataset(source_files=("blind.csv",), chemicals=("Chem-B",), concentrations=(50.0,), replicates=(1, 2, 3))
    profile = FeatureProfile(
        classification_features=list(FEATURES),
        regression_features=list(FEATURES),
        classification_profile={"macro_f1_mean": 0.9},
        regression_profile={"r2_mean": 0.4},
        source="synthetic_integration_profile",
    )

    bundle = train_blind_prediction_bundle(
        training,
        feature_profile=profile,
        config=BlindTrainingConfig(
            min_chemical_specific_rows=4,
            min_chemical_specific_concentrations=2,
            random_state=11,
        ),
    )
    result = predict_blind_sample(blind, bundle=bundle)
    paths = result.write_outputs(tmp_path)

    assert result.predicted_chemical in bundle.class_labels
    assert result.predicted_concentration is not None
    assert result.novelty_status in {"Within Training Distribution", "Borderline", "Out of Distribution"}
    assert not result.chemical_probabilities.empty
    assert not result.concentration_prediction.empty
    assert not result.influential_features.empty
    assert {path.name for path in paths} == set(BLIND_OUTPUT_FILENAMES)


def test_synthetic_ood_sample_is_flagged() -> None:
    training = _canonical_dataset(source_files=("train-a.csv", "train-b.csv"))
    ood = _canonical_dataset(
        source_files=("ood.csv",),
        chemicals=("Chem-Z",),
        concentrations=(500.0,),
        replicates=(1, 2),
        signal_multiplier=75.0,
    )
    profile = FeatureProfile(
        classification_features=list(FEATURES),
        regression_features=list(FEATURES),
        source="synthetic_integration_profile",
    )
    bundle = train_blind_prediction_bundle(
        training,
        feature_profile=profile,
        config=BlindTrainingConfig(
            min_chemical_specific_rows=4,
            min_chemical_specific_concentrations=2,
            random_state=11,
        ),
    )

    result = predict_blind_sample(ood, bundle=bundle)

    assert result.novelty_status == "Out of Distribution"
    assert result.prediction_passed is False


def test_simulated_blind_testing_does_not_split_measurement_units() -> None:
    canonical = _canonical_dataset(source_files=("train-a.csv", "train-b.csv"))
    profile = FeatureProfile(
        classification_features=list(FEATURES),
        regression_features=list(FEATURES),
        source="synthetic_integration_profile",
    )

    simulation = run_simulated_blind_test(
        canonical,
        feature_profile=profile,
        group_column="Source_File",
        config=BlindTrainingConfig(
            min_chemical_specific_rows=4,
            min_chemical_specific_concentrations=2,
            random_state=11,
        ),
    )

    assert simulation["group_leakage_prevented"] is True
    assert simulation["training_measurement_units"] > 0
    assert simulation["blind_measurement_units"] > 0
    assert simulation["prediction"].source_files
    assert "chemical_prediction_correct" in simulation["evaluation"]


def _canonical_dataset(
    *,
    source_files: tuple[str, ...],
    chemicals: tuple[str, ...] = ("Chem-A", "Chem-B", "Chem-C"),
    concentrations: tuple[float, ...] = (5.0, 50.0),
    replicates: tuple[int, ...] = (1, 2),
    signal_multiplier: float = 1.0,
) -> pd.DataFrame:
    frames = []
    for source_file in source_files:
        for chemical_index, chemical in enumerate(chemicals):
            for strain_index, strain in enumerate(("BL011", "BL032")):
                for concentration in concentrations:
                    for replicate in replicates:
                        frames.append(
                            _canonical_dataframe(
                                source_file=source_file,
                                chemical=chemical,
                                chemical_index=chemical_index,
                                strain=strain,
                                strain_index=strain_index,
                                concentration=concentration,
                                replicate=replicate,
                                signal_multiplier=signal_multiplier,
                            )
                        )
    return pd.concat(frames, ignore_index=True)


def _canonical_dataframe(
    *,
    source_file: str,
    chemical: str,
    chemical_index: int,
    strain: str,
    strain_index: int,
    concentration: float,
    replicate: int,
    signal_multiplier: float,
) -> pd.DataFrame:
    base = (10.0 + chemical_index * 18.0 + strain_index * 2.0 + replicate * 0.2) * signal_multiplier
    scale = (concentration / 50.0) * (1.0 + chemical_index * 0.3) * signal_multiplier
    points = [
        (0.0, base),
        (60.0, base + 2.0 * scale),
        (120.0, base + 5.0 * scale),
        (360.0, base + 8.0 * scale),
        (720.0, base + 4.0 * scale),
        (1440.0, base + 1.0 * scale),
    ]
    rows = []
    measurement_unit_id = f"{source_file}-{chemical}-{strain}-{concentration:g}-{replicate}"
    for source_row_id, (time_minutes, luminescence) in enumerate(points, start=1):
        rows.append(
            {
                "Experiment_ID": "EXP-1",
                "Plate_ID": "P1",
                "Source_File": source_file,
                "Source_Path": pd.NA,
                "Source_Type": "csv",
                "Worksheet": pd.NA,
                "Data_Source": "synthetic_stage_9a_integration_test",
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
                "Well_ID": f"A{replicate}",
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
