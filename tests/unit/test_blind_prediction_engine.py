from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.blind_prediction import (
    BlindTrainingConfig,
    FeatureProfile,
    load_model_bundle,
    predict_blind_sample,
    probability_entropy,
    enforce_feature_order,
    evaluate_blind_predictions,
    evaluate_prediction_qc,
    run_simulated_blind_test,
    train_blind_prediction_bundle,
    concentration_range_status,
)
from src.blind_prediction.model_bundle import TRAINING_OUTPUT_FILENAMES
from src.blind_prediction.prediction_report import BLIND_OUTPUT_FILENAMES
from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.feature_selection import build_generated_feature_table
from src.quality_control.canonical_qc import audit_canonical_dataframe


PROFILE_FEATURES = [
    "baseline",
    "peak",
    "endpoint",
    "dynamic_range",
    "time_to_peak",
    "auc",
    "maximum_slope",
    "log2_fold_change",
    "temporal_peak_width",
    "shape_signal_energy",
    "frequency_spectral_entropy",
    "window_0_2h_mean",
    "window_2_6h_mean",
    "window_6_12h_mean",
    "window_12_24h_mean",
]


@pytest.fixture(scope="module")
def training_canonical() -> pd.DataFrame:
    return _canonical_dataset(source_files=("batch-1.csv", "batch-2.csv"))


@pytest.fixture(scope="module")
def feature_profile() -> FeatureProfile:
    return FeatureProfile(
        classification_features=list(PROFILE_FEATURES),
        regression_features=list(PROFILE_FEATURES),
        classification_profile={"macro_f1_mean": 0.9, "balanced_accuracy_mean": 0.9},
        regression_profile={"r2_mean": 0.5, "rmse_mean": 1.0, "mae_mean": 0.5},
        source="synthetic_test_profile",
    )


@pytest.fixture(scope="module")
def bundle(training_canonical: pd.DataFrame, feature_profile: FeatureProfile):
    return train_blind_prediction_bundle(
        training_canonical,
        feature_profile=feature_profile,
        config=BlindTrainingConfig(
            min_chemical_specific_rows=4,
            min_chemical_specific_concentrations=2,
            random_state=7,
        ),
    )


@pytest.fixture(scope="module")
def blind_canonical() -> pd.DataFrame:
    return _canonical_dataset(
        source_files=("blind.csv",),
        chemicals=("Chem-B",),
        concentrations=(50.0,),
        replicates=(1, 2, 3),
    )


@pytest.fixture(scope="module")
def blind_result(bundle, blind_canonical: pd.DataFrame):
    return predict_blind_sample(blind_canonical, bundle=bundle)


def test_model_bundle_saves_and_reloads_correctly(bundle, tmp_path) -> None:
    paths = bundle.save(tmp_path)
    names = {path.name for path in paths}

    assert set(TRAINING_OUTPUT_FILENAMES) == names
    reloaded = load_model_bundle(tmp_path)
    assert reloaded.classification_features == bundle.classification_features
    assert reloaded.regression_features == bundle.regression_features
    assert reloaded.class_labels == bundle.class_labels


def test_feature_order_is_enforced(bundle, blind_canonical: pd.DataFrame) -> None:
    generated = build_generated_feature_table(blind_canonical)["dataframe"]
    ordered, _ = enforce_feature_order(generated, bundle.classification_features)

    assert list(ordered.columns) == bundle.classification_features


def test_missing_required_features_fail_clearly(bundle, blind_canonical: pd.DataFrame) -> None:
    generated = build_generated_feature_table(blind_canonical)["dataframe"].drop(columns=[bundle.classification_features[0]])

    with pytest.raises(ValueError, match="Missing required model features"):
        enforce_feature_order(generated, bundle.classification_features)


def test_extra_features_do_not_silently_alter_model_input(bundle, blind_canonical: pd.DataFrame) -> None:
    generated = build_generated_feature_table(blind_canonical)["dataframe"].copy()
    generated["extra_unapproved_feature"] = 123.0

    ordered, warnings = enforce_feature_order(generated, bundle.classification_features)

    assert "extra_unapproved_feature" not in ordered.columns
    assert any("Extra columns ignored" in warning for warning in warnings)


def test_preprocessing_is_reused_rather_than_refitted(bundle, blind_canonical: pd.DataFrame) -> None:
    before = bundle.classifier_pipeline.named_steps["preprocess"].mean_.copy()

    predict_blind_sample(blind_canonical, bundle=bundle)

    np.testing.assert_allclose(before, bundle.classifier_pipeline.named_steps["preprocess"].mean_)


def test_blind_samples_are_not_appended_to_training_data(bundle, blind_canonical: pd.DataFrame) -> None:
    before = len(bundle.novelty_reference["training_vectors"])

    predict_blind_sample(blind_canonical, bundle=bundle)

    assert len(bundle.novelty_reference["training_vectors"]) == before


def test_class_probabilities_sum_appropriately(blind_result) -> None:
    assert blind_result.chemical_probabilities["probability"].sum() == pytest.approx(1.0)


def test_top_three_predictions_are_ordered(blind_result) -> None:
    probabilities = [row["probability"] for row in blind_result.top_three_candidates]

    assert probabilities == sorted(probabilities, reverse=True)


def test_prediction_margin_is_correct(blind_result) -> None:
    values = sorted(blind_result.chemical_probabilities["probability"].astype(float), reverse=True)

    assert blind_result.prediction_margin == pytest.approx(values[0] - values[1])


def test_entropy_is_calculated_correctly() -> None:
    assert probability_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_chemical_specific_regressor_is_selected(bundle, blind_canonical: pd.DataFrame) -> None:
    result = predict_blind_sample(blind_canonical, bundle=bundle)

    assert result.concentration_prediction.iloc[0]["regressor_used"] == "chemical_specific"


def test_unsupported_chemical_concentration_prediction_is_withheld(bundle, blind_canonical: pd.DataFrame) -> None:
    predicted = predict_blind_sample(blind_canonical, bundle=bundle).predicted_chemical
    reduced_bundle = replace(bundle, chemical_regressors={key: value for key, value in bundle.chemical_regressors.items() if key != predicted})

    result = predict_blind_sample(blind_canonical, bundle=reduced_bundle)

    assert result.predicted_concentration is None
    assert result.concentration_prediction.iloc[0]["withheld_reason"] == "chemical_specific_regressor_unavailable"


def test_training_range_interpolation_is_identified(bundle) -> None:
    status, label = concentration_range_status(50.0, "Chem-B", bundle)

    assert status == "Interpolation"
    assert "to" in label


def test_extrapolation_is_identified(bundle) -> None:
    status, _ = concentration_range_status(5000.0, "Chem-B", bundle)

    assert status == "Extrapolation"


def test_distance_based_novelty_works(bundle, blind_canonical: pd.DataFrame) -> None:
    result = predict_blind_sample(blind_canonical, bundle=bundle)

    assert result.novelty_status in {"Within Training Distribution", "Borderline", "Out of Distribution"}
    assert result.novelty_assessment.iloc[0]["nearest_training_distance"] is not None


def test_confidence_based_novelty_works(bundle) -> None:
    ood = _canonical_dataset(source_files=("ood.csv",), chemicals=("Chem-Z",), concentrations=(500.0,), replicates=(1, 2), signal_multiplier=50.0)
    result = predict_blind_sample(ood, bundle=bundle)

    assert result.novelty_status == "Out of Distribution"


def test_severe_ood_samples_are_flagged(bundle) -> None:
    ood = _canonical_dataset(source_files=("ood.csv",), chemicals=("Chem-Z",), concentrations=(500.0,), replicates=(1, 2), signal_multiplier=50.0)
    result = predict_blind_sample(ood, bundle=bundle)

    assert result.prediction_passed is False
    assert any("Severe novelty" in error for error in result.errors)


def test_qc_gates_prevent_unreliable_predictions(bundle, blind_canonical: pd.DataFrame) -> None:
    generated = build_generated_feature_table(blind_canonical)["dataframe"].drop(columns=[bundle.classification_features[0]])
    qc = evaluate_prediction_qc(
        canonical_qc=audit_canonical_dataframe(blind_canonical),
        canonical_dataframe=blind_canonical,
        feature_dataframe=generated,
        bundle=bundle,
    )

    assert qc.status == "FAIL"


def test_confidence_score_is_deterministic(bundle, blind_canonical: pd.DataFrame) -> None:
    first = predict_blind_sample(blind_canonical, bundle=bundle)
    second = predict_blind_sample(blind_canonical, bundle=bundle)

    pd.testing.assert_frame_equal(first.prediction_confidence, second.prediction_confidence)


def test_model_versions_appear_in_outputs(blind_result) -> None:
    assert blind_result.model_versions["bundle_version"]
    assert blind_result.pipeline_version


def test_input_data_are_not_mutated(bundle, blind_canonical: pd.DataFrame) -> None:
    before = blind_canonical.copy(deep=True)

    predict_blind_sample(blind_canonical, bundle=bundle)

    pd.testing.assert_frame_equal(blind_canonical, before)


def test_prediction_is_reproducible(bundle, blind_canonical: pd.DataFrame) -> None:
    first = predict_blind_sample(blind_canonical, bundle=bundle)
    second = predict_blind_sample(blind_canonical, bundle=bundle)

    pd.testing.assert_frame_equal(first.chemical_probabilities, second.chemical_probabilities)
    assert first.predicted_concentration == pytest.approx(second.predicted_concentration)


def test_true_labels_are_not_required_for_blind_prediction(bundle, blind_canonical: pd.DataFrame) -> None:
    blind = blind_canonical.copy(deep=True)
    blind["Chemical_Name_Original"] = "BlindUnknown"
    blind["Concentration_Label"] = "1"
    blind["Concentration_ug_mL"] = 1.0

    result = predict_blind_sample(blind, bundle=bundle)

    assert result.predicted_chemical in bundle.class_labels
    assert result.summary_dict()["true_labels_included"] is False


def test_evaluation_mode_remains_separate(bundle, blind_canonical: pd.DataFrame, tmp_path) -> None:
    result = predict_blind_sample(blind_canonical, bundle=bundle)
    result.write_outputs(tmp_path)
    truth = pd.DataFrame([{"true_chemical": "Chem-B", "true_concentration": 50.0}])
    truth_path = tmp_path / "truth.csv"
    truth.to_csv(truth_path, index=False)

    evaluation = evaluate_blind_predictions(tmp_path, truth_path)

    assert evaluation["truth_file_read_by_prediction_command"] is False
    assert "chemical_prediction_correct" in evaluation


def test_original_chemical_and_strain_labels_are_preserved(bundle, blind_result) -> None:
    assert set(bundle.class_labels) == {"Chem-A", "Chem-B", "Chem-C"}
    assert set(blind_result.influential_strains["strain"].astype(str)) == {"BL011", "BL032"}


def test_prediction_outputs_use_required_filenames(blind_result, tmp_path) -> None:
    paths = blind_result.write_outputs(tmp_path)

    assert {path.name for path in paths} == set(BLIND_OUTPUT_FILENAMES)
    summary = json.loads((tmp_path / "blind_prediction_summary.json").read_text(encoding="utf-8"))
    assert summary["true_labels_included"] is False


def test_simulated_blind_testing_uses_group_holdout(training_canonical: pd.DataFrame, feature_profile: FeatureProfile) -> None:
    result = run_simulated_blind_test(
        training_canonical,
        feature_profile=feature_profile,
        group_column="Source_File",
        config=BlindTrainingConfig(
            min_chemical_specific_rows=4,
            min_chemical_specific_concentrations=2,
            random_state=7,
        ),
    )

    assert result["group_leakage_prevented"] is True
    assert result["training_rows"] > 0
    assert result["blind_rows"] > 0
    assert "chemical_prediction_correct" in result["evaluation"]


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
                "Data_Source": "synthetic_stage_9a_test",
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
