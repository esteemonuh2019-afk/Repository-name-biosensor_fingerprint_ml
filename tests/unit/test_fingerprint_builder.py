import math

import pandas as pd
import pytest

from src.feature_engine.feature_dataset import FeatureDataset
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.feature_engine.feature_qc import evaluate_feature_qc
from src.feature_validation import validate_features
from src.fingerprint import (
    DEFAULT_DISTANCE_MODE,
    DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS,
    FingerprintDataset,
    build_fingerprint_dataset,
    calculate_distance_matrix,
    correlation_distance,
    cosine_distance,
    euclidean_distance,
    estimate_distance_matrix_size,
    manhattan_distance,
)


def test_fingerprint_creation_from_validated_features() -> None:
    validation = validate_features(_feature_dataframe())

    result = build_fingerprint_dataset(validation, normalization="zscore")

    assert isinstance(result, FingerprintDataset)
    assert len(result.dataframe) == 4
    assert result.summary["feature_rows"] == 4
    assert result.summary["fingerprint_rows"] == 4
    assert result.summary["excluded_rows"] == 0
    assert result.metadata["feature_validation_bypassed"] is False
    assert result.summary["default_distance_mode"] == "consensus"
    assert result.summary["consensus_fingerprint_rows"] == 1


def test_metadata_is_preserved_in_fingerprint_dataset() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[0, "Strain"] = "BL032"
    dataframe.loc[0, "Chemical"] = "Trimethoprim"
    dataframe.loc[0, "Concentration"] = "10 ug/mL"
    dataframe.loc[0, "Source_File"] = "source file.csv"
    validation = validate_features(dataframe)

    result = build_fingerprint_dataset(validation)

    row = result.dataframe.iloc[0]
    assert row["Fingerprint_ID"] == "EXP-1::source file.csv::unit-1"
    assert row["Experiment_ID"] == "EXP-1"
    assert row["Measurement_Unit_ID"] == "unit-1"
    assert row["Strain"] == "BL032"
    assert row["Chemical"] == "Trimethoprim"
    assert row["Concentration"] == "10 ug/mL"


def test_feature_order_is_stable_and_expected() -> None:
    validation = validate_features(_feature_dataframe())

    result = build_fingerprint_dataset(validation)

    actual_order = [
        column
        for column in result.dataframe.columns
        if column in CORE_FEATURE_COLUMNS
    ]
    assert actual_order == list(CORE_FEATURE_COLUMNS)
    assert result.feature_names == list(CORE_FEATURE_COLUMNS)


def test_default_distance_mode_is_consensus() -> None:
    assert DEFAULT_DISTANCE_MODE == "consensus"
    assert DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS == 2000


def test_duplicate_fingerprints_are_detected() -> None:
    dataframe = pd.DataFrame(
        [
            _feature_row("unit-a", "1", 10.0, 20.0, 8.0, 15.0, 12.0, 5.0, 155.0, 2.0, 4.0),
            _feature_row("unit-b", "2", 10.0, 20.0, 8.0, 15.0, 12.0, 5.0, 155.0, 2.0, 4.0),
        ]
    )
    validation = validate_features(dataframe)

    result = build_fingerprint_dataset(validation)

    assert result.qc.summary["duplicate_fingerprint_row_count"] == 2
    assert result.qc.summary["duplicate_fingerprint_group_count"] == 1
    assert any("Duplicate fingerprint vectors" in warning for warning in result.warnings)


def test_duplicated_measurement_unit_ids_are_detected() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[1, "Measurement_Unit_ID"] = dataframe.loc[0, "Measurement_Unit_ID"]
    validation = validate_features(dataframe)

    result = build_fingerprint_dataset(validation)

    assert result.qc.summary["duplicated_measurement_unit_count"] == 1
    assert result.qc.summary["duplicated_measurement_unit_row_count"] == 2


def test_zscore_normalisation_does_not_overwrite_original_values() -> None:
    validation = validate_features(_feature_dataframe())

    result = build_fingerprint_dataset(validation, normalization="zscore")

    original_baseline = result.dataframe["baseline"].copy()
    normalized_baseline = result.normalized_dataframe["baseline"]
    assert original_baseline.tolist() == [10.0, 12.0, 9.0, 15.0]
    assert normalized_baseline.mean() == pytest.approx(0.0)
    assert normalized_baseline.std(ddof=0) == pytest.approx(1.0)


def test_consensus_grouping_preserves_strain_chemical_and_concentration() -> None:
    dataframe = _feature_dataframe_for_consensus()
    validation = validate_features(dataframe)

    result = build_fingerprint_dataset(validation)

    groups = result.consensus_dataframe.loc[:, ["Strain", "Chemical", "Concentration"]]
    assert len(result.consensus_dataframe) == 3
    assert set(map(tuple, groups.to_records(index=False))) == {
        ("BL011", "Diazinon", "5 ug/mL"),
        ("BL011", "Diazinon", "10 ug/mL"),
        ("BL032", "Diazinon", "5 ug/mL"),
    }


def test_consensus_replicate_counts_and_values_are_deterministic() -> None:
    validation = validate_features(_feature_dataframe_for_consensus())

    first = build_fingerprint_dataset(validation)
    second = build_fingerprint_dataset(validation)

    pd.testing.assert_frame_equal(first.consensus_dataframe, second.consensus_dataframe)
    group = first.consensus_dataframe.loc[
        first.consensus_dataframe["Consensus_ID"].eq("BL011::Diazinon::5 ug/mL")
    ].iloc[0]
    assert group["Replicate_Count"] == 2
    assert group["Measurement_Unit_Count"] == 2
    assert group["baseline"] == pytest.approx(11.0)
    summary = first.consensus_summary.loc[
        (first.consensus_summary["Consensus_ID"].eq("BL011::Diazinon::5 ug/mL"))
        & first.consensus_summary["feature"].eq("baseline")
    ].iloc[0]
    assert summary["mean"] == pytest.approx(11.0)
    assert summary["standard_deviation"] == pytest.approx(1.0)
    assert summary["coefficient_of_variation"] == pytest.approx(1.0 / 11.0)


def test_minmax_and_robust_normalisation_are_supported() -> None:
    validation = validate_features(_feature_dataframe())

    minmax = build_fingerprint_dataset(validation, normalization="minmax")
    robust = build_fingerprint_dataset(validation, normalization="robust")

    assert minmax.normalized_dataframe["baseline"].min() == pytest.approx(0.0)
    assert minmax.normalized_dataframe["baseline"].max() == pytest.approx(1.0)
    assert robust.normalization_parameters["method"] == "robust"
    assert "iqr" in robust.normalization_parameters


def test_distance_calculations_match_known_values() -> None:
    assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)
    assert manhattan_distance([0, 0], [3, 4]) == pytest.approx(7.0)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert correlation_distance([1, 2, 3], [3, 2, 1]) == pytest.approx(2.0)

    matrix_input = pd.DataFrame(
        {
            "Fingerprint_ID": ["a", "b"],
            "x": [0.0, 3.0],
            "y": [0.0, 4.0],
        }
    )
    matrix = calculate_distance_matrix(
        matrix_input,
        feature_names=["x", "y"],
        metric="euclidean",
    )
    assert matrix.shape == (2, 2)
    assert matrix.loc["a", "b"] == pytest.approx(5.0)
    assert matrix.loc["b", "a"] == pytest.approx(5.0)


def test_output_is_deterministic() -> None:
    validation = validate_features(_feature_dataframe())

    first = build_fingerprint_dataset(validation)
    second = build_fingerprint_dataset(validation)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    pd.testing.assert_frame_equal(first.normalized_dataframe, second.normalized_dataframe)
    pd.testing.assert_frame_equal(first.consensus_dataframe, second.consensus_dataframe)
    pd.testing.assert_frame_equal(first.consensus_summary, second.consensus_summary)
    assert first.summary == second.summary
    assert first.metadata == second.metadata


def test_input_feature_dataset_is_not_mutated() -> None:
    dataframe = _feature_dataframe()
    feature_dataset = FeatureDataset(
        dataframe=dataframe,
        metadata={"source": "synthetic"},
        summary={"feature_rows": len(dataframe)},
        qc=evaluate_feature_qc(dataframe, feature_columns=CORE_FEATURE_COLUMNS),
    )
    before = feature_dataset.dataframe.copy(deep=True)

    validation = validate_features(feature_dataset)
    build_fingerprint_dataset(validation)

    pd.testing.assert_frame_equal(feature_dataset.dataframe, before)


def test_failed_or_nonfinite_feature_rows_are_excluded_with_qc_context() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[0, "QC_Status"] = "fail"
    dataframe.loc[1, "auc"] = pd.NA
    validation = validate_features(dataframe)

    result = build_fingerprint_dataset(validation)

    assert result.summary["feature_rows"] == 4
    assert result.summary["fingerprint_rows"] == 2
    assert result.summary["excluded_rows"] == 2
    assert set(result.excluded_dataframe["Fingerprint_Exclusion_Reason"]) == {
        "feature_qc_fail",
        "missing_or_nonfinite_core_feature",
    }


def test_builder_requires_validation_result() -> None:
    with pytest.raises(TypeError):
        build_fingerprint_dataset(_feature_dataframe())  # type: ignore[arg-type]


def test_default_outputs_consensus_distance_matrices_only(tmp_path) -> None:
    result = build_fingerprint_dataset(validate_features(_feature_dataframe_for_consensus()))

    paths = result.write_outputs(tmp_path)
    names = {path.name for path in paths}

    assert "consensus_fingerprint_dataset.csv" in names
    assert "consensus_fingerprint_summary.csv" in names
    assert "consensus_distance_matrix_euclidean.csv" in names
    assert "distance_matrix_euclidean.csv" not in names
    assert not (tmp_path / "distance_matrix_euclidean.csv").exists()
    matrix = pd.read_csv(tmp_path / "consensus_distance_matrix_euclidean.csv")
    assert matrix.shape == (3, 4)


def test_none_distance_mode_produces_no_distance_matrices(tmp_path) -> None:
    result = build_fingerprint_dataset(validate_features(_feature_dataframe_for_consensus()))

    paths = result.write_outputs(tmp_path, distance_mode="none")
    names = {path.name for path in paths}

    assert not any("distance_matrix" in name for name in names)
    assert not list(tmp_path.glob("*distance_matrix*.csv"))


def test_individual_distance_mode_works_below_threshold(tmp_path) -> None:
    result = build_fingerprint_dataset(validate_features(_feature_dataframe()))

    paths = result.write_outputs(
        tmp_path,
        distance_mode="individual",
        max_individual_distance_rows=10,
    )
    names = {path.name for path in paths}

    assert "distance_matrix_euclidean.csv" in names
    assert "consensus_distance_matrix_euclidean.csv" not in names


def test_individual_distance_mode_refuses_above_threshold(tmp_path) -> None:
    result = build_fingerprint_dataset(validate_features(_feature_dataframe()))

    with pytest.raises(ValueError, match="Individual distance matrix refused"):
        result.write_outputs(
            tmp_path,
            distance_mode="individual",
            max_individual_distance_rows=2,
        )
    assert not any(tmp_path.iterdir())


def test_override_allows_explicit_large_individual_distance_output(tmp_path) -> None:
    result = build_fingerprint_dataset(validate_features(_feature_dataframe()))

    paths = result.write_outputs(
        tmp_path,
        distance_mode="individual",
        max_individual_distance_rows=2,
        allow_large_distance_matrix=True,
    )

    assert (tmp_path / "distance_matrix_cosine.csv").exists()
    assert any(path.name == "distance_matrix_correlation.csv" for path in paths)


def test_size_estimation_is_deterministic() -> None:
    first = estimate_distance_matrix_size(4)
    second = estimate_distance_matrix_size(4)

    assert first == second
    assert first["rows"] == 4
    assert first["columns"] == 4
    assert first["cells"] == 16
    assert first["estimated_memory_bytes"] == 128


def test_output_overwrite_protection_remains_active(tmp_path) -> None:
    result = build_fingerprint_dataset(validate_features(_feature_dataframe()))
    result.write_outputs(tmp_path, distance_mode="none")

    with pytest.raises(FileExistsError):
        result.write_outputs(tmp_path, distance_mode="none")


def _feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _feature_row("unit-1", "1", 10.0, 20.0, 8.0, 15.0, 12.0, 5.0, 155.0, 2.0, 4.0),
            _feature_row("unit-2", "2", 12.0, 18.0, 9.0, 14.0, 9.0, 4.0, 148.0, 1.2, 2.6),
            _feature_row("unit-3", "3", 9.0, 22.0, 7.0, 13.0, 15.0, 7.0, 166.0, 2.1, 3.4),
            _feature_row("unit-4", "4", 15.0, 24.0, 11.0, 20.0, 13.0, 8.0, 202.0, 0.8, 1.5),
        ]
    )


def _feature_dataframe_for_consensus() -> pd.DataFrame:
    rows = [
        _feature_row("unit-1", "1", 10.0, 20.0, 8.0, 15.0, 12.0, 5.0, 155.0, 2.0, 4.0),
        _feature_row("unit-2", "2", 12.0, 18.0, 9.0, 14.0, 9.0, 4.0, 148.0, 1.2, 2.6),
        _feature_row("unit-3", "1", 9.0, 22.0, 7.0, 13.0, 15.0, 7.0, 166.0, 2.1, 3.4),
        _feature_row("unit-4", "1", 15.0, 24.0, 11.0, 20.0, 13.0, 8.0, 202.0, 0.8, 1.5),
    ]
    rows[2]["Concentration"] = "10 ug/mL"
    rows[3]["Strain"] = "BL032"
    return pd.DataFrame(rows)


def _feature_row(
    measurement_unit_id: str,
    replicate_id: str,
    baseline: float,
    peak: float,
    minimum: float,
    endpoint: float,
    dynamic_range: float,
    time_to_peak: float,
    auc: float,
    initial_slope: float,
    maximum_slope: float,
) -> dict[str, object]:
    return {
        "Experiment_ID": "EXP-1",
        "Measurement_Unit_ID": measurement_unit_id,
        "Source_File": "synthetic.csv",
        "Strain": "BL011",
        "Chemical": "Diazinon",
        "Concentration": "5 ug/mL",
        "Replicate_ID": replicate_id,
        "Duration": 10.0,
        "QC_Status": "pass",
        "Feature_QC_Flags": "",
        "baseline": baseline,
        "peak": peak,
        "minimum": minimum,
        "endpoint": endpoint,
        "dynamic_range": dynamic_range,
        "time_to_peak": time_to_peak,
        "auc": auc,
        "initial_slope": initial_slope,
        "maximum_slope": maximum_slope,
        "fold_change": (peak - baseline) / baseline,
        "log2_fold_change": math.log2(endpoint / baseline),
        "Start_Time": 0.0,
        "End_Time": 10.0,
        "Input_Row_Count": 3,
        "Valid_Observation_Count": 3,
        "Missing_Observation_Count": 0,
        "Duplicate_Timestamp_Count": 0,
        "Duplicate_Timestamp_Group_Count": 0,
        "Source_QC_Statuses": "pass",
        "Source_QC_Flags": "",
    }
