import math

import pandas as pd
import pytest

from src.feature_engine.feature_dataset import FeatureDataset
from src.feature_engine.feature_extractor import CORE_FEATURE_COLUMNS
from src.feature_engine.feature_qc import evaluate_feature_qc
from src.feature_validation import FeatureValidationResult, validate_features


def test_clean_informative_features_pass_validation() -> None:
    result = validate_features(_feature_dataframe())

    assert isinstance(result, FeatureValidationResult)
    assert result.validation_passed is True
    assert result.errors == []
    assert result.metadata["feature_columns_assessed"] == len(CORE_FEATURE_COLUMNS)
    assert len(result.feature_statistics) == len(CORE_FEATURE_COLUMNS)


def test_constant_features_are_detected() -> None:
    dataframe = _feature_dataframe()
    dataframe["maximum_slope"] = 4.0

    result = validate_features(dataframe)

    assert "maximum_slope" in _features(result.constant_feature_summary)
    recommendation = _recommendation_for(result, "maximum_slope")
    assert recommendation == "Exclude"


def test_near_constant_features_are_detected() -> None:
    dataframe = _feature_dataframe()
    dataframe["initial_slope"] = [1.0, 1.0, 1.0, 1.0000001]

    result = validate_features(dataframe)

    assert "initial_slope" in _features(result.low_variance_feature_summary)


def test_missing_values_are_counted_correctly() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[1, "endpoint"] = pd.NA

    result = validate_features(dataframe)

    endpoint = _summary_row(result.missing_value_summary, "endpoint")
    assert endpoint["missing_count"] == 1
    assert endpoint["missing_percentage"] == pytest.approx(25.0)
    assert "Missing feature values detected: 1." in result.warnings


def test_infinite_values_are_detected() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[2, "auc"] = math.inf

    result = validate_features(dataframe)

    auc = _summary_row(result.infinite_value_summary, "auc")
    assert auc["positive_infinity_count"] == 1
    assert auc["nonfinite_count"] == 1
    assert result.validation_passed is False
    assert any("Infinite or non-numeric feature values" in error for error in result.errors)


def test_non_numeric_values_are_handled_safely() -> None:
    dataframe = _feature_dataframe()
    dataframe["initial_slope"] = dataframe["initial_slope"].astype(object)
    dataframe.loc[0, "initial_slope"] = "not-a-number"

    result = validate_features(dataframe)

    initial_slope = _summary_row(result.infinite_value_summary, "initial_slope")
    assert initial_slope["non_numeric_count"] == 1
    assert result.validation_passed is False


def test_metadata_columns_are_excluded_from_feature_statistics() -> None:
    result = validate_features(_feature_dataframe())

    assessed = set(result.feature_statistics["feature"])
    assert assessed == set(CORE_FEATURE_COLUMNS)
    assert "Experiment_ID" not in assessed
    assert "Measurement_Unit_ID" not in assessed
    assert "Replicate_ID" not in assessed


def test_pearson_correlation_is_calculated_correctly() -> None:
    result = validate_features(_perfectly_correlated_dataframe())

    correlation = _correlation_for(result.correlation_summary["pearson"], "baseline", "peak")
    assert correlation == pytest.approx(1.0)


def test_spearman_correlation_is_calculated_correctly() -> None:
    result = validate_features(_perfectly_correlated_dataframe())

    correlation = _correlation_for(result.correlation_summary["spearman"], "baseline", "peak")
    assert correlation == pytest.approx(1.0)


def test_highly_correlated_pairs_are_identified() -> None:
    result = validate_features(_perfectly_correlated_dataframe())

    pairs = result.correlation_summary["highly_correlated_pairs"]
    matching = pairs.loc[
        (pairs["feature_a"].eq("baseline") & pairs["feature_b"].eq("peak"))
        | (pairs["feature_a"].eq("peak") & pairs["feature_b"].eq("baseline"))
    ]
    assert not matching.empty
    assert matching["correlation"].abs().min() >= 0.95


def test_time_to_peak_range_violations_are_detected() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[0, "time_to_peak"] = -1.0

    result = validate_features(dataframe)

    assert _has_violation(result, "time_to_peak", "negative_time_to_peak")
    assert result.validation_passed is False


def test_negative_dynamic_range_is_flagged() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[0, "dynamic_range"] = -1.0

    result = validate_features(dataframe)

    assert _has_violation(result, "dynamic_range", "negative_dynamic_range")
    assert result.validation_passed is False


def test_zero_baseline_fold_change_problems_are_retained_and_reported() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[0, "baseline"] = 0.0
    dataframe.loc[0, "fold_change"] = pd.NA
    dataframe.loc[0, "log2_fold_change"] = pd.NA

    result = validate_features(dataframe)

    assert result.validated_dataframe.loc[0, "baseline"] == 0.0
    assert pd.isna(result.validated_dataframe.loc[0, "fold_change"])
    assert _has_violation(result, "fold_change", "undefined_fold_change_from_zero_baseline")
    assert result.validation_passed is False


def test_replicate_coefficient_of_variation_is_computed_correctly() -> None:
    dataframe = _feature_dataframe().iloc[:2].copy()
    dataframe["Replicate_ID"] = ["1", "2"]

    result = validate_features(dataframe)

    detail = result.replicate_consistency_detail
    baseline = detail.loc[detail["feature"].eq("baseline")].iloc[0]
    assert baseline["replicate_count"] == 2
    assert baseline["mean"] == pytest.approx(11.0)
    assert baseline["standard_deviation"] == pytest.approx(1.0)
    assert baseline["coefficient_of_variation"] == pytest.approx(1.0 / 11.0)


def test_single_replicate_groups_are_marked_insufficient_data() -> None:
    dataframe = _feature_dataframe().iloc[:1].copy()

    result = validate_features(dataframe)

    baseline = result.replicate_consistency_detail.loc[
        result.replicate_consistency_detail["feature"].eq("baseline")
    ].iloc[0]
    assert baseline["stability_flag"] == "Insufficient Data"


def test_replicate_type_unspecified_does_not_claim_biological_reproducibility() -> None:
    dataframe = _feature_dataframe()
    dataframe["Replicate_Type"] = "unspecified"

    result = validate_features(dataframe)

    assert result.metadata["biological_reproducibility_claimed"] is False
    assert result.metadata["replicate_assessment_label"] == "replicate consistency"
    assert result.replicate_consistency_detail["biological_reproducibility_claimed"].eq(False).all()


def test_feature_recommendations_are_deterministic() -> None:
    dataframe = _feature_dataframe()

    first = validate_features(dataframe)
    second = validate_features(dataframe)

    pd.testing.assert_frame_equal(first.feature_recommendations, second.feature_recommendations)
    assert first.retained_feature_candidates == second.retained_feature_candidates
    assert first.excluded_feature_candidates == second.excluded_feature_candidates


def test_input_feature_dataset_is_not_mutated() -> None:
    dataframe = _feature_dataframe()
    qc = evaluate_feature_qc(dataframe, feature_columns=CORE_FEATURE_COLUMNS)
    feature_dataset = FeatureDataset(
        dataframe=dataframe,
        metadata={"source": "synthetic"},
        summary={"feature_rows": len(dataframe)},
        qc=qc,
    )
    before = feature_dataset.dataframe.copy(deep=True)

    validate_features(feature_dataset)

    pd.testing.assert_frame_equal(feature_dataset.dataframe, before)


def test_original_labels_are_preserved() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[0, "Strain"] = "BL032"
    dataframe.loc[0, "Chemical"] = "Trimethoprim"
    dataframe.loc[0, "Concentration"] = "10 ug/mL"
    dataframe.loc[0, "Source_File"] = "synthetic source.csv"

    result = validate_features(dataframe)

    row = result.validated_dataframe.iloc[0]
    assert row["Strain"] == "BL032"
    assert row["Chemical"] == "Trimethoprim"
    assert row["Concentration"] == "10 ug/mL"
    assert row["Source_File"] == "synthetic source.csv"


def test_empty_datasets_are_handled_clearly() -> None:
    result = validate_features(pd.DataFrame())

    assert result.validation_passed is False
    assert "Feature dataset is empty." in result.errors
    assert "No core feature columns are available for validation." in result.errors
    assert result.metadata["feature_rows"] == 0


def _feature_dataframe() -> pd.DataFrame:
    rows = [
        _feature_row("unit-1", "1", 10.0, 20.0, 8.0, 15.0, 12.0, 5.0, 155.0, 2.0, 4.0),
        _feature_row("unit-2", "2", 12.0, 18.0, 9.0, 14.0, 9.0, 4.0, 148.0, 1.2, 2.6),
        _feature_row("unit-3", "3", 9.0, 22.0, 7.0, 13.0, 15.0, 7.0, 166.0, 2.1, 3.4),
        _feature_row("unit-4", "4", 15.0, 24.0, 11.0, 20.0, 13.0, 8.0, 202.0, 0.8, 1.5),
    ]
    return pd.DataFrame(rows)


def _perfectly_correlated_dataframe() -> pd.DataFrame:
    rows = []
    for index, baseline in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        peak = baseline * 2.0
        minimum = baseline - 1.0
        endpoint = baseline * 1.5
        rows.append(
            _feature_row(
                f"corr-{index}",
                str(index),
                baseline,
                peak,
                minimum,
                endpoint,
                peak - minimum,
                float(index),
                baseline * 5.0,
                baseline / 10.0,
                baseline / 5.0,
            )
        )
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
        "baseline": baseline,
        "peak": peak,
        "minimum": minimum,
        "endpoint": endpoint,
        "dynamic_range": dynamic_range,
        "time_to_peak": time_to_peak,
        "auc": auc,
        "initial_slope": initial_slope,
        "maximum_slope": maximum_slope,
        "fold_change": (peak - baseline) / baseline if baseline > 0 else pd.NA,
        "log2_fold_change": math.log2(endpoint / baseline) if baseline > 0 and endpoint > 0 else pd.NA,
        "Start_Time": 0.0,
        "End_Time": 10.0,
        "Input_Row_Count": 3,
        "Valid_Observation_Count": 3,
        "Missing_Observation_Count": 0,
        "Duplicate_Timestamp_Count": 0,
        "Duplicate_Timestamp_Group_Count": 0,
        "Source_QC_Statuses": "pass",
        "Source_QC_Flags": "",
        "Feature_QC_Flags": "",
    }


def _features(dataframe: pd.DataFrame) -> set[str]:
    if dataframe.empty:
        return set()
    return set(dataframe["feature"].astype(str))


def _summary_row(dataframe: pd.DataFrame, feature: str) -> pd.Series:
    return dataframe.loc[dataframe["feature"].astype(str).eq(feature)].iloc[0]


def _recommendation_for(result: FeatureValidationResult, feature: str) -> str:
    match = result.feature_recommendations.loc[
        result.feature_recommendations["feature"].astype(str).eq(feature),
        "recommendation",
    ]
    return str(match.iloc[0])


def _correlation_for(dataframe: pd.DataFrame, feature_a: str, feature_b: str) -> float:
    mask = (
        dataframe["feature_a"].eq(feature_a)
        & dataframe["feature_b"].eq(feature_b)
    ) | (
        dataframe["feature_a"].eq(feature_b)
        & dataframe["feature_b"].eq(feature_a)
    )
    return float(dataframe.loc[mask, "correlation"].iloc[0])


def _has_violation(result: FeatureValidationResult, feature: str, violation: str) -> bool:
    if result.range_validation_summary.empty:
        return False
    return bool(
        (
            result.range_validation_summary["feature"].astype(str).eq(feature)
            & result.range_validation_summary["violation"].astype(str).eq(violation)
        ).any()
    )
