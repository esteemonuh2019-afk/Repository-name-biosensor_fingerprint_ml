import pandas as pd

from src.feature_engineering.features import (
    calculate_auc,
    calculate_final_signal,
    calculate_initial_slope,
    calculate_max_signal,
    calculate_min_signal,
    calculate_time_to_peak,
    extract_features,
)


EXPECTED_FEATURE_COLUMNS = [
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "auc",
    "max_signal",
    "min_signal",
    "time_to_peak",
    "initial_slope",
    "final_signal",
]


def test_auc_calculation_matches_manual_expected_value() -> None:
    auc = calculate_auc(time=[0, 5, 10], values=[100, 200, 150])

    assert auc == 1625.0


def test_max_signal_correct() -> None:
    assert calculate_max_signal([100, 200, 150]) == 200.0


def test_min_signal_correct() -> None:
    assert calculate_min_signal([100, 200, 150]) == 100.0


def test_time_to_peak_correct() -> None:
    assert calculate_time_to_peak(time=[0, 5, 10], values=[100, 200, 150]) == 5.0


def test_initial_slope_correct() -> None:
    assert calculate_initial_slope(time=[0, 5, 10], values=[100, 200, 150]) == 20.0


def test_final_signal_correct() -> None:
    assert calculate_final_signal([100, 200, 150]) == 150.0


def test_extract_features_returns_expected_columns_and_row_count() -> None:
    dataframe = pd.DataFrame(
        {
            "strain": ["BL011", "BL011", "BL011", "BL027", "BL027", "BL027"],
            "chemical": ["Diazinon", "Diazinon", "Diazinon", "DEET", "DEET", "DEET"],
            "concentration": [5, 5, 5, 50, 50, 50],
            "experiment": ["EXP-001"] * 6,
            "replicate": [1, 1, 1, 1, 1, 1],
            "time": [0, 5, 10, 0, 5, 10],
            "luminescence": [1005, 1250, 1180, 990, 1100, 1080],
        }
    )

    features = extract_features(dataframe)

    assert list(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert len(features) == 2
    assert features.loc[0, "auc"] == 11712.5
    assert features.loc[0, "max_signal"] == 1250.0
    assert features.loc[0, "min_signal"] == 1005.0
    assert features.loc[0, "time_to_peak"] == 5.0
    assert features.loc[0, "initial_slope"] == 49.0
    assert features.loc[0, "final_signal"] == 1180.0
