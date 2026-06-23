import pytest
import pandas as pd

from src.feature_engineering.advanced_features import (
    ADVANCED_FEATURE_COLUMNS,
    calculate_auc_early,
    calculate_auc_late,
    calculate_auc_mid,
    calculate_fold_change,
    calculate_max_derivative,
    calculate_min_derivative,
    calculate_peak_to_baseline_ratio,
    calculate_signal_decay_rate,
    extract_advanced_features,
)


def test_all_advanced_features_calculate() -> None:
    time = [0, 6, 12, 24]
    signal = [10, 20, 15, 5]

    assert calculate_peak_to_baseline_ratio(time, signal) == 2.0
    assert calculate_fold_change(time, signal) == -0.5
    assert calculate_signal_decay_rate(time, signal) == pytest.approx(-15 / 18)


def test_segmented_auc_values_calculate() -> None:
    time = [0, 6, 12, 24]
    signal = [10, 20, 15, 5]

    assert calculate_auc_early(time, signal) == 90.0
    assert calculate_auc_mid(time, signal) == 105.0
    assert calculate_auc_late(time, signal) == 120.0


def test_derivative_features_calculate() -> None:
    time = [0, 6, 12, 24]
    signal = [10, 20, 15, 5]

    assert calculate_max_derivative(time, signal) == pytest.approx(10 / 6)
    assert calculate_min_derivative(time, signal) == pytest.approx(-10 / 12)


def test_extract_advanced_features_output_columns_exist() -> None:
    dataframe = pd.DataFrame(
        {
            "strain": ["BL011", "BL011", "BL011", "BL011", "BL027", "BL027", "BL027", "BL027"],
            "chemical": ["Diazinon", "Diazinon", "Diazinon", "Diazinon", "DEET", "DEET", "DEET", "DEET"],
            "concentration": [5, 5, 5, 5, 50, 50, 50, 50],
            "experiment": ["EXP-001"] * 8,
            "replicate": [1, 1, 1, 1, 1, 1, 1, 1],
            "time": [0, 6, 12, 24, 0, 6, 12, 24],
            "luminescence": [10, 20, 15, 5, 12, 18, 16, 14],
        }
    )

    features = extract_advanced_features(dataframe)

    for column in ADVANCED_FEATURE_COLUMNS:
        assert column in features.columns
    assert len(features) == 2
