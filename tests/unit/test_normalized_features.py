import pandas as pd

from src.feature_engineering.normalized_features import (
    BASE_FEATURE_COLUMNS,
    EXPERIMENT_ZSCORE_COLUMNS,
    STRAIN_EXPERIMENT_ZSCORE_COLUMNS,
    add_experiment_zscore_features,
    add_strain_experiment_zscore_features,
    get_normalized_feature_columns,
)


def test_experiment_zscore_columns_are_created() -> None:
    normalized_df = add_experiment_zscore_features(_feature_dataframe())

    for column in EXPERIMENT_ZSCORE_COLUMNS:
        assert column in normalized_df.columns


def test_strain_experiment_zscore_columns_are_created() -> None:
    normalized_df = add_strain_experiment_zscore_features(_feature_dataframe())

    for column in STRAIN_EXPERIMENT_ZSCORE_COLUMNS:
        assert column in normalized_df.columns


def test_zero_standard_deviation_does_not_crash() -> None:
    dataframe = _feature_dataframe()
    dataframe.loc[dataframe["experiment"] == "EXP-003", list(BASE_FEATURE_COLUMNS)] = 1.0

    normalized_df = add_experiment_zscore_features(dataframe)

    zero_std_rows = normalized_df["experiment"] == "EXP-003"
    assert normalized_df.loc[zero_std_rows, list(EXPERIMENT_ZSCORE_COLUMNS)].eq(0.0).all().all()


def test_get_normalized_feature_columns_returns_expected_columns() -> None:
    assert get_normalized_feature_columns() == list(
        EXPERIMENT_ZSCORE_COLUMNS + STRAIN_EXPERIMENT_ZSCORE_COLUMNS
    )


def test_original_columns_are_preserved() -> None:
    dataframe = _feature_dataframe()
    normalized_df = add_strain_experiment_zscore_features(
        add_experiment_zscore_features(dataframe)
    )

    for column in dataframe.columns:
        assert column in normalized_df.columns
    assert normalized_df[list(dataframe.columns)].equals(dataframe)


def _feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strain": ["BL011", "BL011", "BL027", "BL027", "BL011", "BL011"],
            "chemical": ["Diazinon", "DEET", "Diazinon", "DEET", "Diazinon", "DEET"],
            "concentration": [5.0, 50.0, 5.0, 50.0, 5.0, 50.0],
            "experiment": ["EXP-001", "EXP-001", "EXP-002", "EXP-002", "EXP-003", "EXP-003"],
            "replicate": [1, 1, 1, 1, 1, 1],
            "auc": [10.0, 20.0, 30.0, 50.0, 100.0, 100.0],
            "max_signal": [1.0, 2.0, 3.0, 5.0, 10.0, 10.0],
            "min_signal": [0.5, 1.0, 1.5, 2.5, 5.0, 5.0],
            "time_to_peak": [5.0, 10.0, 5.0, 10.0, 7.0, 7.0],
            "initial_slope": [0.1, 0.2, 0.3, 0.5, 1.0, 1.0],
            "final_signal": [0.9, 1.8, 2.7, 4.5, 9.0, 9.0],
        }
    )
