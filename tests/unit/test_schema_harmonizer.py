import pandas as pd

from src.preprocessing.schema_harmonizer import (
    REQUIRED_HARMONIZED_COLUMNS,
    harmonize_schema,
    validate_harmonized_schema,
)


def test_raw_columns_are_renamed_correctly() -> None:
    dataframe = pd.DataFrame(
        {
            " bacteria_id ": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": [5],
            "Experiment": ["EXP-001"],
            "replicate": [1],
            "time_min": [0],
            "luminescence": [1000],
        }
    )

    harmonized = harmonize_schema(dataframe)

    assert list(harmonized.columns) == list(REQUIRED_HARMONIZED_COLUMNS)
    assert harmonized.loc[0, "strain"] == "BL011"
    assert harmonized.loc[0, "chemical"] == "Diazinon"
    assert harmonized.loc[0, "experiment"] == "EXP-001"
    assert harmonized.loc[0, "time"] == 0


def test_unnamed_columns_are_dropped() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": [5],
            "Experiment": ["EXP-001"],
            "replicate": [1],
            "time_min": [0],
            "luminescence": [1000],
            "Unnamed: 7": [None],
        }
    )

    harmonized = harmonize_schema(dataframe)

    assert "Unnamed: 7" not in harmonized.columns


def test_required_harmonized_columns_are_present() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": [5],
            "Experiment": ["EXP-001"],
            "replicate": [1],
            "time_min": [0],
            "luminescence": [1000],
        }
    )

    harmonized = harmonize_schema(dataframe)

    assert validate_harmonized_schema(harmonized) == []


def test_missing_required_columns_are_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": [5],
        }
    )

    harmonized = harmonize_schema(dataframe)

    assert validate_harmonized_schema(harmonized) == [
        "experiment",
        "replicate",
        "time",
        "luminescence",
    ]
