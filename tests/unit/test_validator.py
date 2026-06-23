import pandas as pd

from src.data_validation.validator import (
    validate_concentrations,
    validate_schema,
    validate_strains,
    validate_target_chemicals,
)


def test_valid_schema_passes() -> None:
    dataframe = pd.DataFrame(
        {
            "strain": ["BL011"],
            "chemical": ["Diazinon"],
            "concentration": [5],
        }
    )

    result = validate_schema(dataframe, ["strain", "chemical", "concentration"])

    assert result.valid is True
    assert result.missing_columns == []


def test_missing_column_detected() -> None:
    dataframe = pd.DataFrame(
        {
            "strain": ["BL011"],
            "chemical": ["Diazinon"],
        }
    )

    result = validate_schema(dataframe, ["strain", "chemical", "concentration"])

    assert result.valid is False
    assert result.missing_columns == ["concentration"]


def test_valid_chemicals_pass() -> None:
    dataframe = pd.DataFrame(
        {
            "chemical": [
                "Diazinon",
                "DEET",
                "Propoxur",
                "Metaldehyde",
                "Boric Acid",
                "Trimethoprim",
            ]
        }
    )

    assert validate_target_chemicals(dataframe) == []


def test_invalid_chemical_detected() -> None:
    dataframe = pd.DataFrame({"chemical": ["Diazinon", "Glyphosate"]})

    assert validate_target_chemicals(dataframe) == ["Glyphosate"]


def test_valid_strains_pass() -> None:
    dataframe = pd.DataFrame(
        {
            "strain": [
                "BL011",
                "BL027",
                "BL029",
                "BL030",
                "BL031",
                "BL032",
            ]
        }
    )

    assert validate_strains(dataframe) == []


def test_invalid_strain_detected() -> None:
    dataframe = pd.DataFrame({"strain": ["BL011", "BL999"]})

    assert validate_strains(dataframe) == ["BL999"]


def test_valid_concentrations_pass() -> None:
    dataframe = pd.DataFrame({"concentration": [500, 50, 5, 0.5, 0.05]})

    assert validate_concentrations(dataframe) == []


def test_invalid_concentration_detected() -> None:
    dataframe = pd.DataFrame({"concentration": [500, 25, "not numeric"]})

    assert validate_concentrations(dataframe) == [25, "not numeric"]
