import pandas as pd

from src.preprocessing.cleaner import (
    filter_target_chemicals,
    parse_concentration,
    remove_excluded_chemicals,
    standardize_chemical_names,
    standardize_strain_names,
)


def test_bl027ab_becomes_bl027() -> None:
    dataframe = pd.DataFrame({"strain": ["BL011", "BL027ab", "BL032"]})

    cleaned = standardize_strain_names(dataframe)

    assert cleaned["strain"].tolist() == ["BL011", "BL027", "BL032"]


def test_chemical_names_are_stripped_and_canonicalized() -> None:
    dataframe = pd.DataFrame({"chemical": [" diazinon ", "deet", "Boric acid"]})

    cleaned = standardize_chemical_names(dataframe)

    assert cleaned["chemical"].tolist() == ["Diazinon", "DEET", "Boric Acid"]


def test_monensin_is_removed() -> None:
    dataframe = pd.DataFrame({"chemical": ["Diazinon", " Monensin ", "DEET"]})

    cleaned = remove_excluded_chemicals(dataframe)

    assert cleaned["chemical"].tolist() == ["Diazinon", "DEET"]


def test_target_chemicals_are_retained() -> None:
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

    cleaned = filter_target_chemicals(dataframe)

    assert cleaned["chemical"].tolist() == dataframe["chemical"].tolist()


def test_non_target_chemicals_are_removed() -> None:
    dataframe = pd.DataFrame({"chemical": ["Diazinon", "Glyphosate", "Control", "DEET"]})

    cleaned = filter_target_chemicals(dataframe)

    assert cleaned["chemical"].tolist() == ["Diazinon", "DEET"]


def test_concentration_strings_are_converted_to_numeric() -> None:
    dataframe = pd.DataFrame(
        {
            "concentration": [
                "500",
                "500 \u03bcg/mL",
                "50 ug/mL",
                "0.05",
            ]
        }
    )

    cleaned = parse_concentration(dataframe)

    assert cleaned["concentration"].tolist() == [500.0, 500.0, 50.0, 0.05]


def test_invalid_concentration_becomes_nan() -> None:
    dataframe = pd.DataFrame({"concentration": ["not numeric"]})

    cleaned = parse_concentration(dataframe)

    assert pd.isna(cleaned.loc[0, "concentration"])
