"""Preprocessing helpers for cleaned biosensor datasets."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


TARGET_CHEMICALS: tuple[str, ...] = (
    "Diazinon",
    "DEET",
    "Propoxur",
    "Metaldehyde",
    "Boric Acid",
    "Trimethoprim",
)

EXCLUDED_CHEMICALS: tuple[str, ...] = ("Monensin",)

CHEMICAL_NAME_MAP: dict[str, str] = {
    chemical.casefold(): chemical for chemical in (*TARGET_CHEMICALS, *EXCLUDED_CHEMICALS)
}

STRAIN_NAME_MAP: dict[str, str] = {
    "BL027ab": "BL027",
}


def standardize_strain_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize known strain aliases while preserving valid strain names."""

    cleaned = dataframe.copy()
    cleaned["strain"] = cleaned["strain"].replace(STRAIN_NAME_MAP)
    return cleaned


def standardize_chemical_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and canonicalize recognized chemical names."""

    cleaned = dataframe.copy()
    cleaned["chemical"] = cleaned["chemical"].map(_standardize_chemical_name)
    return cleaned


def remove_excluded_chemicals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove chemicals excluded by the SSDD, including Monensin."""

    cleaned = standardize_chemical_names(dataframe)
    return cleaned[~cleaned["chemical"].isin(EXCLUDED_CHEMICALS)].reset_index(drop=True)


def filter_target_chemicals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep only target chemicals listed in the SSDD."""

    cleaned = standardize_chemical_names(dataframe)
    return cleaned[cleaned["chemical"].isin(TARGET_CHEMICALS)].reset_index(drop=True)


def parse_concentration(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert concentration values to numeric values, coercing invalid values to NaN."""

    cleaned = dataframe.copy()
    cleaned["concentration"] = cleaned["concentration"].map(_parse_concentration_value)
    return cleaned


def _standardize_chemical_name(value: Any) -> Any:
    if pd.isna(value):
        return value

    stripped_value = str(value).strip()
    return CHEMICAL_NAME_MAP.get(stripped_value.casefold(), stripped_value)


def _parse_concentration_value(value: Any) -> float:
    if pd.isna(value):
        return float("nan")

    if isinstance(value, int | float):
        return float(value)

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).strip())
    if not match:
        return float("nan")

    return float(match.group(0))
