"""Dataset validation utilities for biosensor data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import pandas as pd


ALLOWED_CHEMICALS: tuple[str, ...] = (
    "Diazinon",
    "DEET",
    "Propoxur",
    "Metaldehyde",
    "Boric Acid",
    "Trimethoprim",
)

ALLOWED_STRAINS: tuple[str, ...] = (
    "BL011",
    "BL027",
    "BL029",
    "BL030",
    "BL031",
    "BL032",
)

EXPECTED_CONCENTRATIONS: tuple[float, ...] = (500.0, 50.0, 5.0, 0.5, 0.05)


@dataclass(frozen=True)
class SchemaValidationResult:
    valid: bool
    missing_columns: list[str]


def validate_schema(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> SchemaValidationResult:
    """Validate that all required columns are present."""

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    return SchemaValidationResult(valid=not missing_columns, missing_columns=missing_columns)


def validate_target_chemicals(dataframe: pd.DataFrame) -> list[str]:
    """Return chemicals outside the allowed target chemical list."""

    return _unexpected_values(dataframe, "chemical", set(ALLOWED_CHEMICALS))


def validate_strains(dataframe: pd.DataFrame) -> list[str]:
    """Return strains outside the allowed biosensor strain list."""

    return _unexpected_values(dataframe, "strain", set(ALLOWED_STRAINS))


def validate_concentrations(dataframe: pd.DataFrame) -> list[Any]:
    """Return concentrations outside the expected concentration list."""

    unexpected_values = []
    expected = set(EXPECTED_CONCENTRATIONS)
    for concentration in dataframe["concentration"].drop_duplicates().tolist():
        try:
            parsed_concentration = float(concentration)
        except (TypeError, ValueError):
            unexpected_values.append(concentration)
            continue

        if parsed_concentration not in expected:
            unexpected_values.append(concentration)

    return sorted(unexpected_values, key=str)


def _unexpected_values(
    dataframe: pd.DataFrame,
    column: str,
    allowed_values: set[str],
) -> list[str]:
    values = dataframe[column].drop_duplicates().tolist()
    return sorted((value for value in values if value not in allowed_values), key=str)
