"""Raw CSV schema harmonization for biosensor datasets."""

from __future__ import annotations

import pandas as pd


COLUMN_RENAMES: dict[str, str] = {
    "bacteria_id": "strain",
    "antibiotic": "chemical",
    "Experiment": "experiment",
    "time_min": "time",
}

REQUIRED_HARMONIZED_COLUMNS: tuple[str, ...] = (
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "time",
    "luminescence",
)


def harmonize_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Rename inspected raw CSV columns to the SSDD schema and drop Unnamed columns."""

    harmonized = dataframe.copy()
    harmonized.columns = [str(column).strip() for column in harmonized.columns]
    harmonized = harmonized.drop(
        columns=[column for column in harmonized.columns if column.startswith("Unnamed")]
    )
    return harmonized.rename(columns=COLUMN_RENAMES)


def validate_harmonized_schema(dataframe: pd.DataFrame) -> list[str]:
    """Return required SSDD columns missing from a harmonized dataframe."""

    return [
        column
        for column in REQUIRED_HARMONIZED_COLUMNS
        if column not in dataframe.columns
    ]
