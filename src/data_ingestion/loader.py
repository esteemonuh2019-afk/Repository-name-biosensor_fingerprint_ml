"""CSV loading utilities for biosensor raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        dataframe = pd.read_csv(path)
    except EmptyDataError as error:
        raise ValueError(f"CSV file is empty: {path}") from error

    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {path}")

    return dataframe


def load_multiple_csv(file_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and combine multiple CSV files with a source_file column."""

    paths = list(file_paths)
    if not paths:
        raise ValueError("At least one CSV file path is required.")

    dataframes = []
    for file_path in paths:
        dataframe = load_csv(file_path).copy()
        dataframe["source_file"] = str(Path(file_path))
        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> list[str]:
    """Return required columns missing from a DataFrame."""

    return [column for column in required_columns if column not in dataframe.columns]
