import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.data_ingestion.loader import (
    load_csv,
    load_multiple_csv,
    validate_required_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"


@contextmanager
def local_test_workspace(test_name: str) -> Iterator[Path]:
    workspace = TEST_TMP_ROOT / test_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    try:
        yield workspace
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
        if TEST_TMP_ROOT.exists() and not any(TEST_TMP_ROOT.iterdir()):
            TEST_TMP_ROOT.rmdir()


def test_load_valid_csv() -> None:
    with local_test_workspace("loader_valid_csv") as workspace:
        csv_path = workspace / "valid.csv"
        csv_path.write_text(
            "strain,chemical,luminescence\nBL011,Diazinon,1000\n",
            encoding="utf-8",
        )

        dataframe = load_csv(csv_path)

        assert list(dataframe.columns) == ["strain", "chemical", "luminescence"]
        assert len(dataframe) == 1
        assert dataframe.loc[0, "strain"] == "BL011"


def test_missing_file_raises_error() -> None:
    missing_path = PROJECT_ROOT / "tests" / "tmp" / "missing_loader_file.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_path)


def test_empty_csv_raises_error() -> None:
    with local_test_workspace("loader_empty_csv") as workspace:
        csv_path = workspace / "empty.csv"
        csv_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            load_csv(csv_path)


def test_multiple_csv_files_combine_correctly() -> None:
    with local_test_workspace("loader_multiple_csv") as workspace:
        first_path = workspace / "first.csv"
        second_path = workspace / "second.csv"
        first_path.write_text(
            "strain,chemical,luminescence\nBL011,Diazinon,1000\n",
            encoding="utf-8",
        )
        second_path.write_text(
            "strain,chemical,luminescence\nBL027,DEET,980\n",
            encoding="utf-8",
        )

        dataframe = load_multiple_csv([first_path, second_path])

        assert len(dataframe) == 2
        assert "source_file" in dataframe.columns
        assert dataframe["source_file"].tolist() == [str(first_path), str(second_path)]
        assert dataframe["strain"].tolist() == ["BL011", "BL027"]


def test_required_columns_validation_works() -> None:
    dataframe = pd.DataFrame(
        {
            "strain": ["BL011"],
            "chemical": ["Diazinon"],
            "luminescence": [1000],
        }
    )

    missing_columns = validate_required_columns(
        dataframe,
        ["strain", "chemical", "concentration", "luminescence"],
    )

    assert missing_columns == ["concentration"]
    assert validate_required_columns(dataframe, ["strain", "chemical"]) == []
