import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import openpyxl
import pandas as pd
import pytest

from src.data_ingestion.excel_reader import read_biosensor_excel


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


def create_workbook(
    path: Path,
    rows: list[list[object]] | None = None,
    extra_sheet: bool = False,
) -> Path:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    for row in rows or []:
        worksheet.append(row)
    if extra_sheet:
        second = workbook.create_sheet("Second")
        second.append(["note"])
        second.append(["not active"])
    workbook.save(path)
    workbook.close()
    return path


def standard_rows() -> list[list[object]]:
    return [
        ["bacteria_id", "antibiotic", "concentration", "Experiment", "replicate", "time_min", "luminescence"],
        ["BL011", "Lambda Cyclotherin", 5, 1, 1, 0, 100],
    ]


def test_missing_workbook_raises_file_not_found() -> None:
    missing_path = PROJECT_ROOT / "tests" / "tmp" / "missing_workbook.xlsx"

    with pytest.raises(FileNotFoundError):
        read_biosensor_excel(missing_path)


def test_wrong_extension_is_rejected() -> None:
    with local_test_workspace("excel_reader_wrong_extension") as workspace:
        wrong_path = workspace / "BL011.csv"
        wrong_path.write_text("not an xlsx", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\.xlsx"):
            read_biosensor_excel(wrong_path)


def test_empty_workbook_is_rejected() -> None:
    with local_test_workspace("excel_reader_empty") as workspace:
        workbook_path = create_workbook(workspace / "BL011.xlsx")

        with pytest.raises(ValueError, match="empty"):
            read_biosensor_excel(workbook_path)


def test_workbook_with_one_sheet_is_read() -> None:
    with local_test_workspace("excel_reader_one_sheet") as workspace:
        workbook_path = create_workbook(workspace / "BL011.xlsx", standard_rows())

        result = read_biosensor_excel(workbook_path)

        assert result.worksheet_names == ["Sheet1"]
        assert result.active_worksheet == "Sheet1"
        assert result.row_count == 1
        assert result.column_count == 7


def test_workbook_with_multiple_sheets_preserves_names_and_active_sheet() -> None:
    with local_test_workspace("excel_reader_multiple_sheets") as workspace:
        workbook_path = create_workbook(
            workspace / "BL011.xlsx",
            standard_rows(),
            extra_sheet=True,
        )

        result = read_biosensor_excel(workbook_path)

        assert result.worksheet_names == ["Sheet1", "Second"]
        assert result.active_worksheet == "Sheet1"
        assert result.dataframe.loc[0, "antibiotic"] == "Lambda Cyclotherin"


def test_bl011_detection() -> None:
    with local_test_workspace("excel_reader_bl011") as workspace:
        workbook_path = create_workbook(workspace / "BL011.12hrs.xlsx", standard_rows())

        result = read_biosensor_excel(workbook_path)

        assert result.inferred_strain == "BL011"


def test_bl027_preservation() -> None:
    with local_test_workspace("excel_reader_bl027") as workspace:
        workbook_path = create_workbook(workspace / "BL027.12hrs.xlsx", standard_rows())

        result = read_biosensor_excel(workbook_path)

        assert result.inferred_strain == "BL027"


def test_workbook_not_modified() -> None:
    with local_test_workspace("excel_reader_not_modified") as workspace:
        workbook_path = create_workbook(workspace / "BL011.xlsx", standard_rows())
        before = (
            workbook_path.read_bytes(),
            workbook_path.stat().st_mtime_ns,
            workbook_path.stat().st_size,
        )

        read_biosensor_excel(workbook_path)

        after = (
            workbook_path.read_bytes(),
            workbook_path.stat().st_mtime_ns,
            workbook_path.stat().st_size,
        )
        assert after == before


def test_dataframe_returned() -> None:
    with local_test_workspace("excel_reader_dataframe") as workspace:
        workbook_path = create_workbook(workspace / "BL011.xlsx", standard_rows())

        result = read_biosensor_excel(workbook_path)

        assert isinstance(result.dataframe, pd.DataFrame)
        assert result.dataframe.loc[0, "antibiotic"] == "Lambda Cyclotherin"
