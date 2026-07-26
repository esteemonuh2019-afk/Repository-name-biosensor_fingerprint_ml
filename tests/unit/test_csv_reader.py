import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.data_ingestion.csv_reader import read_biosensor_csv


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


def write_text(path: Path, content: str, encoding: str = "utf-8") -> Path:
    path.write_text(content, encoding=encoding)
    return path


def valid_csv_text(delimiter: str = ",") -> str:
    return delimiter.join(
        [
            "bacteria_id",
            "antibiotic",
            "concentration",
            "Experiment",
            "replicate",
            "time_min",
            "luminescence",
        ]
    ) + "\n" + delimiter.join(["BL011", "Diazinon", "5", "1", "1", "0", "100"])


def test_missing_file_raises_file_not_found() -> None:
    missing_path = PROJECT_ROOT / "tests" / "tmp" / "missing_reader_file.csv"

    with pytest.raises(FileNotFoundError):
        read_biosensor_csv(missing_path)


def test_wrong_file_extension_is_rejected() -> None:
    with local_test_workspace("csv_reader_wrong_extension") as workspace:
        text_path = write_text(workspace / "BL011.txt", valid_csv_text())

        with pytest.raises(ValueError, match=r"\.csv"):
            read_biosensor_csv(text_path)


def test_empty_csv_is_rejected() -> None:
    with local_test_workspace("csv_reader_empty") as workspace:
        csv_path = write_text(workspace / "BL011.csv", "")

        with pytest.raises(ValueError, match="empty"):
            read_biosensor_csv(csv_path)


def test_utf8_csv_is_read() -> None:
    with local_test_workspace("csv_reader_utf8") as workspace:
        csv_path = write_text(workspace / "BL011.csv", valid_csv_text(), encoding="utf-8")

        result = read_biosensor_csv(csv_path)

        assert result.encoding == "utf-8"
        assert result.dataframe.loc[0, "bacteria_id"] == "BL011"


def test_utf8_sig_csv_is_read_without_bom_column() -> None:
    with local_test_workspace("csv_reader_utf8_sig") as workspace:
        csv_path = write_text(
            workspace / "BL011.csv",
            valid_csv_text(),
            encoding="utf-8-sig",
        )

        result = read_biosensor_csv(csv_path)

        assert result.encoding == "utf-8-sig"
        assert result.original_columns[0] == "bacteria_id"


def test_cp1252_csv_is_read() -> None:
    with local_test_workspace("csv_reader_cp1252") as workspace:
        csv_path = workspace / "BL011.csv"
        csv_path.write_bytes(
            (
                "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence\n"
                "BL011,Dose\xa0Label,5,1,1,0,100\n"
            ).encode("cp1252")
        )

        result = read_biosensor_csv(csv_path)

        assert result.encoding == "cp1252"
        assert result.dataframe.loc[0, "antibiotic"] == "Dose\xa0Label"


def test_comma_delimiter_is_detected() -> None:
    with local_test_workspace("csv_reader_comma") as workspace:
        csv_path = write_text(workspace / "BL011.csv", valid_csv_text(","))

        result = read_biosensor_csv(csv_path)

        assert result.delimiter == ","


def test_semicolon_delimiter_is_detected() -> None:
    with local_test_workspace("csv_reader_semicolon") as workspace:
        csv_path = write_text(workspace / "BL011.csv", valid_csv_text(";"))

        result = read_biosensor_csv(csv_path)

        assert result.delimiter == ";"
        assert result.dataframe.loc[0, "antibiotic"] == "Diazinon"


def test_strain_inference_for_bl011() -> None:
    with local_test_workspace("csv_reader_bl011") as workspace:
        csv_path = write_text(workspace / "BL011.csv", valid_csv_text())

        result = read_biosensor_csv(csv_path)

        assert result.strain_label_original == "BL011"


def test_bl027ab_is_preserved_from_filename() -> None:
    with local_test_workspace("csv_reader_bl027ab") as workspace:
        csv_path = write_text(workspace / "BL027ab.csv", valid_csv_text())

        result = read_biosensor_csv(csv_path)

        assert result.strain_label_original == "BL027ab"


def test_original_columns_are_preserved() -> None:
    with local_test_workspace("csv_reader_columns") as workspace:
        csv_path = write_text(
            workspace / "BL011.csv",
            "bacteria_id,antibiotic,Original Case\nBL011,Diazinon,value\n",
        )

        result = read_biosensor_csv(csv_path)

        assert result.original_columns == ["bacteria_id", "antibiotic", "Original Case"]
        assert list(result.dataframe.columns) == result.original_columns


def test_duplicate_column_warning_is_returned() -> None:
    with local_test_workspace("csv_reader_duplicate_columns") as workspace:
        csv_path = write_text(
            workspace / "BL011.csv",
            "bacteria_id,bacteria_id,luminescence\nBL011,BL011,100\n",
        )

        result = read_biosensor_csv(csv_path)

        assert any("Duplicate column name detected: bacteria_id" in warning for warning in result.warnings)


def test_empty_column_warning_is_returned() -> None:
    with local_test_workspace("csv_reader_empty_columns") as workspace:
        csv_path = write_text(
            workspace / "BL011.csv",
            "bacteria_id,,luminescence\nBL011,,100\n",
        )

        result = read_biosensor_csv(csv_path)

        assert any("Completely empty columns detected: 2 (<empty>)" in warning for warning in result.warnings)


def test_row_and_column_counts_are_correct() -> None:
    with local_test_workspace("csv_reader_counts") as workspace:
        csv_path = write_text(
            workspace / "BL011.csv",
            valid_csv_text() + "\nBL011,Diazinon,5,1,1,5,105",
        )

        result = read_biosensor_csv(csv_path)

        assert result.row_count == 2
        assert result.column_count == 7


def test_source_file_is_not_modified() -> None:
    with local_test_workspace("csv_reader_no_modification") as workspace:
        csv_path = write_text(workspace / "BL011.csv", valid_csv_text())
        before = (csv_path.read_bytes(), csv_path.stat().st_mtime_ns, csv_path.stat().st_size)

        read_biosensor_csv(csv_path)

        after = (csv_path.read_bytes(), csv_path.stat().st_mtime_ns, csv_path.stat().st_size)
        assert after == before


def test_readable_dataframe_is_returned() -> None:
    with local_test_workspace("csv_reader_dataframe") as workspace:
        csv_path = write_text(workspace / "BL011.csv", valid_csv_text())

        result = read_biosensor_csv(csv_path)

        assert isinstance(result.dataframe, pd.DataFrame)
        assert not result.dataframe.empty


def test_malformed_rows_are_reported_without_dropping_rows() -> None:
    with local_test_workspace("csv_reader_malformed") as workspace:
        csv_path = write_text(
            workspace / "BL011.csv",
            "bacteria_id,antibiotic,luminescence\nBL011,Diazinon,100\nBL011,Diazinon\n",
        )

        result = read_biosensor_csv(csv_path)

        assert result.row_count == 2
        assert result.dataframe.iloc[1].tolist() == ["BL011", "Diazinon", ""]
        assert any("Malformed row lengths detected" in warning for warning in result.warnings)
