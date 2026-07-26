import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from src.data_ingestion.file_discovery import (
    CSV_24H_CANDIDATE,
    EXCEL_12H_CANDIDATE,
    discover_biosensor_files,
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


def touch(path: Path, content: str = "") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_folder_does_not_exist_raises_file_not_found() -> None:
    missing_path = PROJECT_ROOT / "tests" / "tmp" / "missing_discovery_folder"

    with pytest.raises(FileNotFoundError):
        discover_biosensor_files(missing_path)


def test_file_path_is_rejected() -> None:
    with local_test_workspace("discovery_file_path") as workspace:
        file_path = touch(workspace / "not_a_folder.csv")

        with pytest.raises(NotADirectoryError):
            discover_biosensor_files(file_path)


def test_empty_folder_returns_no_files_warning() -> None:
    with local_test_workspace("discovery_empty_folder") as workspace:
        result = discover_biosensor_files(workspace)

        assert result.files == []
        assert result.warnings == [
            f"No supported biosensor source files found in: {workspace.resolve()}"
        ]


def test_csv_discovery_records_expected_metadata() -> None:
    with local_test_workspace("discovery_csv") as workspace:
        csv_path = touch(workspace / "BL011.csv", "placeholder")

        result = discover_biosensor_files(workspace)

        assert len(result.files) == 1
        record = result.files[0]
        assert record.absolute_path == str(csv_path.resolve())
        assert record.filename == "BL011.csv"
        assert record.extension == ".csv"
        assert record.source_type == CSV_24H_CANDIDATE
        assert record.file_size_bytes == len("placeholder")
        assert record.strain_label_from_filename == "BL011"
        assert record.duration_hint_from_filename is None


def test_xlsx_discovery_records_expected_metadata() -> None:
    with local_test_workspace("discovery_xlsx") as workspace:
        touch(workspace / "BL011.12hrs.xlsx")

        result = discover_biosensor_files(workspace)

        assert len(result.files) == 1
        record = result.files[0]
        assert record.extension == ".xlsx"
        assert record.source_type == EXCEL_12H_CANDIDATE
        assert record.strain_label_from_filename == "BL011"
        assert record.duration_hint_from_filename == "12hrs"


def test_mixed_csv_and_xlsx_are_discovered() -> None:
    with local_test_workspace("discovery_mixed") as workspace:
        touch(workspace / "BL011.csv")
        touch(workspace / "BL011.12hrs.xlsx")

        result = discover_biosensor_files(workspace)

        assert [record.source_type for record in result.files] == [
            EXCEL_12H_CANDIDATE,
            CSV_24H_CANDIDATE,
        ]


def test_uppercase_extensions_are_supported() -> None:
    with local_test_workspace("discovery_uppercase") as workspace:
        touch(workspace / "BL011.CSV")
        touch(workspace / "BL032.12H.XLSX")

        result = discover_biosensor_files(workspace)

        assert [record.extension for record in result.files] == [".csv", ".xlsx"]
        assert [record.source_type for record in result.files] == [
            CSV_24H_CANDIDATE,
            EXCEL_12H_CANDIDATE,
        ]


def test_temporary_excel_files_are_ignored() -> None:
    with local_test_workspace("discovery_temp_excel") as workspace:
        touch(workspace / "~$BL011.12hrs.xlsx")
        touch(workspace / "BL011.12hrs.xlsx")

        result = discover_biosensor_files(workspace)

        assert [record.filename for record in result.files] == ["BL011.12hrs.xlsx"]


def test_unrelated_and_hidden_files_are_ignored() -> None:
    with local_test_workspace("discovery_ignored_files") as workspace:
        touch(workspace / ".hidden.csv")
        touch(workspace / "notes.txt")
        touch(workspace / "BL011.csv")
        (workspace / "nested").mkdir()

        result = discover_biosensor_files(workspace)

        assert [record.filename for record in result.files] == ["BL011.csv"]


def test_results_are_sorted_deterministically() -> None:
    with local_test_workspace("discovery_sorting") as workspace:
        touch(workspace / "BL032.csv")
        touch(workspace / "BL011.csv")
        touch(workspace / "BL029.12hrs.xlsx")

        result = discover_biosensor_files(workspace)

        assert [record.filename for record in result.files] == [
            "BL011.csv",
            "BL029.12hrs.xlsx",
            "BL032.csv",
        ]


def test_bl027_is_detected() -> None:
    with local_test_workspace("discovery_bl027") as workspace:
        touch(workspace / "BL027.12hrs.xlsx")

        result = discover_biosensor_files(workspace)

        assert result.files[0].strain_label_from_filename == "BL027"


def test_bl027ab_is_preserved() -> None:
    with local_test_workspace("discovery_bl027ab") as workspace:
        touch(workspace / "BL027ab.csv")

        result = discover_biosensor_files(workspace)

        assert result.files[0].strain_label_from_filename == "BL027ab"


def test_filename_suffixes_do_not_break_detection() -> None:
    with local_test_workspace("discovery_suffixes") as workspace:
        touch(workspace / "BL011(9).csv")
        touch(workspace / "BL032.12hrs(2).xlsx")

        result = discover_biosensor_files(workspace)

        assert [record.strain_label_from_filename for record in result.files] == [
            "BL011",
            "BL032",
        ]


def test_files_without_detectable_strain_generate_warning() -> None:
    with local_test_workspace("discovery_missing_strain") as workspace:
        touch(workspace / "source.csv")

        result = discover_biosensor_files(workspace)

        assert result.files[0].strain_label_from_filename is None
        assert result.warnings == [
            "Could not infer expected strain from filename: source.csv"
        ]


def test_duplicate_strain_source_type_candidates_generate_warning() -> None:
    with local_test_workspace("discovery_duplicates") as workspace:
        touch(workspace / "BL011.csv")
        touch(workspace / "BL011(9).csv")

        result = discover_biosensor_files(workspace)

        assert result.warnings == [
            "Duplicate strain/source-type candidates found for "
            "BL011 / csv_24h_candidate: BL011(9).csv, BL011.csv"
        ]


def test_discovery_does_not_modify_source_files() -> None:
    with local_test_workspace("discovery_no_modification") as workspace:
        csv_path = touch(workspace / "BL011.csv", "synthetic placeholder")
        before = (csv_path.read_bytes(), csv_path.stat().st_mtime_ns, csv_path.stat().st_size)

        discover_biosensor_files(workspace)

        after = (csv_path.read_bytes(), csv_path.stat().st_mtime_ns, csv_path.stat().st_size)
        assert after == before
