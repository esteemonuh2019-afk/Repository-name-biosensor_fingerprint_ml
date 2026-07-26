import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import openpyxl

from src.data_ingestion.canonical_builder import build_canonical_dataset
from src.data_ingestion.csv_reader import read_biosensor_csv
from src.data_ingestion.excel_reader import read_biosensor_excel
from src.data_ingestion.file_discovery import discover_biosensor_files


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


def create_excel_fixture(path: Path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet.append(["", "bacteria_id", "antibiotic", "concentration", "Experiment", "replicate", "time_min", "luminescence"])
    worksheet.append([None, "BL011", "Lambda Cyclotherin", 5, 1, 1, 0, 100])
    workbook.save(path)
    workbook.close()


def test_discovery_reader_builder_schema_flow_with_synthetic_files() -> None:
    with local_test_workspace("canonical_ingestion_flow") as workspace:
        csv_path = workspace / "BL027ab.csv"
        csv_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence\n"
            "BL027,Lambda Cyclotherin,5,1,1,0,100\n",
            encoding="utf-8",
        )
        excel_path = workspace / "BL011.12hrs.xlsx"
        create_excel_fixture(excel_path)

        discovery = discover_biosensor_files(workspace)
        reader_results = []
        for record in discovery.files:
            if record.extension == ".csv":
                reader_results.append(read_biosensor_csv(record.absolute_path))
            elif record.extension == ".xlsx":
                reader_results.append(read_biosensor_excel(record.absolute_path))

        build_result = build_canonical_dataset(reader_results)

        assert len(discovery.files) == 2
        assert build_result.row_count == 2
        assert set(build_result.dataframe["Source_Type"]) == {"csv", "xlsx"}
        assert "BL027ab" in set(build_result.dataframe["Strain_Original"])
        assert "Lambda Cyclotherin" in set(build_result.dataframe["Chemical_Name_Original"])
        assert build_result.schema_validation is not None
