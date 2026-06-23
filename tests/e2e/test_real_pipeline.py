import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.pipeline.run_pipeline import run_analysis_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
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


def test_real_raw_files_run_through_analysis_pipeline_without_modification() -> None:
    input_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    assert input_files, f"No raw CSV files found in {RAW_DATA_DIR}"
    before_metadata = _file_metadata(input_files)

    with local_test_workspace("real_pipeline") as workspace:
        output_dir = workspace / "outputs"
        result = run_analysis_pipeline(input_files, output_dir)
        print(result)

        assert result["status"] == "success"
        assert result["feature_rows"] > 0
        assert Path(result["report_path"]).exists()

    after_metadata = _file_metadata(input_files)
    assert after_metadata == before_metadata


def _file_metadata(csv_files: list[Path]) -> dict[Path, tuple[int, int]]:
    return {
        csv_file: (csv_file.stat().st_size, csv_file.stat().st_mtime_ns)
        for csv_file in csv_files
    }
