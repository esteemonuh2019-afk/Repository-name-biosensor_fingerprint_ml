import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.pipeline.run_pipeline import run_analysis_pipeline


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


def test_pipeline_runs_on_small_valid_csv_fixture() -> None:
    with local_test_workspace("pipeline_valid_csv") as workspace:
        input_path = workspace / "raw.csv"
        output_dir = workspace / "outputs"
        _write_valid_raw_csv(input_path)

        result = run_analysis_pipeline([input_path], output_dir)

        assert result["status"] == "success"
        assert result["raw_rows"] == 6
        assert result["feature_rows"] == 2
        assert result["output_dir"] == str(output_dir)


def test_pipeline_returns_success_status() -> None:
    with local_test_workspace("pipeline_success_status") as workspace:
        input_path = workspace / "raw.csv"
        _write_valid_raw_csv(input_path)

        result = run_analysis_pipeline([input_path], workspace / "outputs")

        assert result["status"] == "success"
        assert result["errors"] == []


def test_pipeline_creates_report_file() -> None:
    with local_test_workspace("pipeline_report_created") as workspace:
        input_path = workspace / "raw.csv"
        _write_valid_raw_csv(input_path)

        result = run_analysis_pipeline([input_path], workspace / "outputs")

        report_path = Path(result["report_path"])
        assert report_path.exists()
        assert "Whole-Cell Biosensor Fingerprint Analysis Report" in report_path.read_text(
            encoding="utf-8"
        )


def test_pipeline_fails_gracefully_when_required_columns_are_missing() -> None:
    with local_test_workspace("pipeline_missing_columns") as workspace:
        input_path = workspace / "missing_columns.csv"
        input_path.write_text(
            "strain,chemical,concentration,experiment,replicate,time\n"
            "BL011,Diazinon,5,EXP-001,1,0\n",
            encoding="utf-8",
        )

        result = run_analysis_pipeline([input_path], workspace / "outputs")

        assert result["status"] == "failed"
        assert result["report_path"] is None
        assert result["feature_rows"] == 0
        assert any("luminescence" in error for error in result["errors"])


def _write_valid_raw_csv(path: Path) -> None:
    path.write_text(
        "strain,chemical,concentration,experiment,replicate,time,luminescence\n"
        "BL011,Diazinon,5,EXP-001,1,0,1005\n"
        "BL011,Diazinon,5,EXP-001,1,5,1250\n"
        "BL011,Diazinon,5,EXP-001,1,10,1180\n"
        "BL027,DEET,50,EXP-001,1,0,990\n"
        "BL027,DEET,50,EXP-001,1,5,1100\n"
        "BL027,DEET,50,EXP-001,1,10,1080\n",
        encoding="utf-8",
    )
