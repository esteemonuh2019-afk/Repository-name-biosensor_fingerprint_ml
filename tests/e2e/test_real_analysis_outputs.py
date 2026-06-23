import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_real_analysis.py"
FEATURES_PATH = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "model_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "scientific_performance_report.md"


def test_real_analysis_outputs_are_created_without_modifying_raw_files() -> None:
    raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    assert raw_files, f"No raw CSV files found in {RAW_DATA_DIR}"
    before_metadata = _file_metadata(raw_files)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)

    assert FEATURES_PATH.exists()
    assert REPORT_PATH.exists()
    assert list(FIGURES_DIR.glob("*.png"))
    assert METRICS_PATH.exists()

    after_metadata = _file_metadata(raw_files)
    assert after_metadata == before_metadata


def _file_metadata(csv_files: list[Path]) -> dict[Path, tuple[int, int]]:
    return {
        csv_file: (csv_file.stat().st_size, csv_file.stat().st_mtime_ns)
        for csv_file in csv_files
    }
