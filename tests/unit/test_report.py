import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.reporting.report import (
    REPORT_TITLE,
    generate_markdown_report,
    generate_validation_summary,
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


def test_markdown_report_file_is_created() -> None:
    with local_test_workspace("report_file_created") as workspace:
        output_path = workspace / "analysis_report.md"

        written_path = generate_markdown_report(output_path, {"Summary": "Complete."})

        assert written_path == output_path
        assert output_path.exists()


def test_report_contains_title() -> None:
    with local_test_workspace("report_contains_title") as workspace:
        output_path = workspace / "analysis_report.md"

        generate_markdown_report(output_path, {"Summary": "Complete."})

        assert f"# {REPORT_TITLE}" in output_path.read_text(encoding="utf-8")


def test_report_contains_all_supplied_sections() -> None:
    with local_test_workspace("report_contains_sections") as workspace:
        output_path = workspace / "analysis_report.md"
        sections = {
            "Data Summary": "Loaded six strains.",
            "Validation Summary": "All checks passed.",
        }

        generate_markdown_report(output_path, sections)
        report_text = output_path.read_text(encoding="utf-8")

        assert "## Data Summary" in report_text
        assert "Loaded six strains." in report_text
        assert "## Validation Summary" in report_text
        assert "All checks passed." in report_text


def test_validation_summary_includes_classification_and_regression_sections() -> None:
    summary = generate_validation_summary(
        {
            "classification": {"accuracy": 0.84, "macro_f1": 0.78},
            "regression": {"r2": 0.79, "rmse": 0.18},
            "scientific_validation": {"cluster_count": 3},
        }
    )

    assert "## Classification Metrics" in summary
    assert "- accuracy: 0.840" in summary
    assert "## Regression Metrics" in summary
    assert "- r2: 0.790" in summary
    assert "## Scientific Validation Metrics" in summary
