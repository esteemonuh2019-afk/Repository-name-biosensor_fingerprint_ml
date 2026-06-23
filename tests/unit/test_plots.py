import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.visualization.plots import (
    plot_dose_response,
    plot_heatmap,
    plot_pca,
    plot_time_course,
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


def test_heatmap_file_is_created() -> None:
    with local_test_workspace("plots_heatmap") as workspace:
        output_path = workspace / "heatmap.png"

        written_path = plot_heatmap(_feature_dataframe(), output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_pca_plot_file_is_created() -> None:
    with local_test_workspace("plots_pca") as workspace:
        output_path = workspace / "pca.png"

        written_path = plot_pca(_feature_dataframe(), output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_dose_response_plot_file_is_created() -> None:
    with local_test_workspace("plots_dose_response") as workspace:
        output_path = workspace / "dose_response.png"

        written_path = plot_dose_response(_feature_dataframe(), output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_time_course_plot_file_is_created() -> None:
    with local_test_workspace("plots_time_course") as workspace:
        output_path = workspace / "time_course.png"

        written_path = plot_time_course(_raw_dataframe(), output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strain": ["BL011", "BL011", "BL027", "BL027"],
            "chemical": ["Diazinon", "DEET", "Diazinon", "DEET"],
            "concentration": [5.0, 50.0, 5.0, 50.0],
            "auc": [6125.0, 5362.5, 4800.0, 5900.0],
            "max_signal": [1250.0, 1100.0, 1180.0, 1210.0],
            "min_signal": [1005.0, 990.0, 970.0, 980.0],
            "time_to_peak": [5.0, 5.0, 10.0, 10.0],
            "initial_slope": [49.0, 22.0, 35.0, 40.0],
            "final_signal": [1180.0, 1080.0, 1175.0, 1200.0],
        }
    )


def _raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strain": ["BL011", "BL011", "BL027", "BL027"],
            "chemical": ["Diazinon", "Diazinon", "DEET", "DEET"],
            "concentration": [5.0, 5.0, 50.0, 50.0],
            "replicate": [1, 1, 1, 1],
            "time": [0.0, 5.0, 0.0, 5.0],
            "luminescence": [1005.0, 1250.0, 990.0, 1100.0],
        }
    )
