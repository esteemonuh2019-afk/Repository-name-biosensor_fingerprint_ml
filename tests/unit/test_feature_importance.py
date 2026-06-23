import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import pandas as pd

from src.model_evaluation.feature_importance import (
    calculate_random_forest_feature_importance,
    generate_pca_by_chemical,
    generate_pca_by_experiment,
    plot_feature_importance,
)
from src.model_training.models import NUMERIC_FEATURE_COLUMNS


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


def test_feature_importance_dataframe_generated() -> None:
    importance_df = calculate_random_forest_feature_importance(_feature_dataframe())

    assert list(importance_df.columns) == ["feature", "importance"]
    assert set(importance_df["feature"]) == set(NUMERIC_FEATURE_COLUMNS)
    assert importance_df["importance"].is_monotonic_decreasing


def test_importance_values_sum_approximately_to_one() -> None:
    importance_df = calculate_random_forest_feature_importance(_feature_dataframe())

    assert importance_df["importance"].sum() == pytest.approx(1.0)


def test_feature_importance_figure_created() -> None:
    with local_test_workspace("feature_importance_plot") as workspace:
        importance_df = calculate_random_forest_feature_importance(_feature_dataframe())
        output_path = workspace / "feature_importance.png"

        written_path = plot_feature_importance(importance_df, output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_pca_by_chemical_figure_created() -> None:
    with local_test_workspace("pca_by_chemical") as workspace:
        output_path = workspace / "pca_by_chemical.png"

        written_path = generate_pca_by_chemical(_feature_dataframe(), output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_pca_by_experiment_figure_created() -> None:
    with local_test_workspace("pca_by_experiment") as workspace:
        output_path = workspace / "pca_by_experiment.png"

        written_path = generate_pca_by_experiment(_feature_dataframe(), output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for experiment, offset in (("EXP-001", 0.0), ("EXP-002", 20.0), ("EXP-003", -20.0)):
        rows.extend(
            [
                _feature_row(experiment, "Diazinon", 5.0, 6000.0 + offset, 1250.0 + offset),
                _feature_row(experiment, "Diazinon", 50.0, 8200.0 + offset, 1500.0 + offset),
                _feature_row(experiment, "DEET", 5.0, 4300.0 + offset, 980.0 + offset),
                _feature_row(experiment, "DEET", 50.0, 5700.0 + offset, 1160.0 + offset),
                _feature_row(experiment, "Propoxur", 5.0, 5200.0 + offset, 1100.0 + offset),
                _feature_row(experiment, "Propoxur", 50.0, 7100.0 + offset, 1380.0 + offset),
            ]
        )
    return pd.DataFrame(rows)


def _feature_row(
    experiment: str,
    chemical: str,
    concentration: float,
    auc: float,
    max_signal: float,
) -> dict[str, float | str]:
    return {
        "strain": "BL011",
        "chemical": chemical,
        "concentration": concentration,
        "experiment": experiment,
        "replicate": 1,
        "auc": auc,
        "max_signal": max_signal,
        "min_signal": max_signal - 300.0,
        "time_to_peak": 5.0 if chemical == "Diazinon" else 10.0,
        "initial_slope": (max_signal - 800.0) / 5.0,
        "final_signal": max_signal - 80.0,
    }
