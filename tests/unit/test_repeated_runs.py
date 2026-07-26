from pathlib import Path
import shutil
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from src.model_evaluation.repeated_runs import (
    DEFAULT_SEEDS,
    REPEATED_RUN_METRICS,
    create_repeated_run_boxplot,
    run_repeated_seed_evaluation,
    summarize_repeated_run_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"
REQUIRED_RUN_COLUMNS = {"seed", "accuracy", "precision", "recall", "f1"}
REQUIRED_SUMMARY_COLUMNS = {"metric", "mean", "std", "min", "max"}


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


def test_repeated_seed_evaluation_returns_required_columns() -> None:
    result = run_repeated_seed_evaluation(_feature_dataframe(), seeds=[1, 7, 11])

    assert REQUIRED_RUN_COLUMNS <= set(result.columns)
    assert result["seed"].tolist() == [1, 7, 11]


def test_default_seed_list_matches_robustness_plan() -> None:
    assert DEFAULT_SEEDS == (1, 7, 11, 21, 42, 101, 123, 202, 555, 999)


def test_summary_contains_required_statistics_for_each_metric() -> None:
    run_metrics = run_repeated_seed_evaluation(_feature_dataframe(), seeds=[1, 7, 11])

    summary = summarize_repeated_run_metrics(run_metrics)

    assert REQUIRED_SUMMARY_COLUMNS <= set(summary.columns)
    assert set(summary["metric"]) == set(REPEATED_RUN_METRICS)
    assert summary["std"].ge(0).all()


def test_boxplot_file_is_created() -> None:
    run_metrics = run_repeated_seed_evaluation(_feature_dataframe(), seeds=[1, 7, 11])
    with local_test_workspace("repeated_run_boxplot") as workspace:
        output_path = workspace / "repeated_run_boxplot.png"

        written_path = create_repeated_run_boxplot(run_metrics, output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for replicate in range(12):
        rows.append(
            _feature_row(
                chemical="Diazinon",
                auc=7000.0 + replicate * 20.0,
                time_to_peak=5.0,
                initial_slope=70.0,
            )
        )
        rows.append(
            _feature_row(
                chemical="DEET",
                auc=4500.0 + replicate * 20.0,
                time_to_peak=10.0,
                initial_slope=25.0,
            )
        )
        rows.append(
            _feature_row(
                chemical="Propoxur",
                auc=5600.0 + replicate * 20.0,
                time_to_peak=15.0,
                initial_slope=45.0,
            )
        )
    return pd.DataFrame(rows)


def _feature_row(
    chemical: str,
    auc: float,
    time_to_peak: float,
    initial_slope: float,
) -> dict[str, float | str]:
    return {
        "chemical": chemical,
        "concentration": 5.0,
        "auc": auc,
        "max_signal": auc / 5,
        "min_signal": auc / 10,
        "time_to_peak": time_to_peak,
        "initial_slope": initial_slope,
        "final_signal": auc / 6,
    }
