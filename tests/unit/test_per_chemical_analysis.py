from pathlib import Path
import shutil
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from src.model_evaluation.per_chemical_analysis import (
    generate_normalized_confusion_matrix,
    run_per_chemical_loeo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"
REQUIRED_COLUMNS = {"chemical", "precision", "recall", "f1", "support"}


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


def test_per_chemical_dataframe_generated() -> None:
    result = run_per_chemical_loeo(_feature_dataframe(), ["BL027", "BL011"])

    assert not result.empty
    assert set(result["chemical"]) == {"Diazinon", "DEET", "Propoxur"}


def test_required_columns_exist() -> None:
    result = run_per_chemical_loeo(_feature_dataframe(), ["BL027", "BL011"])

    assert REQUIRED_COLUMNS <= set(result.columns)


def test_confusion_matrix_figure_created() -> None:
    with local_test_workspace("per_chemical_confusion_matrix") as workspace:
        output_path = workspace / "normalized_confusion_matrix.png"

        written_path = generate_normalized_confusion_matrix(
            _feature_dataframe(),
            ["BL027", "BL011"],
            output_path,
        )

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for strain, strain_offset in (("BL027", 0.0), ("BL011", 40.0), ("BL030", 80.0)):
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 15.0),
            ("EXP-003", -15.0),
        ):
            for chemical, auc_base, peak_time in (
                ("Diazinon", 7000.0, 5.0),
                ("DEET", 4500.0, 10.0),
                ("Propoxur", 5600.0, 15.0),
            ):
                rows.append(
                    _feature_row(
                        strain,
                        experiment,
                        chemical,
                        5.0,
                        auc_base + strain_offset + experiment_offset,
                        peak_time,
                    )
                )
    return pd.DataFrame(rows)


def _feature_row(
    strain: str,
    experiment: str,
    chemical: str,
    concentration: float,
    auc: float,
    time_to_peak: float,
) -> dict[str, float | str]:
    return {
        "strain": strain,
        "chemical": chemical,
        "concentration": concentration,
        "experiment": experiment,
        "replicate": 1,
        "auc": auc,
        "max_signal": auc / 5,
        "min_signal": auc / 10,
        "time_to_peak": time_to_peak,
        "initial_slope": auc / 100,
        "final_signal": auc / 6,
    }
