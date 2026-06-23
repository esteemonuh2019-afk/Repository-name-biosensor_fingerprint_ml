from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Iterator

import pandas as pd

from src.model_evaluation.advanced_per_chemical_analysis import (
    ADVANCED_PER_CHEMICAL_COLUMNS,
    generate_advanced_normalized_confusion_matrix,
    run_advanced_per_chemical_loeo,
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


def test_advanced_per_chemical_dataframe_is_generated() -> None:
    result = run_advanced_per_chemical_loeo(_feature_dataframe(), ["BL027"])

    assert not result.empty
    assert set(result["chemical"]) == {"Diazinon", "DEET", "Propoxur"}


def test_required_columns_exist() -> None:
    result = run_advanced_per_chemical_loeo(_feature_dataframe(), ["BL027"])

    assert REQUIRED_COLUMNS <= set(result.columns)
    assert list(result.columns) == list(ADVANCED_PER_CHEMICAL_COLUMNS)


def test_advanced_normalized_confusion_matrix_figure_created() -> None:
    with local_test_workspace("advanced_per_chemical_confusion_matrix") as workspace:
        output_path = workspace / "advanced_normalized_confusion_matrix_BL027.png"

        written_path = generate_advanced_normalized_confusion_matrix(
            _feature_dataframe(),
            ["BL027"],
            output_path,
        )

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for strain, strain_offset in (("BL027", 0.0), ("BL011", 35.0)):
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 10.0),
            ("EXP-003", -10.0),
        ):
            for chemical, auc_base, peak_time, advanced_offset in (
                ("Diazinon", 7000.0, 5.0, 1.0),
                ("DEET", 4500.0, 10.0, 2.0),
                ("Propoxur", 5600.0, 15.0, 3.0),
            ):
                rows.append(
                    _feature_row(
                        strain=strain,
                        experiment=experiment,
                        chemical=chemical,
                        concentration=5.0,
                        auc=auc_base + strain_offset + experiment_offset,
                        time_to_peak=peak_time,
                        advanced_offset=advanced_offset,
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
    advanced_offset: float,
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
        "peak_to_baseline_ratio": 1.0 + advanced_offset,
        "fold_change": advanced_offset / 10,
        "max_derivative": advanced_offset * 2,
        "min_derivative": -advanced_offset,
        "signal_decay_rate": -advanced_offset / 2,
        "auc_early": auc / (1 + advanced_offset),
        "auc_mid": auc / (2 + advanced_offset),
        "auc_late": auc / (3 + advanced_offset),
    }
