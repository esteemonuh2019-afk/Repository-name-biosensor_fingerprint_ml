from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Iterator

import pandas as pd

from src.model_evaluation.specialist_ensemble import (
    get_specialist_mapping,
    generate_specialist_confusion_matrix,
    run_specialist_ensemble_loeo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"
REQUIRED_METRICS = {"accuracy", "macro_precision", "macro_recall", "macro_f1"}


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


def test_specialist_mapping_matches_expected_assignments() -> None:
    assert get_specialist_mapping() == {
        "Boric Acid": "BL027",
        "DEET": "BL030",
        "Diazinon": "BL029",
        "Metaldehyde": "BL027",
        "Propoxur": "BL027",
        "Trimethoprim": "BL032",
    }


def test_specialist_ensemble_loeo_returns_metrics() -> None:
    result = run_specialist_ensemble_loeo(_feature_dataframe())

    assert REQUIRED_METRICS <= set(result["metrics"])
    assert result["prediction_count"] > 0
    assert result["predictions"]
    for metric in REQUIRED_METRICS:
        assert 0.0 <= result["metrics"][metric] <= 1.0


def test_specialist_confusion_matrix_created() -> None:
    with local_test_workspace("specialist_ensemble_confusion_matrix") as workspace:
        output_path = workspace / "specialist_ensemble_confusion_matrix.png"
        result = run_specialist_ensemble_loeo(_feature_dataframe())

        written_path = generate_specialist_confusion_matrix(result, output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    mapping = get_specialist_mapping()
    strains = sorted(set(mapping.values()))
    chemicals = list(mapping)
    for strain in strains:
        strain_offset = sum(ord(character) for character in strain) % 7
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 0.5),
            ("EXP-003", -0.5),
        ):
            for chemical_index, chemical in enumerate(chemicals, start=1):
                rows.append(
                    _feature_row(
                        strain=strain,
                        experiment=experiment,
                        chemical=chemical,
                        concentration=5.0,
                        replicate=1,
                        signal_base=chemical_index * 10.0 + strain_offset + experiment_offset,
                    )
                )
    return pd.DataFrame(rows)


def _feature_row(
    strain: str,
    experiment: str,
    chemical: str,
    concentration: float,
    replicate: int,
    signal_base: float,
) -> dict[str, float | str]:
    return {
        "strain": strain,
        "chemical": chemical,
        "concentration": concentration,
        "experiment": experiment,
        "replicate": replicate,
        "peak_to_baseline_ratio": signal_base / 10,
        "fold_change": signal_base / 20,
        "max_derivative": signal_base / 5,
        "min_derivative": -signal_base / 8,
        "signal_decay_rate": -signal_base / 12,
        "auc_early": signal_base * 2,
        "auc_mid": signal_base * 3,
        "auc_late": signal_base * 4,
    }
