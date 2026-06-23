from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Iterator

import pandas as pd

from src.model_evaluation.chemical_specific_strains import (
    CHEMICALS,
    RANKING_COLUMNS,
    STRAINS,
    evaluate_strain_for_chemical,
    plot_chemical_specific_strain_heatmap,
    rank_strains_per_chemical,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"
REQUIRED_COLUMNS = {"chemical", "strain", "precision", "recall", "f1", "support"}


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


def test_evaluate_strain_for_chemical_returns_binary_metrics() -> None:
    result = evaluate_strain_for_chemical(_feature_dataframe(), "BL027", "Diazinon")

    assert result["chemical"] == "Diazinon"
    assert result["strain"] == "BL027"
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0
    assert result["support"] > 0


def test_rank_strains_per_chemical_creates_full_ranking_table() -> None:
    rankings = rank_strains_per_chemical(_feature_dataframe())

    assert len(rankings) == len(CHEMICALS) * len(STRAINS)
    assert REQUIRED_COLUMNS <= set(rankings.columns)
    assert list(rankings.columns) == list(RANKING_COLUMNS)


def test_rankings_include_all_chemicals_and_strains() -> None:
    rankings = rank_strains_per_chemical(_feature_dataframe())

    assert set(rankings["chemical"]) == set(CHEMICALS)
    assert set(rankings["strain"]) == set(STRAINS)


def test_chemical_specific_strain_heatmap_created() -> None:
    with local_test_workspace("chemical_specific_strain_heatmap") as workspace:
        output_path = workspace / "chemical_specific_strain_heatmap.png"
        rankings = rank_strains_per_chemical(_feature_dataframe())

        written_path = plot_chemical_specific_strain_heatmap(rankings, output_path)

        assert written_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for strain_index, strain in enumerate(STRAINS):
        strain_offset = strain_index * 1.5
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 0.5),
            ("EXP-003", -0.5),
        ):
            for chemical_index, chemical in enumerate(CHEMICALS, start=1):
                rows.append(
                    _feature_row(
                        strain=strain,
                        experiment=experiment,
                        chemical=chemical,
                        concentration=5.0,
                        signal_base=chemical_index * 10.0 + strain_offset + experiment_offset,
                    )
                )
    return pd.DataFrame(rows)


def _feature_row(
    strain: str,
    experiment: str,
    chemical: str,
    concentration: float,
    signal_base: float,
) -> dict[str, float | str]:
    return {
        "strain": strain,
        "chemical": chemical,
        "concentration": concentration,
        "experiment": experiment,
        "replicate": 1,
        "peak_to_baseline_ratio": signal_base / 10,
        "fold_change": signal_base / 20,
        "max_derivative": signal_base / 5,
        "min_derivative": -signal_base / 8,
        "signal_decay_rate": -signal_base / 12,
        "auc_early": signal_base * 2,
        "auc_mid": signal_base * 3,
        "auc_late": signal_base * 4,
    }
