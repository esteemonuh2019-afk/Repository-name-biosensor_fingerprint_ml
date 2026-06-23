import pandas as pd

from src.model_evaluation.panel_optimization import (
    evaluate_strain_panel,
    run_candidate_panels,
)


REQUIRED_COLUMNS = {
    "panel_name",
    "strains",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "sample_count",
}


def test_candidate_panels_execute() -> None:
    result = run_candidate_panels(_feature_dataframe())

    assert not result.empty
    assert result["status"].eq("success").all()


def test_required_columns_exist() -> None:
    result = run_candidate_panels(_feature_dataframe())

    assert REQUIRED_COLUMNS <= set(result.columns)


def test_panel_count_correct() -> None:
    result = run_candidate_panels(_feature_dataframe())

    assert len(result) == 6
    assert set(result["panel_name"]) == {
        "Panel_A",
        "Panel_B",
        "Panel_C",
        "Panel_D",
        "Panel_E",
        "Panel_F",
    }


def test_metrics_returned() -> None:
    result = evaluate_strain_panel(_feature_dataframe(), ["BL027", "BL011"])

    assert result["status"] == "success"
    assert result["sample_count"] > 0
    assert result["accuracy"] >= 0.0
    assert result["macro_f1"] >= 0.0


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    strains = ("BL027", "BL011", "BL030", "BL029", "BL032", "BL031")
    for strain_index, strain in enumerate(strains):
        strain_offset = strain_index * 50.0
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 15.0),
            ("EXP-003", -15.0),
        ):
            rows.extend(
                [
                    _feature_row(strain, experiment, "Diazinon", 5.0, 5000.0 + strain_offset + experiment_offset),
                    _feature_row(strain, experiment, "Diazinon", 50.0, 7000.0 + strain_offset + experiment_offset),
                    _feature_row(strain, experiment, "DEET", 5.0, 3500.0 + strain_offset + experiment_offset),
                    _feature_row(strain, experiment, "DEET", 50.0, 5200.0 + strain_offset + experiment_offset),
                ]
            )
    return pd.DataFrame(rows)


def _feature_row(
    strain: str,
    experiment: str,
    chemical: str,
    concentration: float,
    auc: float,
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
        "time_to_peak": 5.0 if chemical == "Diazinon" else 10.0,
        "initial_slope": 40.0 if chemical == "Diazinon" else 20.0,
        "final_signal": auc / 6,
    }
