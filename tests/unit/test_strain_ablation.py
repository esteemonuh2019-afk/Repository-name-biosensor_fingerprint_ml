import pandas as pd

from src.model_evaluation.strain_ablation import (
    run_leave_one_strain_out_loeo,
    run_single_strain_loeo,
)


REQUIRED_SINGLE_STRAIN_COLUMNS = {
    "strain",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "number_of_samples",
}

REQUIRED_LEAVE_ONE_STRAIN_OUT_COLUMNS = {
    "removed_strain",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "number_of_samples",
}


def test_single_strain_analysis_runs() -> None:
    result = run_single_strain_loeo(_feature_dataframe())

    assert not result.empty
    assert result["status"].eq("success").all()


def test_leave_one_strain_out_analysis_runs() -> None:
    result = run_leave_one_strain_out_loeo(_feature_dataframe())

    assert not result.empty
    assert result["status"].eq("success").all()


def test_required_output_columns_exist() -> None:
    single_strain_result = run_single_strain_loeo(_feature_dataframe())
    leave_one_strain_out_result = run_leave_one_strain_out_loeo(_feature_dataframe())

    assert REQUIRED_SINGLE_STRAIN_COLUMNS <= set(single_strain_result.columns)
    assert REQUIRED_LEAVE_ONE_STRAIN_OUT_COLUMNS <= set(leave_one_strain_out_result.columns)


def test_output_rows_equal_number_of_strains() -> None:
    feature_df = _feature_dataframe()
    expected_strain_count = feature_df["strain"].nunique()

    assert len(run_single_strain_loeo(feature_df)) == expected_strain_count
    assert len(run_leave_one_strain_out_loeo(feature_df)) == expected_strain_count


def _feature_dataframe() -> pd.DataFrame:
    rows = []
    for strain, strain_offset in (("BL011", 0.0), ("BL027", 100.0), ("BL029", -100.0)):
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 20.0),
            ("EXP-003", -20.0),
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
