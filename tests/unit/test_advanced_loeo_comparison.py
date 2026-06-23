from pathlib import Path
import shutil

import pandas as pd

from src.model_evaluation.advanced_loeo_comparison import (
    ADVANCED_LOEO_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    combine_original_and_advanced_features,
    get_advanced_loeo_feature_columns,
    plot_advanced_panel_macro_f1,
    run_advanced_panel_comparison,
)


def test_advanced_panel_comparison_runs() -> None:
    result = run_advanced_panel_comparison(_combined_feature_dataframe())

    assert len(result) == 3
    assert set(result["panel_name"]) == {"Panel_A", "Panel_B", "Panel_C"}
    assert result["sample_count"].gt(0).all()


def test_output_columns_exist() -> None:
    result = run_advanced_panel_comparison(_combined_feature_dataframe())

    assert list(result.columns) == list(OUTPUT_COLUMNS)


def test_all_original_and_advanced_feature_columns_used() -> None:
    feature_columns = get_advanced_loeo_feature_columns(_combined_feature_dataframe())

    assert feature_columns == list(ADVANCED_LOEO_FEATURE_COLUMNS)


def test_original_and_advanced_feature_tables_combine() -> None:
    combined = _combined_feature_dataframe()
    original = combined[
        [
            "strain",
            "chemical",
            "concentration",
            "experiment",
            "replicate",
            "auc",
            "max_signal",
            "min_signal",
            "time_to_peak",
            "initial_slope",
            "final_signal",
        ]
    ]
    advanced = combined.drop(
        columns=[
            "auc",
            "max_signal",
            "min_signal",
            "time_to_peak",
            "initial_slope",
            "final_signal",
        ]
    )

    merged = combine_original_and_advanced_features(original, advanced)

    assert set(ADVANCED_LOEO_FEATURE_COLUMNS) <= set(merged.columns)
    assert len(merged) == len(combined)


def test_advanced_panel_macro_f1_figure_created() -> None:
    temp_dir = Path("tests") / "tmp" / "advanced_loeo_comparison"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    output_path = temp_dir / "advanced_panel_macro_f1.png"

    try:
        result = run_advanced_panel_comparison(_combined_feature_dataframe())
        plot_advanced_panel_macro_f1(result, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _combined_feature_dataframe() -> pd.DataFrame:
    rows = []
    for strain_index, strain in enumerate(("BL027", "BL011", "BL030")):
        strain_offset = strain_index * 2.0
        for experiment, experiment_offset in (
            ("EXP-001", 0.0),
            ("EXP-002", 1.0),
            ("EXP-003", -1.0),
        ):
            rows.append(
                _feature_row(
                    strain=strain,
                    experiment=experiment,
                    chemical="Diazinon",
                    concentration=5.0,
                    base_signal=100.0 + strain_offset + experiment_offset,
                )
            )
            rows.append(
                _feature_row(
                    strain=strain,
                    experiment=experiment,
                    chemical="DEET",
                    concentration=50.0,
                    base_signal=200.0 + strain_offset + experiment_offset,
                )
            )
    return pd.DataFrame(rows)


def _feature_row(
    strain: str,
    experiment: str,
    chemical: str,
    concentration: float,
    base_signal: float,
) -> dict[str, float | str]:
    is_diazinon = chemical == "Diazinon"
    return {
        "strain": strain,
        "chemical": chemical,
        "concentration": concentration,
        "experiment": experiment,
        "replicate": 1,
        "auc": base_signal * 10,
        "max_signal": base_signal * (1.3 if is_diazinon else 1.8),
        "min_signal": base_signal * (0.6 if is_diazinon else 0.9),
        "time_to_peak": 6.0 if is_diazinon else 12.0,
        "initial_slope": 4.0 if is_diazinon else -2.0,
        "final_signal": base_signal * (0.8 if is_diazinon else 1.5),
        "peak_to_baseline_ratio": 1.3 if is_diazinon else 1.8,
        "fold_change": -0.2 if is_diazinon else 0.5,
        "max_derivative": 4.0 if is_diazinon else 1.5,
        "min_derivative": -2.0 if is_diazinon else -0.5,
        "signal_decay_rate": -1.0 if is_diazinon else 0.2,
        "auc_early": base_signal * (3.0 if is_diazinon else 1.0),
        "auc_mid": base_signal * (2.0 if is_diazinon else 4.0),
        "auc_late": base_signal * (1.0 if is_diazinon else 5.0),
    }
