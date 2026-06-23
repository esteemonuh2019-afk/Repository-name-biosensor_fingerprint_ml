import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "e2e"
RAW_SAMPLE_PATH = FIXTURE_DIR / "raw_sample.csv"

REQUIRED_COLUMNS = {
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "time",
    "luminescence",
}
TARGET_CHEMICALS = {
    "Diazinon",
    "DEET",
    "Propoxur",
    "Metaldehyde",
    "Boric Acid",
    "Trimethoprim",
}
CONTROL_LABELS = {"Control"}
EXCLUDED_CHEMICALS = {"Monensin"}


def test_e2e_001_minimal_raw_data_to_validation_summary_flow() -> None:
    raw_rows = _load_raw_csv(RAW_SAMPLE_PATH)
    schema_result = _validate_schema(raw_rows)
    filtered_rows = _filter_target_chemicals(raw_rows)
    feature_summary = _build_feature_like_summary(filtered_rows)
    classification_output = _mock_classification(feature_summary)
    regression_output = _mock_regression(feature_summary)
    validation_summary = _build_validation_summary(
        classification_output,
        regression_output,
    )

    assert raw_rows
    assert schema_result["passed"] is True
    assert REQUIRED_COLUMNS <= set(raw_rows[0])
    assert all(row["chemical"] not in EXCLUDED_CHEMICALS for row in filtered_rows)
    assert {row["chemical"] for row in filtered_rows if row["chemical"] != "Control"} == {
        "Diazinon",
        "DEET",
    }
    assert feature_summary
    assert all("auc" in row for row in feature_summary)
    assert classification_output["predictions"]
    assert regression_output["predictions"]
    assert "classification" in validation_summary
    assert "regression" in validation_summary


def _load_raw_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as raw_file:
        return list(csv.DictReader(raw_file))


def _validate_schema(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"passed": False, "missing_columns": sorted(REQUIRED_COLUMNS)}

    missing_columns = sorted(REQUIRED_COLUMNS - set(rows[0]))
    return {"passed": not missing_columns, "missing_columns": missing_columns}


def _filter_target_chemicals(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed_chemicals = TARGET_CHEMICALS | CONTROL_LABELS
    return [
        row
        for row in rows
        if row["chemical"] in allowed_chemicals and row["chemical"] not in EXCLUDED_CHEMICALS
    ]


def _build_feature_like_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_key = (
            row["strain"],
            row["chemical"],
            row["concentration"],
            row["experiment"],
            row["replicate"],
        )
        grouped_rows[group_key].append(row)

    feature_rows = []
    for (strain, chemical, concentration, experiment, replicate), group_rows in grouped_rows.items():
        sorted_rows = sorted(group_rows, key=lambda row: float(row["time"]))
        times = [float(row["time"]) for row in sorted_rows]
        signals = [float(row["luminescence"]) for row in sorted_rows]
        peak_index = signals.index(max(signals))

        feature_rows.append(
            {
                "strain": strain,
                "chemical": chemical,
                "concentration": float(concentration),
                "experiment": experiment,
                "replicate": replicate,
                "auc": _trapezoid_auc(times, signals),
                "max_signal": max(signals),
                "time_to_peak": times[peak_index],
            }
        )

    return feature_rows


def _mock_classification(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    treatment_rows = [row for row in feature_rows if row["chemical"] in TARGET_CHEMICALS]
    return {
        "task": "classification",
        "predictions": [
            {"experiment": row["experiment"], "predicted_chemical": row["chemical"]}
            for row in treatment_rows
        ],
    }


def _mock_regression(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    treatment_rows = [row for row in feature_rows if row["chemical"] in TARGET_CHEMICALS]
    return {
        "task": "regression",
        "predictions": [
            {
                "experiment": row["experiment"],
                "predicted_concentration": row["concentration"],
            }
            for row in treatment_rows
        ],
    }


def _build_validation_summary(
    classification_output: dict[str, Any],
    regression_output: dict[str, Any],
) -> dict[str, Any]:
    return {
        "classification": {
            "prediction_count": len(classification_output["predictions"]),
            "status": "mock_validated",
        },
        "regression": {
            "prediction_count": len(regression_output["predictions"]),
            "status": "mock_validated",
        },
    }


def _trapezoid_auc(times: list[float], signals: list[float]) -> float:
    return sum(
        (times[index + 1] - times[index]) * (signals[index] + signals[index + 1]) / 2
        for index in range(len(times) - 1)
    )
