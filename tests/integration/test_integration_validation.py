import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "integration"
CLEAN_DATA_PATH = FIXTURE_DIR / "sample_clean_data.csv"
FEATURES_PATH = FIXTURE_DIR / "sample_features.csv"

CLEAN_DATA_FIELDS = {
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "time",
    "luminescence",
}

FEATURE_FIELDS = {
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
}

CLASSIFIER_REQUIRED_FIELDS = {"chemical", "auc", "max_signal", "min_signal", "time_to_peak"}
REGRESSOR_REQUIRED_FIELDS = {"concentration", "auc", "max_signal", "min_signal", "time_to_peak"}


def test_it_001_cleaned_data_can_pass_into_feature_extraction_like_interface() -> None:
    clean_rows = _load_csv(CLEAN_DATA_PATH)

    feature_rows = _extract_features_like_interface(clean_rows)

    assert feature_rows
    assert set(feature_rows[0]) == FEATURE_FIELDS
    assert feature_rows[0]["chemical"] == "Diazinon"
    assert feature_rows[0]["auc"] > 0


def test_it_002_feature_table_contains_fields_required_by_classifier_and_regressor() -> None:
    feature_rows = _load_csv(FEATURES_PATH)
    feature_columns = set(feature_rows[0])

    assert CLASSIFIER_REQUIRED_FIELDS <= feature_columns
    assert REGRESSOR_REQUIRED_FIELDS <= feature_columns


def test_it_003_chemical_labels_are_usable_for_classification() -> None:
    feature_rows = _load_csv(FEATURES_PATH)
    labels = [row["chemical"] for row in feature_rows]

    assert labels
    assert all(isinstance(label, str) for label in labels)
    assert all(label for label in labels)
    assert set(labels) == {"Diazinon", "DEET"}


def test_it_004_concentration_labels_are_numeric_and_usable_for_regression() -> None:
    feature_rows = _load_csv(FEATURES_PATH)
    targets = [float(row["concentration"]) for row in feature_rows]

    assert targets == [5.0, 50.0]
    assert all(target > 0 for target in targets)


def test_it_005_experiment_ids_are_present_for_experiment_level_validation() -> None:
    feature_rows = _load_csv(FEATURES_PATH)
    experiment_ids = [row["experiment"] for row in feature_rows]

    assert experiment_ids
    assert all(experiment_id.startswith("EXP-") for experiment_id in experiment_ids)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _extract_features_like_interface(clean_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not clean_rows:
        raise ValueError("Clean data must contain at least one row.")
    if not CLEAN_DATA_FIELDS <= set(clean_rows[0]):
        raise ValueError("Clean data is missing required fields.")

    grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        group_key = (
            row["strain"],
            row["chemical"],
            row["concentration"],
            row["experiment"],
            row["replicate"],
        )
        grouped_rows[group_key].append(row)

    feature_rows = []
    for (strain, chemical, concentration, experiment, replicate), rows in grouped_rows.items():
        sorted_rows = sorted(rows, key=lambda row: float(row["time"]))
        times = [float(row["time"]) for row in sorted_rows]
        signals = [float(row["luminescence"]) for row in sorted_rows]
        peak_index = signals.index(max(signals))

        feature_rows.append(
            {
                "strain": strain,
                "chemical": chemical,
                "concentration": concentration,
                "experiment": experiment,
                "replicate": replicate,
                "auc": _trapezoid_auc(times, signals),
                "max_signal": max(signals),
                "min_signal": min(signals),
                "time_to_peak": times[peak_index],
                "initial_slope": (signals[1] - signals[0]) / (times[1] - times[0]),
                "final_signal": signals[-1],
            }
        )

    return feature_rows


def _trapezoid_auc(times: list[float], signals: list[float]) -> float:
    return sum(
        (times[index + 1] - times[index]) * (signals[index] + signals[index + 1]) / 2
        for index in range(len(times) - 1)
    )
