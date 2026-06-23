import csv
import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "regression"
FEATURE_TABLE_PATH = FIXTURE_DIR / "sample_feature_table.csv"
EXPECTED_METRICS_PATH = FIXTURE_DIR / "expected_metrics.json"

APPROVED_FEATURE_COLUMNS = [
    "strain",
    "chemical",
    "concentration",
    "auc",
    "max_signal",
    "time_to_peak",
]

REQUIRED_METRIC_KEYS = {"accuracy", "macro_f1", "r2", "rmse"}
APPROVED_METRIC_RANGES = {
    "accuracy": (0.80, 1.00),
    "macro_f1": (0.75, 1.00),
    "r2": (0.75, 1.00),
    "rmse": (0.00, 0.25),
}


def test_rg_001_sample_feature_table_structure_has_not_changed() -> None:
    with FEATURE_TABLE_PATH.open(newline="", encoding="utf-8") as feature_file:
        reader = csv.DictReader(feature_file)
        rows = list(reader)

    assert reader.fieldnames == APPROVED_FEATURE_COLUMNS
    assert rows


def test_rg_002_expected_metrics_contains_required_keys() -> None:
    metrics = _load_expected_metrics()

    assert set(metrics) == REQUIRED_METRIC_KEYS


def test_rg_003_metric_values_remain_within_approved_ranges() -> None:
    metrics = _load_expected_metrics()

    for metric_name, metric_value in metrics.items():
        lower_bound, upper_bound = APPROVED_METRIC_RANGES[metric_name]
        assert lower_bound <= metric_value <= upper_bound


def test_rg_004_future_schema_changes_are_detected() -> None:
    with FEATURE_TABLE_PATH.open(newline="", encoding="utf-8") as feature_file:
        actual_columns = csv.DictReader(feature_file).fieldnames

    unexpected_columns = set(actual_columns or []) - set(APPROVED_FEATURE_COLUMNS)
    missing_columns = set(APPROVED_FEATURE_COLUMNS) - set(actual_columns or [])

    assert unexpected_columns == set()
    assert missing_columns == set()
    assert actual_columns == APPROVED_FEATURE_COLUMNS


def _load_expected_metrics() -> dict[str, float]:
    return json.loads(EXPECTED_METRICS_PATH.read_text(encoding="utf-8"))
