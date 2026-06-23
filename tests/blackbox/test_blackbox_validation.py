import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "blackbox"

REQUIRED_COLUMNS = {
    "strain",
    "chemical",
    "concentration",
    "replicate",
    "experiment",
    "time",
    "luminescence",
}

VALID_STRAINS = {"BL011", "BL027", "BL029", "BL030", "BL031", "BL032"}
TARGET_CHEMICALS = {
    "Diazinon",
    "DEET",
    "Propoxur",
    "Metaldehyde",
    "Boric Acid",
    "Trimethoprim",
}
CONTROL_LABELS = {"Control"}


@dataclass(frozen=True)
class BlackBoxValidationResult:
    rows: list[dict[str, str]]
    passed: bool
    failures: list[str]
    warnings: list[str]


def _load_csv_fixture(file_name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (FIXTURE_DIR / file_name).open(newline="", encoding="utf-8") as fixture_file:
        reader = csv.DictReader(fixture_file)
        return reader.fieldnames or [], list(reader)


def _validate_blackbox_fixture(file_name: str) -> BlackBoxValidationResult:
    columns, rows = _load_csv_fixture(file_name)
    failures: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
    if missing_columns:
        failures.append(f"Missing required columns: {', '.join(missing_columns)}")
        return BlackBoxValidationResult(
            rows=rows,
            passed=False,
            failures=failures,
            warnings=warnings,
        )

    invalid_strains = _unexpected_values(rows, "strain", VALID_STRAINS)
    if invalid_strains:
        failures.append(f"Invalid strain labels: {', '.join(invalid_strains)}")

    valid_chemical_labels = TARGET_CHEMICALS | CONTROL_LABELS
    unknown_chemicals = _unexpected_values(rows, "chemical", valid_chemical_labels)
    if unknown_chemicals:
        warnings.append(f"Unknown chemical label(s): {', '.join(unknown_chemicals)}")

    has_treatment = any(row["chemical"] in TARGET_CHEMICALS for row in rows)
    has_control = any(row["chemical"] in CONTROL_LABELS for row in rows)
    if has_treatment and not has_control:
        warnings.append("Missing control row; normalization cannot be performed.")

    return BlackBoxValidationResult(
        rows=rows,
        passed=not failures,
        failures=failures,
        warnings=warnings,
    )


def _unexpected_values(
    rows: list[dict[str, Any]],
    column: str,
    allowed_values: set[str],
) -> list[str]:
    return sorted({row[column] for row in rows if row[column] not in allowed_values})


def test_bb_001_valid_sample_loads_and_passes_basic_validation() -> None:
    result = _validate_blackbox_fixture("valid_sample.csv")

    assert result.rows
    assert result.passed is True
    assert result.failures == []
    assert result.warnings == []


def test_bb_002_missing_columns_produces_validation_failure() -> None:
    result = _validate_blackbox_fixture("missing_columns.csv")

    assert result.passed is False
    assert any("luminescence" in failure for failure in result.failures)


def test_bb_003_unknown_chemical_produces_warning_or_failure() -> None:
    result = _validate_blackbox_fixture("unknown_chemical.csv")

    assert result.warnings or result.failures
    assert any("Glyphosate" in message for message in result.warnings + result.failures)


def test_bb_004_missing_control_warns_normalization_cannot_be_performed() -> None:
    result = _validate_blackbox_fixture("missing_control.csv")

    assert result.passed is True
    assert any("normalization cannot be performed" in warning for warning in result.warnings)
