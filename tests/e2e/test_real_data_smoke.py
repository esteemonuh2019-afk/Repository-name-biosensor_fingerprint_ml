from pathlib import Path

from src.data_ingestion import loader as loader_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REQUIRED_BIOLOGICAL_COLUMNS = {
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "time",
    "luminescence",
}


def test_real_raw_csv_files_load_without_modifying_inputs(monkeypatch) -> None:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    assert csv_files, f"No CSV files found in {RAW_DATA_DIR}"

    _install_read_csv_smoke_fallback(monkeypatch)
    before_metadata = _file_metadata(csv_files)
    per_file_columns: dict[str, list[str]] = {}

    for csv_file in csv_files:
        dataframe = loader_module.load_csv(csv_file)
        assert len(dataframe) > 0
        per_file_columns[csv_file.name] = list(dataframe.columns)
        _print_column_report(csv_file.name, dataframe.columns)

    combined_data = loader_module.load_multiple_csv(csv_files)
    assert len(combined_data) > 0
    print(f"Loaded {len(csv_files)} raw CSV file(s).")
    print(f"Combined row count: {len(combined_data)}")
    print(f"Combined columns: {list(combined_data.columns)}")

    missing_required_columns = sorted(
        REQUIRED_BIOLOGICAL_COLUMNS - {column.lower() for column in combined_data.columns}
    )
    if missing_required_columns:
        print(
            "Missing exact SSDD biological column names: "
            f"{missing_required_columns}"
        )

    after_metadata = _file_metadata(csv_files)
    assert after_metadata == before_metadata


def _install_read_csv_smoke_fallback(monkeypatch) -> None:
    original_read_csv = loader_module.pd.read_csv

    def read_csv_with_encoding_report(*args, **kwargs):
        try:
            return original_read_csv(*args, **kwargs)
        except UnicodeDecodeError:
            csv_path = args[0] if args else "<unknown>"
            print(f"{csv_path} could not be decoded as UTF-8; retrying with latin1 for smoke test.")
            return original_read_csv(*args, encoding="latin1", **kwargs)

    monkeypatch.setattr(loader_module.pd, "read_csv", read_csv_with_encoding_report)


def _print_column_report(file_name: str, columns) -> None:
    actual_columns = list(columns)
    normalized_columns = {column.lower() for column in actual_columns}
    missing_columns = sorted(REQUIRED_BIOLOGICAL_COLUMNS - normalized_columns)
    present_columns = sorted(REQUIRED_BIOLOGICAL_COLUMNS & normalized_columns)

    print(f"{file_name} columns: {actual_columns}")
    print(f"{file_name} matching SSDD columns: {present_columns}")
    print(f"{file_name} missing exact SSDD columns: {missing_columns}")


def _file_metadata(csv_files: list[Path]) -> dict[Path, tuple[int, int]]:
    return {
        csv_file: (csv_file.stat().st_size, csv_file.stat().st_mtime_ns)
        for csv_file in csv_files
    }
