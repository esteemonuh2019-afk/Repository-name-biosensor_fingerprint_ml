"""Full analysis pipeline runner for biosensor fingerprint analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data_ingestion.loader import load_multiple_csv
from src.data_validation.validator import validate_schema
from src.feature_engineering.features import extract_features
from src.preprocessing.cleaner import (
    filter_target_chemicals,
    parse_concentration,
    remove_excluded_chemicals,
    standardize_chemical_names,
    standardize_strain_names,
)
from src.preprocessing.schema_harmonizer import harmonize_schema
from src.reporting.report import generate_markdown_report


REQUIRED_COLUMNS: tuple[str, ...] = (
    "strain",
    "chemical",
    "concentration",
    "experiment",
    "replicate",
    "time",
    "luminescence",
)


def run_analysis_pipeline(
    input_files: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the raw-data-to-report analysis flow using existing production modules."""

    destination = Path(output_dir)
    result: dict[str, Any] = {
        "raw_rows": 0,
        "feature_rows": 0,
        "output_dir": str(destination),
        "report_path": None,
        "status": "failed",
        "errors": [],
    }

    try:
        raw_data = _load_input_files(input_files)
        result["raw_rows"] = len(raw_data)
        harmonized_data = harmonize_schema(raw_data)
        harmonized_data = _fill_missing_strain_from_source_file(harmonized_data)

        schema_result = validate_schema(harmonized_data, REQUIRED_COLUMNS)
        if not schema_result.valid:
            result["errors"] = [
                f"Missing required columns: {', '.join(schema_result.missing_columns)}"
            ]
            return result

        cleaned_data = standardize_strain_names(harmonized_data)
        cleaned_data = standardize_chemical_names(cleaned_data)
        cleaned_data = remove_excluded_chemicals(cleaned_data)
        cleaned_data = filter_target_chemicals(cleaned_data)
        cleaned_data = parse_concentration(cleaned_data)
        cleaned_data = cleaned_data.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
        cleaned_data = _prepare_feature_input(cleaned_data)

        if cleaned_data.empty:
            result["errors"] = ["No target chemical rows remain after preprocessing."]
            return result

        feature_data = extract_features(cleaned_data)
        result["feature_rows"] = len(feature_data)

        report_path = destination / "analysis_report.md"
        generate_markdown_report(
            report_path,
            {
                "Data Summary": (
                    f"Raw rows loaded: {len(raw_data)}\n\n"
                    f"Rows after preprocessing: {len(cleaned_data)}"
                ),
                "Feature Summary": f"Feature rows generated: {len(feature_data)}",
            },
        )

        result.update(
            {
                "report_path": str(report_path),
                "status": "success",
                "errors": [],
            }
        )
        return result
    except Exception as error:
        result["errors"] = [str(error)]
        return result


def _load_input_files(input_files: Iterable[str | Path]) -> pd.DataFrame:
    paths = list(input_files)
    try:
        return load_multiple_csv(paths)
    except UnicodeDecodeError:
        return _load_multiple_csv_with_encoding_fallback(paths)


def _load_multiple_csv_with_encoding_fallback(file_paths: list[str | Path]) -> pd.DataFrame:
    if not file_paths:
        raise ValueError("At least one CSV file path is required.")

    dataframes = []
    for file_path in file_paths:
        path = Path(file_path)
        dataframe = pd.read_csv(path, encoding="latin1", low_memory=False)
        if dataframe.empty:
            raise ValueError(f"CSV file is empty: {path}")
        dataframe["source_file"] = str(path)
        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


def _fill_missing_strain_from_source_file(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "strain" not in dataframe.columns or "source_file" not in dataframe.columns:
        return dataframe

    harmonized = dataframe.copy()
    missing_strain = harmonized["strain"].isna()
    if missing_strain.any():
        harmonized.loc[missing_strain, "strain"] = harmonized.loc[
            missing_strain,
            "source_file",
        ].map(lambda source_file: Path(source_file).stem)

    return harmonized


def _prepare_feature_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "strain",
        "chemical",
        "concentration",
        "experiment",
        "replicate",
        "time",
    ]
    averaged_data = (
        dataframe.groupby(group_columns, as_index=False, sort=False)["luminescence"]
        .mean()
        .reset_index(drop=True)
    )

    feature_groups = ["strain", "chemical", "concentration", "experiment", "replicate"]
    time_counts = averaged_data.groupby(feature_groups, sort=False)["time"].transform("nunique")
    return averaged_data[time_counts >= 2].reset_index(drop=True)
