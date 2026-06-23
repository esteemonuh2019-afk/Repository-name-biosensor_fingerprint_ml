"""Run the real biosensor dataset through the analysis modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.features import extract_features
from src.model_evaluation.evaluate import evaluate_classification, evaluate_regression
from src.model_training.models import (
    predict_classifier,
    predict_regressor,
    train_classifier,
    train_regressor,
)
from src.pipeline.run_pipeline import (
    REQUIRED_COLUMNS,
    _fill_missing_strain_from_source_file,
    _load_input_files,
    _prepare_feature_input,
    run_analysis_pipeline,
)
from src.preprocessing.cleaner import (
    filter_target_chemicals,
    parse_concentration,
    remove_excluded_chemicals,
    standardize_chemical_names,
    standardize_strain_names,
)
from src.preprocessing.schema_harmonizer import harmonize_schema
from src.reporting.report import generate_markdown_report, generate_validation_summary
from src.visualization.plots import (
    plot_dose_response,
    plot_heatmap,
    plot_pca,
    plot_time_course,
)


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

REAL_CHEMICAL_ALIASES = {
    "n,n-diethyl-m-toluamide (deet)": "DEET",
    "n,n-diethyl-m-toluamide": "DEET",
    "o,o-diethyl o-(2-isopropyl-6-methylpyrimidinyl) (diazinon)": "Diazinon",
}


def run_real_analysis() -> dict[str, Any]:
    raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_result = run_analysis_pipeline(raw_files, REPORTS_DIR)
    if pipeline_result["status"] != "success":
        raise RuntimeError(f"Pipeline failed: {pipeline_result.get('errors', [])}")

    raw_data = _load_input_files(raw_files)
    harmonized_data = _fill_missing_strain_from_source_file(harmonize_schema(raw_data))
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in harmonized_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned_data = standardize_strain_names(harmonized_data)
    cleaned_data = _normalize_real_chemical_aliases(cleaned_data)
    cleaned_data = standardize_chemical_names(cleaned_data)
    cleaned_data = remove_excluded_chemicals(cleaned_data)
    cleaned_data = filter_target_chemicals(cleaned_data)
    cleaned_data = parse_concentration(cleaned_data)
    cleaned_data = cleaned_data.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    processed_data = _prepare_feature_input(cleaned_data)
    feature_data = extract_features(processed_data)

    cleaned_data.to_csv(TABLES_DIR / "cleaned_data.csv", index=False)
    processed_data.to_csv(TABLES_DIR / "processed_data.csv", index=False)
    features_path = TABLES_DIR / "features.csv"
    feature_data.to_csv(features_path, index=False)

    figure_paths = _generate_figures(feature_data, processed_data)
    metrics = _train_and_evaluate_if_possible(feature_data)
    metrics_path = None
    if metrics:
        metrics_path = TABLES_DIR / "model_metrics.json"
        metrics_path.write_text(
            json.dumps(_json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )

    validation_summary = generate_validation_summary(metrics) if metrics else "ML metrics were not generated.\n"
    report_path = REPORTS_DIR / "scientific_performance_report.md"
    generate_markdown_report(
        report_path,
        {
            "Data Summary": (
                f"Raw files analyzed: {len(raw_files)}\n\n"
                f"Raw rows loaded: {len(raw_data)}\n\n"
                f"Rows after preprocessing: {len(cleaned_data)}"
            ),
            "Feature Summary": f"Feature rows generated: {len(feature_data)}",
            "Figures": "\n".join(f"- {path}" for path in figure_paths),
            "Validation Summary": validation_summary,
        },
    )

    return {
        "raw_files": [str(path) for path in raw_files],
        "feature_rows": len(feature_data),
        "features_path": str(features_path),
        "figure_paths": [str(path) for path in figure_paths],
        "metrics_path": str(metrics_path) if metrics_path else None,
        "report_path": str(report_path),
        "pipeline_result": pipeline_result,
    }


def _normalize_real_chemical_aliases(dataframe):
    normalized = dataframe.copy()
    normalized["chemical"] = normalized["chemical"].map(_normalize_chemical_name)
    return normalized


def _normalize_chemical_name(value: Any) -> Any:
    if value is None:
        return value

    normalized_value = " ".join(str(value).strip().split())
    return REAL_CHEMICAL_ALIASES.get(normalized_value.casefold(), normalized_value)


def _generate_figures(feature_data, processed_data) -> list[Path]:
    figure_paths = [
        plot_heatmap(feature_data, FIGURES_DIR / "heatmap.png"),
        plot_pca(feature_data, FIGURES_DIR / "pca.png"),
        plot_dose_response(feature_data, FIGURES_DIR / "dose_response.png"),
        plot_time_course(_time_course_subset(processed_data), FIGURES_DIR / "time_course.png"),
    ]
    return figure_paths


def _time_course_subset(processed_data):
    group_columns = ["strain", "chemical", "concentration", "replicate"]
    first_groups = processed_data[group_columns].drop_duplicates().head(4)
    return processed_data.merge(first_groups, on=group_columns, how="inner")


def _train_and_evaluate_if_possible(feature_data) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if len(feature_data) >= 2 and feature_data["chemical"].nunique() >= 2:
        classifier, classifier_columns = train_classifier(feature_data)
        classifier_predictions = predict_classifier(
            classifier,
            feature_data,
            classifier_columns,
        )
        metrics["classification"] = evaluate_classification(
            feature_data["chemical"],
            classifier_predictions,
        )

    if len(feature_data) >= 2 and feature_data["concentration"].nunique() >= 2:
        regressor, regressor_columns = train_regressor(feature_data)
        regressor_predictions = predict_regressor(
            regressor,
            feature_data,
            regressor_columns,
        )
        metrics["regression"] = evaluate_regression(
            feature_data["concentration"],
            regressor_predictions,
        )

    return metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    result = run_real_analysis()
    print(json.dumps(_json_ready(result), indent=2))
