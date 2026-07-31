"""Registry values for the Observation Engine."""

from __future__ import annotations

from pathlib import Path


CATEGORIES: tuple[str, ...] = (
    "QC",
    "Dataset",
    "Fingerprint",
    "PCA",
    "Classification",
    "Regression",
    "Feature Engineering",
    "Feature Selection",
    "Strain Contribution",
    "Blind Prediction",
)


ANALYSIS_STAGES: dict[str, str] = {
    "QC": "Stages 5C, 6B, 7A",
    "Dataset": "Stage 6B",
    "Fingerprint": "Stage 7A",
    "PCA": "Stage 7B",
    "Classification": "Stage 8A",
    "Regression": "Stage 8B",
    "Feature Engineering": "Stage 8C",
    "Feature Selection": "Stage 8D",
    "Strain Contribution": "Strain ablation outputs",
    "Blind Prediction": "Stage 9A",
}


DEFAULT_SUMMARY_PATH = Path("outputs/supervisor_results/supervisor_results_summary.json")
DEFAULT_VALIDATION_PATH = Path("outputs/supervisor_results/report_validation.json")
DEFAULT_OUTPUT_DIR = Path("outputs/scientific_observations")


OUTPUT_FILENAMES: tuple[str, ...] = (
    "observations.json",
    "observations.csv",
    "observations.md",
)
