from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FIXED_TIMESTAMP = "2026-07-31T00:00:00+00:00"


def create_supervisor_fixture(
    tmp_path: Path,
    *,
    missing_required: str | None = None,
    missing_optional: str | None = None,
    remove_provenance_metric: str | None = None,
    classification_metric_model: str = "Extra Trees",
    regression_metric_model: str = "Extra Trees Regressor",
) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    supervisor = project_root / "outputs" / "supervisor_results_2"
    supervisor.mkdir(parents=True)

    summary = _summary(classification_metric_model, regression_metric_model)
    provenance = [
        *_dataset_provenance(),
        *_qc_provenance(),
        *_fingerprint_provenance(),
        *_exploratory_provenance(),
        *_classification_provenance(classification_metric_model),
        *_regression_provenance(regression_metric_model),
        *_feature_engineering_provenance(),
        *_feature_selection_provenance(),
        *_strain_provenance(),
        _prov("P9001", "Limitations", "true_labels_included", False, "", "", "blind_prediction/summary.json"),
    ]
    if remove_provenance_metric:
        provenance = [row for row in provenance if row["metric_name"] != remove_provenance_metric]

    files: dict[str, Any] = {
        "supervisor_results_summary.json": summary,
        "report_validation.json": {"passed": True, "checks": []},
        "provenance_index.csv": provenance,
        "selected_tables.csv": [
            {"table_id": "fingerprint_summary", "title": "Fingerprint Summary", "source_file": "fingerprints/summary.json", "row_count": 3, "status": "POPULATED", "notes": ""},
            {"table_id": "feature_selection_results", "title": "Feature Selection", "source_file": "feature_selection/summary.csv", "row_count": 4, "status": "POPULATED", "notes": ""},
            {"table_id": "strain_contribution", "title": "Strain Contribution", "source_file": "tables/strain.csv", "row_count": 2, "status": "POPULATED", "notes": ""},
        ],
        "selected_figures.csv": [
            {"figure_id": "consensus_fingerprint_heatmap", "title": "Consensus Fingerprint Heatmap", "source_file": "figures/fingerprint.png", "source_run": "", "output_file": "figures/fingerprint.png", "status": "SELECTED", "notes": ""},
            {"figure_id": "hierarchical_dendrogram", "title": "Hierarchical Dendrogram", "source_file": "figures/dendrogram.png", "source_run": "", "output_file": "figures/dendrogram.png", "status": "SELECTED", "notes": ""},
            {"figure_id": "strain_contribution_heatmap", "title": "Strain Contribution Heatmap", "source_file": "figures/strain.png", "source_run": "", "output_file": "figures/strain.png", "status": "SELECTED", "notes": ""},
        ],
    }

    for filename, payload in files.items():
        if filename in {missing_required, missing_optional}:
            continue
        path = supervisor / filename
        if filename.endswith(".json"):
            _write_json(path, payload)
        else:
            _write_csv(path, payload)
    return project_root, supervisor


def _summary(classification_metric_model: str, regression_metric_model: str) -> dict[str, Any]:
    return {
        "package_passed": True,
        "project_summary": {
            "blind_prediction_context": {
                "prediction_passed": False,
                "true_labels_included": False,
                "predicted_chemical": "Boric Acid",
                "chemical_confidence": 0.21,
                "predicted_concentration": 346.4,
                "concentration_units": "ug/mL",
                "novelty_status": "Out of Distribution",
                "source_file": "blind_prediction/summary.json",
            }
        },
        "dataset_summary": {
            "input_canonical_rows": 100,
            "feature_rows": 10,
            "core_feature_count": 2,
            "chemical_count": 3,
            "strain_count": 2,
            "chemicals_detected": ["A", "B", "C"],
            "strains_detected": ["BL011", "BL032"],
            "concentration_levels": ["0.5", "Control"],
            "source_file_count": 1,
            "feature_summary_source": "features/summary.json",
            "canonical_qc_source": "qc/summary.json",
        },
        "quality_control_summary": {
            "canonical_error_count": 0,
            "canonical_warning_count": 1,
            "feature_failed_rows": 0,
            "fingerprint_excluded_rows": 1,
            "canonical_qc_source": "qc/summary.json",
            "feature_summary_source": "features/summary.json",
            "fingerprint_summary_source": "fingerprints/summary.json",
        },
        "fingerprint_summary": {
            "fingerprint_rows": 10,
            "consensus_fingerprint_rows": 4,
            "feature_count": 2,
            "normalization_method": "zscore",
            "source_file": "fingerprints/summary.json",
        },
        "exploratory_results": {
            "cumulative_variance_pc3": 0.85,
            "cluster_count": 2,
            "cluster_assignment_rows": 4,
            "pca_source": "exploratory/pca.csv",
            "cluster_source": "exploratory/clusters.csv",
        },
        "classification_results": {
            "selected_model": {
                "model_name": "Extra Trees",
                "rank": 1,
                "selection_metric": "f1_macro_mean",
                "source_file": "classification/best.json",
            },
            "selected_metrics": [
                _metric("accuracy_mean", 0.73, model=classification_metric_model, source="classification/best.json"),
                _metric("balanced_accuracy_mean", 0.72, model=classification_metric_model, source="classification/best.json"),
                _metric("f1_macro_mean", 0.71, model=classification_metric_model, source="classification/best.json"),
                _metric("f1_weighted_mean", 0.74, model=classification_metric_model, source="classification/best.json"),
                _metric("precision_macro_mean", 0.75, model=classification_metric_model, source="classification/best.json"),
                _metric("recall_macro_mean", 0.76, model=classification_metric_model, source="classification/best.json"),
                _metric("roc_auc_ovr_weighted_mean", 0.9, model=classification_metric_model, source="classification/best.json"),
                _metric("log_loss_mean", 1.1, model=classification_metric_model, source="classification/best.json"),
                _metric("fold_count", 3, model=classification_metric_model, source="classification/rankings.csv"),
                _metric("sample_count", 90, model=classification_metric_model, source="classification/best.json"),
                _metric("class_count", 3, model=classification_metric_model, source="classification/best.json"),
            ],
            "model_comparison": [{"model_name": "Extra Trees"}, {"model_name": "Random Forest"}],
        },
        "regression_results": {
            "selected_model": {
                "model_name": "Extra Trees Regressor",
                "rank": 1,
                "selection_metric": "r2_mean",
                "target_units": "ug/mL",
                "source_file": "regression/best.json",
            },
            "selected_metrics": [
                _metric("r2_mean", 0.31, model=regression_metric_model, source="regression/best.json"),
                _metric("explained_variance_mean", 0.32, model=regression_metric_model, source="regression/best.json"),
                _metric("rmse_mean", 11.0, "ug/mL", regression_metric_model, "regression/best.json"),
                _metric("mae_mean", 5.0, "ug/mL", regression_metric_model, "regression/best.json"),
                _metric("median_absolute_error_mean", 4.0, "ug/mL", regression_metric_model, "regression/best.json"),
                _metric("fold_count", 3, model=regression_metric_model, source="regression/rankings.csv"),
                _metric("sample_count", 80, model=regression_metric_model, source="regression/best.json"),
                _metric("concentration_min", 0.5, "ug/mL", regression_metric_model, "regression/best.json"),
                _metric("concentration_max", 50.0, "ug/mL", regression_metric_model, "regression/best.json"),
            ],
            "model_comparison": [{"model_name": "Extra Trees Regressor"}, {"model_name": "XGBoost Regressor"}],
        },
        "feature_engineering_results": {
            "best_feature_family": "window_features",
            "classification_improvement": 0.1,
            "regression_improvement": 0.2,
            "runtime_increase_seconds": 3.0,
            "feature_family_count": 2,
            "summary_source": "feature_engineering/summary.json",
            "benchmark_source": "feature_engineering/benchmark.csv",
        },
        "feature_selection_results": {
            "summary_rows": [{"task": "classification"} for _ in range(4)],
            "selected_feature_count": 4,
            "recommended_defaults": [],
            "summary_source": "feature_selection/summary.csv",
            "selected_features_source": "feature_selection/selected.csv",
        },
        "strain_results": {
            "leave_one_strain_count": 2,
            "chemical_specific_count": 3,
            "single_strain_count": 2,
            "loeo_source": "tables/loeo.csv",
            "chemical_source": "tables/chemical_strain.csv",
            "single_source": "tables/single.csv",
        },
        "limitations": [{"limitation": "Active QC limitation", "status": "ACTIVE"}],
    }


def _metric(name: str, value: Any, units: str | None = None, model: str | None = None, source: str | None = None) -> dict[str, Any]:
    return {
        "metric_name": name,
        "metric_value": value,
        "metric_units": units,
        "model_name": model,
        "source_file": source,
        "source_run": "fixture",
        "status": "SUPPORTED",
    }


def _dataset_provenance() -> list[dict[str, Any]]:
    return [
        _prov("P0001", "Dataset and Experimental Scope", "input_canonical_rows", 100, "rows", "", "features/summary.json"),
        _prov("P0002", "Dataset and Experimental Scope", "feature_rows", 10, "rows", "", "features/summary.json"),
        _prov("P0003", "Dataset and Experimental Scope", "core_feature_count", 2, "features", "", "features/summary.json"),
        _prov("P0004", "Dataset and Experimental Scope", "chemical_count", 3, "count", "", "qc/summary.json"),
        _prov("P0005", "Dataset and Experimental Scope", "strain_count", 2, "count", "", "qc/summary.json"),
    ]


def _qc_provenance() -> list[dict[str, Any]]:
    return [
        _prov("P0010", "Data Quality and Preprocessing", "canonical_error_count", 0, "count", "", "qc/summary.json"),
        _prov("P0011", "Data Quality and Preprocessing", "canonical_warning_count", 1, "count", "", "qc/summary.json"),
        _prov("P0012", "Data Quality and Preprocessing", "feature_failed_rows", 0, "rows", "", "features/summary.json"),
        _prov("P0013", "Data Quality and Preprocessing", "fingerprint_excluded_rows", 1, "rows", "", "fingerprints/summary.json"),
    ]


def _fingerprint_provenance() -> list[dict[str, Any]]:
    return [
        _prov("P0020", "Biosensor Fingerprint Analysis", "fingerprint_rows", 10, "rows", "", "fingerprints/summary.json"),
        _prov("P0021", "Biosensor Fingerprint Analysis", "consensus_fingerprint_rows", 4, "rows", "", "fingerprints/summary.json"),
        _prov("P0022", "Biosensor Fingerprint Analysis", "feature_count", 2, "features", "", "fingerprints/summary.json"),
    ]


def _exploratory_provenance() -> list[dict[str, Any]]:
    return [
        _prov("P0030", "Exploratory Analysis", "cumulative_explained_variance_ratio_pc3", 0.85, "", "", "exploratory/pca.csv"),
        _prov("P0031", "Exploratory Analysis", "cluster_count", 2, "count", "", "exploratory/clusters.csv"),
    ]


def _classification_provenance(model: str) -> list[dict[str, Any]]:
    metrics = {
        "accuracy_mean": 0.73,
        "balanced_accuracy_mean": 0.72,
        "f1_macro_mean": 0.71,
        "f1_weighted_mean": 0.74,
        "precision_macro_mean": 0.75,
        "recall_macro_mean": 0.76,
        "roc_auc_ovr_weighted_mean": 0.9,
        "log_loss_mean": 1.1,
        "fold_count": 3,
        "sample_count": 90,
        "class_count": 3,
    }
    rows = []
    for i, (name, value) in enumerate(metrics.items(), start=40):
        source = "classification/rankings.csv" if name == "fold_count" else "classification/best.json"
        rows.append(_prov(f"P{i:04d}", "Chemical Classification", name, value, "", model, source))
    return rows


def _regression_provenance(model: str) -> list[dict[str, Any]]:
    metrics = {
        "r2_mean": (0.31, ""),
        "explained_variance_mean": (0.32, ""),
        "rmse_mean": (11.0, "ug/mL"),
        "mae_mean": (5.0, "ug/mL"),
        "median_absolute_error_mean": (4.0, "ug/mL"),
        "fold_count": (3, ""),
        "sample_count": (80, ""),
        "concentration_min": (0.5, "ug/mL"),
        "concentration_max": (50.0, "ug/mL"),
    }
    rows = []
    for i, (name, (value, units)) in enumerate(metrics.items(), start=60):
        source = "regression/rankings.csv" if name == "fold_count" else "regression/best.json"
        rows.append(_prov(f"P{i:04d}", "Concentration Regression", name, value, units, model, source))
    return rows


def _feature_engineering_provenance() -> list[dict[str, Any]]:
    return [
        _prov("P0080", "Advanced Feature Engineering", "classification_improvement", 0.1, "", "", "feature_engineering/summary.json"),
        _prov("P0081", "Advanced Feature Engineering", "regression_improvement", 0.2, "", "", "feature_engineering/summary.json"),
        _prov("P0082", "Advanced Feature Engineering", "runtime_increase_seconds", 3.0, "seconds", "", "feature_engineering/summary.json"),
        _prov("P0083", "Advanced Feature Engineering", "feature_family_count", 2, "count", "", "feature_engineering/summary.json"),
    ]


def _feature_selection_provenance() -> list[dict[str, Any]]:
    return [_prov("P0090", "Feature Selection", "selected_feature_rows", 4, "rows", "", "feature_selection/selected.csv")]


def _strain_provenance() -> list[dict[str, Any]]:
    return [
        _prov("P0100", "Strain Contribution", "leave_one_strain_count", 2, "rows", "", "tables/loeo.csv"),
        _prov("P0101", "Strain Contribution", "chemical_specific_count", 3, "rows", "", "tables/chemical_strain.csv"),
    ]


def _prov(
    provenance_id: str,
    section: str,
    metric_name: str,
    metric_value: Any,
    units: str,
    model_name: str,
    source_file: str,
) -> dict[str, Any]:
    return {
        "provenance_id": provenance_id,
        "record_type": "metric",
        "section": section,
        "claim": f"{metric_name} claim",
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_units": units,
        "model_name": model_name,
        "source_file": source_file,
        "source_run": "fixture",
        "table_reference": "",
        "figure_reference": "",
        "status": "SUPPORTED",
        "notes": "",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
