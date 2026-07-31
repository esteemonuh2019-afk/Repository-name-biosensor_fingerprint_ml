"""Build a supervisor-ready results package from selected authoritative outputs."""

from __future__ import annotations

import csv
import json
import math
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

from .authoritative_source_loader import (
    filter_sources,
    first_source,
    load_selected_sources,
    read_csv_rows,
    read_json,
    read_text,
)
from .figure_selector import select_figures
from .provenance_tracker import ProvenanceTracker, write_provenance_csv
from .report_models import SelectedSource, SupervisorResultsPackage, empty_metric
from .scientific_interpreter import compose_scientific_interpretation, derive_limitations
from .table_builder import write_selected_tables_csv, write_tables_xlsx


CLASSIFICATION_PRIMARY_METRICS = [
    "f1_macro_mean",
    "f1_macro_std",
    "f1_weighted_mean",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "accuracy_mean",
    "accuracy_std",
    "precision_macro_mean",
    "recall_macro_mean",
    "roc_auc_ovr_weighted_mean",
    "log_loss_mean",
    "class_count",
    "sample_count",
    "fold_count",
]

REGRESSION_PRIMARY_METRICS = [
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "median_absolute_error_mean",
    "explained_variance_mean",
    "fold_count",
    "sample_count",
    "concentration_min",
    "concentration_max",
]


def build_supervisor_results_package(
    project_root: Path,
    selected_results_path: Path,
    title: str = "Biosensor Fingerprint ML Supervisor Results",
    author: str = "",
    supervisor_name: str = "",
) -> SupervisorResultsPackage:
    project_root = Path(project_root)
    selected_results_path = Path(selected_results_path)
    sources = load_selected_sources(project_root, selected_results_path)
    tracker = ProvenanceTracker()
    warnings = _source_warnings(sources)

    package = SupervisorResultsPackage(
        title=title,
        author=author,
        supervisor_name=supervisor_name,
        selected_results_file=str(selected_results_path),
        warnings=warnings,
    )

    package.dataset_summary = _build_dataset_summary(sources, tracker)
    package.quality_control_summary = _build_qc_summary(sources, tracker)
    package.fingerprint_summary = _build_fingerprint_summary(sources, tracker)
    package.exploratory_results = _build_exploratory_results(sources, tracker)
    package.classification_results = _build_classification_results(sources, tracker)
    package.regression_results = _build_regression_results(sources, tracker)
    package.feature_engineering_results = _build_feature_engineering_results(sources, tracker)
    package.feature_selection_results = _build_feature_selection_results(sources, tracker)
    package.strain_results = _build_strain_results(sources, tracker)

    blind_summary = _build_blind_prediction_context(sources, tracker)
    documentation_sources = [source.source_file for source in sources if source.analysis_type == "documentation" and source.exists]
    package.project_summary = {
        "title": title,
        "author": author,
        "supervisor_name": supervisor_name,
        "selected_results_file": str(selected_results_path),
        "listed_source_count": len(sources),
        "existing_source_count": sum(1 for source in sources if source.exists),
        "machine_readable_source_count": sum(
            1 for source in sources if source.exists and source.source_kind in {"csv", "json", "text"}
        ),
        "documentation_sources": documentation_sources,
        "blind_prediction_context": blind_summary,
        "_selected_sources": [source.__dict__ for source in sources],
    }

    package.limitations = derive_limitations(
        package.dataset_summary,
        package.quality_control_summary,
        blind_summary,
        documentation_sources,
    )
    package.conclusions = _build_conclusions(package)
    package.provenance = tracker.records
    package.selected_tables = _build_selected_tables(package)
    package.validation = validate_package(package)
    package.package_passed = bool(package.validation.get("passed"))
    return package


def write_supervisor_results_package(
    package: SupervisorResultsPackage,
    output_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_figures = select_figures(_sources_from_package(package), output_dir)
    package.selected_figures = selected_figures
    _add_figure_provenance(package)
    package.selected_tables = _build_selected_tables(package)
    package.validation = validate_package(package, output_dir=output_dir)
    package.package_passed = bool(package.validation.get("passed"))

    summary_json = output_dir / "supervisor_results_summary.json"
    tables_xlsx = output_dir / "supervisor_results_tables.xlsx"
    report_md = output_dir / "supervisor_results_report.md"
    report_docx = output_dir / "supervisor_results_report.docx"
    report_pdf = output_dir / "supervisor_results_report.pdf"
    figures_csv = output_dir / "selected_figures.csv"
    tables_csv = output_dir / "selected_tables.csv"
    provenance_csv = output_dir / "provenance_index.csv"
    validation_json = output_dir / "report_validation.json"

    report_text = render_markdown(package)
    summary_json.write_text(json.dumps(package.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    write_tables_xlsx(package.selected_tables, tables_xlsx)
    report_md.write_text(report_text, encoding="utf-8")
    _write_csv(package.selected_figures, figures_csv)
    write_selected_tables_csv(package.selected_tables, tables_csv)
    write_provenance_csv(package.provenance, provenance_csv)
    validation_json.write_text(json.dumps(package.validation, indent=2), encoding="utf-8")

    if package.package_passed:
        write_minimal_docx(package, report_docx)
        write_simple_pdf(report_text, report_pdf, title=package.title)

    return {
        "output_dir": str(output_dir),
        "summary_json": str(summary_json),
        "tables_xlsx": str(tables_xlsx),
        "report_md": str(report_md),
        "report_docx": str(report_docx) if report_docx.exists() else None,
        "report_pdf": str(report_pdf) if report_pdf.exists() else None,
        "selected_figures_csv": str(figures_csv),
        "selected_tables_csv": str(tables_csv),
        "provenance_csv": str(provenance_csv),
        "validation_json": str(validation_json),
        "package_passed": package.package_passed,
        "validation": package.validation,
    }


def _source_warnings(sources: Iterable[SelectedSource]) -> List[str]:
    warnings = []
    for source in sources:
        if not source.exists:
            warnings.append(f"Listed source is unavailable: {source.source_file}")
        elif source.source_kind == "unsupported":
            warnings.append(f"Unsupported listed source format: {source.source_file}")
    return warnings


def _sources_from_package(package: SupervisorResultsPackage) -> List[SelectedSource]:
    payload = package.project_summary.get("_selected_sources", [])
    return [SelectedSource(**source) for source in payload]


def _store_sources(package_section: Dict[str, Any], sources: List[SelectedSource]) -> None:
    package_section["_selected_sources"] = [source.__dict__ for source in sources]


def _build_dataset_summary(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    feature_summary_source = first_source(sources, "feature extraction", "feature_summary.json")
    feature_summary = read_json(feature_summary_source)
    qc_source = first_source(sources, "canonical QC; feature validation", "qc_summary.json")
    qc_summary = read_json(qc_source)
    feature_csv_source = first_source(sources, "feature extraction", "feature_dataset.csv", primary=True)

    summary = feature_summary.get("summary", {})
    qc = qc_summary
    dataset = {
        "feature_rows": summary.get("feature_rows"),
        "core_feature_count": summary.get("core_feature_count"),
        "core_features": summary.get("core_features", []),
        "input_canonical_rows": summary.get("input_canonical_rows") or qc.get("row_count"),
        "chemical_count": len(qc.get("chemicals_detected", [])),
        "chemicals_detected": qc.get("chemicals_detected", []),
        "strain_count": len(qc.get("strains_detected", [])),
        "strains_detected": qc.get("strains_detected", []),
        "concentration_levels": qc.get("concentrations_detected", []),
        "source_file_count": len(qc.get("source_files", [])),
        "source_files": qc.get("source_files", []),
        "feature_dataset_source": feature_csv_source.source_file if feature_csv_source else None,
        "feature_summary_source": feature_summary_source.source_file if feature_summary_source else None,
        "canonical_qc_source": qc_source.source_file if qc_source else None,
    }
    for metric in ["feature_rows", "core_feature_count", "input_canonical_rows", "chemical_count", "strain_count", "source_file_count"]:
        tracker.add(
            "metric",
            "Dataset and Experimental Scope",
            f"Dataset summary metric {metric}",
            feature_summary_source if metric in {"feature_rows", "core_feature_count", "input_canonical_rows"} else qc_source,
            metric_name=metric,
            metric_value=dataset.get(metric),
            table_reference="dataset_overview",
        )
    return dataset


def _build_qc_summary(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    canonical_source = first_source(sources, "canonical QC; feature validation", "qc_summary.json")
    feature_source = first_source(sources, "feature extraction", "feature_summary.json")
    fingerprint_source = first_source(sources, "fingerprint generation", "fingerprint_summary.json")
    canonical = read_json(canonical_source)
    feature = read_json(feature_source)
    fingerprint = read_json(fingerprint_source)
    feature_qc = feature.get("qc", {})
    fingerprint_qc = fingerprint.get("qc", {})

    summary = {
        "canonical_qc_passed": canonical.get("qc_passed"),
        "canonical_error_count": len(canonical.get("errors", [])),
        "canonical_warning_count": len(canonical.get("warnings", [])),
        "canonical_row_count": canonical.get("row_count"),
        "feature_qc_passed": feature_qc.get("passed"),
        "feature_failed_rows": feature_qc.get("failed_feature_rows"),
        "feature_warning_rows": feature_qc.get("warning_feature_rows"),
        "feature_missing_value_count": feature_qc.get("missing_feature_value_count"),
        "fingerprint_qc_passed": fingerprint_qc.get("passed"),
        "fingerprint_excluded_rows": fingerprint_qc.get("excluded_rows"),
        "fingerprint_warning_count": len(fingerprint_qc.get("warnings", [])),
        "canonical_qc_source": canonical_source.source_file if canonical_source else None,
        "feature_summary_source": feature_source.source_file if feature_source else None,
        "fingerprint_summary_source": fingerprint_source.source_file if fingerprint_source else None,
    }
    for metric, source in [
        ("canonical_row_count", canonical_source),
        ("canonical_error_count", canonical_source),
        ("canonical_warning_count", canonical_source),
        ("feature_failed_rows", feature_source),
        ("feature_warning_rows", feature_source),
        ("feature_missing_value_count", feature_source),
        ("fingerprint_excluded_rows", fingerprint_source),
        ("fingerprint_warning_count", fingerprint_source),
    ]:
        tracker.add(
            "metric",
            "Data Quality and Preprocessing",
            f"QC summary metric {metric}",
            source,
            metric_name=metric,
            metric_value=summary.get(metric),
            table_reference="qc_summary",
        )
    return summary


def _build_fingerprint_summary(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    source = first_source(sources, "fingerprint generation", "fingerprint_summary.json")
    data = read_json(source)
    summary = data.get("summary", {})
    fingerprint = {
        "fingerprint_rows": summary.get("fingerprint_rows"),
        "consensus_fingerprint_rows": summary.get("consensus_fingerprint_rows"),
        "feature_count": summary.get("feature_count"),
        "normalization_method": summary.get("normalization_method"),
        "distance_matrix_rows": summary.get("distance_matrix_rows"),
        "distance_matrix_columns": summary.get("distance_matrix_columns"),
        "duplicate_fingerprint_row_count": summary.get("duplicate_fingerprint_row_count"),
        "excluded_rows": summary.get("excluded_rows"),
        "source_file": source.source_file if source else None,
    }
    for metric in [
        "fingerprint_rows",
        "consensus_fingerprint_rows",
        "feature_count",
        "distance_matrix_rows",
        "distance_matrix_columns",
        "duplicate_fingerprint_row_count",
        "excluded_rows",
    ]:
        tracker.add(
            "metric",
            "Biosensor Fingerprint Analysis",
            f"Fingerprint summary metric {metric}",
            source,
            metric_name=metric,
            metric_value=fingerprint.get(metric),
            table_reference="fingerprint_summary",
        )
    return fingerprint


def _build_exploratory_results(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    pca_source = first_source(sources, "exploratory analysis", "pca_explained_variance.csv", primary=True)
    cluster_source = first_source(sources, "exploratory analysis", "cluster_assignments.csv")
    composition_source = first_source(sources, "exploratory analysis", "cluster_composition.csv")
    pca_rows = read_csv_rows(pca_source)
    cluster_rows = read_csv_rows(cluster_source)
    cluster_ids = sorted({row.get("cluster_id") for row in cluster_rows if row.get("cluster_id")})
    first_three = pca_rows[:3]
    cumulative_pc3 = _to_number(first_three[-1].get("cumulative_explained_variance_ratio")) if len(first_three) >= 3 else None
    result = {
        "pca_explained_variance": first_three,
        "cumulative_variance_pc3": cumulative_pc3,
        "cluster_count": len(cluster_ids),
        "cluster_assignment_rows": len(cluster_rows),
        "cluster_composition_rows": len(read_csv_rows(composition_source)),
        "pca_source": pca_source.source_file if pca_source else None,
        "cluster_source": cluster_source.source_file if cluster_source else None,
    }
    tracker.add(
        "metric",
        "Exploratory Analysis",
        "Cumulative explained variance through PC3",
        pca_source,
        metric_name="cumulative_explained_variance_ratio_pc3",
        metric_value=cumulative_pc3,
        table_reference="exploratory_pca",
    )
    tracker.add(
        "metric",
        "Exploratory Analysis",
        "Number of clusters in selected cluster assignments",
        cluster_source,
        metric_name="cluster_count",
        metric_value=len(cluster_ids),
        table_reference="exploratory_clusters",
    )
    return result


def _build_classification_results(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    best_source = first_source(sources, "classification", "best_model_metrics.json", primary=True)
    ranking_source = first_source(sources, "classification", "model_rankings.csv")
    per_class_source = first_source(sources, "classification", "per_class_metrics.csv")
    confusion_source = first_source(sources, "classification", "confusion_matrix.csv")
    best = read_json(best_source)
    rankings = read_csv_rows(ranking_source)
    selected_model = best.get("model_name") or _first_nonempty(best.get("best_model"), best.get("model_id"))
    selected_row = _find_model_row(rankings, selected_model)

    metrics = []
    for metric in CLASSIFICATION_PRIMARY_METRICS:
        value = best.get(metric)
        source = best_source
        if value is None and selected_row and metric in selected_row:
            value = _clean_value(selected_row.get(metric))
            source = ranking_source
        metrics.append(_metric_record(metric, value, None, selected_model, source, selected_model))
        tracker.add(
            "metric",
            "Chemical Classification",
            f"Selected classifier metric {metric}",
            source,
            metric_name=metric,
            metric_value=value,
            model_name=selected_model,
            table_reference="selected_classifier_metrics",
            status="SUPPORTED" if value is not None else "MISSING",
        )

    comparison = _with_selection_status(rankings, selected_model)
    per_class = read_csv_rows(per_class_source)
    result = {
        "selected_model": {
            "model_name": selected_model,
            "model_id": best.get("model_id"),
            "rank": best.get("rank"),
            "selection_metric": best.get("selection_metric"),
            "source_file": best_source.source_file if best_source else None,
        },
        "selected_metrics": metrics,
        "model_comparison": comparison,
        "per_class_metrics": per_class,
        "confusion_matrix": read_csv_rows(confusion_source),
        "comparison_source": ranking_source.source_file if ranking_source else None,
        "per_class_source": per_class_source.source_file if per_class_source else None,
        "confusion_source": confusion_source.source_file if confusion_source else None,
    }
    return result


def _build_regression_results(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    best_source = first_source(sources, "regression", "best_regression_model.json", primary=True)
    ranking_source = first_source(sources, "regression", "model_rankings.csv")
    residual_source = first_source(sources, "regression", "residuals.csv")
    prediction_source = first_source(sources, "regression", "prediction_vs_actual.csv")
    best = read_json(best_source)
    rankings = read_csv_rows(ranking_source)
    selected_model = best.get("model_name") or _first_nonempty(best.get("best_model"), best.get("model_id"))
    selected_row = _find_model_row(rankings, selected_model)
    units = best.get("target_units")

    metrics = []
    for metric in REGRESSION_PRIMARY_METRICS:
        value = best.get(metric)
        source = best_source
        if value is None and selected_row and metric in selected_row:
            value = _clean_value(selected_row.get(metric))
            source = ranking_source
        metrics.append(_metric_record(metric, value, units if _metric_uses_target_units(metric) else None, selected_model, source, selected_model))
        tracker.add(
            "metric",
            "Concentration Regression",
            f"Selected regressor metric {metric}",
            source,
            metric_name=metric,
            metric_value=value,
            metric_units=units if _metric_uses_target_units(metric) else None,
            model_name=selected_model,
            table_reference="selected_regressor_metrics",
            status="SUPPORTED" if value is not None else "MISSING",
        )

    result = {
        "selected_model": {
            "model_name": selected_model,
            "model_id": best.get("model_id"),
            "rank": best.get("rank"),
            "selection_metric": best.get("selection_metric"),
            "target_units": units,
            "source_file": best_source.source_file if best_source else None,
        },
        "selected_metrics": metrics,
        "model_comparison": _with_selection_status(rankings, selected_model),
        "residual_rows": len(read_csv_rows(residual_source)),
        "prediction_vs_actual_rows": len(read_csv_rows(prediction_source)),
        "comparison_source": ranking_source.source_file if ranking_source else None,
        "residual_source": residual_source.source_file if residual_source else None,
        "prediction_vs_actual_source": prediction_source.source_file if prediction_source else None,
    }
    return result


def _build_feature_engineering_results(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    summary_source = first_source(sources, "advanced feature engineering", "stage_8c_summary.json")
    ablation_source = first_source(sources, "advanced feature engineering", "feature_family_ablation_summary.csv")
    summary = read_json(summary_source)
    metadata = summary.get("metadata", {})
    rows = read_csv_rows(ablation_source)
    result = {
        "best_feature_family": summary.get("best_feature_family") or metadata.get("best_feature_family"),
        "classification_improvement": summary.get("classification_improvement"),
        "regression_improvement": summary.get("regression_improvement"),
        "runtime_increase_seconds": summary.get("runtime_increase_seconds"),
        "feature_family_count": metadata.get("feature_family_count"),
        "feature_set_count": metadata.get("feature_set_count"),
        "feature_family_benchmark": rows,
        "summary_source": summary_source.source_file if summary_source else None,
        "benchmark_source": ablation_source.source_file if ablation_source else None,
    }
    for metric in ["classification_improvement", "regression_improvement", "runtime_increase_seconds", "feature_family_count"]:
        tracker.add(
            "metric",
            "Advanced Feature Engineering",
            f"Feature engineering metric {metric}",
            summary_source,
            metric_name=metric,
            metric_value=result.get(metric),
            table_reference="feature_family_benchmark",
        )
    return result


def _build_feature_selection_results(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    summary_source = first_source(sources, "feature selection", "feature_selection_summary.csv")
    selected_source = first_source(sources, "feature selection", "selected_features.csv")
    performance_source = first_source(sources, "feature selection", "performance_vs_feature_count.csv")
    rows = read_csv_rows(summary_source)
    recommended = [row for row in rows if str(row.get("recommended_default", "")).lower() == "true"]
    selected_features = read_csv_rows(selected_source)
    result = {
        "recommended_defaults": recommended,
        "summary_rows": rows,
        "selected_feature_rows": selected_features,
        "selected_feature_count": len(selected_features),
        "performance_vs_feature_count": read_csv_rows(performance_source),
        "summary_source": summary_source.source_file if summary_source else None,
        "selected_features_source": selected_source.source_file if selected_source else None,
        "performance_source": performance_source.source_file if performance_source else None,
    }
    tracker.add(
        "metric",
        "Feature Selection",
        "Number of selected feature records",
        selected_source,
        metric_name="selected_feature_rows",
        metric_value=len(selected_features),
        table_reference="feature_selection_results",
    )
    for row in recommended:
        metric_name = "macro_f1_mean" if row.get("task") == "classification" else "r2_mean"
        tracker.add(
            "metric",
            "Feature Selection",
            f"Recommended {row.get('task')} feature-selection metric",
            summary_source,
            metric_name=metric_name,
            metric_value=_clean_value(row.get(metric_name)),
            model_name=row.get("model_name"),
            table_reference="feature_selection_results",
        )
    return result


def _build_strain_results(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    loeo_source = first_source(sources, "strain ablation", "leave_one_strain_out_loeo.csv")
    chemical_source = first_source(sources, "strain ablation", "chemical_specific_strain_rankings.csv")
    single_source = first_source(sources, "strain ablation", "single_strain_loeo.csv")
    loeo_rows = read_csv_rows(loeo_source)
    chemical_rows = read_csv_rows(chemical_source)
    single_rows = read_csv_rows(single_source)
    result = {
        "leave_one_strain_rows": loeo_rows,
        "chemical_specific_rows": chemical_rows,
        "single_strain_rows": single_rows,
        "leave_one_strain_count": len(loeo_rows),
        "chemical_specific_count": len(chemical_rows),
        "single_strain_count": len(single_rows),
        "loeo_source": loeo_source.source_file if loeo_source else None,
        "chemical_source": chemical_source.source_file if chemical_source else None,
        "single_source": single_source.source_file if single_source else None,
    }
    tracker.add(
        "metric",
        "Strain Contribution",
        "Leave-one-strain result count",
        loeo_source,
        metric_name="leave_one_strain_count",
        metric_value=len(loeo_rows),
        table_reference="strain_contribution",
    )
    tracker.add(
        "metric",
        "Strain Contribution",
        "Chemical-specific strain ranking count",
        chemical_source,
        metric_name="chemical_specific_count",
        metric_value=len(chemical_rows),
        table_reference="strain_contribution",
    )
    return result


def _build_blind_prediction_context(sources: List[SelectedSource], tracker: ProvenanceTracker) -> Dict[str, Any]:
    source = first_source(sources, "real blind validation", "blind_prediction_summary.json")
    data = read_json(source)
    if not data:
        return {"true_labels_included": None, "source_file": source.source_file if source else None}
    result = {
        "prediction_passed": data.get("prediction_passed"),
        "true_labels_included": data.get("true_labels_included"),
        "predicted_chemical": data.get("predicted_chemical"),
        "chemical_confidence": data.get("chemical_confidence"),
        "predicted_concentration": data.get("predicted_concentration"),
        "concentration_units": data.get("concentration_units"),
        "novelty_status": data.get("novelty_status"),
        "source_file": source.source_file if source else None,
    }
    tracker.add(
        "metric",
        "Limitations",
        "Blind-prediction true-label availability",
        source,
        metric_name="true_labels_included",
        metric_value=data.get("true_labels_included"),
        table_reference="limitations_status",
        notes="False means prediction context only, not blind-validation performance.",
    )
    return result


def _build_conclusions(package: SupervisorResultsPackage) -> List[str]:
    classifier = package.classification_results.get("selected_model", {}).get("model_name") or "MISSING"
    regressor = package.regression_results.get("selected_model", {}).get("model_name") or "MISSING"
    return [
        f"Authoritative benchmark outputs selected {classifier} for chemical classification.",
        f"Authoritative benchmark outputs selected {regressor} for concentration regression.",
        "Primary model metrics are reported separately from feature-family, feature-selection, and strain-ablation results.",
        "External validation remains a future requirement because selected blind-prediction output did not include true labels.",
    ]


def _build_selected_tables(package: SupervisorResultsPackage) -> List[Dict[str, Any]]:
    rows = [
        _table("dataset_overview", "Dataset Overview", package.dataset_summary.get("feature_summary_source"), _dataset_rows(package)),
        _table("qc_summary", "QC Summary", package.quality_control_summary.get("canonical_qc_source"), _qc_rows(package)),
        _table("fingerprint_summary", "Fingerprint Summary", package.fingerprint_summary.get("source_file"), _fingerprint_rows(package)),
        _table("exploratory_pca", "Exploratory PCA", package.exploratory_results.get("pca_source"), package.exploratory_results.get("pca_explained_variance", [])),
        _table("classifier_comparison", "Classifier Comparison", package.classification_results.get("comparison_source"), package.classification_results.get("model_comparison", [])),
        _table("selected_classifier_metrics", "Selected Classifier Metrics", package.classification_results.get("selected_model", {}).get("source_file"), package.classification_results.get("selected_metrics", [])),
        _table("per_class_metrics", "Per-Class Metrics", package.classification_results.get("per_class_source"), package.classification_results.get("per_class_metrics", [])),
        _table("regressor_comparison", "Regressor Comparison", package.regression_results.get("comparison_source"), package.regression_results.get("model_comparison", [])),
        _table("selected_regressor_metrics", "Selected Regressor Metrics", package.regression_results.get("selected_model", {}).get("source_file"), package.regression_results.get("selected_metrics", [])),
        _table("feature_family_benchmark", "Feature-Family Benchmark", package.feature_engineering_results.get("benchmark_source"), package.feature_engineering_results.get("feature_family_benchmark", [])),
        _table("feature_selection_results", "Feature-Selection Results", package.feature_selection_results.get("summary_source"), package.feature_selection_results.get("summary_rows", [])),
        _table("strain_contribution", "Strain Contribution", package.strain_results.get("loeo_source"), package.strain_results.get("leave_one_strain_rows", [])),
        _table("limitations_status", "Limitations and Status", None, package.limitations),
    ]
    return rows


def _table(table_id: str, title: str, source_file: Optional[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "table_id": table_id,
        "title": title,
        "source_file": source_file,
        "row_count": len(rows),
        "rows": rows,
        "status": "POPULATED" if rows else "MISSING",
        "notes": "" if rows else "No rows available from selected sources.",
    }


def _dataset_rows(package: SupervisorResultsPackage) -> List[Dict[str, Any]]:
    summary = package.dataset_summary
    return [
        {"metric": "input_canonical_rows", "value": summary.get("input_canonical_rows"), "units": "rows"},
        {"metric": "feature_rows", "value": summary.get("feature_rows"), "units": "rows"},
        {"metric": "core_feature_count", "value": summary.get("core_feature_count"), "units": "features"},
        {"metric": "chemical_count", "value": summary.get("chemical_count"), "units": "chemicals"},
        {"metric": "strain_count", "value": summary.get("strain_count"), "units": "strains"},
        {"metric": "source_file_count", "value": summary.get("source_file_count"), "units": "files"},
        {"metric": "concentration_levels", "value": "; ".join(summary.get("concentration_levels", [])), "units": None},
        {"metric": "chemicals_detected", "value": "; ".join(summary.get("chemicals_detected", [])), "units": None},
        {"metric": "strains_detected", "value": "; ".join(summary.get("strains_detected", [])), "units": None},
    ]


def _qc_rows(package: SupervisorResultsPackage) -> List[Dict[str, Any]]:
    qc = package.quality_control_summary
    return [
        {"qc_stage": "canonical", "status": qc.get("canonical_qc_passed"), "metric": "row_count", "value": qc.get("canonical_row_count"), "source_file": qc.get("canonical_qc_source")},
        {"qc_stage": "canonical", "status": qc.get("canonical_qc_passed"), "metric": "error_count", "value": qc.get("canonical_error_count"), "source_file": qc.get("canonical_qc_source")},
        {"qc_stage": "canonical", "status": qc.get("canonical_qc_passed"), "metric": "warning_count", "value": qc.get("canonical_warning_count"), "source_file": qc.get("canonical_qc_source")},
        {"qc_stage": "feature", "status": qc.get("feature_qc_passed"), "metric": "failed_rows", "value": qc.get("feature_failed_rows"), "source_file": qc.get("feature_summary_source")},
        {"qc_stage": "feature", "status": qc.get("feature_qc_passed"), "metric": "warning_rows", "value": qc.get("feature_warning_rows"), "source_file": qc.get("feature_summary_source")},
        {"qc_stage": "fingerprint", "status": qc.get("fingerprint_qc_passed"), "metric": "excluded_rows", "value": qc.get("fingerprint_excluded_rows"), "source_file": qc.get("fingerprint_summary_source")},
    ]


def _fingerprint_rows(package: SupervisorResultsPackage) -> List[Dict[str, Any]]:
    summary = package.fingerprint_summary
    return [
        {"metric": key, "value": value, "source_file": summary.get("source_file")}
        for key, value in summary.items()
        if key != "source_file"
    ]


def _metric_record(
    metric_name: str,
    value: Any,
    units: Optional[str],
    model_name: Optional[str],
    source: Optional[SelectedSource],
    selected_model: Optional[str],
) -> Dict[str, Any]:
    if value is None or value == "":
        return empty_metric(metric_name, selected_model)
    return {
        "metric_name": metric_name,
        "metric_value": _clean_value(value),
        "metric_units": units,
        "model_name": model_name,
        "source_file": source.source_file if source else None,
        "source_run": source.selected_run if source else None,
        "status": "SUPPORTED",
        "notes": "Authoritative selected-model metric.",
    }


def _find_model_row(rows: List[Dict[str, Any]], model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not model_name:
        return rows[0] if rows else None
    for row in rows:
        if row.get("model_name") == model_name:
            return row
    return None


def _with_selection_status(rows: List[Dict[str, Any]], selected_model: Optional[str]) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        copied["selection_status"] = "SELECTED" if row.get("model_name") == selected_model else "COMPARISON"
        enriched.append(copied)
    return enriched


def _first_nonempty(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _to_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _clean_value(value: Any) -> Any:
    number = _to_number(value)
    if number is not None:
        if number.is_integer():
            return int(number)
        return number
    if value == "":
        return None
    return value


def _metric_uses_target_units(metric: str) -> bool:
    return metric.startswith("rmse") or metric.startswith("mae") or metric.startswith("median_absolute_error") or metric.startswith("concentration_")


def validate_package(package: SupervisorResultsPackage, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    checks = []
    selected_classifier = package.classification_results.get("selected_model", {}).get("model_name")
    selected_regressor = package.regression_results.get("selected_model", {}).get("model_name")
    checks.append(_check_model_metrics("classification_primary_model_coherent", selected_classifier, package.classification_results.get("selected_metrics", [])))
    checks.append(_check_model_metrics("regression_primary_model_coherent", selected_regressor, package.regression_results.get("selected_metrics", [])))
    checks.append(
        {
            "check": "no_model_metric_mixing",
            "passed": checks[0]["passed"] and checks[1]["passed"],
            "details": "Primary metrics reference one selected classifier and one selected regressor.",
        }
    )
    checks.append(_check_figures(package, output_dir))
    checks.append(_check_table_sources(package))
    checks.append(_check_provenance(package))
    checks.append(
        {
            "check": "real_blind_validation_not_claimed",
            "passed": package.project_summary.get("blind_prediction_context", {}).get("true_labels_included") is not True,
            "details": "Blind-prediction output is described as prediction context unless true labels are present.",
        }
    )
    checks.append(
        {
            "check": "chemical_and_strain_names_preserved",
            "passed": bool(package.dataset_summary.get("chemicals_detected")) and bool(package.dataset_summary.get("strains_detected")),
            "details": "Chemical and strain names are copied from selected QC summary without normalization.",
        }
    )
    regression_units = [
        metric.get("metric_units")
        for metric in package.regression_results.get("selected_metrics", [])
        if _metric_uses_target_units(metric.get("metric_name", ""))
        and metric.get("metric_value") is not None
    ]
    checks.append(
        {
            "check": "units_reported_where_available",
            "passed": all(unit for unit in regression_units),
            "details": "Target-unit regression metrics include units when values are present.",
        }
    )
    checks.append(
        {
            "check": "missing_information_marked",
            "passed": all(table.get("status") in {"POPULATED", "MISSING"} for table in package.selected_tables),
            "details": "Tables and missing selected metrics carry status fields.",
        }
    )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _check_model_metrics(check_name: str, selected_model: Optional[str], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    populated = [metric for metric in metrics if metric.get("status") == "SUPPORTED"]
    models = sorted({metric.get("model_name") for metric in populated if metric.get("model_name")})
    passed = bool(selected_model) and set(models).issubset({selected_model})
    return {
        "check": check_name,
        "passed": passed,
        "details": f"selected_model={selected_model}; populated_metric_models={models}",
    }


def _check_figures(package: SupervisorResultsPackage, output_dir: Optional[Path]) -> Dict[str, Any]:
    if output_dir is None:
        return {"check": "selected_figures_exist", "passed": True, "details": "Figure files checked during write."}
    missing = []
    for figure in package.selected_figures:
        output_file = figure.get("output_file")
        if output_file and not (output_dir / output_file).exists():
            missing.append(output_file)
    return {
        "check": "selected_figures_exist",
        "passed": not missing,
        "details": "All selected figures exist." if not missing else f"Missing figures: {missing}",
    }


def _check_table_sources(package: SupervisorResultsPackage) -> Dict[str, Any]:
    missing = [
        table["table_id"]
        for table in package.selected_tables
        if table.get("status") == "POPULATED" and table.get("source_file") is None and table["table_id"] != "limitations_status"
    ]
    return {
        "check": "table_sources_exist",
        "passed": not missing,
        "details": "All populated data tables have source files." if not missing else f"Tables without source: {missing}",
    }


def _check_provenance(package: SupervisorResultsPackage) -> Dict[str, Any]:
    unsupported = [
        record
        for record in package.provenance
        if record.get("record_type") == "metric" and record.get("status") == "SUPPORTED" and not record.get("source_file")
    ]
    return {
        "check": "no_unsupported_quantitative_claims",
        "passed": not unsupported,
        "details": "All supported quantitative metric claims have source files.",
    }


def _add_figure_provenance(package: SupervisorResultsPackage) -> None:
    existing_ids = {record.get("figure_reference") for record in package.provenance}
    for figure in package.selected_figures:
        figure_id = figure.get("figure_id")
        if figure_id in existing_ids:
            continue
        package.provenance.append(
            {
                "provenance_id": f"P{len(package.provenance) + 1:04d}",
                "record_type": "figure",
                "section": "Selected Figures",
                "claim": f"Selected figure {figure_id}",
                "metric_name": None,
                "metric_value": None,
                "metric_units": None,
                "model_name": None,
                "source_file": figure.get("source_file"),
                "source_run": figure.get("source_run"),
                "table_reference": None,
                "figure_reference": figure_id,
                "status": figure.get("status"),
                "notes": figure.get("notes"),
            }
        )


def render_markdown(package: SupervisorResultsPackage) -> str:
    classification_model = package.classification_results.get("selected_model", {})
    regression_model = package.regression_results.get("selected_model", {})
    lines = [
        f"# {package.title}",
        "",
        f"Generated: {package.generated_at}",
    ]
    if package.author:
        lines.append(f"Author: {package.author}")
    if package.supervisor_name:
        lines.append(f"Supervisor: {package.supervisor_name}")
    lines.extend(
        [
            "",
            "## Executive Summary",
            "",
            f"- Selected classifier: {classification_model.get('model_name') or 'MISSING'}.",
            f"- Selected regressor: {regression_model.get('model_name') or 'MISSING'}.",
            f"- Listed sources parsed: {package.project_summary.get('existing_source_count')} of {package.project_summary.get('listed_source_count')}.",
            f"- Package validation: {'PASS' if package.package_passed else 'FAIL'}.",
            "",
            "## Dataset and Experimental Scope",
            "",
            _sentence_metric("Input canonical rows", package.dataset_summary.get("input_canonical_rows"), "rows"),
            _sentence_metric("Feature rows", package.dataset_summary.get("feature_rows"), "rows"),
            _sentence_metric("Core features", package.dataset_summary.get("core_feature_count"), "features"),
            f"Chemicals: {', '.join(package.dataset_summary.get('chemicals_detected', [])) or 'MISSING'}.",
            f"Strains: {', '.join(package.dataset_summary.get('strains_detected', [])) or 'MISSING'}.",
            "",
            "## Data Quality and Preprocessing",
            "",
            _sentence_metric("Canonical QC errors", package.quality_control_summary.get("canonical_error_count"), "records"),
            _sentence_metric("Canonical QC warnings", package.quality_control_summary.get("canonical_warning_count"), "warnings"),
            _sentence_metric("Feature QC failed rows", package.quality_control_summary.get("feature_failed_rows"), "rows"),
            _sentence_metric("Fingerprint excluded rows", package.quality_control_summary.get("fingerprint_excluded_rows"), "rows"),
            "",
            "## Biosensor Fingerprint Analysis",
            "",
            _sentence_metric("Fingerprint rows", package.fingerprint_summary.get("fingerprint_rows"), "rows"),
            _sentence_metric("Consensus fingerprints", package.fingerprint_summary.get("consensus_fingerprint_rows"), "rows"),
            f"Normalization method: {package.fingerprint_summary.get('normalization_method') or 'MISSING'}.",
            "",
            "## Exploratory Analysis",
            "",
            _sentence_metric("Cumulative variance through PC3", package.exploratory_results.get("cumulative_variance_pc3"), None),
            _sentence_metric("Cluster assignments", package.exploratory_results.get("cluster_assignment_rows"), "rows"),
            "",
            "## Chemical Classification",
            "",
            f"The authoritative classification best-model file selected {classification_model.get('model_name') or 'MISSING'}.",
            _metric_lines(package.classification_results.get("selected_metrics", [])),
            "",
            "## Concentration Regression",
            "",
            f"The authoritative regression best-model file selected {regression_model.get('model_name') or 'MISSING'}.",
            _metric_lines(package.regression_results.get("selected_metrics", [])),
            "",
            "## Advanced Feature Engineering",
            "",
            f"Best feature family reported by Stage 8C: {package.feature_engineering_results.get('best_feature_family') or 'MISSING'}.",
            _sentence_metric("Classification improvement", package.feature_engineering_results.get("classification_improvement"), None),
            _sentence_metric("Regression improvement", package.feature_engineering_results.get("regression_improvement"), None),
            "",
            "## Feature Selection",
            "",
            _sentence_metric("Feature-selection summary rows", len(package.feature_selection_results.get("summary_rows", [])), "rows"),
            _sentence_metric("Selected feature records", package.feature_selection_results.get("selected_feature_count"), "rows"),
            "",
            "## Strain Contribution",
            "",
            _sentence_metric("Leave-one-strain rows", package.strain_results.get("leave_one_strain_count"), "rows"),
            _sentence_metric("Chemical-specific strain rows", package.strain_results.get("chemical_specific_count"), "rows"),
            "",
            "## Scientific Interpretation",
            "",
        ]
    )
    for statement in compose_scientific_interpretation(package):
        lines.append(f"- {statement}")
    lines.extend(["", "## Limitations", ""])
    for limitation in package.limitations:
        lines.append(f"- {limitation.get('limitation')} Status: {limitation.get('status')}.")
    lines.extend(["", "## Conclusions and Next Steps", ""])
    for conclusion in package.conclusions:
        lines.append(f"- {conclusion}")
    lines.extend(["", "## Selected Figures", ""])
    if package.selected_figures:
        for figure in package.selected_figures:
            lines.append(f"- {figure.get('figure_id')}: {figure.get('output_file')} (source: {figure.get('source_file')})")
    else:
        lines.append("- No readable inventory-listed figures were selected.")
    lines.extend(["", "## Selected Tables", ""])
    for table in package.selected_tables:
        lines.append(f"- {table['table_id']}: {table['title']} ({table['row_count']} rows; {table['status']})")
    lines.extend(["", "## Provenance", ""])
    lines.append(f"Quantitative provenance records: {len([p for p in package.provenance if p.get('record_type') == 'metric'])}.")
    return "\n".join(lines) + "\n"


def _sentence_metric(label: str, value: Any, units: Optional[str]) -> str:
    if value in (None, ""):
        return f"{label}: MISSING."
    suffix = f" {units}" if units else ""
    return f"{label}: {_format_value(value)}{suffix}."


def _metric_lines(metrics: List[Dict[str, Any]]) -> str:
    lines = []
    for metric in metrics:
        value = metric.get("metric_value")
        units = metric.get("metric_units")
        if value is None:
            lines.append(f"- {metric.get('metric_name')}: MISSING.")
        else:
            suffix = f" {units}" if units else ""
            lines.append(f"- {metric.get('metric_name')}: {_format_value(value)}{suffix}.")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else [
        "figure_id",
        "title",
        "source_file",
        "source_run",
        "output_file",
        "status",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_minimal_docx(package: SupervisorResultsPackage, output_path: Path) -> None:
    body = []
    body.append(_docx_paragraph(package.title, style="Title"))
    if package.author or package.supervisor_name:
        body.append(_docx_paragraph(f"Author: {package.author or 'MISSING'}    Supervisor: {package.supervisor_name or 'MISSING'}"))
    body.append(_docx_paragraph(f"Generated: {package.generated_at}"))
    sections = [
        ("Executive Summary", [
            f"Selected classifier: {package.classification_results.get('selected_model', {}).get('model_name') or 'MISSING'}.",
            f"Selected regressor: {package.regression_results.get('selected_model', {}).get('model_name') or 'MISSING'}.",
            f"Package validation: {'PASS' if package.package_passed else 'FAIL'}.",
        ]),
        ("Dataset and Experimental Scope", _plain_dataset_lines(package)),
        ("Data Quality and Preprocessing", _plain_qc_lines(package)),
        ("Biosensor Fingerprint Analysis", _plain_fingerprint_lines(package)),
        ("Exploratory Analysis", _plain_exploratory_lines(package)),
        ("Chemical Classification", _plain_metric_lines(package.classification_results.get("selected_metrics", []))),
        ("Concentration Regression", _plain_metric_lines(package.regression_results.get("selected_metrics", []))),
        ("Advanced Feature Engineering", [
            f"Best feature family: {package.feature_engineering_results.get('best_feature_family') or 'MISSING'}.",
            _sentence_metric("Classification improvement", package.feature_engineering_results.get("classification_improvement"), None),
            _sentence_metric("Regression improvement", package.feature_engineering_results.get("regression_improvement"), None),
        ]),
        ("Feature Selection", [
            _sentence_metric("Summary rows", len(package.feature_selection_results.get("summary_rows", [])), "rows"),
            _sentence_metric("Selected feature records", package.feature_selection_results.get("selected_feature_count"), "rows"),
        ]),
        ("Strain Contribution", [
            _sentence_metric("Leave-one-strain rows", package.strain_results.get("leave_one_strain_count"), "rows"),
            _sentence_metric("Chemical-specific strain rows", package.strain_results.get("chemical_specific_count"), "rows"),
        ]),
        ("Scientific Interpretation", compose_scientific_interpretation(package)),
        ("Limitations", [item.get("limitation", "") for item in package.limitations]),
        ("Conclusions and Next Steps", package.conclusions),
    ]
    for heading, paragraphs in sections:
        body.append(_docx_paragraph(heading, style="Heading1"))
        for paragraph in paragraphs:
            body.append(_docx_paragraph(paragraph))
    body.append(_docx_paragraph("Selected Tables", style="Heading1"))
    body.append(_docx_table(
        ["Table ID", "Title", "Rows", "Status"],
        [[table["table_id"], table["title"], table["row_count"], table["status"]] for table in package.selected_tables],
    ))
    body.append(_sect_pr())

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}</w:body></w:document>"
    )
    styles_xml = _styles_xml()
    content_types = _content_types_xml()
    rels = _rels_xml()
    doc_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)


def _docx_paragraph(text: Any, style: Optional[str] = None) -> str:
    safe = escape(str(text))
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{safe}</w:t></w:r></w:p>"


def _docx_table(headers: List[str], rows: List[List[Any]]) -> str:
    widths = [2340] * len(headers)
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    header_row = _docx_table_row(headers, widths, bold=True)
    data_rows = "".join(_docx_table_row(row, widths) for row in rows)
    props = (
        '<w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblInd w:w="120" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="D9D9D9"/>'
        '<w:left w:val="single" w:sz="4" w:color="D9D9D9"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="D9D9D9"/>'
        '<w:right w:val="single" w:sz="4" w:color="D9D9D9"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D9D9D9"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D9D9D9"/></w:tblBorders></w:tblPr>'
    )
    return f"<w:tbl>{props}<w:tblGrid>{grid}</w:tblGrid>{header_row}{data_rows}</w:tbl>"


def _docx_table_row(values: Iterable[Any], widths: List[int], bold: bool = False) -> str:
    cells = []
    for value, width in zip(values, widths):
        bold_xml = "<w:b/>" if bold else ""
        safe = escape(str(value))
        cells.append(
            f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
            f"<w:p><w:r><w:rPr>{bold_xml}</w:rPr><w:t>{safe}</w:t></w:r></w:p></w:tc>"
        )
    return f"<w:tr>{''.join(cells)}</w:tr>"


def _sect_pr() -> str:
    return (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '<w:name w:val="Normal"/><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:pPr><w:spacing w:after="160"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:sz w:val="32"/><w:color w:val="0B2545"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:pPr><w:spacing w:before="320" w:after="160"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:sz w:val="32"/><w:color w:val="2E74B5"/></w:rPr></w:style>'
        '</w:styles>'
    )


def _content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )


def _rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )


def _plain_dataset_lines(package: SupervisorResultsPackage) -> List[str]:
    return [
        _sentence_metric("Input canonical rows", package.dataset_summary.get("input_canonical_rows"), "rows"),
        _sentence_metric("Feature rows", package.dataset_summary.get("feature_rows"), "rows"),
        _sentence_metric("Core features", package.dataset_summary.get("core_feature_count"), "features"),
        _sentence_metric("Chemicals", package.dataset_summary.get("chemical_count"), "chemicals"),
        _sentence_metric("Strains", package.dataset_summary.get("strain_count"), "strains"),
    ]


def _plain_qc_lines(package: SupervisorResultsPackage) -> List[str]:
    return [
        _sentence_metric("Canonical QC errors", package.quality_control_summary.get("canonical_error_count"), "records"),
        _sentence_metric("Canonical QC warnings", package.quality_control_summary.get("canonical_warning_count"), "warnings"),
        _sentence_metric("Feature failed rows", package.quality_control_summary.get("feature_failed_rows"), "rows"),
        _sentence_metric("Fingerprint excluded rows", package.quality_control_summary.get("fingerprint_excluded_rows"), "rows"),
    ]


def _plain_fingerprint_lines(package: SupervisorResultsPackage) -> List[str]:
    return [
        _sentence_metric("Fingerprint rows", package.fingerprint_summary.get("fingerprint_rows"), "rows"),
        _sentence_metric("Consensus fingerprints", package.fingerprint_summary.get("consensus_fingerprint_rows"), "rows"),
        f"Normalization method: {package.fingerprint_summary.get('normalization_method') or 'MISSING'}.",
    ]


def _plain_exploratory_lines(package: SupervisorResultsPackage) -> List[str]:
    return [
        _sentence_metric("Cumulative variance through PC3", package.exploratory_results.get("cumulative_variance_pc3"), None),
        _sentence_metric("Cluster assignments", package.exploratory_results.get("cluster_assignment_rows"), "rows"),
    ]


def _plain_metric_lines(metrics: List[Dict[str, Any]]) -> List[str]:
    return [_sentence_metric(metric.get("metric_name", "metric"), metric.get("metric_value"), metric.get("metric_units")) for metric in metrics]


def write_simple_pdf(markdown_text: str, output_path: Path, title: str = "Report") -> None:
    text_lines = _markdown_to_plain_lines(markdown_text)
    pages: List[List[str]] = []
    current: List[str] = []
    for line in text_lines:
        wrapped = textwrap.wrap(line, width=92) or [""]
        for item in wrapped:
            if len(current) >= 46:
                pages.append(current)
                current = []
            current.append(_pdf_safe(item))
    if current:
        pages.append(current)

    objects: List[bytes] = []
    page_refs = []
    font_obj_num = 3
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page in pages:
        content = ["BT", "/F1 10 Tf", "72 750 Td", "14 TL"]
        for line in page:
            content.append(f"({line}) Tj")
            content.append("T*")
        content.append("ET")
        content_bytes = "\n".join(content).encode("latin-1", errors="replace")
        content_obj_num = len(objects) + 1
        objects.append(b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"\nendstream")
        page_obj_num = len(objects) + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                f"/Contents {content_obj_num} 0 R >>"
            ).encode("ascii")
        )
        page_refs.append(page_obj_num)
    kids = " ".join(f"{page_ref} 0 R" for page_ref in page_refs)
    objects[1] = f"<< /Type /Pages /Count {len(page_refs)} /Kids [{kids}] >>".encode("ascii")
    _write_pdf_objects(objects, output_path)


def _markdown_to_plain_lines(markdown_text: str) -> List[str]:
    lines = []
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            line = line.lstrip("#").strip().upper()
        if line.startswith("- "):
            line = "* " + line[2:]
        lines.append(line)
    return lines


def _pdf_safe(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf_objects(objects: List[bytes], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = []
    with output_path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        for index, payload in enumerate(objects, start=1):
            offsets.append(handle.tell())
            handle.write(f"{index} 0 obj\n".encode("ascii"))
            handle.write(payload)
            handle.write(b"\nendobj\n")
        xref_offset = handle.tell()
        handle.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        handle.write(b"0000000000 65535 f \n")
        for offset in offsets:
            handle.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        handle.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
