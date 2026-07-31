"""Factual observation rules.

Rules in this module create observations from a validated supervisor summary.
They do not interpret metrics or recommend actions.
"""

from __future__ import annotations

from typing import Any, Iterable

from .observation_models import Observation
from .observation_registry import ANALYSIS_STAGES, CATEGORIES


def build_observations_from_summary(summary: dict[str, Any]) -> list[Observation]:
    """Build one or more factual observations for supported categories."""

    return [
        _qc_observation(summary),
        _dataset_observation(summary),
        _fingerprint_observation(summary),
        _pca_observation(summary),
        _classification_observation(summary),
        _regression_observation(summary),
        _feature_engineering_observation(summary),
        _feature_selection_observation(summary),
        _strain_observation(summary),
        _blind_prediction_observation(summary),
    ]


def incomplete_observations(reason: str, supporting_files: Iterable[str] | None = None) -> list[Observation]:
    """Return incomplete observations for all supported categories."""

    files = list(supporting_files or [])
    return [
        Observation(
            id="",
            category=category,
            title=f"{category} observation incomplete",
            statement=f"No complete {category.lower()} observation was generated.",
            analysis_stage=ANALYSIS_STAGES[category],
            supporting_files=files,
            supporting_metrics=[],
            confidence="Low",
            notes=reason,
            status="INCOMPLETE",
        )
        for category in CATEGORIES
    ]


def _qc_observation(summary: dict[str, Any]) -> Observation:
    qc = summary.get("quality_control_summary", {})
    metrics = [
        _metric("canonical_qc_passed", qc.get("canonical_qc_passed"), source=qc.get("canonical_qc_source")),
        _metric("canonical_error_count", qc.get("canonical_error_count"), "count", qc.get("canonical_qc_source")),
        _metric("canonical_warning_count", qc.get("canonical_warning_count"), "count", qc.get("canonical_qc_source")),
        _metric("feature_qc_passed", qc.get("feature_qc_passed"), source=qc.get("feature_summary_source")),
        _metric("feature_failed_rows", qc.get("feature_failed_rows"), "rows", qc.get("feature_summary_source")),
        _metric("fingerprint_qc_passed", qc.get("fingerprint_qc_passed"), source=qc.get("fingerprint_summary_source")),
    ]
    statement = (
        "Validated QC summaries report canonical_qc_passed="
        f"{_value(qc.get('canonical_qc_passed'))}, feature_qc_passed={_value(qc.get('feature_qc_passed'))}, "
        f"and fingerprint_qc_passed={_value(qc.get('fingerprint_qc_passed'))}."
    )
    return _observation(
        category="QC",
        title="Validated QC status",
        statement=statement,
        files=[qc.get("canonical_qc_source"), qc.get("feature_summary_source"), qc.get("fingerprint_summary_source")],
        metrics=metrics,
    )


def _dataset_observation(summary: dict[str, Any]) -> Observation:
    dataset = summary.get("dataset_summary", {})
    metrics = [
        _metric("input_canonical_rows", dataset.get("input_canonical_rows"), "rows", dataset.get("canonical_qc_source")),
        _metric("feature_rows", dataset.get("feature_rows"), "rows", dataset.get("feature_summary_source")),
        _metric("core_feature_count", dataset.get("core_feature_count"), "features", dataset.get("feature_summary_source")),
        _metric("chemical_count", dataset.get("chemical_count"), "labels", dataset.get("canonical_qc_source")),
        _metric("strain_count", dataset.get("strain_count"), "strains", dataset.get("canonical_qc_source")),
    ]
    statement = (
        "Validated dataset outputs report "
        f"{_value(dataset.get('input_canonical_rows'))} canonical rows, "
        f"{_value(dataset.get('feature_rows'))} feature rows, "
        f"{_value(dataset.get('core_feature_count'))} core features, "
        f"{_value(dataset.get('chemical_count'))} chemical labels, and "
        f"{_value(dataset.get('strain_count'))} strains."
    )
    return _observation(
        category="Dataset",
        title="Dataset scope counts",
        statement=statement,
        files=[dataset.get("feature_summary_source"), dataset.get("canonical_qc_source")],
        metrics=metrics,
    )


def _fingerprint_observation(summary: dict[str, Any]) -> Observation:
    fingerprint = summary.get("fingerprint_summary", {})
    metrics = [
        _metric("fingerprint_rows", fingerprint.get("fingerprint_rows"), "rows", fingerprint.get("source_file")),
        _metric("consensus_fingerprint_rows", fingerprint.get("consensus_fingerprint_rows"), "rows", fingerprint.get("source_file")),
        _metric("feature_count", fingerprint.get("feature_count"), "features", fingerprint.get("source_file")),
        _metric("distance_matrix_rows", fingerprint.get("distance_matrix_rows"), "rows", fingerprint.get("source_file")),
        _metric("distance_matrix_columns", fingerprint.get("distance_matrix_columns"), "columns", fingerprint.get("source_file")),
    ]
    statement = (
        "Validated fingerprint outputs report "
        f"{_value(fingerprint.get('fingerprint_rows'))} fingerprint rows, "
        f"{_value(fingerprint.get('consensus_fingerprint_rows'))} consensus fingerprints, "
        f"{_value(fingerprint.get('feature_count'))} features, and "
        f"normalization_method={_value(fingerprint.get('normalization_method'))}."
    )
    return _observation(
        category="Fingerprint",
        title="Fingerprint matrix counts",
        statement=statement,
        files=[fingerprint.get("source_file")],
        metrics=metrics + [_metric("normalization_method", fingerprint.get("normalization_method"), source=fingerprint.get("source_file"))],
    )


def _pca_observation(summary: dict[str, Any]) -> Observation:
    exploratory = summary.get("exploratory_results", {})
    metrics = [
        _metric("cumulative_variance_pc3", exploratory.get("cumulative_variance_pc3"), source=exploratory.get("pca_source")),
        _metric("cluster_count", exploratory.get("cluster_count"), "count", exploratory.get("cluster_source")),
        _metric("cluster_assignment_rows", exploratory.get("cluster_assignment_rows"), "rows", exploratory.get("cluster_source")),
    ]
    statement = (
        "Validated exploratory outputs report cumulative_variance_pc3="
        f"{_value(exploratory.get('cumulative_variance_pc3'))}, "
        f"cluster_count={_value(exploratory.get('cluster_count'))}, and "
        f"cluster_assignment_rows={_value(exploratory.get('cluster_assignment_rows'))}."
    )
    return _observation(
        category="PCA",
        title="PCA and clustering summary",
        statement=statement,
        files=[exploratory.get("pca_source"), exploratory.get("cluster_source")],
        metrics=metrics,
    )


def _classification_observation(summary: dict[str, Any]) -> Observation:
    classification = summary.get("classification_results", {})
    model = classification.get("selected_model", {})
    metric_rows = classification.get("selected_metrics", [])
    wanted = [
        "f1_macro_mean",
        "balanced_accuracy_mean",
        "accuracy_mean",
        "precision_macro_mean",
        "recall_macro_mean",
        "roc_auc_ovr_weighted_mean",
        "sample_count",
        "fold_count",
    ]
    metrics = [_selected_metric(metric_rows, name, model.get("source_file")) for name in wanted]
    statement = (
        "Validated classification metadata lists "
        f"{_value(model.get('model_name'))} as rank {_value(model.get('rank'))} "
        f"with selection_metric={_value(model.get('selection_metric'))}."
    )
    return _observation(
        category="Classification",
        title="Selected classification model metrics",
        statement=statement,
        files=[model.get("source_file")],
        metrics=metrics + [_metric("selected_model", model.get("model_name"), source=model.get("source_file"))],
    )


def _regression_observation(summary: dict[str, Any]) -> Observation:
    regression = summary.get("regression_results", {})
    model = regression.get("selected_model", {})
    metric_rows = regression.get("selected_metrics", [])
    wanted = [
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "median_absolute_error_mean",
        "explained_variance_mean",
        "sample_count",
        "fold_count",
        "concentration_min",
        "concentration_max",
    ]
    metrics = [_selected_metric(metric_rows, name, model.get("source_file")) for name in wanted]
    statement = (
        "Validated regression metadata lists "
        f"{_value(model.get('model_name'))} as rank {_value(model.get('rank'))} "
        f"with selection_metric={_value(model.get('selection_metric'))}."
    )
    return _observation(
        category="Regression",
        title="Selected regression model metrics",
        statement=statement,
        files=[model.get("source_file")],
        metrics=metrics + [_metric("selected_model", model.get("model_name"), source=model.get("source_file"))],
    )


def _feature_engineering_observation(summary: dict[str, Any]) -> Observation:
    features = summary.get("feature_engineering_results", {})
    metrics = [
        _metric("best_feature_family", features.get("best_feature_family"), source=features.get("summary_source")),
        _metric("classification_improvement", features.get("classification_improvement"), source=features.get("summary_source")),
        _metric("regression_improvement", features.get("regression_improvement"), source=features.get("summary_source")),
        _metric("runtime_increase_seconds", features.get("runtime_increase_seconds"), "seconds", features.get("summary_source")),
        _metric("feature_family_count", features.get("feature_family_count"), "count", features.get("summary_source")),
    ]
    statement = (
        "Validated feature-engineering summary reports "
        f"best_feature_family={_value(features.get('best_feature_family'))}, "
        f"classification_improvement={_value(features.get('classification_improvement'))}, "
        f"regression_improvement={_value(features.get('regression_improvement'))}, and "
        f"runtime_increase_seconds={_value(features.get('runtime_increase_seconds'))}."
    )
    return _observation(
        category="Feature Engineering",
        title="Feature-family benchmark summary",
        statement=statement,
        files=[features.get("summary_source"), features.get("benchmark_source")],
        metrics=metrics,
    )


def _feature_selection_observation(summary: dict[str, Any]) -> Observation:
    selection = summary.get("feature_selection_results", {})
    recommended = selection.get("recommended_defaults", [])
    metrics = [
        _metric("summary_rows", len(selection.get("summary_rows", [])), "rows", selection.get("summary_source")),
        _metric("selected_feature_count", selection.get("selected_feature_count"), "rows", selection.get("selected_features_source")),
        _metric("recommended_default_count", len(recommended), "count", selection.get("summary_source")),
    ]
    for row in recommended:
        task = row.get("task", "unknown")
        metric_name = "macro_f1_mean" if task == "classification" else "r2_mean"
        metrics.append(
            _metric(
                f"{task}_{metric_name}",
                row.get(metric_name),
                source=selection.get("summary_source"),
                notes=f"feature_count={row.get('feature_count')}; selector_method={row.get('selector_method')}",
            )
        )
    statement = (
        "Validated feature-selection outputs report "
        f"{_value(len(selection.get('summary_rows', [])))} summary rows, "
        f"{_value(selection.get('selected_feature_count'))} selected-feature records, and "
        f"{_value(len(recommended))} recommended_default rows."
    )
    return _observation(
        category="Feature Selection",
        title="Feature-selection output counts",
        statement=statement,
        files=[selection.get("summary_source"), selection.get("selected_features_source")],
        metrics=metrics,
    )


def _strain_observation(summary: dict[str, Any]) -> Observation:
    strain = summary.get("strain_results", {})
    metrics = [
        _metric("leave_one_strain_count", strain.get("leave_one_strain_count"), "rows", strain.get("loeo_source")),
        _metric("chemical_specific_count", strain.get("chemical_specific_count"), "rows", strain.get("chemical_source")),
        _metric("single_strain_count", strain.get("single_strain_count"), "rows", strain.get("single_source")),
    ]
    statement = (
        "Validated strain-contribution outputs report "
        f"{_value(strain.get('leave_one_strain_count'))} leave-one-strain rows, "
        f"{_value(strain.get('chemical_specific_count'))} chemical-specific rows, and "
        f"{_value(strain.get('single_strain_count'))} single-strain rows."
    )
    return _observation(
        category="Strain Contribution",
        title="Strain contribution table counts",
        statement=statement,
        files=[strain.get("loeo_source"), strain.get("chemical_source"), strain.get("single_source")],
        metrics=metrics,
    )


def _blind_prediction_observation(summary: dict[str, Any]) -> Observation:
    blind = summary.get("project_summary", {}).get("blind_prediction_context", {})
    metrics = [
        _metric("prediction_passed", blind.get("prediction_passed"), source=blind.get("source_file")),
        _metric("true_labels_included", blind.get("true_labels_included"), source=blind.get("source_file")),
        _metric("predicted_chemical", blind.get("predicted_chemical"), source=blind.get("source_file")),
        _metric("chemical_confidence", blind.get("chemical_confidence"), source=blind.get("source_file")),
        _metric("predicted_concentration", blind.get("predicted_concentration"), blind.get("concentration_units"), blind.get("source_file")),
        _metric("novelty_status", blind.get("novelty_status"), source=blind.get("source_file")),
    ]
    statement = (
        "Validated blind-prediction context reports "
        f"prediction_passed={_value(blind.get('prediction_passed'))}, "
        f"true_labels_included={_value(blind.get('true_labels_included'))}, "
        f"predicted_chemical={_value(blind.get('predicted_chemical'))}, and "
        f"novelty_status={_value(blind.get('novelty_status'))}."
    )
    return _observation(
        category="Blind Prediction",
        title="Blind-prediction context",
        statement=statement,
        files=[blind.get("source_file")],
        metrics=metrics,
        notes="This observation records prediction context only; it does not state validation performance.",
    )


def _observation(
    category: str,
    title: str,
    statement: str,
    files: Iterable[Any],
    metrics: list[dict[str, Any]],
    notes: str = "",
) -> Observation:
    supporting_files = [str(path) for path in files if path]
    missing_metrics = [metric["metric_name"] for metric in metrics if metric.get("metric_value") is None]
    status = "INCOMPLETE" if missing_metrics or not supporting_files else "COMPLETE"
    confidence = "High" if status == "COMPLETE" else "Low"
    if missing_metrics:
        suffix = "Missing metrics: " + ", ".join(missing_metrics) + "."
        notes = f"{notes} {suffix}".strip()
    return Observation(
        id="",
        category=category,
        title=title,
        statement=statement,
        analysis_stage=ANALYSIS_STAGES[category],
        supporting_files=supporting_files,
        supporting_metrics=metrics,
        confidence=confidence,
        notes=notes,
        status=status,
    )


def _selected_metric(metric_rows: list[dict[str, Any]], metric_name: str, fallback_source: str | None = None) -> dict[str, Any]:
    for row in metric_rows:
        if row.get("metric_name") == metric_name:
            return _metric(
                metric_name,
                row.get("metric_value"),
                row.get("metric_units"),
                row.get("source_file") or fallback_source,
                notes=f"status={row.get('status')}",
            )
    return _metric(metric_name, None, source=fallback_source, notes="Metric absent from selected metrics.")


def _metric(
    name: str,
    value: Any,
    units: str | None = None,
    source: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "metric_name": name,
        "metric_value": value,
        "metric_units": units,
        "source_file": source,
        "notes": notes,
    }


def _value(value: Any) -> str:
    if value is None or value == "":
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
