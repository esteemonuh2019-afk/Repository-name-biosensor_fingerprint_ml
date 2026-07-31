from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.scientific_narrative import aggregate_scientific_evidence, write_aggregation_outputs
from src.scientific_narrative.aggregation_rules import (
    aggregate_fold_metric,
    classification_model_rankings,
    metric_direction,
    parse_metric_value,
    regression_model_rankings,
    select_preferred_records,
    smallest_within_tolerance,
)
from src.scientific_narrative.evidence_aggregator import load_evidence_records
from src.scientific_narrative.evidence_traceability import build_traceability_index, traceability_coverage
from src.scientific_narrative.scientific_summary import (
    AggregatedEvidence,
    EvidenceInputRecord,
    OUTPUT_FILENAMES,
    SummaryRecord,
)


EVIDENCE_FIELDS = [
    "analysis_type",
    "source_file",
    "source_run",
    "metric_name",
    "metric_value",
    "metric_units",
    "figure_reference",
    "table_reference",
    "biological_entity",
    "model_name",
    "confidence",
    "extraction_status",
    "notes",
]


def test_load_evidence_assigns_deterministic_ids(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, [_row(metric_name="row_count", metric_value=12), _row(metric_name="qc_passed", metric_value=True)])

    records = load_evidence_records(path)

    assert [record.evidence_id for record in records] == ["EV000001", "EV000002"]


def test_parse_metric_value_handles_null_numbers_bools_and_strings() -> None:
    assert parse_metric_value("") is None
    assert parse_metric_value("3") == 3
    assert parse_metric_value("3.5") == 3.5
    assert parse_metric_value("true") is True
    assert parse_metric_value("Extra Trees") == "Extra Trees"


def test_empty_evidence_file_is_rejected(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, [])

    with pytest.raises(ValueError):
        aggregate_scientific_evidence(path)


def test_dataset_summary_extracts_counts_without_interpretation(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, [_row(metric_name="row_count", metric_value=96, metric_units="count")])

    aggregation = aggregate_scientific_evidence(path)

    row_count = _summary_by_metric(aggregation.dataset_summary, "total_canonical_rows")
    assert row_count.metric_value == 96
    assert row_count.metric_units == "count"
    assert row_count.source_evidence_ids == ["EV000001"]


def test_qc_summary_extracts_pass_rates_failures_and_warnings(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(metric_name="qc_passed", metric_value=False),
            _row(metric_name="failed_feature_rows", metric_value=3, analysis_type="feature extraction"),
            _row(metric_name="warning_count", metric_value=2),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    metrics = {record.metric_name: record.metric_value for record in aggregation.qc_summary}
    assert metrics["qc_passed"] is False
    assert metrics["failed_feature_rows"] == 3
    assert metrics["warning_count"] == 2


def test_fingerprint_summary_is_kept_in_json_aggregate_not_separate_csv(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [_row(analysis_type="fingerprint generation", metric_name="fingerprint_rows", metric_value=44)],
    )

    aggregation = aggregate_scientific_evidence(path)

    assert _summary_by_metric(aggregation.fingerprint_summary, "fingerprint_rows").metric_value == 44
    assert "fingerprint_summary.csv" not in OUTPUT_FILENAMES


def test_pca_variance_summary_uses_component_rows(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(
                analysis_type="exploratory analysis",
                source_file="exploratory/stage_7b_3/pca_explained_variance.csv",
                metric_name="explained_variance_ratio",
                metric_value=0.62,
                biological_entity="PC1",
            )
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    pca = _summary_by_metric(aggregation.exploratory_summary, "explained_variance_ratio")
    assert pca.metric_value == 0.62
    assert pca.biological_entity == "PC1"


def test_matrix_extrema_summarises_similarity_values(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _matrix("exploratory/stage_7b_3/chemical_similarity_heatmap_table.csv", "Ampicillin", "Kanamycin", 0.2),
            _matrix("exploratory/stage_7b_3/chemical_similarity_heatmap_table.csv", "Ampicillin", "Ciprofloxacin", 0.9),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    extrema = {record.metric_name: record.metric_value for record in aggregation.exploratory_summary}
    assert extrema["minimum_matrix_value"] == 0.2
    assert extrema["maximum_matrix_value"] == 0.9


def test_classification_ranking_prefers_macro_f1_then_balanced_accuracy() -> None:
    records = [
        _record(analysis_type="classification", model_name="Model A", metric_name="f1_macro_mean", metric_value=0.8),
        _record(analysis_type="classification", model_name="Model A", metric_name="balanced_accuracy_mean", metric_value=0.9),
        _record(analysis_type="classification", model_name="Model B", metric_name="f1_macro_mean", metric_value=0.8),
        _record(analysis_type="classification", model_name="Model B", metric_name="balanced_accuracy_mean", metric_value=0.95),
    ]

    rankings = classification_model_rankings(records)

    assert rankings[0]["model_name"] == "Model B"


def test_classification_summary_contains_ranked_model_metrics(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="A", metric_name="f1_macro_mean", metric_value=0.7),
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="B", metric_name="f1_macro_mean", metric_value=0.9),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    best = [record for record in aggregation.classification_summary if record.model_name == "B" and record.metric_name == "f1_macro_mean"][0]
    assert best.rank == 1
    assert best.direction == "higher_is_better"


def test_primary_classification_macro_f1_uses_classification_source_only(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="Classifier", metric_name="f1_macro_mean", metric_value=0.88),
            _row(analysis_type="strain ablation", source_file="classification/stage_8a/leave_one_strain_importance.csv", model_name="Classifier", metric_name="held_out_f1_macro", metric_value=0.11),
            _row(analysis_type="feature selection", source_file="feature_selection_3/feature_selection_summary.csv", model_name="Classifier", metric_name="macro_f1_mean", metric_value=0.99, biological_entity="feature_subset_id=fs; task=classification"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    macro_f1 = [record for record in aggregation.classification_summary if record.metric_name == "f1_macro_mean" and record.status != "MISSING"]
    assert len(macro_f1) == 1
    assert macro_f1[0].metric_value == 0.88
    assert macro_f1[0].source_files == ["classification/stage_8a/model_rankings.csv"]


def test_held_out_f1_macro_stays_in_strain_summary(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="Classifier", metric_name="f1_macro_mean", metric_value=0.88),
            _row(analysis_type="strain ablation", source_file="classification/stage_8a/leave_one_strain_importance.csv", model_name="Classifier", metric_name="held_out_f1_macro", metric_value=0.42, biological_entity="BL011"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    strain_metric = [record for record in aggregation.strain_summary if record.metric_name == "held_out_f1_macro"][0]
    assert strain_metric.metric_value == 0.42
    assert all(record.metric_value != 0.42 for record in aggregation.classification_summary if record.metric_name == "f1_macro_mean")


def test_conflicting_classification_aliases_are_flagged(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="Classifier", metric_name="f1_macro_mean", metric_value=0.88),
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="Classifier", metric_name="macro_f1_mean", metric_value=0.77),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    conflicts = [record for record in aggregation.classification_summary if record.metric_name == "f1_macro_mean" and record.status == "CONFLICT"]
    assert {record.metric_value for record in conflicts} == {0.88, 0.77}


def test_per_class_best_and_worst_f1_are_ranked(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _matrix("classification/stage_8a/per_class_metrics.csv", "BL011", "f1", 0.9, analysis_type="classification"),
            _matrix("classification/stage_8a/per_class_metrics.csv", "BL027", "f1", 0.2, analysis_type="classification"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    best = [record for record in aggregation.classification_summary if record.summary_type == "per_class_best_f1"][0]
    worst = [record for record in aggregation.classification_summary if record.summary_type == "per_class_worst_f1"][0]
    assert best.metric_value == 0.9
    assert worst.metric_value == 0.2


def test_confusion_matrix_summary_excludes_diagonal(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _matrix("classification/stage_8a/confusion_matrix.csv", "A", "A", 99, analysis_type="classification"),
            _matrix("classification/stage_8a/confusion_matrix.csv", "A", "B", 4, analysis_type="classification"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    confusions = [record for record in aggregation.classification_summary if record.summary_type == "most_frequent_confusions"]
    assert len(confusions) == 1
    assert confusions[0].metric_value == 4


def test_regression_ranking_prefers_r2_then_rmse_then_mae() -> None:
    records = [
        _record(analysis_type="regression", model_name="A", metric_name="r2_mean", metric_value=0.5),
        _record(analysis_type="regression", model_name="A", metric_name="rmse_mean", metric_value=2.0),
        _record(analysis_type="regression", model_name="B", metric_name="r2_mean", metric_value=0.5),
        _record(analysis_type="regression", model_name="B", metric_name="rmse_mean", metric_value=1.0),
    ]

    rankings = regression_model_rankings(records)

    assert rankings[0]["model_name"] == "B"


def test_one_regression_model_is_selected_deterministically(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="regression", source_file="regression/model_rankings.csv", model_name="Beta", metric_name="r2_mean", metric_value=0.5),
            _row(analysis_type="regression", source_file="regression/model_rankings.csv", model_name="Beta", metric_name="rmse_mean", metric_value=2.0),
            _row(analysis_type="regression", source_file="regression/model_rankings.csv", model_name="Alpha", metric_name="r2_mean", metric_value=0.5),
            _row(analysis_type="regression", source_file="regression/model_rankings.csv", model_name="Alpha", metric_name="rmse_mean", metric_value=2.0),
            _row(analysis_type="regression", source_file="regression/model_rankings.csv", model_name="Alpha", metric_name="mae_mean", metric_value=1.0),
            _row(analysis_type="regression", source_file="regression/model_rankings.csv", model_name="Beta", metric_name="mae_mean", metric_value=1.0),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    assert _summary_by_metric(aggregation.regression_summary, "best_model").metric_value == "Alpha"
    selected = [row for row in aggregation.regression_model_comparison if row["selection_status"] == "SELECTED"]
    assert selected == [
        {
            "model_name": "Alpha",
            "rank": 1,
            "r2_mean": 0.5,
            "r2_std": "",
            "rmse_mean": 2.0,
            "rmse_std": "",
            "mae_mean": 1.0,
            "mae_std": "",
            "median_absolute_error_mean": "",
            "explained_variance_mean": "",
            "fold_count": "",
            "sample_count": "",
            "selection_status": "SELECTED",
        }
    ]


def test_regression_summary_flags_negative_r2_without_interpretation(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [_row(analysis_type="regression", source_file="regression/stage_8b_2/model_rankings.csv", model_name="R", metric_name="r2_mean", metric_value=-0.1)],
    )

    aggregation = aggregate_scientific_evidence(path)

    negative = [record for record in aggregation.regression_summary if record.summary_type == "negative_r2_model"]
    assert negative[0].metric_value == -0.1
    assert negative[0].direction == "negative"


def test_all_primary_regression_metrics_come_from_selected_model(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _reg_row("Selected", "r2_mean", 0.9),
            _reg_row("Selected", "r2_std", 0.01),
            _reg_row("Selected", "rmse_mean", 1.0),
            _reg_row("Selected", "rmse_std", 0.1),
            _reg_row("Selected", "mae_mean", 0.5),
            _reg_row("Selected", "mae_std", 0.05),
            _reg_row("Selected", "median_absolute_error_mean", 0.4),
            _reg_row("Selected", "explained_variance_mean", 0.91),
            _reg_row("Selected", "fold_count", 10),
            _reg_row("Selected", "sample_count", 100),
            _reg_row("Selected", "concentration_min", 0.1),
            _reg_row("Selected", "concentration_max", 10),
            _reg_row("Other", "r2_mean", 0.1),
            _reg_row("Other", "rmse_mean", 99),
            _reg_row("Other", "mae_mean", 88),
            _reg_row("Other", "median_absolute_error_mean", 77),
            _reg_row("Other", "explained_variance_mean", 0.2),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    primary = [
        record
        for record in aggregation.regression_summary
        if record.summary_type == "model_ranking_metric" and record.status != "MISSING"
    ]
    assert {record.model_name for record in primary} == {"Selected"}
    values = {record.metric_name: record.metric_value for record in primary}
    assert values["median_absolute_error_mean"] == 0.4
    assert values["explained_variance_mean"] == 0.91
    assert all("Other" not in "; ".join(record.source_files) for record in primary)


def test_other_model_metrics_are_not_substituted_for_selected_model(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _reg_row("Selected", "r2_mean", 0.9),
            _reg_row("Selected", "rmse_mean", 1.0),
            _reg_row("Selected", "mae_mean", 0.5),
            _reg_row("Other", "r2_mean", 0.1),
            _reg_row("Other", "rmse_mean", 2.0),
            _reg_row("Other", "mae_mean", 1.0),
            _reg_row("Other", "median_absolute_error_mean", 77),
            _reg_row("Other", "explained_variance_mean", 0.2),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    missing = {record.metric_name: record for record in aggregation.regression_summary if record.status == "MISSING"}
    assert missing["median_absolute_error_mean"].model_name == "Selected"
    assert missing["explained_variance_mean"].model_name == "Selected"


def test_comparison_output_includes_every_regression_model_and_one_selected(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _reg_row("A", "r2_mean", 0.3),
            _reg_row("A", "rmse_mean", 3.0),
            _reg_row("A", "mae_mean", 2.0),
            _reg_row("B", "r2_mean", 0.5),
            _reg_row("B", "rmse_mean", 2.0),
            _reg_row("B", "mae_mean", 1.0),
            _reg_row("C", "r2_mean", 0.1),
            _reg_row("C", "rmse_mean", 4.0),
            _reg_row("C", "mae_mean", 3.0),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    assert {row["model_name"] for row in aggregation.regression_model_comparison} == {"A", "B", "C"}
    assert [row["selection_status"] for row in aggregation.regression_model_comparison].count("SELECTED") == 1
    assert [row for row in aggregation.regression_model_comparison if row["selection_status"] == "SELECTED"][0]["model_name"] == "B"


def test_selected_regression_traceability_references_selected_model(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _reg_row("Selected", "r2_mean", 0.9),
            _reg_row("Selected", "rmse_mean", 1.0),
            _reg_row("Selected", "mae_mean", 0.5),
            _reg_row("Other", "r2_mean", 0.1),
            _reg_row("Other", "rmse_mean", 2.0),
            _reg_row("Other", "mae_mean", 1.0),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)
    records_by_id = {record.evidence_id: record for record in load_evidence_records(path)}
    selected_primary = [
        record
        for record in aggregation.regression_summary
        if record.summary_type == "model_ranking_metric" and record.status != "MISSING"
    ]

    assert selected_primary
    for summary in selected_primary:
        assert summary.model_name == "Selected"
        for evidence_id in summary.source_evidence_ids:
            assert records_by_id[evidence_id].model_name == "Selected"


def test_classification_aggregation_remains_unchanged_by_regression_lock(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/model_rankings.csv", model_name="Classifier", metric_name="f1_macro_mean", metric_value=0.8),
            _row(analysis_type="classification", source_file="classification/model_rankings.csv", model_name="Classifier", metric_name="balanced_accuracy_mean", metric_value=0.7),
            _reg_row("Regressor", "r2_mean", 0.3),
            _reg_row("Regressor", "rmse_mean", 1.0),
            _reg_row("Regressor", "mae_mean", 0.5),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    assert any(record.metric_name == "f1_macro_mean" and record.metric_value == 0.8 for record in aggregation.classification_summary)
    assert any(record.metric_name == "balanced_accuracy_mean" and record.metric_value == 0.7 for record in aggregation.classification_summary)


def test_primary_regression_r2_uses_regression_source_only(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="regression", source_file="regression/stage_8b_2/model_rankings.csv", model_name="Regressor", metric_name="r2_mean", metric_value=0.31),
            _row(analysis_type="feature selection", source_file="feature_selection_3/feature_selection_summary.csv", model_name="Regressor", metric_name="r2_mean", metric_value=0.91, biological_entity="feature_subset_id=fs; task=regression"),
            _row(analysis_type="advanced feature engineering", source_file="feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", model_name="family_a", metric_name="r2_mean", metric_value=0.71, biological_entity="feature_family=family_a"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    r2 = [record for record in aggregation.regression_summary if record.metric_name == "r2_mean" and record.status != "MISSING"]
    assert len(r2) == 1
    assert r2[0].metric_value == 0.31
    assert r2[0].source_files == ["regression/stage_8b_2/model_rankings.csv"]


def test_fold_metrics_aggregate_by_arithmetic_mean() -> None:
    records = [
        _record(metric_name="fold_f1_macro", metric_value=0.4),
        _record(metric_name="fold_f1_macro", metric_value=0.6, evidence_id="EV000002"),
        _record(metric_name="other_metric", metric_value=1.0, evidence_id="EV000003"),
    ]

    value, used = aggregate_fold_metric(records, "fold_f1_macro")

    assert value == 0.5
    assert [record.evidence_id for record in used] == ["EV000001", "EV000002"]


def test_confidence_intervals_are_preserved_for_ranked_models(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="A", metric_name="f1_macro_mean", metric_value=0.8),
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="A", metric_name="f1_macro_ci95_low", metric_value=0.7),
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="A", metric_name="f1_macro_ci95_high", metric_value=0.9),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    metrics = {record.metric_name: record.metric_value for record in aggregation.classification_summary}
    assert metrics["f1_macro_ci95_low"] == 0.7
    assert metrics["f1_macro_ci95_high"] == 0.9


def test_feature_family_table_and_extrema_are_extracted(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_a", "regression_r2", 0.2, analysis_type="advanced feature engineering"),
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_b", "regression_r2", 0.8, analysis_type="advanced feature engineering"),
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_b", "regression_rmse", 1.0, analysis_type="advanced feature engineering"),
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_b", "regression_mae", 0.6, analysis_type="advanced feature engineering"),
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_b", "runtime_increase_seconds", 5, analysis_type="advanced feature engineering"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    highest = _summary_by_metric(aggregation.feature_engineering_summary, "highest_r2_family")
    assert highest.metric_value == 0.8
    assert highest.comparison_group == "family_b"


def test_feature_family_deltas_are_aggregated_directly(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_a", "regression_rmse_delta", -0.2, analysis_type="advanced feature engineering"),
            _matrix("feature_engineering/stage_8c_real/feature_family_ablation_summary.csv", "family_a", "regression_mae_delta", -0.1, analysis_type="advanced feature engineering"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    deltas = {
        record.metric_name: record.metric_value
        for record in aggregation.feature_engineering_summary
        if record.summary_type == "feature_family_metric"
    }
    assert deltas["regression_rmse_delta"] == -0.2
    assert deltas["regression_mae_delta"] == -0.1


def test_feature_family_r2_mean_remains_feature_engineering(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="advanced feature engineering", source_file="feature_engineering/stage_8c_real/family.csv", model_name="family_a", metric_name="r2_mean", metric_value=0.71, biological_entity="feature_family=family_a"),
            _row(analysis_type="regression", source_file="regression/stage_8b_2/model_rankings.csv", model_name="Regressor", metric_name="r2_mean", metric_value=0.31),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    feature_r2 = [record for record in aggregation.feature_engineering_summary if record.metric_name == "regression_r2"]
    regression_r2 = [record for record in aggregation.regression_summary if record.metric_name == "r2_mean" and record.status != "MISSING"]
    assert feature_r2[0].metric_value == 0.71
    assert feature_r2[0].analysis_section == "feature_engineering"
    assert regression_r2[0].metric_value == 0.31


def test_feature_engineering_headlines_are_direct_values(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [_row(analysis_type="advanced feature engineering", metric_name="best_feature_family", metric_value="advanced_windows")],
    )

    aggregation = aggregate_scientific_evidence(path)

    headline = _summary_by_metric(aggregation.feature_engineering_summary, "best_feature_family")
    assert headline.metric_value == "advanced_windows"


def test_feature_selection_best_rows_and_reductions_are_summarised(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _fs_row("classification__90pct", "classification", "mutual_info", "macro_f1_mean", 0.7, 20),
            _fs_row("classification__80pct", "classification", "mutual_info", "macro_f1_mean", 0.8, 10),
            _fs_row("classification__80pct", "classification", "mutual_info", "feature_count", 10, 10),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    best = [record for record in aggregation.feature_selection_summary if record.summary_type == "best_classification_macro_f1_mean"][0]
    reductions = _summary_by_metric(aggregation.feature_selection_summary, "tested_reduction_levels")
    assert best.metric_value == 0.8
    assert reductions.metric_value == "80;90"


def test_smallest_within_tolerance_returns_smaller_subset() -> None:
    rows = [
        {"task": "classification", "macro_f1_mean": 0.99, "feature_count": 100, "feature_subset_id": "large"},
        {"task": "classification", "macro_f1_mean": 0.985, "feature_count": 10, "feature_subset_id": "small"},
    ]

    smallest = smallest_within_tolerance(rows, task="classification", metric="macro_f1_mean", higher_is_better=True)

    assert smallest["feature_subset_id"] == "small"


def test_recommended_feature_set_is_preserved(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [_row(analysis_type="feature selection", metric_name="recommended_feature_set", metric_value="research", biological_entity="task=classification")],
    )

    aggregation = aggregate_scientific_evidence(path)

    recommended = _summary_by_metric(aggregation.feature_selection_summary, "recommended_feature_set")
    assert recommended.metric_value == "research"


def test_strain_summary_ranks_leave_one_strain_metrics(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="strain ablation", metric_name="held_out_f1_macro", metric_value=0.2, biological_entity="BL011"),
            _row(analysis_type="strain ablation", metric_name="held_out_f1_macro", metric_value=0.8, biological_entity="BL027"),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    top = [record for record in aggregation.strain_summary if record.summary_type == "leave_one_strain_classification"][0]
    assert top.biological_entity == "BL027"
    assert top.rank == 1


def test_limitations_summary_uses_evidence_rows_only(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [_row(metric_name="error_count", metric_value=1)],
    )
    path.with_suffix(".json").write_text(json.dumps({"metadata": {"unsupported_file_count": 7}}), encoding="utf-8")

    aggregation = aggregate_scientific_evidence(path)

    limitations = [record for record in aggregation.limitations_summary if record.metric_name == "unsupported_file_count"]
    assert limitations == []
    assert _summary_by_metric(aggregation.limitations_summary, "error_count").source_evidence_ids
    assert aggregation.metadata["unsupported_file_count_from_json"] == 7


def test_blind_validation_status_separates_prediction_and_real_validation(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(
                analysis_type="real blind validation",
                source_file="blind_validation/real_subset_prediction.csv",
                source_run="not yet available",
                metric_name="prediction_passed",
                metric_value=True,
            )
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    statuses = {record.metric_name: record.metric_value for record in aggregation.blind_validation_status}
    assert statuses["blind_prediction_infrastructure_evidence_available"] is True
    assert statuses["real_blind_experimental_validation_status"] == "not yet available"


def test_source_priority_prefers_best_model_json_over_csv() -> None:
    records = [
        _record(source_file="classification/stage_8a/classification_summary.csv", analysis_type="classification", model_name="A", metric_name="f1_macro_mean", metric_value=0.7),
        _record(source_file="classification/stage_8a/best_model_metrics.json", analysis_type="classification", model_name="A", metric_name="f1_macro_mean", metric_value=0.8),
    ]

    selected = select_preferred_records(records)

    assert len(selected) == 1
    assert selected[0].metric_value == 0.8


def test_conflicting_scalar_values_are_preserved_as_conflicts(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(metric_name="qc_passed", metric_value=True),
            _row(metric_name="qc_passed", metric_value=False),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    conflicts = [record for record in aggregation.qc_summary if record.metric_name == "qc_passed" and record.status == "CONFLICT"]
    assert {record.metric_value for record in conflicts} == {True, False}


def test_null_and_non_extracted_rows_are_excluded(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(metric_name="qc_passed", metric_value="", extraction_status="NULL", notes="value unavailable"),
            _row(metric_name="warning_count", metric_value=1),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    assert aggregation.metadata["unsupported_or_null_evidence_records"] == 1
    assert aggregation.metadata["excluded_evidence_reasons"] == ["value unavailable"]
    assert _summary_by_metric(aggregation.qc_summary, "warning_count").metric_value == 1


def test_source_files_are_retained_in_summaries(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, [_row(metric_name="warning_count", metric_value=1, source_file="qc/source.json")])

    aggregation = aggregate_scientific_evidence(path)

    warning = _summary_by_metric(aggregation.qc_summary, "warning_count")
    assert warning.source_files == ["qc/source.json"]


def test_aggregation_is_reproducible(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, [_row(metric_name="warning_count", metric_value=1)])

    first = aggregate_scientific_evidence(path)
    second = aggregate_scientific_evidence(path)

    assert first.to_dict() == second.to_dict()


def test_input_evidence_file_is_not_modified(tmp_path: Path) -> None:
    path = _write_evidence(tmp_path, [_row(metric_name="warning_count", metric_value=1)])
    before = path.read_bytes()

    aggregate_scientific_evidence(path)

    assert path.read_bytes() == before


def test_real_evidence_schema_is_supported(tmp_path: Path) -> None:
    path = _write_evidence(
        tmp_path,
        [
            _row(analysis_type="classification", source_file="classification/stage_8a/model_rankings.csv", model_name="Classifier", metric_name="balanced_accuracy_mean", metric_value=0.9),
            _row(analysis_type="regression", source_file="regression/stage_8b_2/model_rankings.csv", model_name="Regressor", metric_name="mae_mean", metric_value=1.2),
            _row(analysis_type="strain ablation", source_file="classification/stage_8a/leave_one_strain_importance.csv", metric_name="held_out_f1_macro", metric_value=0.5),
        ],
    )

    aggregation = aggregate_scientific_evidence(path)

    assert any(record.metric_name == "balanced_accuracy_mean" for record in aggregation.classification_summary)
    assert any(record.metric_name == "mae_mean" for record in aggregation.regression_summary)
    assert any(record.metric_name == "held_out_f1_macro" for record in aggregation.strain_summary)


def test_traceability_index_maps_summary_to_original_evidence() -> None:
    evidence = _record(metric_name="r2_mean", metric_value=0.4, evidence_id="EV000123")
    summary = SummaryRecord(
        summary_id="SUM0001",
        analysis_section="regression",
        summary_type="model_ranking_metric",
        metric_name="r2_mean",
        metric_value=0.4,
        source_evidence_ids=["EV000123"],
    )

    rows = build_traceability_index([summary], {"EV000123": evidence})

    assert rows[0]["summary_id"] == "SUM0001"
    assert rows[0]["source_evidence_id"] == "EV000123"
    assert rows[0]["original_metric_name"] == "r2_mean"


def test_traceability_coverage_ignores_missing_summaries() -> None:
    ok = SummaryRecord(summary_id="SUM0001", analysis_section="qc", summary_type="x", metric_name="warning_count", metric_value=1, source_evidence_ids=["EV000001"])
    missing = SummaryRecord(summary_id="SUM0002", analysis_section="qc", summary_type="x", metric_name="error_count", metric_value=None, status="MISSING")

    assert traceability_coverage([ok, missing]) == 1.0


def test_write_outputs_has_exact_stage_9b2b_filenames(tmp_path: Path) -> None:
    paths = write_aggregation_outputs(_aggregation_with_one_summary(), tmp_path / "aggregation")

    assert {path.name for path in paths} == set(OUTPUT_FILENAMES)
    assert (tmp_path / "aggregation" / "aggregated_evidence.csv").exists()
    assert not (tmp_path / "aggregation" / "fingerprint_summary.csv").exists()


def test_write_outputs_blocks_existing_directory_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "aggregation"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_aggregation_outputs(_aggregation_with_one_summary(), target)


def test_write_outputs_overwrite_replaces_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "aggregation"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    write_aggregation_outputs(_aggregation_with_one_summary(), target, overwrite=True)

    assert not (target / "old.txt").exists()
    assert (target / "aggregation_report.md").exists()


def test_maximum_source_ids_per_cell_clips_long_cells(tmp_path: Path) -> None:
    source_ids = [f"EV{index:06d}" for index in range(1, 5)]
    aggregation = AggregatedEvidence(
        dataset_summary=[
            SummaryRecord(
                summary_id="SUM0001",
                analysis_section="dataset_summary",
                summary_type="dataset_metric",
                metric_name="row_count",
                metric_value=4,
                source_evidence_ids=source_ids,
            )
        ],
        aggregation_passed=True,
    )

    write_aggregation_outputs(aggregation, tmp_path / "aggregation", maximum_source_ids_per_cell=2)

    rows = _read_csv(tmp_path / "aggregation" / "dataset_summary.csv")
    assert rows[0]["source_evidence_ids"] == "EV000001; EV000002; ... (2 more)"


def test_metric_direction_marks_lower_and_higher_metrics() -> None:
    assert metric_direction("rmse_mean", 1.2) == "lower_is_better"
    assert metric_direction("f1_macro_mean", 0.8) == "higher_is_better"


def _write_evidence(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "scientific_evidence.csv"
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{field: "" for field in EVIDENCE_FIELDS}, **row})
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _row(
    *,
    analysis_type: str = "canonical QC; feature validation",
    source_file: str = "qc/stage_5c_2/qc_summary.json",
    source_run: str = "stage_test",
    metric_name: str,
    metric_value: object,
    metric_units: str = "unitless",
    biological_entity: str = "",
    model_name: str = "",
    extraction_status: str = "EXTRACTED",
    notes: str = "",
) -> dict[str, object]:
    return {
        "analysis_type": analysis_type,
        "source_file": source_file,
        "source_run": source_run,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_units": metric_units,
        "biological_entity": biological_entity,
        "model_name": model_name,
        "confidence": "HIGH",
        "extraction_status": extraction_status,
        "notes": notes,
    }


def _matrix(
    source_file: str,
    row: str,
    column: str,
    value: object,
    *,
    analysis_type: str = "exploratory analysis",
) -> dict[str, object]:
    return _row(
        analysis_type=analysis_type,
        source_file=source_file,
        metric_name="matrix_value",
        metric_value=value,
        biological_entity=f"row={row}; column={column}",
    )


def _fs_row(
    subset: str,
    task: str,
    selector: str,
    metric_name: str,
    metric_value: object,
    feature_count: int,
) -> dict[str, object]:
    return _row(
        analysis_type="feature selection",
        source_file="feature_selection_3/feature_selection_summary.csv",
        metric_name=metric_name,
        metric_value=metric_value,
        biological_entity=f"feature_subset_id={subset}; task={task}; selector_method={selector}; feature_count={feature_count}",
    )


def _reg_row(model_name: str, metric_name: str, metric_value: object) -> dict[str, object]:
    return _row(
        analysis_type="regression",
        source_file=f"regression/{model_name}/model_rankings.csv",
        model_name=model_name,
        metric_name=metric_name,
        metric_value=metric_value,
    )


def _record(
    *,
    evidence_id: str = "EV000001",
    analysis_type: str = "canonical QC; feature validation",
    source_file: str = "qc/stage_5c_2/qc_summary.json",
    source_run: str = "stage_test",
    metric_name: str,
    metric_value: object,
    model_name: str = "",
    biological_entity: str = "",
) -> EvidenceInputRecord:
    return EvidenceInputRecord(
        evidence_id=evidence_id,
        analysis_type=analysis_type,
        source_file=source_file,
        source_run=source_run,
        metric_name=metric_name,
        metric_value=metric_value,
        model_name=model_name,
        biological_entity=biological_entity,
        extraction_status="EXTRACTED",
    )


def _summary_by_metric(records: list[SummaryRecord], metric_name: str) -> SummaryRecord:
    for record in records:
        if record.metric_name == metric_name:
            return record
    raise AssertionError(f"Missing summary metric: {metric_name}")


def _aggregation_with_one_summary() -> AggregatedEvidence:
    summary = SummaryRecord(
        summary_id="SUM0001",
        analysis_section="dataset_summary",
        summary_type="dataset_metric",
        metric_name="row_count",
        metric_value=10,
        source_evidence_ids=["EV000001"],
        source_files=["qc/stage_5c_2/qc_summary.json"],
    )
    return AggregatedEvidence(dataset_summary=[summary], aggregation_passed=True)
