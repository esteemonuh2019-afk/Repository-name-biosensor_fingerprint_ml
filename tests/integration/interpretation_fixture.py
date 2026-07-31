from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    Observation,
    ObservationCategory,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
)


FIXED_TIME = "2026-07-31T00:00:00+00:00"


def make_observation(
    *,
    observation_id: str,
    category: ObservationCategory,
    title: str,
    statement: str,
    metrics: dict[str, Any] | None = None,
    limitations: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    status: ObservationStatus = ObservationStatus.COMPLETE,
) -> Observation:
    metrics = metrics or {}
    provenance_records = []
    supporting_metrics = []
    for index, (metric_name, metric_value) in enumerate(metrics.items(), start=1):
        provenance_id = f"P-{observation_id}-{index:04d}"
        provenance_records.append(
            ProvenanceRecord(
                provenance_id=provenance_id,
                source_file=f"validated/{observation_id}.json",
                source_run="synthetic-observations",
                section=category.value,
                claim_text=f"Synthetic evidence for {metric_name}",
                metric_name=metric_name,
                metric_value=metric_value,
                support_status="SUPPORTED",
            )
        )
        supporting_metrics.append(
            SupportingMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                source_file=f"validated/{observation_id}.json",
                source_run="synthetic-observations",
                provenance_id=provenance_id,
            )
        )
    return Observation(
        observation_id=observation_id,
        category=category,
        title=title,
        statement=statement,
        status=status,
        analysis_stage="Synthetic Observation Package",
        supporting_metrics=tuple(supporting_metrics),
        supporting_files=(f"validated/{observation_id}.json",),
        provenance_records=tuple(provenance_records),
        confidence=confidence,
        limitations=limitations,
        created_at=FIXED_TIME,
        software_version="BSIP-2.0-test",
        tags=("synthetic",),
        metadata=metadata or {},
    )


def realistic_observations(*, regression_r2: float | None = 0.28, blind_labels: bool = False) -> tuple[Observation, ...]:
    regression_metrics: dict[str, Any] = {
        "rmse_mean": 1.8,
        "mae_mean": 1.1,
        "concentration_min": 0.0,
        "concentration_max": 500.0,
    }
    if regression_r2 is not None:
        regression_metrics["r2_mean"] = regression_r2
        regression_metrics["explained_variance_mean"] = regression_r2 + 0.02
    return tuple(
        sorted(
            (
                make_observation(
                    observation_id="OBS-QC-0001",
                    category=ObservationCategory.QUALITY_CONTROL,
                    title="Quality-control facts",
                    statement="Synthetic QC observation reports warnings.",
                    metrics={
                        "canonical_error_count": 1,
                        "canonical_warning_count": 2,
                        "feature_failed_rows": 3,
                        "fingerprint_excluded_rows": 4,
                    },
                    limitations=("Synthetic QC limitations are active.",),
                    metadata={"package_validation_passed": False},
                ),
                make_observation(
                    observation_id="OBS-VALIDATION-0001",
                    category=ObservationCategory.VALIDATION,
                    title="Validation facts",
                    statement="Synthetic package validation facts are available.",
                    metrics={},
                    metadata={"validation_issue_count": 0, "package_validation_passed": True},
                ),
                make_observation(
                    observation_id="OBS-FINGERPRINT-0001",
                    category=ObservationCategory.FINGERPRINT,
                    title="Fingerprint facts",
                    statement="Synthetic fingerprint observation reports fingerprint rows.",
                    metrics={"fingerprint_rows": 100, "feature_count": 24},
                ),
                make_observation(
                    observation_id="OBS-EXPLORATORY_ANALYSIS-0001",
                    category=ObservationCategory.EXPLORATORY_ANALYSIS,
                    title="Exploratory facts",
                    statement="Synthetic exploratory observation reports PCA and cluster metrics.",
                    metrics={"cumulative_explained_variance_ratio_pc3": 0.62, "cluster_count": 4},
                ),
                make_observation(
                    observation_id="OBS-CLASSIFICATION-0001",
                    category=ObservationCategory.CLASSIFICATION,
                    title="Classification facts",
                    statement="Synthetic classification observation reports benchmark metrics.",
                    metrics={
                        "accuracy_mean": 0.74,
                        "balanced_accuracy_mean": 0.71,
                        "f1_macro_mean": 0.69,
                        "f1_weighted_mean": 0.74,
                        "roc_auc_ovr_weighted_mean": 0.83,
                        "class_count": 4,
                        "fold_count": 10,
                        "sample_count": 900,
                    },
                    metadata={"selected_model": "Extra Trees"},
                ),
                make_observation(
                    observation_id="OBS-REGRESSION-0001",
                    category=ObservationCategory.REGRESSION,
                    title="Regression facts",
                    statement="Synthetic regression observation reports benchmark metrics.",
                    metrics=regression_metrics,
                    metadata={"selected_model": "Extra Trees Regressor"},
                ),
                make_observation(
                    observation_id="OBS-FEATURE_ENGINEERING-0001",
                    category=ObservationCategory.FEATURE_ENGINEERING,
                    title="Feature-engineering facts",
                    statement="Synthetic feature-engineering observation reports benchmark changes.",
                    metrics={"classification_improvement": 0.04, "regression_improvement": 0.03},
                    metadata={"best_feature_family": "temporal"},
                ),
                make_observation(
                    observation_id="OBS-FEATURE_SELECTION-0001",
                    category=ObservationCategory.FEATURE_SELECTION,
                    title="Feature-selection facts",
                    statement="Synthetic feature-selection observation reports selected feature rows.",
                    metrics={"selected_feature_rows": 18},
                ),
                make_observation(
                    observation_id="OBS-STRAIN_CONTRIBUTION-0001",
                    category=ObservationCategory.STRAIN_CONTRIBUTION,
                    title="Strain-contribution facts",
                    statement="Synthetic strain-contribution observation reports ablation counts.",
                    metrics={"leave_one_strain_count": 12, "chemical_specific_count": 4},
                ),
                make_observation(
                    observation_id="OBS-BLIND_PREDICTION-0001",
                    category=ObservationCategory.BLIND_PREDICTION,
                    title="Blind-prediction facts",
                    statement="Synthetic blind-prediction observation reports label availability.",
                    metrics={"true_labels_included": blind_labels},
                    metadata={"prediction_output_available": True},
                ),
            ),
            key=lambda observation: observation.observation_id,
        )
    )


def write_observation_package(
    directory: Path,
    observations: tuple[Observation, ...] | None = None,
    *,
    validation_passed: bool = True,
    critical_issue_count: int = 0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    observations = observations or realistic_observations()
    ordered = tuple(sorted(observations, key=lambda observation: observation.observation_id))
    (directory / "observations.json").write_text(
        json.dumps(
            {
                "schema_version": "BSIP-2.0",
                "software_version": "BSIP-2.0-test",
                "generated_at": FIXED_TIME,
                "observations": [observation.to_record() for observation in ordered],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "observation_validation.json").write_text(
        json.dumps(
            {
                "validation_passed": validation_passed,
                "critical_issue_count": critical_issue_count,
                "warning_count": 0,
                "structured_validation_issues": [
                    {
                        "code": "SYNTHETIC_CRITICAL",
                        "severity": "CRITICAL",
                        "message": "Synthetic critical validation issue.",
                        "observation_id": None,
                        "field": "observation_validation.json",
                        "source_file": None,
                    }
                ]
                if critical_issue_count
                else [],
                "output_readability_checks": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    category_counts = Counter(observation.category.value for observation in ordered)
    (directory / "observation_summary.json").write_text(
        json.dumps(
            {
                "total_observations": len(ordered),
                "count_by_category": dict(sorted(category_counts.items())),
                "source_files_missing": [],
                "validation_passed": validation_passed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (directory / "observation_provenance.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "observation_id",
            "provenance_id",
            "source_file",
            "source_run",
            "section",
            "claim_text",
            "metric_name",
            "metric_value",
            "units",
            "model_name",
            "table_or_figure_reference",
            "support_status",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for observation in ordered:
            for provenance in observation.provenance_records:
                row = provenance.to_record()
                row["observation_id"] = observation.observation_id
                writer.writerow(row)
    return directory
