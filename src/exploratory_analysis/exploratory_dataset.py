"""Structured result object and output writing for Stage 7B."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExploratoryAnalysisResult:
    """Container for exploratory fingerprint analysis outputs."""

    pca_scores: pd.DataFrame
    pca_loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    clustering_results: dict[str, pd.DataFrame]
    distance_summaries: dict[str, pd.DataFrame]
    heatmap_tables: dict[str, pd.DataFrame]
    metadata: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    analysis_passed: bool
    top_component_features: pd.DataFrame
    concentration_trajectories: pd.DataFrame
    strain_dispersion: pd.DataFrame
    replicate_to_consensus_distances: pd.DataFrame
    individual_pca_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    individual_pca_loadings: pd.DataFrame = field(default_factory=pd.DataFrame)
    individual_explained_variance: pd.DataFrame = field(default_factory=pd.DataFrame)
    figure_paths: list[Path] = field(default_factory=list)

    def write_outputs(self, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Write all Stage 7B tables, report, summary, and figures."""

        from src.exploratory_analysis.fingerprint_heatmaps import write_exploratory_figures

        target = Path(output_dir)
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory already exists and is not empty: {target}")
        target.mkdir(parents=True, exist_ok=True)

        outputs = {
            "pca_scores.csv": self.pca_scores,
            "pca_loadings.csv": self.pca_loadings,
            "pca_explained_variance.csv": self.explained_variance,
            "top_component_features.csv": self.top_component_features,
            "cluster_assignments.csv": self.clustering_results.get("cluster_assignments", pd.DataFrame()),
            "cluster_composition.csv": self.clustering_results.get("cluster_composition", pd.DataFrame()),
            "concentration_trajectories.csv": self.concentration_trajectories,
            "strain_dispersion.csv": self.strain_dispersion,
            "replicate_to_consensus_distances.csv": self.replicate_to_consensus_distances,
        }

        created: list[Path] = []
        for filename, dataframe in outputs.items():
            path = target / filename
            _ensure_can_write(path, overwrite=overwrite)
            dataframe.to_csv(path, index=False)
            created.append(path)

        for key, dataframe in self.heatmap_tables.items():
            path = target / f"{key}.csv"
            _ensure_can_write(path, overwrite=overwrite)
            dataframe.to_csv(path, index=True)
            created.append(path)

        if not self.individual_pca_scores.empty:
            path = target / "individual_pca_scores.csv"
            _ensure_can_write(path, overwrite=overwrite)
            self.individual_pca_scores.to_csv(path, index=False)
            created.append(path)

        summary_path = target / "exploratory_summary.json"
        _ensure_can_write(summary_path, overwrite=overwrite)
        summary_path.write_text(
            json.dumps(self.to_summary_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        created.append(summary_path)

        report_path = target / "exploratory_analysis_report.md"
        _ensure_can_write(report_path, overwrite=overwrite)
        report_path.write_text(render_exploratory_report(self), encoding="utf-8")
        created.append(report_path)

        figure_paths = write_exploratory_figures(self, target, overwrite=overwrite)
        created.extend(figure_paths)
        return created

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""

        return {
            "metadata": self.metadata,
            "analysis_passed": self.analysis_passed,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "pca_components": int(self.explained_variance["component"].nunique())
            if not self.explained_variance.empty
            else 0,
            "explained_variance": self.explained_variance.to_dict("records"),
            "cluster_count": int(
                self.clustering_results.get("cluster_assignments", pd.DataFrame())
                .get("cluster_id", pd.Series(dtype=int))
                .nunique()
            ),
            "top_component_features": self.top_component_features.to_dict("records"),
            "concentration_trajectory_rows": int(len(self.concentration_trajectories)),
            "strain_dispersion_rows": int(len(self.strain_dispersion)),
            "replicate_to_consensus_rows": int(len(self.replicate_to_consensus_distances)),
        }


def run_exploratory_analysis(
    individual_fingerprints: pd.DataFrame,
    consensus_fingerprints: pd.DataFrame,
    *,
    scaling: str = "zscore",
    distance: str = "euclidean",
    linkage_method: str = "ward",
    individual_pca: bool = False,
    upstream_warnings: list[str] | tuple[str, ...] = (),
    upstream_errors: list[str] | tuple[str, ...] = (),
) -> ExploratoryAnalysisResult:
    """Run the complete Stage 7B exploratory analysis workflow."""

    from src.exploratory_analysis.clustering_analysis import run_hierarchical_clustering
    from src.exploratory_analysis.exploratory_qc import (
        calculate_concentration_trajectories,
        calculate_replicate_to_consensus_distances,
        calculate_strain_dispersion,
        enrich_consensus_metadata,
        validate_exploratory_inputs,
    )
    from src.exploratory_analysis.fingerprint_heatmaps import create_heatmap_tables
    from src.exploratory_analysis.pca_analysis import FEATURE_COLUMNS, run_pca_analysis

    individual = individual_fingerprints.copy(deep=True)
    consensus = enrich_consensus_metadata(consensus_fingerprints.copy(deep=True), individual)
    warnings, errors, metadata = validate_exploratory_inputs(
        individual,
        consensus,
        feature_columns=FEATURE_COLUMNS,
    )
    warnings.extend(f"Upstream QC warning: {warning}" for warning in upstream_warnings)
    warnings.extend(f"Upstream QC error retained as context: {error}" for error in upstream_errors)
    metadata.update(
        {
            "scaling_method": scaling,
            "distance_metric": distance,
            "linkage_method": linkage_method,
            "consensus_primary": True,
            "individual_pca_requested": bool(individual_pca),
            "supervised_machine_learning_performed": False,
            "target_based_feature_selection_performed": False,
        }
    )
    if errors:
        return _empty_result(metadata=metadata, warnings=warnings, errors=errors)

    pca_scores = pd.DataFrame()
    pca_loadings = pd.DataFrame()
    explained = pd.DataFrame()
    top_features = pd.DataFrame()
    individual_scores = pd.DataFrame()
    individual_loadings = pd.DataFrame()
    individual_explained = pd.DataFrame()
    clustering_results: dict[str, pd.DataFrame] = {}
    distance_summaries: dict[str, pd.DataFrame] = {}
    heatmap_tables: dict[str, pd.DataFrame] = {}
    trajectories = pd.DataFrame()
    strain_dispersion = pd.DataFrame()
    replicate_distances = pd.DataFrame()

    try:
        pca_scores, pca_loadings, explained, top_features, pca_warnings, pca_metadata = run_pca_analysis(
            consensus,
            feature_columns=FEATURE_COLUMNS,
            scaling=scaling,
        )
        warnings.extend(pca_warnings)
        metadata.update(pca_metadata)
    except ValueError as error:
        errors.append(f"PCA failed: {error}")

    if individual_pca:
        try:
            individual_scores, individual_loadings, individual_explained, _, individual_warnings, _ = run_pca_analysis(
                individual,
                feature_columns=FEATURE_COLUMNS,
                scaling=scaling,
            )
            warnings.extend(f"Individual PCA warning: {warning}" for warning in individual_warnings)
        except ValueError as error:
            warnings.append(f"Individual PCA unavailable: {error}")

    try:
        clustering_results, distance_summaries, cluster_warnings = run_hierarchical_clustering(
            consensus,
            feature_columns=FEATURE_COLUMNS,
            distance=distance,
            linkage_method=linkage_method,
            scaling=scaling,
        )
        warnings.extend(cluster_warnings)
    except ValueError as error:
        errors.append(f"Clustering failed: {error}")

    trajectories, trajectory_warnings = calculate_concentration_trajectories(
        consensus,
        feature_columns=FEATURE_COLUMNS,
        scaling=scaling,
    )
    warnings.extend(trajectory_warnings)
    strain_dispersion = calculate_strain_dispersion(
        consensus,
        feature_columns=FEATURE_COLUMNS,
        scaling=scaling,
    )
    replicate_distances = calculate_replicate_to_consensus_distances(
        individual,
        consensus,
        feature_columns=FEATURE_COLUMNS,
        distance=distance,
        scaling=scaling,
    )
    heatmap_tables = create_heatmap_tables(
        consensus,
        pca_loadings,
        feature_columns=FEATURE_COLUMNS,
        scaling=scaling,
        distance=distance,
    )

    metadata.update(
        {
            "pca_score_rows": int(len(pca_scores)),
            "cluster_assignment_rows": int(len(clustering_results.get("cluster_assignments", pd.DataFrame()))),
            "concentration_trajectory_rows": int(len(trajectories)),
            "replicate_to_consensus_rows": int(len(replicate_distances)),
        }
    )
    return ExploratoryAnalysisResult(
        pca_scores=pca_scores,
        pca_loadings=pca_loadings,
        explained_variance=explained,
        clustering_results=clustering_results,
        distance_summaries=distance_summaries,
        heatmap_tables=heatmap_tables,
        metadata=metadata,
        warnings=warnings,
        errors=errors,
        analysis_passed=not errors,
        top_component_features=top_features,
        concentration_trajectories=trajectories,
        strain_dispersion=strain_dispersion,
        replicate_to_consensus_distances=replicate_distances,
        individual_pca_scores=individual_scores,
        individual_pca_loadings=individual_loadings,
        individual_explained_variance=individual_explained,
    )


def render_exploratory_report(result: ExploratoryAnalysisResult) -> str:
    """Render a Markdown report with careful scientific language."""

    metadata = result.metadata
    lines = [
        "# Stage 7B Exploratory Fingerprint Analysis Report",
        "",
        "## Scope",
        "This report describes exploratory structure in validated biosensor fingerprints. It does not perform supervised classification, regression, blind prediction, or target-based feature selection.",
        "",
        "## Input Summary",
        f"- individual fingerprints: {metadata.get('individual_fingerprint_count', 0)}",
        f"- consensus fingerprints: {metadata.get('consensus_fingerprint_count', 0)}",
        f"- features analysed: {metadata.get('feature_count', 0)}",
        f"- rows excluded for PCA/QC: {metadata.get('excluded_for_analysis_count', 0)}",
        f"- scaling method: {metadata.get('scaling_method')}",
        "",
        "## PCA",
    ]
    if result.explained_variance.empty:
        lines.append("- PCA was not available for this dataset.")
    else:
        for row in result.explained_variance.head(3).to_dict("records"):
            lines.append(
                f"- {row['component']}: explained variance ratio "
                f"{float(row['explained_variance_ratio']):.4f}, cumulative "
                f"{float(row['cumulative_explained_variance_ratio']):.4f}"
            )

    lines.extend(["", "## Clustering"])
    cluster_assignments = result.clustering_results.get("cluster_assignments", pd.DataFrame())
    if cluster_assignments.empty:
        lines.append("- Hierarchical clustering was not available.")
    else:
        lines.append(
            f"- clusters generated: {int(cluster_assignments['cluster_id'].nunique())}"
        )
        lines.append(
            "- Cluster assignments are exploratory observations and are not biological classifications."
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "- Observation: PCA and clustering can show geometry in fingerprint space.",
            "- Statistical evidence: this stage reports variance, distances, and cluster membership only.",
            "- Hypothesis: visible separation may motivate later validation experiments.",
            "- Conclusion: no chemical classifiability or dose-response success is claimed here.",
        ]
    )

    if result.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines) + "\n"


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")


def _empty_result(
    *,
    metadata: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> ExploratoryAnalysisResult:
    return ExploratoryAnalysisResult(
        pca_scores=pd.DataFrame(),
        pca_loadings=pd.DataFrame(),
        explained_variance=pd.DataFrame(),
        clustering_results={},
        distance_summaries={},
        heatmap_tables={},
        metadata=metadata,
        warnings=warnings,
        errors=errors,
        analysis_passed=False,
        top_component_features=pd.DataFrame(),
        concentration_trajectories=pd.DataFrame(),
        strain_dispersion=pd.DataFrame(),
        replicate_to_consensus_distances=pd.DataFrame(),
    )
