import math

import pandas as pd
import pytest

from src.exploratory_analysis import run_exploratory_analysis, run_pca_analysis
from src.exploratory_analysis.clustering_analysis import run_hierarchical_clustering
from src.exploratory_analysis.exploratory_qc import (
    calculate_concentration_trajectories,
    calculate_replicate_to_consensus_distances,
)
from src.fingerprint.fingerprint_builder import FINGERPRINT_FEATURE_COLUMNS


def test_pca_excludes_metadata() -> None:
    scores, loadings, _, _, _, metadata = run_pca_analysis(_consensus_dataframe())

    assert "Chemical" in scores.columns
    assert "Chemical" not in set(loadings["feature"])
    assert "Strain" not in set(loadings["feature"])
    assert metadata["feature_columns_used"]


def test_pca_returns_deterministic_scores_up_to_fixed_sign_convention() -> None:
    first = run_pca_analysis(_consensus_dataframe())
    second = run_pca_analysis(_consensus_dataframe())

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_explained_variance_sums_appropriately() -> None:
    _, _, explained, _, _, _ = run_pca_analysis(_consensus_dataframe())

    assert explained["explained_variance_ratio"].sum() <= 1.0 + 1e-12
    assert explained["cumulative_explained_variance_ratio"].is_monotonic_increasing


def test_loadings_have_correct_dimensions() -> None:
    _, loadings, explained, _, _, _ = run_pca_analysis(_consensus_dataframe())

    assert len(loadings) == len(FINGERPRINT_FEATURE_COLUMNS)
    assert set(explained["component"]).issubset(set(loadings.columns))


def test_constant_features_are_handled() -> None:
    dataframe = _consensus_dataframe()
    dataframe["maximum_slope"] = 1.0

    _, loadings, _, _, warnings, metadata = run_pca_analysis(dataframe)

    assert "maximum_slope" in metadata["constant_features_excluded"]
    assert "maximum_slope" not in set(loadings["feature"])
    assert any("Constant features excluded" in warning for warning in warnings)


def test_non_finite_rows_are_reported() -> None:
    dataframe = _consensus_dataframe()
    dataframe["auc"] = dataframe["auc"].astype(float)
    dataframe.loc[0, "auc"] = math.inf

    _, _, _, _, warnings, metadata = run_pca_analysis(dataframe)

    assert metadata["excluded_nonfinite_rows"] == 1
    assert any("non-finite rows" in warning for warning in warnings)


def test_consensus_fingerprints_are_used_by_default() -> None:
    result = run_exploratory_analysis(_individual_dataframe(), _consensus_dataframe())

    assert result.metadata["consensus_primary"] is True
    assert result.metadata["pca_score_rows"] == len(_consensus_dataframe())
    assert result.individual_pca_scores.empty


def test_individual_pca_is_optional() -> None:
    result = run_exploratory_analysis(
        _individual_dataframe(),
        _consensus_dataframe(),
        individual_pca=True,
    )

    assert not result.individual_pca_scores.empty


def test_ward_linkage_rejects_non_euclidean_distance() -> None:
    with pytest.raises(ValueError, match="Ward linkage"):
        run_hierarchical_clustering(
            _consensus_dataframe(),
            distance="cosine",
            linkage_method="ward",
        )


def test_cluster_assignments_are_deterministic() -> None:
    first = run_hierarchical_clustering(_consensus_dataframe())
    second = run_hierarchical_clustering(_consensus_dataframe())

    pd.testing.assert_frame_equal(
        first[0]["cluster_assignments"],
        second[0]["cluster_assignments"],
    )


def test_chemical_similarity_matrix_has_correct_dimensions() -> None:
    result = run_exploratory_analysis(_individual_dataframe(), _consensus_dataframe())
    matrix = result.heatmap_tables["chemical_similarity_heatmap_table"]

    assert matrix.shape == (3, 3)
    assert list(matrix.index) == list(matrix.columns)


def test_concentration_ordering_is_numeric_not_lexical() -> None:
    trajectories, _ = calculate_concentration_trajectories(_consensus_dataframe(), scaling="none")
    row = trajectories.loc[
        trajectories["Chemical"].eq("Diazinon")
        & trajectories["Strain"].eq("BL011")
    ].iloc[0]

    assert row["from_concentration"] == "2 ug/mL"
    assert row["to_concentration"] == "10 ug/mL"


def test_missing_concentrations_are_reported() -> None:
    dataframe = _consensus_dataframe()
    dataframe.loc[0, "Concentration"] = "control"

    _, warnings = calculate_concentration_trajectories(dataframe)

    assert any("Missing numeric concentration" in warning for warning in warnings)


def test_replicate_to_consensus_distances_are_correct() -> None:
    individual = _replicate_distance_individual()
    consensus = _replicate_distance_consensus()

    distances = calculate_replicate_to_consensus_distances(
        individual,
        consensus,
        scaling="none",
    )

    assert sorted(distances["distance_to_consensus"].round(6).tolist()) == [1.0, 1.0]
    assert distances["group_replicate_count"].eq(2).all()


def test_original_labels_are_preserved() -> None:
    consensus = _consensus_dataframe()
    consensus.loc[0, "Chemical"] = "Lambda Cyclotherin"
    consensus.loc[0, "Strain"] = "BL027ab"
    individual = _individual_dataframe()
    individual.loc[0, "Chemical"] = "Lambda Cyclotherin"
    individual.loc[0, "Strain"] = "BL027ab"

    result = run_exploratory_analysis(individual, consensus)

    assert "Lambda Cyclotherin" in set(result.pca_scores["Chemical"])
    assert "BL027ab" in set(result.pca_scores["Strain"])


def test_input_datasets_are_not_mutated() -> None:
    individual = _individual_dataframe()
    consensus = _consensus_dataframe()
    individual_before = individual.copy(deep=True)
    consensus_before = consensus.copy(deep=True)

    run_exploratory_analysis(individual, consensus)

    pd.testing.assert_frame_equal(individual, individual_before)
    pd.testing.assert_frame_equal(consensus, consensus_before)


def test_empty_or_insufficient_datasets_fail_clearly() -> None:
    result = run_exploratory_analysis(pd.DataFrame(), pd.DataFrame())

    assert result.analysis_passed is False
    assert result.errors


def test_figure_creation_does_not_overwrite_by_default(tmp_path) -> None:
    result = run_exploratory_analysis(_individual_dataframe(), _consensus_dataframe())
    result.write_outputs(tmp_path)

    with pytest.raises(FileExistsError):
        result.write_outputs(tmp_path)


def _consensus_dataframe() -> pd.DataFrame:
    rows = [
        _row("c1", "BL011", "Diazinon", "2 ug/mL", 10, 20, 8, 15, 12, 5, 155, 2, 4),
        _row("c2", "BL011", "Diazinon", "10 ug/mL", 12, 25, 7, 18, 18, 9, 210, 3, 5),
        _row("c3", "BL032", "DEET", "2 ug/mL", 8, 12, 6, 10, 6, 3, 95, 1, 2),
        _row("c4", "BL032", "Glyphosate", "5 ug/mL", 20, 34, 15, 28, 19, 7, 310, 4, 6),
    ]
    return pd.DataFrame(rows)


def _individual_dataframe() -> pd.DataFrame:
    rows = []
    for index, row in enumerate(_consensus_dataframe().to_dict("records"), start=1):
        for replicate in (1, 2):
            copy = dict(row)
            copy.pop("Consensus_ID")
            copy["Fingerprint_ID"] = f"fp-{index}-{replicate}"
            copy["Measurement_Unit_ID"] = f"unit-{index}-{replicate}"
            copy["Experiment_ID"] = "EXP-1"
            copy["Source_File"] = "synthetic.csv"
            copy["Replicate_ID"] = str(replicate)
            copy["Duration"] = 10.0
            copy["baseline"] = float(copy["baseline"]) + (replicate - 1) * 0.2
            rows.append(copy)
    return pd.DataFrame(rows)


def _replicate_distance_individual() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        [
            _row("fp-1", "BL011", "Diazinon", "1 ug/mL", 0, 0, 0, 0, 0, 0, 0, 0, 0),
            _row("fp-2", "BL011", "Diazinon", "1 ug/mL", 2, 0, 0, 0, 0, 0, 0, 0, 0),
        ]
    )
    dataframe["fold_change"] = 0.0
    dataframe["log2_fold_change"] = 0.0
    return dataframe


def _replicate_distance_consensus() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        [_row("c-1", "BL011", "Diazinon", "1 ug/mL", 1, 0, 0, 0, 0, 0, 0, 0, 0)]
    )
    dataframe["fold_change"] = 0.0
    dataframe["log2_fold_change"] = 0.0
    return dataframe


def _row(
    identifier: str,
    strain: str,
    chemical: str,
    concentration: str,
    baseline: float,
    peak: float,
    minimum: float,
    endpoint: float,
    dynamic_range: float,
    time_to_peak: float,
    auc: float,
    initial_slope: float,
    maximum_slope: float,
) -> dict[str, object]:
    return {
        "Consensus_ID": identifier,
        "Strain": strain,
        "Chemical": chemical,
        "Concentration": concentration,
        "Replicate_Count": 2,
        "Measurement_Unit_Count": 2,
        "Source_File_Count": 1,
        "QC_Status": "warning",
        "baseline": baseline,
        "peak": peak,
        "minimum": minimum,
        "endpoint": endpoint,
        "dynamic_range": dynamic_range,
        "time_to_peak": time_to_peak,
        "auc": auc,
        "initial_slope": initial_slope,
        "maximum_slope": maximum_slope,
        "fold_change": (peak - baseline) / baseline if baseline else 0,
        "log2_fold_change": math.log2(endpoint / baseline) if baseline and endpoint else 0,
    }
