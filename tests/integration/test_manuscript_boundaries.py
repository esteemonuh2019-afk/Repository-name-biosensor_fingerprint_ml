import csv
import json

from src.scientific_reasoning.manuscript import ManuscriptEngine
from tests.integration.manuscript_fixture import create_manuscript_source_fixture


def test_results_sentence_comes_from_quantitative_observation(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()

    with (project_root / "outputs" / "scientific_manuscript" / "manuscript_sentence_traceability.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    result_rows = [row for row in rows if row["sentence_type"] == "RESULT" and "10 canonical rows" in row["text"]]
    assert result_rows
    assert "OBS-DATASET-0001" in result_rows[0]["observation_ids"]


def test_discussion_only_claim_stays_out_of_results(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path, results_ready=False)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    claim_matrix = _claim_matrix(project_root)

    row = claim_matrix["CLM-CHEMICAL_DISCRIMINATION-0001"]
    assert row["publication_boundary"] == "DISCUSSION_ONLY"
    assert row["result_sentence_ids"] == "[]"
    assert row["discussion_sentence_ids"] != "[]"


def test_results_ready_boundary_is_recorded(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path, results_ready=True)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    claim_matrix = _claim_matrix(project_root)

    assert claim_matrix["CLM-CHEMICAL_DISCRIMINATION-0001"]["publication_boundary"] == "RESULTS_ALLOWED"


def test_limitation_only_claim_goes_to_limitations(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    claim_matrix = _claim_matrix(project_root)

    row = claim_matrix["CLM-DATA_QUALITY-0001"]
    assert row["publication_boundary"] == "LIMITATION_ONLY"
    assert row["limitation_sentence_ids"] != "[]"


def test_withheld_claim_is_recorded_as_withheld(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path, include_withheld=True)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    claim_matrix = _claim_matrix(project_root)

    assert claim_matrix["CLM-WITHHELD-0001"]["publication_boundary"] == "WITHHELD"
    assert claim_matrix["CLM-WITHHELD-0001"]["withheld"] == "True"


def test_no_external_validation_overclaim_or_unsupported_significance(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()

    validation = json.loads((project_root / "outputs" / "scientific_manuscript" / "manuscript_validation.json").read_text(encoding="utf-8"))

    assert validation["external_validation_overclaim_count"] == 0
    assert validation["statistical_significance_issue_count"] == 0
    assert validation["causal_language_issue_count"] == 0
    assert validation["mechanism_language_issue_count"] == 0
    assert validation["novelty_language_issue_count"] == 0


def _claim_matrix(project_root):
    with (project_root / "outputs" / "scientific_manuscript" / "manuscript_claim_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        return {row["claim_id"]: row for row in csv.DictReader(handle)}
