import json

from src.scientific_reasoning.reviewer import ReviewerEngine
from tests.integration.test_reviewer_engine_integration import create_reviewer_project


def test_reviewer_does_not_generate_new_scientific_boundaries(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path)
    ReviewerEngine(project_root=project_root, overwrite=True).run()

    validation = json.loads((project_root / "outputs" / "scientific_review" / "reviewer_validation.json").read_text(encoding="utf-8"))
    report = (project_root / "outputs" / "scientific_review" / "reviewer_report.md").read_text(encoding="utf-8")

    assert validation["new_claim_issue_count"] == 0
    assert validation["experimental_protocol_issue_count"] == 0
    assert validation["novelty_language_issue_count"] == 0
    assert validation["journal_prediction_issue_count"] == 0
    assert "No manuscript prose" in report
