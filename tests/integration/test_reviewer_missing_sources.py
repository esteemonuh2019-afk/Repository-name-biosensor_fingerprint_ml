from src.scientific_reasoning.reviewer import ReviewerEngine
from tests.integration.test_reviewer_engine_integration import create_reviewer_project


def test_missing_required_source_fails_gracefully_without_outputs(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path)
    (project_root / "outputs" / "evidence_scoring" / "evidence_scores.json").unlink()

    result = ReviewerEngine(project_root=project_root, overwrite=True).run()

    assert not result.output_paths
    assert any(issue.code == "MISSING_SOURCE_FILE" for issue in result.validation_issues)
    assert result.metadata["validation_passed"] is False
