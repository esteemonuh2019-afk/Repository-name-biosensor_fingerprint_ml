import csv
import json

from src.scientific_reasoning.reviewer import ReviewerEngine
from src.scientific_reasoning.reviewer.writers import OUTPUT_FILENAMES
from tests.integration.test_reviewer_engine_integration import create_reviewer_project


def test_reviewer_outputs_are_readable(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path)
    ReviewerEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "scientific_review"

    for filename in OUTPUT_FILENAMES:
        path = output_dir / filename
        assert path.exists(), filename
        if filename.endswith(".json"):
            assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
        elif filename.endswith(".csv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                assert list(csv.DictReader(handle)) or filename == "reviewer_blockers.csv"
        else:
            assert path.read_text(encoding="utf-8")


def test_publication_assessment_contains_required_flags(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path)
    ReviewerEngine(project_root=project_root, overwrite=True).run()

    assessment = json.loads((project_root / "outputs" / "scientific_review" / "reviewer_publication_assessment.json").read_text(encoding="utf-8"))

    assert assessment["manuscript_drafting_allowed"] is True
    assert assessment["definitive_generalization_allowed"] is False
    assert "overall_readiness_score" in assessment
