import json

from src.scientific_reasoning.manuscript import ManuscriptEngine
from src.scientific_reasoning.manuscript.writers import OUTPUT_FILENAMES
from tests.integration.manuscript_fixture import create_manuscript_source_fixture


def test_manuscript_engine_generates_required_outputs(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)

    result = ManuscriptEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["validation_passed"] is True
    assert result.metadata["document_status"] == "REVISION_REQUIRED"
    assert {path.name for path in result.output_paths} == set(OUTPUT_FILENAMES)
    summary = json.loads((project_root / "outputs" / "scientific_manuscript" / "manuscript_summary.json").read_text(encoding="utf-8"))
    assert summary["sentence_count"] == result.metadata["sentence_count"]


def test_source_gate_stops_when_drafting_is_not_allowed(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path, drafting_allowed=False)

    result = ManuscriptEngine(project_root=project_root, overwrite=True).run()

    assert not result.output_paths
    assert any(issue.code == "MANUSCRIPT_DRAFTING_PROHIBITED" for issue in result.validation_issues)


def test_missing_required_source_fails_gracefully(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    (project_root / "outputs" / "scientific_review" / "review_findings.json").unlink()

    result = ManuscriptEngine(project_root=project_root, overwrite=True).run()

    assert not result.output_paths
    assert any(issue.code == "MISSING_SOURCE_FILE" for issue in result.validation_issues)
