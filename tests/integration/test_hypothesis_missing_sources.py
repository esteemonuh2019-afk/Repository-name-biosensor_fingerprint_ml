from src.scientific_reasoning.hypothesis import HypothesisEngine

from tests.integration.hypothesis_fixture import write_interpretation_package


def test_missing_required_source_file_returns_critical_issue(tmp_path) -> None:
    interpretations_dir = write_interpretation_package(tmp_path / "interpretations")
    (interpretations_dir / "interpretation_dependencies.csv").unlink()
    result = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=tmp_path / "hypotheses",
        overwrite=True,
    ).run()
    assert result.metadata["validation_passed"] is False
    assert result.output_paths == tuple()
    assert "MISSING_INTERPRETATION_SOURCE_FILE" in {issue.code for issue in result.validation_issues}


def test_critically_invalid_interpretation_package_is_rejected(tmp_path) -> None:
    interpretations_dir = write_interpretation_package(
        tmp_path / "interpretations",
        validation_passed=False,
        critical_issue_count=1,
    )
    result = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=tmp_path / "hypotheses",
        overwrite=True,
    ).run()
    codes = {issue.code for issue in result.validation_issues}
    assert result.metadata["validation_passed"] is False
    assert result.output_paths == tuple()
    assert "CRITICALLY_INVALID_INTERPRETATION_PACKAGE" in codes
    assert "INTERPRETATION_PACKAGE_VALIDATION_FAILED" in codes
