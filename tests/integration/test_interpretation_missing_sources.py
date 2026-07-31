from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationEvidenceLink,
    ScientificInterpretationEngine,
    validate_interpretations,
)

from tests.integration.interpretation_fixture import write_observation_package


def test_missing_required_observation_file_returns_critical_issue(tmp_path) -> None:
    observations_dir = write_observation_package(tmp_path / "observations")
    (observations_dir / "observation_provenance.csv").unlink()
    result = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=tmp_path / "interpretations",
        overwrite=True,
    ).run()
    assert result.metadata["validation_passed"] is False
    assert result.output_paths == tuple()
    assert "MISSING_OBSERVATION_SOURCE_FILE" in {issue.code for issue in result.validation_issues}


def test_critically_invalid_observation_package_is_rejected(tmp_path) -> None:
    observations_dir = write_observation_package(
        tmp_path / "observations",
        validation_passed=False,
        critical_issue_count=1,
    )
    result = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=tmp_path / "interpretations",
        overwrite=True,
    ).run()
    codes = {issue.code for issue in result.validation_issues}
    assert result.metadata["validation_passed"] is False
    assert result.output_paths == tuple()
    assert "CRITICALLY_INVALID_OBSERVATION_PACKAGE" in codes
    assert "OBSERVATION_PACKAGE_VALIDATION_FAILED" in codes


def test_missing_observation_dependency_is_reported_by_validation(tmp_path) -> None:
    observations_dir = write_observation_package(tmp_path / "observations")
    output_dir = tmp_path / "interpretations"
    result = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=output_dir,
        overwrite=True,
    ).run()
    original = result.interpretations[0]
    broken = Interpretation(
        interpretation_id=original.interpretation_id,
        category=original.category,
        title=original.title,
        claim=original.claim,
        status=original.status,
        confidence=original.confidence,
        supporting_observation_ids=("OBS-MISSING-0001",),
        evidence_summary=(
            InterpretationEvidenceLink(
                observation_id="OBS-MISSING-0001",
                direction=EvidenceDirection.SUPPORTING,
                rationale="Synthetic missing dependency.",
            ),
        ),
        reasoning_rule_ids=original.reasoning_rule_ids,
        created_at=original.created_at,
        software_version=original.software_version,
        source_observation_schema_version=original.source_observation_schema_version,
    )
    issues = validate_interpretations((broken,), observations=())
    assert "NONEXISTENT_OBSERVATION_DEPENDENCY" in {issue.code for issue in issues}
