from pathlib import Path

import pytest

from src.scientific_reasoning.claim import ClaimEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def test_missing_required_source_fails_gracefully_without_outputs(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    (project_root / "outputs" / "reasoning_graph" / "reasoning_graph.json").unlink()

    result = ClaimEngine(project_root=project_root, overwrite=True).run()

    assert not result.output_paths
    assert result.metadata["critical_issue_count"] > 0
    assert any(issue.code == "MISSING_CLAIM_SOURCE_FILE" for issue in result.validation_issues)


def test_non_overwrite_refuses_non_empty_claim_output_directory(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()

    with pytest.raises(FileExistsError):
        ClaimEngine(project_root=project_root, overwrite=False).run()


def test_overwrite_replaces_claim_output_directory(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    marker = project_root / "outputs" / "scientific_claims" / "marker.txt"
    marker.write_text("old", encoding="utf-8")

    result = ClaimEngine(project_root=project_root, overwrite=True).run()

    assert result.output_paths
    assert not marker.exists()
