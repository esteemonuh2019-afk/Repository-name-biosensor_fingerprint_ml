import csv
import json

import pytest

from src.scientific_reasoning.hypothesis import HypothesisEngine

from tests.integration.hypothesis_fixture import write_interpretation_package


def generate_outputs(tmp_path):
    interpretations_dir = write_interpretation_package(tmp_path / "interpretations")
    output_dir = tmp_path / "hypotheses"
    result = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=output_dir,
        overwrite=True,
        software_version="BSIP-2.2.0-test",
    ).run()
    return result, output_dir, interpretations_dir


def test_json_csv_and_markdown_readability(tmp_path) -> None:
    result, output_dir, _interpretations_dir = generate_outputs(tmp_path)
    assert result.metadata["validation_passed"] is True
    json.loads((output_dir / "hypotheses.json").read_text(encoding="utf-8"))
    with (output_dir / "hypotheses.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.hypotheses)
    markdown = (output_dir / "hypotheses.md").read_text(encoding="utf-8")
    assert "## CHEMICAL_DISCRIMINATION" in markdown
    assert "Falsifiability statement" in markdown


def test_summary_count_consistency(tmp_path) -> None:
    result, output_dir, _interpretations_dir = generate_outputs(tmp_path)
    summary = json.loads((output_dir / "hypothesis_summary.json").read_text(encoding="utf-8"))
    assert summary["total_hypotheses"] == len(result.hypotheses)
    assert sum(summary["count_by_category"].values()) == summary["total_hypotheses"]
    assert sum(summary["count_by_status"].values()) == summary["total_hypotheses"]
    assert sum(summary["count_by_priority"].values()) == summary["total_hypotheses"]
    assert summary["competing_count"] == 4


def test_dependency_table_consistency(tmp_path) -> None:
    result, output_dir, _interpretations_dir = generate_outputs(tmp_path)
    with (output_dir / "hypothesis_dependencies.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = sum(len(hypothesis.supporting_interpretation_ids) for hypothesis in result.hypotheses)
    assert len(rows) == expected
    hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in result.hypotheses}
    assert {row["hypothesis_id"] for row in rows}.issubset(hypothesis_ids)
    assert all(row["interpretation_id"].startswith("INT-") for row in rows)


def test_competition_map_consistency(tmp_path) -> None:
    _result, output_dir, _interpretations_dir = generate_outputs(tmp_path)
    with (output_dir / "hypothesis_competition_map.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["reciprocal_link"] == "True" for row in rows)


def test_non_overwrite_behavior(tmp_path) -> None:
    interpretations_dir = write_interpretation_package(tmp_path / "interpretations")
    output_dir = tmp_path / "hypotheses"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    engine = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=output_dir,
        overwrite=False,
    )
    with pytest.raises(FileExistsError):
        engine.run()
    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_overwrite_behavior_replaces_only_output_directory(tmp_path) -> None:
    interpretations_dir = write_interpretation_package(tmp_path / "interpretations")
    output_dir = tmp_path / "hypotheses"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("replace", encoding="utf-8")
    result = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=output_dir,
        overwrite=True,
    ).run()
    assert result.metadata["validation_passed"] is True
    assert not (output_dir / "old.txt").exists()
    assert (interpretations_dir / "interpretations.json").exists()


def test_validation_report_includes_output_readability_checks(tmp_path) -> None:
    _result, output_dir, _interpretations_dir = generate_outputs(tmp_path)
    validation = json.loads((output_dir / "hypothesis_validation.json").read_text(encoding="utf-8"))
    assert validation["validation_passed"] is True
    assert set(validation["output_readability_checks"]) == {
        "hypotheses.csv",
        "hypotheses.json",
        "hypotheses.md",
        "hypothesis_competition_map.csv",
        "hypothesis_dependencies.csv",
        "hypothesis_summary.json",
        "hypothesis_validation.json",
    }
