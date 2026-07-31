import csv
import json

import pytest

from src.scientific_reasoning.interpretation import ScientificInterpretationEngine

from tests.integration.interpretation_fixture import write_observation_package


def generate_outputs(tmp_path):
    observations_dir = write_observation_package(tmp_path / "observations")
    output_dir = tmp_path / "interpretations"
    result = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=output_dir,
        overwrite=True,
        software_version="BSIP-2.1.0-test",
    ).run()
    return result, output_dir


def test_json_csv_and_markdown_outputs_are_readable(tmp_path) -> None:
    result, output_dir = generate_outputs(tmp_path)
    assert result.metadata["validation_passed"] is True
    json.loads((output_dir / "interpretations.json").read_text(encoding="utf-8"))
    with (output_dir / "interpretations.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.interpretations)
    markdown = (output_dir / "interpretations.md").read_text(encoding="utf-8")
    assert "## CHEMICAL_CLASSIFICATION" in markdown
    assert "Interpretation ID" not in markdown or "INT-CHEMICAL_CLASSIFICATION-0001" in markdown


def test_summary_count_consistency(tmp_path) -> None:
    result, output_dir = generate_outputs(tmp_path)
    summary = json.loads((output_dir / "interpretation_summary.json").read_text(encoding="utf-8"))
    assert summary["total_interpretations"] == len(result.interpretations)
    assert sum(summary["count_by_category"].values()) == summary["total_interpretations"]
    assert sum(summary["count_by_status"].values()) == summary["total_interpretations"]
    assert summary["supported_interpretation_count"] + summary["partially_supported_count"] == summary[
        "total_interpretations"
    ]


def test_dependency_table_consistency(tmp_path) -> None:
    result, output_dir = generate_outputs(tmp_path)
    with (output_dir / "interpretation_dependencies.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = sum(len(interpretation.supporting_observation_ids) for interpretation in result.interpretations)
    assert len(rows) == expected
    interpretation_ids = {interpretation.interpretation_id for interpretation in result.interpretations}
    assert {row["interpretation_id"] for row in rows}.issubset(interpretation_ids)
    assert all(row["observation_id"].startswith("OBS-") for row in rows)


def test_non_overwrite_behavior_refuses_non_empty_output_directory(tmp_path) -> None:
    observations_dir = write_observation_package(tmp_path / "observations")
    output_dir = tmp_path / "interpretations"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not replace", encoding="utf-8")
    engine = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=output_dir,
        overwrite=False,
    )
    with pytest.raises(FileExistsError):
        engine.run()
    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "do not replace"


def test_overwrite_behavior_replaces_only_specified_output_directory(tmp_path) -> None:
    observations_dir = write_observation_package(tmp_path / "observations")
    output_dir = tmp_path / "interpretations"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("replace me", encoding="utf-8")
    result = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=output_dir,
        overwrite=True,
    ).run()
    assert result.metadata["validation_passed"] is True
    assert not (output_dir / "old.txt").exists()
    assert (observations_dir / "observations.json").exists()


def test_validation_report_includes_readability_checks(tmp_path) -> None:
    _result, output_dir = generate_outputs(tmp_path)
    validation = json.loads((output_dir / "interpretation_validation.json").read_text(encoding="utf-8"))
    assert validation["validation_passed"] is True
    assert set(validation["output_readability_checks"]) == {
        "interpretation_dependencies.csv",
        "interpretation_summary.json",
        "interpretation_validation.json",
        "interpretations.csv",
        "interpretations.json",
        "interpretations.md",
    }
