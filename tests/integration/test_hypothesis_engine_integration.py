from src.scientific_reasoning.hypothesis import HypothesisEngine

from tests.integration.hypothesis_fixture import realistic_interpretations, write_interpretation_package


def run_engine(tmp_path, *, interpretations=None, output_name="hypotheses", overwrite=True):
    interpretations_dir = write_interpretation_package(tmp_path / "interpretations", interpretations)
    output_dir = tmp_path / output_name
    engine = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        software_version="BSIP-2.2.0-test",
    )
    return engine.run(), output_dir


def test_successful_generation_from_synthetic_interpretations(tmp_path) -> None:
    result, output_dir = run_engine(tmp_path)
    assert result.metadata["validation_passed"] is True
    assert len(result.hypotheses) == 12
    assert result.metadata["competing_hypothesis_count"] == 4
    assert len(result.output_paths) == 7
    assert (output_dir / "hypotheses.json").exists()


def test_hypothesis_ordering_is_deterministic(tmp_path) -> None:
    result, _output_dir = run_engine(tmp_path, interpretations=tuple(reversed(realistic_interpretations())))
    ids = [hypothesis.hypothesis_id for hypothesis in result.hypotheses]
    assert ids == sorted(ids)


def test_every_hypothesis_links_to_valid_interpretation_ids(tmp_path) -> None:
    result, _output_dir = run_engine(tmp_path)
    interpretation_ids = {interpretation.interpretation_id for interpretation in realistic_interpretations()}
    for hypothesis in result.hypotheses:
        assert hypothesis.supporting_interpretation_ids
        assert set(hypothesis.supporting_interpretation_ids).issubset(interpretation_ids)


def test_supporting_observation_ids_remain_traceable(tmp_path) -> None:
    result, _output_dir = run_engine(tmp_path)
    assert all(hypothesis.supporting_observation_ids for hypothesis in result.hypotheses)
    overall = [item for item in result.hypotheses if item.hypothesis_id == "HYP-OVERALL_SYSTEM_BEHAVIOR-0001"][0]
    assert "OBS-CLASSIFICATION-0001" in overall.supporting_observation_ids
    assert "OBS-REGRESSION-0001" in overall.supporting_observation_ids
