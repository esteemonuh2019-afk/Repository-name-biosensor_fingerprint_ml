import json

from src.scientific_reasoning.interpretation import ScientificInterpretationEngine

from tests.integration.interpretation_fixture import realistic_observations, write_observation_package


def run_engine(tmp_path, *, observations=None, output_name="interpretations", overwrite=True):
    observations_dir = write_observation_package(tmp_path / "observations", observations)
    output_dir = tmp_path / output_name
    engine = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        software_version="BSIP-2.1.0-test",
    )
    return engine.run(), output_dir


def test_successful_interpretation_generation_from_realistic_synthetic_observations(tmp_path) -> None:
    result, output_dir = run_engine(tmp_path)
    assert result.metadata["validation_passed"] is True
    assert len(result.interpretations) == 9
    assert len(result.output_paths) == 6
    assert (output_dir / "interpretations.json").exists()


def test_interpretation_ordering_is_deterministic(tmp_path) -> None:
    result, _output_dir = run_engine(tmp_path, observations=tuple(reversed(realistic_observations())))
    ids = [interpretation.interpretation_id for interpretation in result.interpretations]
    assert ids == sorted(ids)


def test_interpretation_serialization_is_deterministic_in_ordering(tmp_path) -> None:
    result, output_dir = run_engine(tmp_path)
    payload = json.loads((output_dir / "interpretations.json").read_text(encoding="utf-8"))
    ids = [item["interpretation_id"] for item in payload["interpretations"]]
    assert ids == sorted(ids)
    assert payload["interpretations"][0]["supporting_observation_ids"] == sorted(
        payload["interpretations"][0]["supporting_observation_ids"]
    )


def test_every_interpretation_links_to_valid_observation_ids(tmp_path) -> None:
    result, _output_dir = run_engine(tmp_path)
    observation_ids = {observation.observation_id for observation in realistic_observations()}
    for interpretation in result.interpretations:
        assert interpretation.supporting_observation_ids
        assert set(interpretation.supporting_observation_ids).issubset(observation_ids)


def test_confidence_policy_behavior_for_generated_interpretations(tmp_path) -> None:
    result, _output_dir = run_engine(tmp_path)
    by_id = {interpretation.interpretation_id: interpretation for interpretation in result.interpretations}
    assert by_id["INT-FINGERPRINT_STRUCTURE-0001"].confidence.value == "HIGH"
    assert by_id["INT-CHEMICAL_CLASSIFICATION-0001"].confidence.value == "MODERATE"
    assert by_id["INT-BLIND_VALIDATION-0001"].confidence.value == "MODERATE"
