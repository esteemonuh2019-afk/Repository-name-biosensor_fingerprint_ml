import json
import subprocess
import sys
from pathlib import Path

from src.scientific_reasoning.observation.engine import ScientificObservationEngine
from tests.integration.observation_fixture import FIXED_TIMESTAMP, create_supervisor_fixture


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_successful_generation_from_realistic_fixture(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    engine = ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir="outputs/scientific_observations",
        overwrite=True,
        generated_at=FIXED_TIMESTAMP,
    )
    result = engine.run()
    assert result.validation_passed is True
    assert len(result.observations) == 11
    assert all(path.exists() for path in result.output_paths)


def test_deterministic_observation_ordering(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result = ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir="outputs/scientific_observations",
        overwrite=True,
        generated_at=FIXED_TIMESTAMP,
    ).run()
    ids = [observation.observation_id for observation in result.observations]
    assert ids == sorted(ids)


def test_deterministic_serialized_outputs(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    first = ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir="outputs/observations_first",
        overwrite=True,
        generated_at=FIXED_TIMESTAMP,
    ).run()
    second = ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir="outputs/observations_second",
        overwrite=True,
        generated_at=FIXED_TIMESTAMP,
    ).run()
    assert first.validation_passed and second.validation_passed
    for filename in ["observations.json", "observations.csv", "observations.md", "observation_summary.json"]:
        assert (first.write_result.output_dir / filename).read_bytes() == (second.write_result.output_dir / filename).read_bytes()


def test_cli_runs_and_prints_summary(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_scientific_observations.py"),
            "--project-root",
            str(project_root),
            "--supervisor-results",
            str(supervisor.relative_to(project_root)),
            "--output-dir",
            "outputs/scientific_observations",
            "--overwrite",
            "--software-version",
            "fixture-version",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["validation_passed"] is True
    assert summary["observation_count"] == 11
