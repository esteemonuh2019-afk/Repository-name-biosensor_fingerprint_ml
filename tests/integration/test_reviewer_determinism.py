import json

from src.scientific_reasoning.reviewer import ReviewerEngine
from tests.integration.test_reviewer_engine_integration import create_reviewer_project


def test_reviewer_outputs_are_deterministic_except_timestamps(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path)
    ReviewerEngine(project_root=project_root, output_dir="outputs/review_a", overwrite=True).run()
    ReviewerEngine(project_root=project_root, output_dir="outputs/review_b", overwrite=True).run()

    first = json.loads((project_root / "outputs" / "review_a" / "review_findings.json").read_text(encoding="utf-8"))
    second = json.loads((project_root / "outputs" / "review_b" / "review_findings.json").read_text(encoding="utf-8"))

    _scrub_timestamps(first)
    _scrub_timestamps(second)

    assert first == second


def _scrub_timestamps(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"generated_at", "created_at"}:
                payload[key] = "<timestamp>"
            elif key == "path":
                payload[key] = "<path>"
            else:
                _scrub_timestamps(value)
    elif isinstance(payload, list):
        for item in payload:
            _scrub_timestamps(item)
