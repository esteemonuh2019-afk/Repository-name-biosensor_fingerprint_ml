import json

from src.scientific_reasoning.manuscript import ManuscriptEngine
from tests.integration.manuscript_fixture import create_manuscript_source_fixture


def test_manuscript_structured_outputs_are_deterministic_except_environment_fields(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, output_dir="outputs/ms_a", overwrite=True).run()
    ManuscriptEngine(project_root=project_root, output_dir="outputs/ms_b", overwrite=True).run()

    first = json.loads((project_root / "outputs" / "ms_a" / "manuscript_manifest.json").read_text(encoding="utf-8"))
    second = json.loads((project_root / "outputs" / "ms_b" / "manuscript_manifest.json").read_text(encoding="utf-8"))

    _scrub(first)
    _scrub(second)

    assert first["summary"] == second["summary"]
    assert first["validation_summary"]["validation_passed"] == second["validation_summary"]["validation_passed"]


def _scrub(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"generated_at", "created_at", "path", "generated_output_files", "generated_output_checksums"}:
                payload[key] = "<scrubbed>"
            else:
                _scrub(value)
    elif isinstance(payload, list):
        for item in payload:
            _scrub(item)
