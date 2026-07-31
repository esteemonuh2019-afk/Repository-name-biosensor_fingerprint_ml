import json
import subprocess
import sys
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def test_evidence_scoring_cli_runs_and_prints_summary(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()

    result = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts" / "build_evidence_scoring.py"),
            "--project-root",
            str(project_root),
            "--overwrite",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["claims_loaded"] == 8
    assert summary["claims_scored"] == 8
    assert summary["validation_status"] is True
