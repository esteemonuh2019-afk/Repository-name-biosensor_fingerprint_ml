import json
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from src.scientific_reasoning.evidence_scoring import EvidenceScoringEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def test_repeated_runs_are_equivalent_except_timestamps(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    EvidenceScoringEngine(project_root=project_root, output_dir="outputs/evidence_scoring_a", overwrite=True).run()
    EvidenceScoringEngine(project_root=project_root, output_dir="outputs/evidence_scoring_b", overwrite=True).run()

    first = json.loads((project_root / "outputs" / "evidence_scoring_a" / "evidence_scores.json").read_text(encoding="utf-8"))
    second = json.loads((project_root / "outputs" / "evidence_scoring_b" / "evidence_scores.json").read_text(encoding="utf-8"))
    first["generated_at"] = "<timestamp>"
    second["generated_at"] = "<timestamp>"

    assert first == second
