import csv
import json
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from src.scientific_reasoning.evidence_scoring import EvidenceScoringEngine
from src.scientific_reasoning.evidence_scoring.serializers import OUTPUT_FILENAMES
from tests.integration.claim_fixture import create_claim_source_fixture


def create_claim_outputs(tmp_path: Path) -> Path:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    return project_root


def test_evidence_scoring_generates_required_outputs(tmp_path: Path) -> None:
    project_root = create_claim_outputs(tmp_path)
    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["validation_passed"] is True
    assert result.metadata["claims_loaded"] == 8
    assert result.metadata["claims_scored"] == 8
    assert {path.name for path in result.output_paths} == set(OUTPUT_FILENAMES)

    output_dir = project_root / "outputs" / "evidence_scoring"
    scores = json.loads((output_dir / "evidence_scores.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "evidence_scoring_validation.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "evidence_scoring_summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "evidence_scoring.md").read_text(encoding="utf-8")

    assert len(scores["evidence_scores"]) == 8
    assert validation["validation_passed"] is True
    assert summary["total_claims_scored"] == 8
    assert "not probabilities" in report
    assert "Internal validation is not external validation" in report


def test_evidence_scoring_csv_outputs_are_readable(tmp_path: Path) -> None:
    project_root = create_claim_outputs(tmp_path)
    EvidenceScoringEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "evidence_scoring"
    for filename in ("evidence_scores.csv", "claim_confidence_matrix.csv", "evidence_dimension_breakdown.csv"):
        with (output_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, filename


def test_absence_of_external_validation_caps_high_confidence_readiness(tmp_path: Path) -> None:
    project_root = create_claim_outputs(tmp_path)
    result = EvidenceScoringEngine(project_root=project_root, overwrite=True).run()

    assert all(record.publication_readiness.value != "HIGH_CONFIDENCE_RESULTS_READY" for record in result.records)
    summary = json.loads((project_root / "outputs" / "evidence_scoring" / "evidence_scoring_summary.json").read_text(encoding="utf-8"))
    assert summary["claims_with_external_validation"] == 0
    assert summary["claims_without_external_validation"] == 8
