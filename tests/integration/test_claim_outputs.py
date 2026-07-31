import csv
import json
from pathlib import Path

from src.scientific_reasoning.claim.writers import OUTPUT_FILENAMES
from src.scientific_reasoning.claim import ClaimEngine
from tests.integration.claim_fixture import create_claim_source_fixture


def test_claim_outputs_are_readable_and_consistent(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    result = ClaimEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "scientific_claims"

    assert {path.name for path in result.output_paths} == set(OUTPUT_FILENAMES)
    claims_doc = json.loads((output_dir / "claims.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "claim_validation.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "claim_summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "claims.md").read_text(encoding="utf-8")

    assert len(claims_doc["claims"]) == summary["total_claims"] == 8
    assert validation["validation_passed"] is True
    assert validation["withheld_claim_count"] == summary["withheld_count"]
    assert "CLM-CHEMICAL_DISCRIMINATION-0001" in markdown
    assert "**Claim text:**" in markdown

    for filename in ("claims.csv", "claim_dependencies.csv", "claim_evidence_scores.csv", "claim_publication_matrix.csv"):
        with (output_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, filename


def test_dependency_and_publication_matrix_consistency(tmp_path: Path) -> None:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "scientific_claims"
    with (output_dir / "claim_dependencies.csv").open("r", encoding="utf-8", newline="") as handle:
        dependency_rows = list(csv.DictReader(handle))
    with (output_dir / "claim_publication_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))

    claim_ids = {row["claim_id"] for row in matrix_rows}
    assert claim_ids == {row["claim_id"] for row in json.loads((output_dir / "claims.json").read_text(encoding="utf-8"))["claims"]}
    assert {"supporting_hypothesis", "supporting_interpretation", "supporting_observation"}.issubset(
        {row["dependency_type"] for row in dependency_rows}
    )
    assert all(row["publication_use"] for row in matrix_rows)
