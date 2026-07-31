import json
from pathlib import Path

from src.scientific_reasoning.claim import ClaimEngine
from src.scientific_reasoning.evidence_scoring import EvidenceScoringEngine
from src.scientific_reasoning.reviewer import ReviewerEngine
from src.scientific_reasoning.reviewer.writers import OUTPUT_FILENAMES
from tests.integration.claim_fixture import create_claim_source_fixture


def create_reviewer_project(tmp_path: Path, *, supervisor_links: bool = False) -> Path:
    project_root = create_claim_source_fixture(tmp_path)
    ClaimEngine(project_root=project_root, overwrite=True).run()
    EvidenceScoringEngine(project_root=project_root, overwrite=True).run()
    supervisor_dir = project_root / "outputs" / "supervisor_results_2"
    supervisor_dir.mkdir(parents=True)
    if supervisor_links:
        (supervisor_dir / "selected_figures.csv").write_text("figure_id,title,claim_id\nfig-chemical,Chemical figure,CLM-CHEMICAL_DISCRIMINATION-0001\n", encoding="utf-8")
        (supervisor_dir / "selected_tables.csv").write_text("table_id,title,claim_id\ntbl-system,System table,CLM-SYSTEM_LEVEL_PERFORMANCE-0001\n", encoding="utf-8")
    else:
        (supervisor_dir / "selected_figures.csv").write_text("figure_id,title\nfig-chemical,Chemical figure\n", encoding="utf-8")
        (supervisor_dir / "selected_tables.csv").write_text("table_id,title\ntbl-system,System table\n", encoding="utf-8")
    (supervisor_dir / "report_validation.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    return project_root


def test_reviewer_engine_generates_findings_and_outputs(tmp_path: Path) -> None:
    project_root = create_reviewer_project(tmp_path)

    result = ReviewerEngine(project_root=project_root, overwrite=True).run()

    assert result.metadata["validation_passed"] is True
    assert result.metadata["findings_generated"] > 0
    assert result.metadata["overall_recommendation"] == "NEEDS_MAJOR_REVISION"
    assert {path.name for path in result.output_paths} == set(OUTPUT_FILENAMES)
