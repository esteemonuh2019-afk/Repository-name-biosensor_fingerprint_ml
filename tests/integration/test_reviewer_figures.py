import csv
import json

from src.scientific_reasoning.reviewer import ReviewerEngine
from tests.integration.test_reviewer_engine_integration import create_reviewer_project


def test_missing_claim_level_visual_links_are_informational(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path, supervisor_links=False)
    ReviewerEngine(project_root=project_root, overwrite=True).run()

    findings = json.loads((project_root / "outputs" / "scientific_review" / "review_findings.json").read_text(encoding="utf-8"))["review_findings"]

    figure_findings = [finding for finding in findings if finding["reviewer_type"] == "FIGURE"]
    assert figure_findings
    assert figure_findings[0]["severity"] == "INFORMATION"


def test_claim_level_visual_link_metadata_populates_figure_matrix(tmp_path) -> None:
    project_root = create_reviewer_project(tmp_path, supervisor_links=True)
    ReviewerEngine(project_root=project_root, overwrite=True).run()

    with (project_root / "outputs" / "scientific_review" / "reviewer_figure_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = {row["claim_id"]: row for row in csv.DictReader(handle)}

    assert rows["CLM-CHEMICAL_DISCRIMINATION-0001"]["visual_support_status"] == "LINKED"
    assert rows["CLM-SYSTEM_LEVEL_PERFORMANCE-0001"]["visual_support_status"] == "LINKED"
