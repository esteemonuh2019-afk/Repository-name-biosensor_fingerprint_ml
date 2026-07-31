import csv
import json

from src.scientific_reasoning.manuscript import ManuscriptEngine
from tests.integration.manuscript_fixture import create_manuscript_source_fixture


def test_figure_and_table_captions_are_generated_from_metadata(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "scientific_manuscript"

    assert "Figure 1." in (output_dir / "figure_captions.md").read_text(encoding="utf-8")
    assert "Table 1." in (output_dir / "table_captions.md").read_text(encoding="utf-8")
    with (output_dir / "manuscript_figure_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["caption_status"] == "GENERATED"


def test_missing_figure_or_table_metadata_yields_zero_captions_without_crash(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path, empty_figures=True, empty_tables=True)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()

    summary = json.loads((project_root / "outputs" / "scientific_manuscript" / "manuscript_summary.json").read_text(encoding="utf-8"))

    assert summary["figure_caption_count"] == 0
    assert summary["table_caption_count"] == 0
