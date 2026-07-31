import csv
import json
import zipfile

from src.scientific_reasoning.manuscript import ManuscriptEngine
from src.scientific_reasoning.manuscript.writers import OUTPUT_FILENAMES
from tests.integration.manuscript_fixture import create_manuscript_source_fixture


def test_markdown_csv_json_and_docx_outputs_are_readable(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "scientific_manuscript"

    for filename in OUTPUT_FILENAMES:
        path = output_dir / filename
        assert path.exists(), filename
        if filename.endswith(".json"):
            assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
        elif filename.endswith(".csv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                assert list(csv.DictReader(handle)) or filename == "manuscript_revision_flags.csv"
        elif filename.endswith(".docx"):
            with zipfile.ZipFile(path) as archive:
                assert "word/document.xml" in archive.namelist()
        else:
            assert path.read_text(encoding="utf-8")


def test_summary_and_manifest_are_consistent(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    output_dir = project_root / "outputs" / "scientific_manuscript"

    summary = json.loads((output_dir / "manuscript_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manuscript_manifest.json").read_text(encoding="utf-8"))

    assert manifest["summary"]["sentence_count"] == summary["sentence_count"]
    assert len(manifest["generated_output_files"]) == len(OUTPUT_FILENAMES)


def test_non_overwrite_refuses_non_empty_directory(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()

    try:
        ManuscriptEngine(project_root=project_root, overwrite=False).run()
    except FileExistsError as exc:
        assert "Use --overwrite" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError")


def test_overwrite_replaces_output_directory(tmp_path) -> None:
    project_root = create_manuscript_source_fixture(tmp_path)
    ManuscriptEngine(project_root=project_root, overwrite=True).run()
    marker = project_root / "outputs" / "scientific_manuscript" / "marker.txt"
    marker.write_text("old", encoding="utf-8")

    ManuscriptEngine(project_root=project_root, overwrite=True).run()

    assert not marker.exists()
