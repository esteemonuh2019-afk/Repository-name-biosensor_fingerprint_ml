import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.validation.requirements import ALL_REQUIREMENT_IDS
from src.validation.traceability import TraceabilityRecord, generate_traceability_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"

EXPECTED_REQUIREMENT_IDS = {
    "REQ-DATA-001",
    "REQ-DATA-002",
    "REQ-DATA-003",
    "REQ-DATA-004",
    "REQ-PRE-001",
    "REQ-PRE-002",
    "REQ-PRE-003",
    "REQ-PRE-004",
    "REQ-FEAT-001",
    "REQ-FEAT-002",
    "REQ-FEAT-003",
    "REQ-FEAT-004",
    "REQ-FEAT-005",
    "REQ-FEAT-006",
    "REQ-VIS-001",
    "REQ-VIS-002",
    "REQ-VIS-003",
    "REQ-VIS-004",
    "REQ-MLC-001",
    "REQ-MLC-002",
    "REQ-MLC-003",
    "REQ-MLR-001",
    "REQ-MLR-002",
}


@contextmanager
def local_test_workspace(test_name: str) -> Iterator[Path]:
    workspace = TEST_TMP_ROOT / test_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    try:
        yield workspace
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
        if TEST_TMP_ROOT.exists() and not any(TEST_TMP_ROOT.iterdir()):
            TEST_TMP_ROOT.rmdir()


def test_all_vv_plan_requirement_ids_exist() -> None:
    assert set(ALL_REQUIREMENT_IDS) == EXPECTED_REQUIREMENT_IDS


def test_traceability_record_can_be_created() -> None:
    record = TraceabilityRecord(
        requirement_id="REQ-DATA-001",
        description="Load CSV files",
        validation_method="Unit + Black-box",
        acceptance_criteria="100% valid files loaded",
        test_ids=("UT-001",),
        evidence="Unit test",
    )

    assert record.requirement_id == "REQ-DATA-001"
    assert record.test_ids == ("UT-001",)


def test_generate_traceability_matrix_writes_markdown_file() -> None:
    with local_test_workspace("traceability_matrix") as workspace:
        output_path = workspace / "traceability_matrix.md"

        written_path = generate_traceability_matrix(output_path)

        assert written_path == output_path
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("# Traceability Matrix")
        assert "| Requirement ID | Description | Validation Method | Acceptance Criteria | Test IDs | Evidence |" in content
        assert "REQ-DATA-001" in content
        assert "REQ-MLR-002" in content
