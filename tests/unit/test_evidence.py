import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.validation.evidence import (
    EVIDENCE_LOG_PATH,
    EvidenceRecord,
    load_evidence_log,
    save_validation_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"


@contextmanager
def local_test_workspace(test_name: str) -> Iterator[Path]:
    workspace = TEST_TMP_ROOT / test_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        yield workspace
    finally:
        os.chdir(previous_cwd)
        if workspace.exists():
            shutil.rmtree(workspace)
        if TEST_TMP_ROOT.exists() and not any(TEST_TMP_ROOT.iterdir()):
            TEST_TMP_ROOT.rmdir()


def test_evidence_record_can_be_created() -> None:
    record = EvidenceRecord(
        evidence_type="test_log",
        timestamp="2026-06-22T10:00:00+00:00",
        file_path="outputs/logs/unit.log",
        requirement_id="REQ-DATA-001",
        description="Valid CSV load test passed",
    )

    assert record.evidence_type == "test_log"
    assert record.requirement_id == "REQ-DATA-001"


def test_save_validation_evidence_creates_log() -> None:
    with local_test_workspace("evidence_creates_log"):
        record = save_validation_evidence(
            evidence_type="test_log",
            requirement_id="REQ-DATA-001",
            source_file="outputs/logs/unit.log",
            description="Valid CSV load test passed",
        )

        assert EVIDENCE_LOG_PATH.exists()
        log_data = json.loads(EVIDENCE_LOG_PATH.read_text(encoding="utf-8"))
        assert log_data == [
            {
                "evidence_type": "test_log",
                "timestamp": record.timestamp,
                "file_path": "outputs/logs/unit.log",
                "requirement_id": "REQ-DATA-001",
                "description": "Valid CSV load test passed",
            }
        ]


def test_multiple_validation_evidence_records_append() -> None:
    with local_test_workspace("evidence_appends_records"):
        save_validation_evidence(
            evidence_type="test_log",
            requirement_id="REQ-DATA-001",
            source_file="outputs/logs/unit.log",
            description="Valid CSV load test passed",
        )
        save_validation_evidence(
            evidence_type="figure",
            requirement_id="REQ-VIS-001",
            source_file="outputs/figures/heatmap.png",
            description="Heatmap generated",
        )

        log_data = json.loads(EVIDENCE_LOG_PATH.read_text(encoding="utf-8"))
        assert len(log_data) == 2
        assert log_data[0]["requirement_id"] == "REQ-DATA-001"
        assert log_data[1]["requirement_id"] == "REQ-VIS-001"


def test_load_evidence_log_returns_records() -> None:
    with local_test_workspace("evidence_loads_records"):
        saved_record = save_validation_evidence(
            evidence_type="metrics_table",
            requirement_id="REQ-MLR-002",
            source_file="outputs/tables/regression_metrics.csv",
            description="Regression metrics captured",
        )

        records = load_evidence_log()

        assert records == [saved_record]
