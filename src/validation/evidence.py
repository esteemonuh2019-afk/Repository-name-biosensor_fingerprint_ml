"""Evidence collection for validation and verification runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


EVIDENCE_LOG_PATH = Path("outputs") / "validation" / "evidence_log.json"


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_type: str
    timestamp: str
    file_path: str
    requirement_id: str
    description: str


def save_validation_evidence(
    evidence_type: str,
    requirement_id: str,
    source_file: str | Path,
    description: str,
) -> EvidenceRecord:
    """Append validation evidence metadata to the evidence log."""

    EVIDENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = load_evidence_log()
    record = EvidenceRecord(
        evidence_type=evidence_type,
        timestamp=datetime.now(UTC).isoformat(),
        file_path=str(source_file),
        requirement_id=requirement_id,
        description=description,
    )

    serialized_records = [asdict(existing_record) for existing_record in records]
    serialized_records.append(asdict(record))
    EVIDENCE_LOG_PATH.write_text(
        json.dumps(serialized_records, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def load_evidence_log() -> list[EvidenceRecord]:
    """Load validation evidence records from the evidence log."""

    if not EVIDENCE_LOG_PATH.exists():
        return []

    raw_records = json.loads(EVIDENCE_LOG_PATH.read_text(encoding="utf-8"))
    return [EvidenceRecord(**raw_record) for raw_record in raw_records]
