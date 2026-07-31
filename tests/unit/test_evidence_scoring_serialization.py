import json

from src.scientific_reasoning.evidence_scoring.models import json_ready
from tests.unit.test_evidence_scoring_models import record


def test_json_ready_serializes_nested_evidence_record() -> None:
    payload = json_ready(record())
    json.dumps(payload, sort_keys=True)
    assert payload["evidence_level"] == "VERY_STRONG"
