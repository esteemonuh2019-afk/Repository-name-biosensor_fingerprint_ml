from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.manuscript import ManuscriptSentence, SectionType, SentenceType, TraceabilityStatus
from src.scientific_reasoning.manuscript.policies import sentence_id


def test_manuscript_sentence_is_immutable_and_serializes() -> None:
    sentence = ManuscriptSentence(
        sentence_id="SENT-RESULTS-0001",
        text="The source observation listed 10 rows.",
        sentence_type=SentenceType.RESULT,
        section_id="SEC-RESULTS-0001",
        observation_ids=("OBS-2", "OBS-1"),
    )

    assert sentence.to_dict()["sentence_type"] == "RESULT"
    assert sentence.to_dict()["observation_ids"] == ["OBS-1", "OBS-2"]
    with pytest.raises(FrozenInstanceError):
        sentence.text = "changed"


def test_sentence_ids_are_deterministic() -> None:
    assert sentence_id(SectionType.RESULTS, 1) == "SENT-RESULTS-0001"
    assert sentence_id(SectionType.DISCUSSION, 12) == "SENT-DISCUSSION-0012"
    assert sentence_id(SectionType.TABLE_CAPTIONS, 2) == "SENT-TABLE-0002"


def test_traceability_status_serializes_as_string() -> None:
    sentence = ManuscriptSentence(
        sentence_id="SENT-ABSTRACT-0001",
        text="ABSTRACT PLACEHOLDER.",
        sentence_type="CONTEXT",
        section_id="SEC-ABSTRACT-0001",
        traceability_status=TraceabilityStatus.NOT_APPLICABLE,
        metadata={"placeholder": True},
    )

    assert sentence.to_dict()["traceability_status"] == "NOT_APPLICABLE"
