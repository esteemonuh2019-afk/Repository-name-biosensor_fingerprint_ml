"""Scientific narrative evidence extraction and aggregation."""

from src.scientific_narrative.evidence_database import (
    EvidenceDatabase,
    EvidenceRecord,
    SourceParseStatus,
    write_evidence_outputs,
)
from src.scientific_narrative.evidence_extractor import build_scientific_evidence
from src.scientific_narrative.evidence_aggregator import aggregate_scientific_evidence
from src.scientific_narrative.scientific_summary import AggregatedEvidence, write_aggregation_outputs

__all__ = [
    "AggregatedEvidence",
    "EvidenceDatabase",
    "EvidenceRecord",
    "SourceParseStatus",
    "aggregate_scientific_evidence",
    "build_scientific_evidence",
    "write_aggregation_outputs",
    "write_evidence_outputs",
]
