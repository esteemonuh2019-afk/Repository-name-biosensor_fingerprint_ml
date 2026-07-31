"""BSIP 2.2.0 Scientific Hypothesis Engine public contract and implementation."""

from .engine import (
    DEFAULT_SOFTWARE_VERSION,
    HYPOTHESIS_SCHEMA_VERSION,
    HypothesisEngine,
    InterpretationSourcePackage,
    ScientificHypothesisEngine,
    load_interpretation_package,
)
from .enums import (
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    HypothesisSeverity,
    HypothesisStatus,
)
from .models import Hypothesis, HypothesisRunResult, HypothesisValidationIssue
from .policies import (
    assign_confidence,
    falsifiability_is_valid,
    find_forbidden_hypothesis_terms,
    find_literature_comparison_terms,
    find_protocol_terms,
    find_recommendation_terms,
    priority_from_score,
    priority_score,
    supports_confidence_assignment,
)
from .rules import build_hypotheses
from .validators import validate_hypothesis, validate_hypotheses

__all__ = [
    "DEFAULT_SOFTWARE_VERSION",
    "HYPOTHESIS_SCHEMA_VERSION",
    "Hypothesis",
    "HypothesisCategory",
    "HypothesisConfidence",
    "HypothesisEngine",
    "HypothesisPriority",
    "HypothesisRunResult",
    "HypothesisSeverity",
    "HypothesisStatus",
    "HypothesisValidationIssue",
    "InterpretationSourcePackage",
    "ScientificHypothesisEngine",
    "assign_confidence",
    "build_hypotheses",
    "falsifiability_is_valid",
    "find_forbidden_hypothesis_terms",
    "find_literature_comparison_terms",
    "find_protocol_terms",
    "find_recommendation_terms",
    "load_interpretation_package",
    "priority_from_score",
    "priority_score",
    "supports_confidence_assignment",
    "validate_hypothesis",
    "validate_hypotheses",
]
