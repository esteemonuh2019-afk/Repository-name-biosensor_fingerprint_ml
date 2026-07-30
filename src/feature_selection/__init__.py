"""Stage 8D automatic feature selection."""

from src.feature_selection.feature_selector import (
    FEATURE_SELECTION_VERSION,
    REDUCTION_LEVELS,
    REQUIRED_SELECTOR_METHODS,
    FeatureSelectionConfig,
    build_generated_feature_table,
    run_feature_selection,
)
from src.feature_selection.selection_result import FeatureSelectionResult

__all__ = [
    "FEATURE_SELECTION_VERSION",
    "REDUCTION_LEVELS",
    "REQUIRED_SELECTOR_METHODS",
    "FeatureSelectionConfig",
    "FeatureSelectionResult",
    "build_generated_feature_table",
    "run_feature_selection",
]
