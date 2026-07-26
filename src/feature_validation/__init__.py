"""Stage 6C feature validation package."""

from src.feature_validation.feature_selection_report import (
    render_validation_report,
    write_validation_outputs,
)
from src.feature_validation.feature_statistics import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_DOMINANT_PROPORTION_THRESHOLD,
    DEFAULT_LOW_VARIANCE_THRESHOLD,
)
from src.feature_validation.feature_validator import (
    FeatureValidationResult,
    validate_features,
)
from src.feature_validation.replicate_reproducibility import (
    DEFAULT_ACCEPTABLE_CV_THRESHOLD,
    DEFAULT_STABLE_CV_THRESHOLD,
)

__all__ = [
    "DEFAULT_ACCEPTABLE_CV_THRESHOLD",
    "DEFAULT_CORRELATION_THRESHOLD",
    "DEFAULT_DOMINANT_PROPORTION_THRESHOLD",
    "DEFAULT_LOW_VARIANCE_THRESHOLD",
    "DEFAULT_STABLE_CV_THRESHOLD",
    "FeatureValidationResult",
    "render_validation_report",
    "validate_features",
    "write_validation_outputs",
]

