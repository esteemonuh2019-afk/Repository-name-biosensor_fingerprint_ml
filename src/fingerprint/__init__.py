"""Stage 7A fingerprint dataset builder package."""

from src.fingerprint.fingerprint_builder import (
    DEFAULT_CONSENSUS_GROUP_COLUMNS,
    DEFAULT_DISTANCE_MODE,
    DEFAULT_NORMALIZATION,
    FINGERPRINT_FEATURE_COLUMNS,
    FINGERPRINT_VERSION,
    build_consensus_fingerprints,
    build_fingerprint_dataset,
    normalize_fingerprint_dataframe,
)
from src.fingerprint.fingerprint_dataset import (
    DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS,
    FingerprintDataset,
)
from src.fingerprint.fingerprint_qc import FingerprintQCResult, evaluate_fingerprint_qc
from src.fingerprint.fingerprint_similarity import (
    calculate_distance_matrix,
    correlation_distance,
    cosine_distance,
    euclidean_distance,
    estimate_distance_matrix_size,
    manhattan_distance,
    write_distance_matrix_csv,
)

__all__ = [
    "DEFAULT_CONSENSUS_GROUP_COLUMNS",
    "DEFAULT_DISTANCE_MODE",
    "DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS",
    "DEFAULT_NORMALIZATION",
    "FINGERPRINT_FEATURE_COLUMNS",
    "FINGERPRINT_VERSION",
    "FingerprintDataset",
    "FingerprintQCResult",
    "build_consensus_fingerprints",
    "build_fingerprint_dataset",
    "calculate_distance_matrix",
    "correlation_distance",
    "cosine_distance",
    "euclidean_distance",
    "estimate_distance_matrix_size",
    "evaluate_fingerprint_qc",
    "manhattan_distance",
    "normalize_fingerprint_dataframe",
    "write_distance_matrix_csv",
]
