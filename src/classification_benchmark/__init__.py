"""Stage 8A chemical classification benchmark framework."""

from src.classification_benchmark.benchmark_runner import (
    BENCHMARK_VERSION,
    DEFAULT_PREPROCESSING,
    DEFAULT_VALIDATION_STRATEGY,
    SUPPORTED_PREPROCESSING,
    SUPPORTED_VALIDATION_STRATEGIES,
    BenchmarkConfig,
    ClassificationPreparedData,
    make_validation_splits,
    prepare_classification_data,
    rank_models,
    run_classification_benchmark,
)
from src.classification_benchmark.classification_dataset import (
    ClassificationBenchmarkResult,
    render_classification_report,
)
from src.classification_benchmark.models import (
    ModelSpec,
    available_model_specs,
    required_model_ids,
)

__all__ = [
    "BENCHMARK_VERSION",
    "DEFAULT_PREPROCESSING",
    "DEFAULT_VALIDATION_STRATEGY",
    "SUPPORTED_PREPROCESSING",
    "SUPPORTED_VALIDATION_STRATEGIES",
    "BenchmarkConfig",
    "ClassificationBenchmarkResult",
    "ClassificationPreparedData",
    "ModelSpec",
    "available_model_specs",
    "make_validation_splits",
    "prepare_classification_data",
    "rank_models",
    "render_classification_report",
    "required_model_ids",
    "run_classification_benchmark",
]
