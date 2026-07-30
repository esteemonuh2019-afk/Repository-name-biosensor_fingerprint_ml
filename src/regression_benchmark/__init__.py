"""Stage 8B concentration regression benchmark framework."""

from src.regression_benchmark.benchmark_runner import (
    BENCHMARK_VERSION,
    DEFAULT_PREPROCESSING,
    DEFAULT_VALIDATION_STRATEGY,
    SUPPORTED_PREPROCESSING,
    SUPPORTED_VALIDATION_STRATEGIES,
    RegressionBenchmarkConfig,
    RegressionPreparedData,
    make_validation_splits,
    prepare_regression_data,
    rank_regression_models,
    run_regression_benchmark,
)
from src.regression_benchmark.models import (
    RegressionModelSpec,
    available_regression_model_specs,
    required_regression_model_ids,
)
from src.regression_benchmark.regression_dataset import (
    RegressionBenchmarkResult,
    render_regression_report,
)

__all__ = [
    "BENCHMARK_VERSION",
    "DEFAULT_PREPROCESSING",
    "DEFAULT_VALIDATION_STRATEGY",
    "SUPPORTED_PREPROCESSING",
    "SUPPORTED_VALIDATION_STRATEGIES",
    "RegressionBenchmarkConfig",
    "RegressionBenchmarkResult",
    "RegressionModelSpec",
    "RegressionPreparedData",
    "available_regression_model_specs",
    "make_validation_splits",
    "prepare_regression_data",
    "rank_regression_models",
    "render_regression_report",
    "required_regression_model_ids",
    "run_regression_benchmark",
]
