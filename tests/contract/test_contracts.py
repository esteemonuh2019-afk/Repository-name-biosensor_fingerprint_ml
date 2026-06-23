"""Contract placeholders for SSDD-defined pipeline interfaces.

These tests document expected module boundaries without implementing the
biosensor pipeline. They skip until production modules provide the named or
equivalent callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

import pytest


@dataclass(frozen=True)
class InterfaceContract:
    contract_id: str
    candidate_modules: tuple[str, ...]
    candidate_callables: tuple[str, ...]
    required_input_type: str
    required_output_type: str
    required_failure_behavior: str


SCHEMA_VALIDATION_CONTRACT = InterfaceContract(
    contract_id="CT-001",
    candidate_modules=("src.validation", "src.validation.schema", "src.validation.validation"),
    candidate_callables=("validate_schema",),
    required_input_type="tabular raw luminescence data with SSDD-required columns",
    required_output_type="validation result/report indicating pass/fail status and issues",
    required_failure_behavior="must reject or report missing required columns",
)

PREPROCESSING_CONTRACT = InterfaceContract(
    contract_id="CT-002",
    candidate_modules=("src.preprocessing",),
    candidate_callables=("preprocess", "clean_preprocess"),
    required_input_type="validated tabular biosensor data",
    required_output_type="cleaned tabular data with Monensin removed and target chemicals retained",
    required_failure_behavior="must report invalid labels or missing controls without silent data loss",
)

FEATURE_EXTRACTION_CONTRACT = InterfaceContract(
    contract_id="CT-003",
    candidate_modules=("src.features", "src.feature_extraction"),
    candidate_callables=("extract_features",),
    required_input_type="preprocessed normalized biosensor time-series data",
    required_output_type="feature table containing kinetic fingerprint features",
    required_failure_behavior="must fail or report when required time-series fields are unavailable",
)

CLASSIFIER_TRAINING_CONTRACT = InterfaceContract(
    contract_id="CT-004",
    candidate_modules=("src.models", "src.classification"),
    candidate_callables=("train_classifier",),
    required_input_type="feature table and chemical identity labels",
    required_output_type="trained classifier object or classifier training result",
    required_failure_behavior="must fail or report when labels are missing or class sizes are invalid",
)

REGRESSOR_TRAINING_CONTRACT = InterfaceContract(
    contract_id="CT-005",
    candidate_modules=("src.models", "src.regression"),
    candidate_callables=("train_regressor",),
    required_input_type="feature table and numeric concentration targets",
    required_output_type="trained regressor object or regressor training result",
    required_failure_behavior="must fail or report when targets are missing or non-numeric",
)


def test_validate_schema_contract_placeholder() -> None:
    """Required input: raw table; output: validation result; failure: missing columns."""

    _assert_contract_documented(SCHEMA_VALIDATION_CONTRACT)
    _resolve_contract_callable_or_skip(SCHEMA_VALIDATION_CONTRACT)


def test_preprocess_contract_placeholder() -> None:
    """Required input: validated table; output: cleaned table; failure: invalid labels/controls."""

    _assert_contract_documented(PREPROCESSING_CONTRACT)
    _resolve_contract_callable_or_skip(PREPROCESSING_CONTRACT)


def test_extract_features_contract_placeholder() -> None:
    """Required input: time series; output: feature table; failure: missing time-series fields."""

    _assert_contract_documented(FEATURE_EXTRACTION_CONTRACT)
    _resolve_contract_callable_or_skip(FEATURE_EXTRACTION_CONTRACT)


def test_train_classifier_contract_placeholder() -> None:
    """Required input: features and labels; output: classifier; failure: invalid labels/classes."""

    _assert_contract_documented(CLASSIFIER_TRAINING_CONTRACT)
    _resolve_contract_callable_or_skip(CLASSIFIER_TRAINING_CONTRACT)


def test_train_regressor_contract_placeholder() -> None:
    """Required input: features and targets; output: regressor; failure: invalid targets."""

    _assert_contract_documented(REGRESSOR_TRAINING_CONTRACT)
    _resolve_contract_callable_or_skip(REGRESSOR_TRAINING_CONTRACT)


def _assert_contract_documented(contract: InterfaceContract) -> None:
    assert contract.contract_id
    assert contract.candidate_modules
    assert contract.candidate_callables
    assert contract.required_input_type
    assert contract.required_output_type
    assert contract.required_failure_behavior


def _resolve_contract_callable_or_skip(contract: InterfaceContract) -> Callable[..., object]:
    for module_name in contract.candidate_modules:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue

        for callable_name in contract.candidate_callables:
            candidate = getattr(module, callable_name, None)
            if callable(candidate):
                return candidate

    pytest.skip(
        f"{contract.contract_id} placeholder: expected callable "
        f"{' or '.join(contract.candidate_callables)} in "
        f"{', '.join(contract.candidate_modules)}. "
        f"Input: {contract.required_input_type}. "
        f"Output: {contract.required_output_type}. "
        f"Failure behavior: {contract.required_failure_behavior}."
    )
