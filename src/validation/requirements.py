"""Requirement identifiers from the V&V plan."""

from __future__ import annotations


DATA_REQUIREMENT_IDS: tuple[str, ...] = (
    "REQ-DATA-001",
    "REQ-DATA-002",
    "REQ-DATA-003",
    "REQ-DATA-004",
)

PREPROCESSING_REQUIREMENT_IDS: tuple[str, ...] = (
    "REQ-PRE-001",
    "REQ-PRE-002",
    "REQ-PRE-003",
    "REQ-PRE-004",
)

FEATURE_REQUIREMENT_IDS: tuple[str, ...] = (
    "REQ-FEAT-001",
    "REQ-FEAT-002",
    "REQ-FEAT-003",
    "REQ-FEAT-004",
    "REQ-FEAT-005",
    "REQ-FEAT-006",
)

VISUALIZATION_REQUIREMENT_IDS: tuple[str, ...] = (
    "REQ-VIS-001",
    "REQ-VIS-002",
    "REQ-VIS-003",
    "REQ-VIS-004",
)

CLASSIFICATION_REQUIREMENT_IDS: tuple[str, ...] = (
    "REQ-MLC-001",
    "REQ-MLC-002",
    "REQ-MLC-003",
)

REGRESSION_REQUIREMENT_IDS: tuple[str, ...] = (
    "REQ-MLR-001",
    "REQ-MLR-002",
)

ALL_REQUIREMENT_IDS: tuple[str, ...] = (
    *DATA_REQUIREMENT_IDS,
    *PREPROCESSING_REQUIREMENT_IDS,
    *FEATURE_REQUIREMENT_IDS,
    *VISUALIZATION_REQUIREMENT_IDS,
    *CLASSIFICATION_REQUIREMENT_IDS,
    *REGRESSION_REQUIREMENT_IDS,
)

REQUIREMENT_DESCRIPTIONS: dict[str, str] = {
    "REQ-DATA-001": "Load CSV files",
    "REQ-DATA-002": "Validate schema",
    "REQ-DATA-003": "Validate strain labels",
    "REQ-DATA-004": "Validate concentration labels",
    "REQ-PRE-001": "Remove Monensin",
    "REQ-PRE-002": "Retain target chemicals",
    "REQ-PRE-003": "Normalize to controls",
    "REQ-PRE-004": "Handle missing values",
    "REQ-FEAT-001": "Calculate AUC",
    "REQ-FEAT-002": "Calculate max signal",
    "REQ-FEAT-003": "Calculate min signal",
    "REQ-FEAT-004": "Calculate time-to-peak",
    "REQ-FEAT-005": "Calculate slope",
    "REQ-FEAT-006": "Calculate fold-change",
    "REQ-VIS-001": "Heatmaps generated",
    "REQ-VIS-002": "PCA generated",
    "REQ-VIS-003": "Dose-response plots generated",
    "REQ-VIS-004": "Time-course plots generated",
    "REQ-MLC-001": "Train classifier",
    "REQ-MLC-002": "Predict chemical identity",
    "REQ-MLC-003": "Generate confusion matrix",
    "REQ-MLR-001": "Train concentration regressor",
    "REQ-MLR-002": "Predict concentration",
}

REQUIREMENT_VALIDATION_METHODS: dict[str, str] = {
    "REQ-DATA-001": "Unit + Black-box",
    "REQ-DATA-002": "Unit Test",
    "REQ-DATA-003": "Black-box",
    "REQ-DATA-004": "Unit Test",
    "REQ-PRE-001": "Unit Test",
    "REQ-PRE-002": "Unit Test",
    "REQ-PRE-003": "Unit Test",
    "REQ-PRE-004": "Black-box",
    "REQ-FEAT-001": "Unit Test",
    "REQ-FEAT-002": "Unit Test",
    "REQ-FEAT-003": "Unit Test",
    "REQ-FEAT-004": "Unit Test",
    "REQ-FEAT-005": "Unit Test",
    "REQ-FEAT-006": "Unit Test",
    "REQ-VIS-001": "Black-box",
    "REQ-VIS-002": "Black-box",
    "REQ-VIS-003": "Black-box",
    "REQ-VIS-004": "Black-box",
    "REQ-MLC-001": "Integration Test",
    "REQ-MLC-002": "ML Validation",
    "REQ-MLC-003": "Black-box",
    "REQ-MLR-001": "Integration Test",
    "REQ-MLR-002": "ML Validation",
}

REQUIREMENT_ACCEPTANCE_CRITERIA: dict[str, str] = {
    "REQ-DATA-001": "100% valid files loaded",
    "REQ-DATA-002": "Missing columns detected",
    "REQ-DATA-003": "Invalid labels rejected",
    "REQ-DATA-004": "All concentrations parsed correctly",
    "REQ-PRE-001": "100% removed",
    "REQ-PRE-002": "Only six target chemicals retained",
    "REQ-PRE-003": "Expected normalized output",
    "REQ-PRE-004": "Missing values reported correctly",
    "REQ-FEAT-001": "Matches manual calculation",
    "REQ-FEAT-002": "Exact match",
    "REQ-FEAT-003": "Exact match",
    "REQ-FEAT-004": "Exact match",
    "REQ-FEAT-005": "Exact match",
    "REQ-FEAT-006": "Exact match",
    "REQ-VIS-001": "Figure saved",
    "REQ-VIS-002": "Figure saved",
    "REQ-VIS-003": "Figure saved",
    "REQ-VIS-004": "Figure saved",
    "REQ-MLC-001": "Model successfully trained",
    "REQ-MLC-002": "Accuracy >= 80%",
    "REQ-MLC-003": "Figure generated",
    "REQ-MLR-001": "Model trained",
    "REQ-MLR-002": "R^2 >= 0.75",
}
