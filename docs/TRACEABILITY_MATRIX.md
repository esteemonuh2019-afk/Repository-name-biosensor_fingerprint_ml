# Requirement Traceability Matrix

## Project

Whole-Cell Biosensor Fingerprint ML Platform

---

| Req ID | Requirement                         | SSDD Area              | Implementation                               | Test Evidence            | Acceptance Criterion                | Status |
| ------ | ----------------------------------- | ---------------------- | -------------------------------------------- | ------------------------ | ----------------------------------- | ------ |
| R-001  | Load biosensor CSV data             | Data Ingestion         | src/data_ingestion                           | Unit Tests               | CSV files loaded successfully       | Passed |
| R-002  | Validate dataset schema             | Validation             | src/validation                               | Unit + Black-box Tests   | Invalid files rejected              | Passed |
| R-003  | Preprocess luminescence data        | Preprocessing          | src/preprocessing                            | Unit + Integration Tests | Clean dataset generated             | Passed |
| R-004  | Generate primary features           | Feature Engineering    | src/feature_engineering                      | Unit Tests               | Feature table created               | Passed |
| R-005  | Generate advanced temporal features | Feature Engineering    | src/feature_engineering/advanced_features.py | Unit Tests               | Advanced features generated         | Passed |
| R-006  | Train contaminant classifier        | Model Training         | src/model_training                           | Integration Tests        | Classification model trained        | Passed |
| R-007  | Compute evaluation metrics          | Model Evaluation       | src/model_evaluation                         | Unit Tests               | Metrics generated                   | Passed |
| R-008  | Generate confusion matrices         | Visualization          | src/visualization                            | Integration Tests        | Figures generated                   | Passed |
| R-009  | Perform LOEO validation             | Model Evaluation       | src/model_evaluation                         | Integration Tests        | LOEO metrics produced               | Passed |
| R-010  | Perform strain ablation analysis    | Model Evaluation       | src/model_evaluation                         | Regression Tests         | Ablation results generated          | Passed |
| R-011  | Perform panel optimization          | Model Evaluation       | src/model_evaluation                         | Regression Tests         | Panel ranking generated             | Passed |
| R-012  | Generate per-chemical analysis      | Model Evaluation       | src/model_evaluation                         | Regression Tests         | Chemical metrics generated          | Passed |
| R-013  | Generate scientific report          | Reporting              | scripts/reporting                            | E2E Tests                | Report produced                     | Passed |
| R-014  | Produce reproducible outputs        | Pipeline               | scripts                                      | E2E Tests                | Outputs recreated from scripts      | Passed |
| R-015  | Execute complete pipeline           | Pipeline Orchestration | scripts                                      | End-to-End Tests         | Full workflow executes successfully | Passed |

---

## Test Coverage Summary

### Unit Tests

* Data validation
* Feature generation
* Metrics calculation
* Advanced feature extraction

### Black-Box Tests

* Input/output validation
* Error handling
* Invalid schema detection

### Mock Tests

* Controlled datasets
* Simulated inputs

### Regression Tests

* Protection of approved behavior
* Prevention of feature regressions

### Integration Tests

* Module-to-module compatibility
* Data flow validation

### End-to-End Tests

* Full pipeline execution
* Output generation

---

## Evidence

Evidence generated during validation includes:

* pytest reports
* model_metrics.json
* loeo_metrics.json
* panel_optimization.csv
* per_chemical_loeo.csv
* scientific_performance_report.md
* generated figures
* generated tables

---

## Current Status

All major SSDD requirements have corresponding implementation modules, validation tests, acceptance criteria, and generated evidence.

Prepared for SSDD/V&V review.
