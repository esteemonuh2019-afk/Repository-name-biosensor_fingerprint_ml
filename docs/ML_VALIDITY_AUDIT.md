# ML Validity Audit

## Purpose

This document summarizes the scientific and statistical validity of the machine-learning framework developed for contaminant identification using whole-cell biosensor fingerprints.

The objective is to demonstrate that model selection, dataset design, feature engineering, validation strategy, and performance interpretation are appropriate for the research question.

---

# Research Objective

Determine whether luminescent biosensor fingerprints can be used to identify agricultural contaminants.

Machine-learning models are used to map biosensor response patterns to contaminant identity.

---

# Dataset Suitability

## Dataset Characteristics

The dataset consists of:

- Multiple biosensor strains
- Multiple contaminants
- Multiple concentrations
- Multiple experiments
- Time-series luminescence measurements

This structure provides sufficient variability to evaluate contaminant classification performance.

---

## Strengths

- Multiple biological replicates
- Multiple experiments
- Multiple strains
- Multiple contaminants

These characteristics reduce the likelihood of overfitting to a single biological condition.

---

## Limitations

- Limited contaminant diversity
- Limited strain diversity
- Laboratory-generated data only

Future work should expand biological diversity and external validation.

---

# Label Validity

## Label Definition

Labels correspond to experimentally applied contaminants.

Classes include:

- Boric Acid
- DEET
- Diazinon
- Metaldehyde
- Propoxur
- Trimethoprim

Labels are generated directly from experimental design rather than inferred post hoc.

---

## Scientific Justification

Each label corresponds to a known contaminant treatment and therefore represents a biologically meaningful classification target.

---

# Feature Engineering Validity

## Rationale

Raw luminescence time-series data contain thousands of measurements.

Feature engineering converts these responses into biologically interpretable descriptors.

---

## Features Used

Examples include:

- Area Under Curve (AUC)
- Maximum Signal
- Minimum Signal
- Final Signal
- Initial Slope
- Time-to-Peak

---

## Biological Interpretation

These features capture:

- Response magnitude
- Response dynamics
- Temporal behavior
- Induction and inhibition patterns

---

# Model Selection

## Selected Model

Random Forest Classifier

---

## Justification

Random Forest was selected because:

- Handles nonlinear relationships
- Robust to noise
- Performs well with moderate dataset sizes
- Provides feature importance estimates
- Requires limited parameter tuning

---

## Alternative Models Considered

Potential future comparisons include:

- XGBoost
- Gradient Boosting
- Support Vector Machines
- Temporal Neural Networks

---

# Validation Strategy

## Standard Holdout Validation

Train/Test split used for initial performance estimation.

---

## Leave-One-Experiment-Out (LOEO)

Purpose:

Evaluate generalization across independent experiments.

Results demonstrated performance degradation under experiment holdout conditions, providing a realistic estimate of deployment robustness.

---

## Leave-One-Strain-Out Validation

Purpose:

Evaluate biological generalization.

Each biosensor strain was removed during training and used exclusively for testing.

This prevents the model from relying on strain-specific fingerprints.

---

## Panel Optimization

Purpose:

Determine the smallest biosensor panel capable of maintaining acceptable performance.

Results support rational biosensor panel design.

---

# Leakage Control

The following controls were implemented:

- Experiment-level separation
- Strain-level separation
- Independent evaluation datasets

These controls reduce risk of information leakage.

---

# Performance Evaluation

## Classification Metrics

- Accuracy
- Precision
- Recall
- Macro F1

---

## Error Analysis

Evaluation includes:

- Confusion matrices
- Per-chemical performance
- LOEO performance
- Leave-one-strain-out performance

---

# Robustness Assessment

Robustness was evaluated using:

- LOEO validation
- Leave-One-Strain-Out validation
- Feature importance analysis
- Panel ablation studies

These analyses demonstrate that performance is not solely dependent on a single experiment or biosensor strain.

---

# Interpretability

Interpretability was evaluated using:

- Feature importance rankings
- Chemical-specific performance analysis
- Biological interpretation of temporal response features

The model therefore provides both predictive performance and biological insight.

---

# Conclusion

The machine-learning framework demonstrates strong scientific validity for contaminant classification using biosensor fingerprints.

Model selection, validation strategy, feature engineering, and robustness analyses are aligned with the research objective and provide a defensible basis for future publication-oriented development.