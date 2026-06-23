# Statistical Analysis Plan

## Primary Objective

Identify agricultural contaminants using luminescent whole-cell biosensor fingerprints.

## Classification Task

Input:
- Luminescence time-series measurements

Output:
- Contaminant identity

Target Classes:
- Boric Acid
- DEET
- Diazinon
- Metaldehyde
- Propoxur
- Trimethoprim

## Primary Metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

## Secondary Metrics

- Confusion Matrix
- Per-Chemical Performance
- Feature Importance Rankings

## Regression Metrics

Where concentration prediction is performed:

- R²
- RMSE
- MAE

## Validation Strategy

### Standard Holdout Validation

Train/Test split performed on processed feature data.

### Leave-One-Experiment-Out (LOEO)

Each experiment is held out individually and evaluated using models trained on remaining experiments.

Purpose:
- Assess experimental generalization.

### Leave-One-Strain-Out Validation

Each bacterial strain is removed during training and used exclusively for testing.

Purpose:
- Assess biological generalization.

### Panel Ablation Analysis

Different strain panels are evaluated to determine minimum biosensor requirements.

Purpose:
- Identify efficient sensing configurations.

## Leakage Prevention

The project implements:

- Experiment-level separation
- Strain-level separation
- Independent evaluation sets

to reduce information leakage between training and testing data.

## Error Analysis

Performance evaluation includes:

- Confusion matrices
- Per-class F1 scores
- Feature importance analysis
- Cross-experiment validation
- Cross-strain validation