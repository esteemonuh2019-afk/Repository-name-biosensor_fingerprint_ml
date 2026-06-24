# Sample Size and Replication Justification

## Purpose

This document justifies the dataset structure, biosensor strain panel, contaminant selection, replication strategy, and validation design used in the Whole-Cell Biosensor Fingerprint ML Platform.

---

## Dataset Structure

The project evaluates luminescent whole-cell biosensor responses across:

- Six biosensor strains
- Six target contaminants
- Multiple concentrations
- Multiple experimental runs
- Time-series luminescence measurements

This design supports both biological interpretation and machine-learning model development.

---

## Biosensor Strain Panel

The study includes six bacterial biosensor strains:

- BL011
- BL027
- BL029
- BL030
- BL031
- BL032

Using multiple strains is scientifically justified because different biosensors may respond differently to the same contaminant.

The analysis showed that strain-specific responses are biologically meaningful, with some strains performing better for specific contaminants.

---

## Contaminant Panel

The contaminant panel includes:

- Boric Acid
- DEET
- Diazinon
- Metaldehyde
- Propoxur
- Trimethoprim

These targets provide a chemically diverse set of agricultural and antimicrobial contaminants.

The inclusion of both pesticides and Trimethoprim allows the platform to test whether luminescence fingerprints can distinguish different biological stress patterns.

---

## Replication Strategy

Replication is necessary because whole-cell biosensors are living systems and may show biological variability.

The project includes repeated measurements and multiple experimental runs to support:

- Signal reproducibility assessment
- Model training
- Experiment-level validation
- Detection of batch effects
- Robustness testing

---

## Feature-Level Sample Size

The raw dataset contains approximately:

- 1,044,502 processed luminescence measurements
- 3,645 derived fingerprint-level samples

This provides sufficient feature-level data for classical machine-learning models such as Random Forest.

---

## Validation Justification

### Random Train/Test Split

Used to estimate initial model performance.

### Leave-One-Experiment-Out Validation

Used to test whether the model generalizes across independent experiments.

This is critical because random splits can overestimate performance when samples from the same experiment appear in both training and testing sets.

### Leave-One-Strain-Out Validation

Used to test whether model performance depends on individual biosensor strains.

This helps determine whether the biosensor panel is broadly informative or strain-specific.

### Panel Optimization

Used to identify the smallest informative biosensor strain set.

This is important for practical biosensor design because smaller panels may be cheaper, simpler, and more robust.

---

## Statistical Suitability

The dataset is appropriate for proof-of-concept machine-learning analysis because it includes:

- Multiple biological conditions
- Multiple chemical classes
- Multiple biosensor strains
- Repeated experimental measurements
- Independent validation strategies

---

## Limitations

The current dataset remains limited by:

- Laboratory-only conditions
- No external laboratory validation
- Limited number of contaminants
- Limited biosensor strain diversity

Therefore, the dataset is sufficient for proof-of-concept validation but not yet sufficient for regulatory or field deployment claims.

---

## Conclusion

The sample size and replication structure are appropriate for a proof-of-concept biosensor fingerprinting study.

The combination of multiple strains, multiple contaminants, time-series measurements, feature extraction, LOEO validation, and repeated-run testing provides a scientifically defensible basis for model development and evaluation.