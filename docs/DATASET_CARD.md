# Dataset Card

## Dataset Name

Whole-Cell Biosensor Luminescence Fingerprint Dataset

---

## Purpose

This dataset was generated to investigate whether temporal luminescence responses from engineered whole-cell biosensors can be used to identify environmental contaminants.

The dataset supports:

* Contaminant classification
* Biosensor strain comparison
* Feature engineering
* Machine learning model development
* Experiment-independent validation

---

## Biological System

Luminescent bacterial biosensor strains:

* BL011
* BL027
* BL029
* BL030
* BL031
* BL032

---

## Target Chemicals

* Boric Acid
* DEET
* Diazinon
* Metaldehyde
* Propoxur
* Trimethoprim

---

## Data Collection

Measurements were collected using luminescence-based biosensor assays.

Each experiment produced time-series luminescence measurements recorded across multiple time points.

---

## Raw Data

Approximate raw observations:

* ~1,044,502 measurements

Data format:

* CSV files
* Time-series luminescence values
* Multiple strains
* Multiple contaminants
* Multiple concentrations

---

## Processed Dataset

Derived fingerprints:

* 3,645 feature-level samples

Features include:

### Original Features

* AUC
* Maximum Signal
* Minimum Signal
* Time-to-Peak
* Initial Slope
* Final Signal

### Advanced Features

* Peak-to-Baseline Ratio
* Fold Change
* Maximum Derivative
* Minimum Derivative
* Signal Decay Rate
* Early AUC
* Mid AUC
* Late AUC

---

## Preprocessing

The preprocessing pipeline includes:

* Data validation
* Missing value handling
* Signal normalization
* Feature extraction
* Advanced feature generation

---

## Labels

Target labels represent contaminant identity.

Label categories:

* Boric Acid
* DEET
* Diazinon
* Metaldehyde
* Propoxur
* Trimethoprim

---

## Validation Strategy

The project uses:

* Random train/test evaluation
* Leave-One-Experiment-Out (LOEO) validation
* Strain ablation studies
* Panel optimization
* Per-chemical analysis

---

## Known Limitations

* Dataset originates from a single biosensor platform.
* External validation has not yet been performed.
* Some contaminants exhibit overlapping fingerprints.
* Experiment-to-experiment variability exists.

---

## Intended Use

This dataset is intended for:

* Biosensor fingerprinting research
* Contaminant classification
* Feature engineering research
* Biosensor optimization studies

---

## Out-of-Scope Uses

The dataset should not be used for:

* Clinical diagnosis
* Regulatory decision-making
* Human health risk assessment

without additional validation.

---

## Version

Version 1.0

Prepared for SSDD/V&V submission package.
