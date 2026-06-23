# Limitations and Risks

## Purpose

This document identifies known limitations, risks, assumptions, and future development requirements associated with the Whole-Cell Biosensor Fingerprint Machine Learning Platform.

---

## Dataset Limitations

### Limited Contaminant Diversity

The current dataset contains six contaminants:

* Boric Acid
* DEET
* Diazinon
* Metaldehyde
* Propoxur
* Trimethoprim

Performance on additional contaminants remains unknown.

---

### Limited Biosensor Diversity

The study evaluates six biosensor strains:

* BL011
* BL027
* BL029
* BL030
* BL031
* BL032

Additional strains may alter classification performance.

---

### Laboratory Dataset

All measurements were generated under controlled laboratory conditions.

Real-world environmental variability has not yet been evaluated.

---

## Biological Risks

### Biosensor Drift

Whole-cell biosensors may exhibit changes in response characteristics over time due to:

* Genetic instability
* Metabolic adaptation
* Growth-state variation

---

### Environmental Sensitivity

Biosensor responses may be affected by:

* Temperature
* pH
* Humidity
* Nutrient availability

These factors were not explicitly modeled.

---

### Strain-Specific Behavior

Results demonstrate that some strains specialize in recognition of specific contaminants.

Consequently, performance may vary substantially depending on strain selection.

---

## Machine Learning Limitations

### Generalization Risk

Although LOEO and Leave-One-Strain-Out validation were performed, true external validation has not yet been conducted.

---

### Unseen Contaminants

The model cannot currently identify contaminants not represented in the training data.

---

### Temporal Compression

Feature engineering reduces complex time-series responses into summary features.

Some biological information may be lost during this process.

---

## Statistical Limitations

### Class Distribution

Performance may be influenced by class imbalance and contaminant-specific variability.

---

### Experimental Variability

Differences between experiments may influence classification performance despite normalization procedures.

---

## Deployment Risks

Potential deployment challenges include:

* Sensor degradation
* Storage effects
* Transportation stress
* Environmental interference
* Hardware variability

---

## Future Risk Mitigation

Future work should include:

1. External validation datasets
2. Additional contaminants
3. Additional biosensor strains
4. Field deployment studies
5. Confidence interval estimation
6. Uncertainty quantification
7. Multi-laboratory validation

---

## Conclusion

The current platform demonstrates strong research and validation evidence but should be considered a laboratory-stage biosensor analytics framework rather than a deployment-ready monitoring system.
