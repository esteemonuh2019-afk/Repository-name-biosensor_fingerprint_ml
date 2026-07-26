# Feature Validation Strategy

Stage: 6A

Status: validation specification only. This document defines how the future feature extraction engine should be verified. It does not implement tests or algorithms.

## Purpose

The future feature extraction engine will compress each canonical luminescence time series into one feature vector. Validation must prove that the feature values are mathematically correct, biologically interpretable, reproducible, and honest about uncertainty.

The validation strategy must protect three scientific requirements:

- Raw measurements are never modified.
- Duplicate timestamps are never averaged silently.
- Low-quality or ambiguous series remain visible through feature-level quality metrics and confidence labels.

## Validation Scope

Validation applies to features computed from one canonical measurement series grouped by:

- `Experiment_ID`
- `Source_File`
- `Measurement_Unit_ID`

The engine must preserve these identifiers in feature outputs so every feature vector can be traced back to its source rows.

Validation must cover:

- signal intensity features
- temporal features
- area features
- slope features
- relative-response features
- quality metrics
- confidence scoring
- canonical schema compatibility
- Stage 5C duplicate-timepoint risk

## Reference Datasets

Synthetic curves should be the primary mathematical validation source because their expected values are known exactly. Real canonical data should be used for integration and audit validation, not for proving exact formulas.

Required synthetic curve families:

| Curve family | Purpose | Expected behavior |
|---|---|---|
| Constant curve | Baseline, endpoint, AUC, standard deviation, no response | Dynamic range, slopes, positive/negative area, and SD are zero; endpoint/baseline is one when baseline is positive. |
| Linear increasing curve | Slope, AUC, time to peak, fold change | Initial, average, and all adjacent slopes match the known slope; peak occurs at endpoint. |
| Linear decreasing curve | Minimum, negative response, minimum slope, endpoint ratio | Minimum occurs at endpoint; average slope is negative; endpoint/baseline is below one. |
| Triangular induction peak | Peak timing, peak width, positive area, recovery slope | Peak time, half-maximum duration, and AUC match analytic values. |
| Inhibitory trough | Minimum timing, negative area, minimum fold change | Minimum and negative area match expected values; recovery features behave as specified. |
| Step response | Response latency and sustained endpoint response | Latency is the known step time once threshold rules are approved. |
| Pulse response | Transient response and recovery | Endpoint returns near baseline; recovery time and positive or negative area match expectations. |
| Multi-peak curve | Multiple-peak policy | Earliest global maximum or approved dominant peak is selected reproducibly. |
| Multi-minimum curve | Multiple-minimum policy | Earliest global minimum or approved dominant trough is selected reproducibly. |
| Noisy no-response curve | False-positive resistance | Dynamic features remain low confidence if threshold crossings are noise-driven. |
| Sparse curve | Missing and insufficient evidence | Features requiring two or more points return null; quality counts explain why. |
| Duplicate timestamp curve | Stage 5C protection | Conflicting duplicate timestamps block or lower confidence for time-dependent features and are counted explicitly. |
| Irregular time-spacing curve | Trapezoidal and slope correctness | AUC and slopes use actual time intervals, not observation index. |
| Zero-baseline curve | Relative-feature denominator handling | Ratio and fold-change features return null or approved offset behavior. |
| Negative raw-value curve | Retention and warning behavior | Negative values are retained for calculations where valid, with QC/confidence penalties. |

## Feature-Level Validation Matrix

| Feature group | Features | Primary validation checks |
|---|---|---|
| Signal Intensity | Baseline, maximum, minimum, endpoint, dynamic range, mean, median, standard deviation, coefficient of variation, baseline variability | Validate exact values on constant, increasing, decreasing, shifted-baseline, noisy-baseline, and missing-baseline curves. Confirm extrema tie handling and near-zero denominator behavior. |
| Temporal Features | Time to peak, time to minimum, response latency, recovery time, peak width, half-maximum duration | Validate known peak/trough times, tied extrema, multi-peak policy, missing crossing behavior, threshold persistence, and null behavior for no-response curves. |
| Area Features | Total AUC, positive area, negative area, normalised area | Validate trapezoidal integration on analytic curves, signed negative area, baseline-centered integration, irregular time spacing, and duration normalization. |
| Slope Features | Initial slope, maximum slope, minimum slope, average slope, recovery slope | Validate known adjacent slopes, zero-duration rejection, duplicate timestamp handling, endpoint slope, and recovery from peak or trough. |
| Relative Response | Maximum fold change, minimum fold change, log2 fold change, normalised endpoint, peak/baseline ratio, endpoint/baseline ratio | Validate baseline denominator handling, scaled-curve invariance, zero or negative denominator behavior, and endpoint versus peak distinction. |
| Quality Metrics | Signal-to-noise ratio, valid observations, missing observations, duplicate timestamp count, QC flag count, confidence score | Validate exact counts, parsed QC flags, SNR under known noise, and confidence thresholds for clean, warning, sparse, noisy, invalid, and duplicate-timestamp curves. |

## Mathematical Acceptance Criteria

For deterministic synthetic curves, feature values should match analytic expectations within a documented numeric tolerance. Suggested default tolerance is strict equality for integer counts and exact extrema, and a small floating-point tolerance for integration, slopes, and ratios.

Acceptance rules:

- Constant curve AUC must equal constant signal multiplied by duration.
- Linear curve AUC must match trapezoidal integration exactly for sampled endpoints.
- Dynamic range must equal maximum minus minimum.
- Time to peak and time to minimum must return the earliest tied extremum unless the implementation documents a different approved rule.
- Slopes must use actual time differences.
- Duplicate timestamps must not be resolved by averaging inside feature extraction.
- Relative features with invalid denominators must return null and lower confidence.
- Quality metrics must still be emitted when scientific features are null.

## Canonical Integration Validation

Integration tests should use canonical-format fixtures with all 35 canonical columns. Fixtures should include:

- one clean series with unique timestamps
- one series with missing optional identifiers but no ambiguity
- one series with synthetic `Measurement_Unit_ID`
- one series with `Record_Valid = False` rows
- one series with missing `Luminescence_Raw`
- one series with duplicate timestamps and identical values
- one series with duplicate timestamps and conflicting values
- one series with missing `Replicate_ID`, `Plate_ID`, and `Well_ID`
- one series with negative raw luminescence
- one series with zero baseline

Expected outcomes:

- Series grouping uses `Experiment_ID`, `Source_File`, and `Measurement_Unit_ID`.
- Measurement identity fields are preserved in output.
- No feature output is grouped only by strain, chemical, concentration, replicate, and time.
- Duplicate-timepoint series do not get averaged by the feature engine.
- Feature rows retain enough provenance to trace back to source files and measurement units.

## Real-Data Verification

The first real-data verification run should use the canonical dataset built from the same source files audited in Stage 5C.

Minimum checks:

- Feature row count equals the number of eligible measurement series after documented exclusions or low-confidence retention.
- The 26 Stage 5C conflicting duplicate-timepoint groups are not silently averaged.
- Affected BL030 and BL032 series are marked low confidence or excluded from mathematical features with explicit reason fields.
- The one ambiguous measurement identity row is visible in quality metrics.
- Missing `Plate_ID` and `Well_ID` do not prevent feature creation by themselves, but they lower provenance confidence where appropriate.
- No raw data files are modified.
- No canonical input rows are deleted by the feature engine.

The real-data run should be considered a scientific smoke test, not a proof of formula correctness. Exact formula validation must remain synthetic.

## Confidence Validation

Confidence scoring must be deterministic and versioned. Validation should include a table of scenarios with expected bands.

| Scenario | Expected confidence |
|---|---|
| Clean, dense, unique timestamps, stable baseline | High |
| Synthetic measurement unit but clean signal | Medium or High depending on provenance weighting |
| Missing plate and well but unambiguous source-position unit | Medium unless project policy assigns stronger penalty |
| Sparse curve with two valid points | Medium or Low depending on feature |
| One valid point only | Low for most scientific features; quality metrics still emitted |
| High baseline noise | Medium or Low for relative, latency, width, and SNR features |
| Unresolved conflicting duplicate timestamp | Low for time-dependent feature vector |
| Missing or invalid baseline | Low for baseline-dependent features |
| QC failure affecting required measurement fields | Low or null scientific features |

Confidence validation must check both the numeric score and categorical label: High, Medium, or Low.

## Missing-Data Validation

Missing-data validation must distinguish between:

- absent rows that are known from an expected sampling schedule
- present rows with missing time
- present rows with missing raw luminescence
- invalid retained records
- optional missing identifiers

Feature behavior:

- Quality metrics must count missingness.
- Features requiring missing inputs must return null.
- Features that remain mathematically valid may be calculated from valid observations but should receive reduced confidence.
- Missing endpoint observations must lower endpoint and sustained-response confidence.
- Missing early observations must lower baseline, latency, initial-slope, and fold-change confidence.

## Duplicate Timestamp Validation

Duplicate timestamp validation must include two separate cases:

- identical duplicate values
- conflicting duplicate values

Identical duplicates may represent duplicate source records or repeated export behavior. Conflicting duplicates may represent legitimate repeated measurements, missing identifiers, importer defects, or source ambiguity. In both cases, the feature engine must not average them silently.

Required assertions:

- Duplicate timestamp count is non-zero.
- Time-dependent features either return null or receive explicit low confidence under the approved policy.
- Conflicting duplicates are distinguishable from identical duplicates in QC provenance.
- Source row identifiers remain traceable.

## Multiple-Peak and Multiple-Minimum Validation

Synthetic multi-peak curves must test:

- equal-height peaks
- one dominant peak and one smaller peak
- a noisy spike followed by a biologically plausible broad peak
- two separated transient responses

Synthetic multi-minimum curves must test:

- equal-depth minima
- one dominant inhibitory trough
- a noisy downward spike
- inhibition followed by recovery and re-inhibition

Before peak-detection algorithms are implemented, core features must use earliest global maximum/minimum. After peak detection is approved, validation must compare global-extremum features with dominant-event features to avoid silent definition drift.

## Noisy-Curve Validation

Noisy-curve validation should include:

- additive Gaussian-like noise around a constant baseline
- isolated spike artifacts
- isolated downward artifacts
- noisy induction with true sustained response
- noisy transient response

Expected behavior:

- Robust baseline and SNR features reflect the known noise level.
- Extrema and slope confidence is lowered for unsupported isolated spikes.
- Response latency avoids false threshold crossing when persistence rules are not met.
- Smoothing-dependent features document whether smoothing was applied.

## Biological Plausibility Validation

Feature outputs should be checked for biological plausibility across known response types:

- strong induction: high maximum, high positive area, high endpoint ratio if sustained
- strong inhibition: low minimum, negative area, endpoint ratio below one if sustained
- transient induction: high peak, recovery toward baseline, endpoint ratio near one
- delayed response: high latency and late time to peak
- no response: low dynamic range, low slopes, low positive and negative area, confidence driven by adequate observations

These checks should not enforce a biological result on real data. They should ensure that feature signs and magnitudes support sensible interpretation.

## Regression and Backward-Compatibility Validation

Existing historical outputs include older features such as `auc`, `max_signal`, `min_signal`, `time_to_peak`, `initial_slope`, `final_signal`, peak-to-baseline ratio, fold change, derivative features, and segmented AUC. The Stage 6B canonical engine may produce differently named and more strictly validated features.

Validation should therefore separate:

- mathematical equivalence tests on clean synthetic data where definitions match
- schema compatibility tests for the new canonical feature table
- non-regression tests proving existing legacy tests still pass if legacy modules remain unchanged

The canonical feature engine should not inherit legacy preprocessing that averages duplicate measurements before feature extraction.

## Documentation Validation

Every implemented feature must have:

- name
- short name
- category
- units
- priority
- mathematical definition
- dependency declaration
- missing-data behavior
- duplicate-timestamp behavior
- confidence behavior
- validation fixture coverage

The implementation should fail documentation review if a feature exists in code but not in the feature dictionary, or if the dictionary names a feature that is not implemented after its target stage.

## Recommended Stage 6B Validation Order

1. Validate canonical series grouping and provenance preservation.
2. Validate quality metrics and duplicate-timestamp detection.
3. Validate baseline estimation and denominator safety.
4. Validate core signal intensity features.
5. Validate core slope features on clean unique timestamps.
6. Validate total AUC on constant, linear, triangular, and irregularly spaced curves.
7. Validate core relative-response features.
8. Validate confidence score and label mapping.
9. Validate real-data behavior on Stage 5C conflict cases.
10. Add recommended and optional feature tests only after core behavior is stable.

## Acceptance Criteria for Stage 6B

Stage 6B should be accepted only when:

- Core feature formulas pass synthetic validation.
- Quality metrics are emitted for every series, including low-quality series.
- Duplicate conflicting timestamps are not averaged.
- Feature outputs preserve canonical series identifiers.
- Feature confidence is deterministic and documented.
- Existing tests continue to pass.
- Real-data verification reports how many series are high, medium, and low confidence.
- No feature extraction code modifies raw data, canonical input data, or Stage 5 QC artifacts.

