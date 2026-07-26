# Biosensor Feature Extraction Scientific Specification

Stage: 6A

Status: scientific specification only. This document does not implement the feature extraction engine and does not alter any existing calculation.

## Purpose

This document defines the scientific contract for the future biosensor feature extraction engine. A feature is a biologically interpretable descriptor calculated from one luminescence time series. Each independent measurement series produces one feature vector.

The specification is designed to be reproducible enough that a researcher could independently implement the feature engine without referring to project code. It is also designed to protect the raw scientific signal: measurements must not be deleted, averaged, or suppressed to make features easier to compute.

## Canonical Input Contract

The feature engine must consume the canonical long-format dataset defined by schema version `1.1.0`. One canonical row represents one luminescence value for one experimental unit at one measured time point.

The future feature engine must group rows into one series using:

- `Experiment_ID`
- `Source_File`
- `Measurement_Unit_ID`

The measurement key for individual rows is:

- `Experiment_ID`
- `Source_File`
- `Measurement_Unit_ID`
- `Time_Minutes`

The feature engine must use `Luminescence_Raw` as the primary signal unless a feature explicitly requires a separately generated normalized signal. `Luminescence_Normalized` must never overwrite the raw signal.

Stage 5C showed that the current real canonical dataset has 6,041 synthetic measurement units, missing `Plate_ID` and `Well_ID` for all rows, one ambiguous measurement identity row, and 26 duplicate time-point groups with conflicting raw luminescence values. Therefore, Stage 6B implementation must not average duplicate timestamps. Duplicate timestamps must either block affected time-dependent feature calculations or be represented by explicit low-confidence flags until Stage 5D resolves the builder defect.

## Scientific Assumptions

Luminescence is treated as a proxy for biosensor physiological state, reporter expression, cell viability, metabolic activity, or stress-response activation depending on the strain construct. A single feature does not identify a mechanism by itself. Biological interpretation must come from the pattern across strains, chemicals, concentrations, experiments, and complementary feature groups.

The baseline is the early reference signal for the same measurement series. Until a baseline window is formally parameterized, the default baseline definition is the first valid observation. For noisy or dense early curves, the preferred scientific definition is the median of a pre-specified early baseline window.

Time is measured in minutes using `Time_Minutes`. AUC units are raw luminescence units multiplied by minutes. Slope units are raw luminescence units per minute. Relative features are unitless.

The engine must retain uncertainty. Missing rows, duplicate timestamps, invalid records, warning flags, synthetic identifiers, sparse curves, noisy curves, and unresolved baseline problems must lower confidence rather than being hidden.

## Notation

For one series, let valid observations after row-level QC be:

`S = {(t_1, y_1), ..., (t_n, y_n)}`

where `t_i` is `Time_Minutes`, `y_i` is `Luminescence_Raw`, and observations are sorted by increasing time. If two valid observations have the same `Time_Minutes`, the series contains a duplicate timestamp.

Definitions used below:

- `B`: baseline estimate.
- `N_B`: baseline noise estimate, preferably standard deviation or robust median absolute deviation within the baseline window.
- `P`: maximum raw signal, `max(y_i)`.
- `M`: minimum raw signal, `min(y_i)`.
- `E`: endpoint raw signal, the last valid signal in time order.
- `D`: observed duration, `max(t_i) - min(t_i)`.
- `z_i`: baseline-centered signal, `y_i - B`.
- `r_i`: baseline ratio, `y_i / B`, defined only when `B > 0`.
- `s_i`: adjacent slope, `(y_(i+1) - y_i) / (t_(i+1) - t_i)`, defined only for distinct adjacent times.

All calculations must use finite numeric values. Infinite values are invalid. Negative raw luminescence values are retained but flagged because they may indicate background subtraction, instrument behavior, or source-data problems.

## Edge-Case Policy

Missing data: Features requiring a missing time, signal, or baseline must be returned as null with a low-confidence reason. Features that can be calculated from the remaining valid points may be reported, but the missing-observation features must expose the loss.

Duplicate timestamps: The engine must not average duplicated time points. If duplicate timestamps are unresolved and values differ, time-dependent features such as AUC, slopes, latency, peak width, and recovery time must be null or explicitly low confidence. Count features must report the duplicate timestamp count.

Multiple peaks: Maximum-based features use the earliest occurrence of the global maximum unless a future peak-detection module defines a dominant-peak rule. Duration and recovery features must identify whether they refer to the global peak or a detected dominant peak.

Multiple minima: Minimum-based features use the earliest occurrence of the global minimum unless a future trough-detection module defines a dominant-minimum rule. Inhibition features must preserve the sign of the response.

Noisy curves: Core raw summary features may be computed from unsmoothed data. Peak, latency, recovery, width, and slope extrema are noise-sensitive; they should either use an approved smoothing method or receive lower confidence when signal roughness is high.

No response: A no-response curve is one whose dynamic range and relative response remain within a pre-specified noise tolerance around baseline.

Negative response: A negative response means the signal decreases below baseline or below control-relative expectation. It does not require negative raw luminescence.

Transient response: A transient response rises or falls away from baseline and then returns toward baseline.

Sustained response: A sustained response remains elevated or suppressed through the endpoint.

## Feature Categories

The feature engine will define six groups:

- Signal Intensity
- Temporal Features
- Area Features
- Slope Features
- Relative Response
- Quality Metrics

Core features should be implemented first. Recommended features should follow once baseline and duplicate-time policies are stable. Optional features are useful but not required for the first scientific engine. Experimental features require additional threshold, smoothing, or biological calibration before they should be used as primary model inputs.

## Signal Intensity Features

### Baseline Luminescence

- Feature name: Baseline Luminescence
- Short name: `baseline_luminescence`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Estimates the starting reporter state before measurable exposure-driven dynamics.
- Mathematical definition: `B`, the first valid `y_i` by default; preferably the median of observations within a pre-specified baseline window.
- Units: raw luminescence units, RLU.
- Expected range: finite numeric; usually non-negative for raw assays.
- Interpretation: Larger values suggest stronger starting luminescence, greater biomass, higher constitutive reporter activity, or early induction. Smaller values suggest weaker starting signal, inhibition, low biomass, or background-level emission. No response means later values remain near this baseline. Negative response is measured relative to this baseline. Transient and sustained responses are interpreted by comparing later features to `B`.
- Advantages: Simple, interpretable, and required by relative-response features.
- Limitations: Sensitive to first-point artifacts if only one baseline observation exists.
- Missing data behavior: Null if no valid baseline observation exists.
- Multiple peaks behavior: Not affected except when the first point is part of an unresolved transient peak.
- Multiple minima behavior: Not affected unless the baseline itself is an early inhibitory artifact.
- Noisy curve behavior: Use median baseline when a baseline window exists; lower confidence when baseline noise is high.
- Validation strategy: Constant curves should return the constant value; curves with missing first observations should use the first valid baseline rule or return null if no baseline exists.
- Dependencies: time ordering, baseline estimation, QC.

### Baseline Variability

- Feature name: Baseline Variability
- Short name: `baseline_variability`
- Category: Signal Intensity
- Priority: Recommended
- Scientific purpose: Quantifies early measurement instability before interpreting treatment-induced changes.
- Mathematical definition: Standard deviation or robust spread of `y_i` within the baseline window.
- Units: RLU.
- Expected range: `>= 0`.
- Interpretation: Larger values indicate unstable starting signal, growth variability, instrument fluctuation, or early response onset. Smaller values indicate a stable reference state. No response is easier to trust when baseline variability is low. Negative, transient, and sustained responses all become less certain when baseline variability is high.
- Advantages: Directly informs response thresholds and confidence.
- Limitations: Requires at least two baseline-window observations.
- Missing data behavior: Null when fewer than two baseline observations are available.
- Multiple peaks behavior: Early peaks inflate this feature and should lower confidence for latency and fold-change features.
- Multiple minima behavior: Early troughs inflate this feature and may signal early inhibition.
- Noisy curve behavior: Robust spread should be preferred for noisy curves.
- Validation strategy: Constant baseline should give zero; synthetic noisy baseline should recover the injected noise scale.
- Dependencies: time ordering, baseline estimation, QC.

### Maximum Luminescence

- Feature name: Maximum Luminescence
- Short name: `max_luminescence`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Captures the highest observed reporter output in the series.
- Mathematical definition: `P = max(y_i)`.
- Units: RLU.
- Expected range: finite numeric; usually `>= 0` in raw assays.
- Interpretation: Larger values suggest strong induction, growth-supported signal, or high reporter activation. Smaller values suggest weak induction or suppressed luminescence. No response has `P` close to `B`. Negative response may still have a low maximum if the full curve is suppressed. Transient response has high `P` with later decline; sustained response has high `P` near endpoint.
- Advantages: Easy to interpret and robust to uneven time spacing.
- Limitations: Sensitive to single-point spikes and longer observation windows.
- Missing data behavior: Null if no valid signal exists; otherwise calculated from valid observations with lowered confidence if missingness is substantial.
- Multiple peaks behavior: Uses the global maximum; earliest time is used for time-to-peak tie handling.
- Multiple minima behavior: Independent of minima except dynamic range.
- Noisy curve behavior: Spike-sensitive; confidence lowered when isolated peak prominence is not supported by adjacent points.
- Validation strategy: Synthetic curves with known peaks must return the known maximum.
- Dependencies: QC.

### Minimum Luminescence

- Feature name: Minimum Luminescence
- Short name: `min_luminescence`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Captures the strongest observed signal suppression or trough.
- Mathematical definition: `M = min(y_i)`.
- Units: RLU.
- Expected range: finite numeric; raw values may be negative but must be flagged.
- Interpretation: Larger minima imply the curve never strongly suppressed luminescence. Smaller minima suggest inhibition, cell stress, reporter repression, or background-level signal. No response has `M` close to `B`. Negative response is reflected by `M < B`. Transient inhibition has a low `M` followed by recovery; sustained inhibition has low endpoint and low `M`.
- Advantages: Direct marker of inhibitory responses.
- Limitations: Sensitive to one-point downward artifacts.
- Missing data behavior: Null if no valid signal exists; confidence lowered when missingness could hide a trough.
- Multiple peaks behavior: Independent of peaks except dynamic range.
- Multiple minima behavior: Uses global minimum and earliest tie.
- Noisy curve behavior: Downward spikes can dominate; confidence lowered when not supported by neighboring points.
- Validation strategy: Synthetic inhibitory curves must return the known trough.
- Dependencies: QC.

### Endpoint Luminescence

- Feature name: Endpoint Luminescence
- Short name: `endpoint_luminescence`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Describes the terminal response state at the end of the observed assay.
- Mathematical definition: `E = y_i` at the largest valid `t_i`.
- Units: RLU.
- Expected range: finite numeric; usually non-negative.
- Interpretation: Larger endpoint values suggest sustained induction or strong late growth-linked signal. Smaller endpoint values suggest sustained inhibition, signal decay, or lack of recovery. No response has `E` close to `B`. Negative response has `E < B`. Transient response has `E` closer to baseline than the peak. Sustained response has `E` close to the maximum or minimum direction of change.
- Advantages: Simple marker of persistent outcome.
- Limitations: Strongly depends on assay duration and missing late time points.
- Missing data behavior: Null if no valid final time exists; confidence lowered if late observations are missing.
- Multiple peaks behavior: Distinguishes transient peaks from sustained peaks.
- Multiple minima behavior: Distinguishes recovered troughs from sustained inhibition.
- Noisy curve behavior: Last-point noise can dominate; future implementation may allow terminal-window median as a companion feature.
- Validation strategy: Synthetic curves with known final value must return that value.
- Dependencies: time ordering, QC.

### Dynamic Range

- Feature name: Dynamic Range
- Short name: `dynamic_range`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Quantifies total response amplitude within the observed series.
- Mathematical definition: `P - M`.
- Units: RLU.
- Expected range: `>= 0`.
- Interpretation: Larger values indicate stronger change over time, either induction, inhibition and recovery, or oscillatory response. Smaller values indicate stable or no-response behavior. Negative response contributes through lower minima. Transient responses often have large dynamic range. Sustained responses may have moderate or large range depending on baseline.
- Advantages: Direction-neutral measure of response strength.
- Limitations: Does not distinguish induction from inhibition and is sensitive to outliers.
- Missing data behavior: Null if fewer than one valid signal exists; confidence lowered when missing points could hide extrema.
- Multiple peaks behavior: Captures only global span, not number of peaks.
- Multiple minima behavior: Captures only global span, not number of troughs.
- Noisy curve behavior: Spike-sensitive; should be interpreted with standard deviation and SNR.
- Validation strategy: Known max/min synthetic curves should return exact difference.
- Dependencies: QC.

### Mean Luminescence

- Feature name: Mean Luminescence
- Short name: `mean_luminescence`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Estimates average reporter output over sampled observations.
- Mathematical definition: Arithmetic mean of valid `y_i`.
- Units: RLU.
- Expected range: finite numeric.
- Interpretation: Larger values suggest generally elevated luminescence across the series. Smaller values suggest broad suppression. No response has mean near baseline. Negative response lowers the mean relative to baseline. Transient responses affect the mean less than sustained responses with similar extrema.
- Advantages: Stable summary of central signal level.
- Limitations: Depends on sampling density and can overweight densely sampled periods.
- Missing data behavior: Calculated from valid observations when at least one exists; confidence lowered when missingness is non-random.
- Multiple peaks behavior: Multiple peaks raise the mean if they occupy substantial duration.
- Multiple minima behavior: Multiple troughs lower the mean if sustained.
- Noisy curve behavior: More robust than extrema but still affected by extreme outliers.
- Validation strategy: Constant curves should return the constant value; mixed curves should match manual mean.
- Dependencies: QC.

### Median Luminescence

- Feature name: Median Luminescence
- Short name: `median_luminescence`
- Category: Signal Intensity
- Priority: Recommended
- Scientific purpose: Provides a robust central tendency for skewed or spike-prone curves.
- Mathematical definition: Median of valid `y_i`.
- Units: RLU.
- Expected range: finite numeric.
- Interpretation: Larger values suggest the typical state is elevated. Smaller values suggest typical suppression. No response has median near baseline. Negative transient events influence the median only when frequent or sustained. Transient responses affect the median less than sustained responses.
- Advantages: Less sensitive to isolated spikes than the mean.
- Limitations: Ignores time spacing and duration.
- Missing data behavior: Null if no valid signal exists; confidence lowered when missingness clusters in one phase.
- Multiple peaks behavior: Robust to isolated peaks but underrepresents brief strong induction.
- Multiple minima behavior: Robust to isolated troughs but underrepresents brief strong inhibition.
- Noisy curve behavior: Good companion to mean and extrema.
- Validation strategy: Synthetic odd/even-length curves should match standard median definitions.
- Dependencies: QC.

### Signal Standard Deviation

- Feature name: Signal Standard Deviation
- Short name: `signal_sd`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Quantifies total variability of luminescence within a series.
- Mathematical definition: Sample or population standard deviation of valid `y_i`; implementation must document degrees of freedom.
- Units: RLU.
- Expected range: `>= 0`.
- Interpretation: Larger values suggest dynamic response, noise, or heterogeneous measurement phases. Smaller values suggest stable signal. No response should have low standard deviation relative to baseline noise. Transient responses often raise this feature. Sustained plateaus after an early shift may produce moderate values.
- Advantages: Useful for separating flat curves from dynamic curves.
- Limitations: Does not distinguish biological response from measurement noise.
- Missing data behavior: Null when fewer than two valid observations exist.
- Multiple peaks behavior: Increases with repeated excursions from the mean.
- Multiple minima behavior: Increases with repeated suppressions.
- Noisy curve behavior: High values can indicate either biological dynamics or noise; interpret with SNR.
- Validation strategy: Constant curve should return zero; synthetic noisy curve should match known spread.
- Dependencies: QC.

### Signal Coefficient of Variation

- Feature name: Signal Coefficient of Variation
- Short name: `signal_cv`
- Category: Signal Intensity
- Priority: Core
- Scientific purpose: Scales within-series variability by average signal magnitude.
- Mathematical definition: `signal_sd / abs(mean_luminescence)` when mean is non-zero.
- Units: unitless.
- Expected range: `>= 0`; null when mean is zero or too close to zero.
- Interpretation: Larger values indicate high relative variability, strong dynamics, or noisy low-signal behavior. Smaller values indicate stable signal relative to magnitude. No response has low CV if baseline is stable. Negative responses can increase CV when signal crosses baseline or approaches zero. Transient responses usually raise CV more than sustained plateaus.
- Advantages: Enables comparison across strains with different absolute brightness.
- Limitations: Unstable when mean signal is near zero.
- Missing data behavior: Null when fewer than two valid observations exist or mean denominator is invalid.
- Multiple peaks behavior: Increases with repeated peaks.
- Multiple minima behavior: Increases with repeated troughs.
- Noisy curve behavior: Can be inflated by low denominators and instrument noise.
- Validation strategy: Scaled versions of the same curve should have similar CV.
- Dependencies: QC.

## Temporal Features

### Time to Peak

- Feature name: Time to Peak
- Short name: `time_to_peak`
- Category: Temporal Features
- Priority: Core
- Scientific purpose: Identifies when maximum induction or maximum observed luminescence occurs.
- Mathematical definition: Earliest `t_i` where `y_i = P`.
- Units: minutes.
- Expected range: from first valid time to last valid time.
- Interpretation: Larger values indicate delayed peak response, slow induction, or late growth-supported signal. Smaller values indicate rapid activation or an early artifact. No response makes time to peak biologically weak and confidence should depend on dynamic range. Negative response can still have an early weak peak before suppression. Transient response has a peak before endpoint; sustained response often peaks at or near endpoint.
- Advantages: Direct kinetic descriptor.
- Limitations: Sensitive to noise and assay duration.
- Missing data behavior: Null if no valid signal; confidence lowered when missing points occur near expected peak.
- Multiple peaks behavior: Uses earliest global maximum; future peak-detection may choose a dominant peak by prominence.
- Multiple minima behavior: Not directly affected.
- Noisy curve behavior: Isolated noisy maxima lower confidence.
- Validation strategy: Synthetic curves with known peak time and tied peaks should return the expected earliest time.
- Dependencies: time ordering, peak detection, QC.

### Time to Minimum

- Feature name: Time to Minimum
- Short name: `time_to_minimum`
- Category: Temporal Features
- Priority: Recommended
- Scientific purpose: Identifies when strongest observed inhibition or trough occurs.
- Mathematical definition: Earliest `t_i` where `y_i = M`.
- Units: minutes.
- Expected range: from first valid time to last valid time.
- Interpretation: Larger values indicate delayed suppression. Smaller values indicate immediate inhibition or early low signal. No response makes the value weakly meaningful. Negative response is biologically reflected when the minimum is below baseline. Transient inhibition has minimum before recovery; sustained inhibition often has minimum near endpoint.
- Advantages: Captures inhibitory kinetics complementary to time to peak.
- Limitations: Sensitive to downward spikes.
- Missing data behavior: Null if no valid signal; confidence lowered when missing points could hide a trough.
- Multiple peaks behavior: Not directly affected.
- Multiple minima behavior: Uses earliest global minimum.
- Noisy curve behavior: Isolated troughs lower confidence.
- Validation strategy: Synthetic inhibitory curves and tied minima should return the known earliest minimum time.
- Dependencies: time ordering, trough detection, QC.

### Response Latency

- Feature name: Response Latency
- Short name: `response_latency`
- Category: Temporal Features
- Priority: Experimental
- Scientific purpose: Estimates the time before a biologically meaningful departure from baseline begins.
- Mathematical definition: First `t_i` at which the smoothed signal exceeds an approved threshold such as `B +/- k * N_B` and persists for an approved duration or point count.
- Units: minutes.
- Expected range: non-negative within observed duration; null when no threshold crossing occurs.
- Interpretation: Larger values indicate delayed sensing, slow transcriptional response, or delayed toxicity. Smaller values indicate rapid recognition or immediate stress. No response returns null. Negative response latency uses the inhibitory threshold below baseline. Transient and sustained responses both have latency, but sustained responses should maintain direction longer.
- Advantages: Biologically close to response onset.
- Limitations: Requires threshold, baseline-noise, persistence, and smoothing choices.
- Missing data behavior: Null or low confidence when early points are missing.
- Multiple peaks behavior: Uses first qualified response, not necessarily global peak.
- Multiple minima behavior: Uses first qualified inhibitory departure when inhibition is the dominant response.
- Noisy curve behavior: Highly sensitive; smoothing and persistence rules are required.
- Validation strategy: Synthetic step-response curves should return known onset; noisy no-response curves should avoid false latency calls.
- Dependencies: time ordering, baseline estimation, peak/trough detection, smoothing, QC.

### Recovery Time

- Feature name: Recovery Time
- Short name: `recovery_time`
- Category: Temporal Features
- Priority: Experimental
- Scientific purpose: Estimates when a disturbed biosensor response returns close to baseline.
- Mathematical definition: First `t_i` after the dominant peak or trough where signal returns within an approved tolerance of `B` and remains there for an approved persistence rule.
- Units: minutes.
- Expected range: from response extremum to last valid time; null when recovery is not observed.
- Interpretation: Larger values indicate slow recovery, persistent stress, or sustained reporter activation. Smaller values indicate rapid adaptation or transient response. No response has no recovery event. Negative response recovery means suppressed signal rises back toward baseline. Sustained response yields null or censored recovery time.
- Advantages: Directly separates transient from sustained behavior.
- Limitations: Requires baseline tolerance, dominant-extremum selection, and enough late observations.
- Missing data behavior: Null or low confidence when late observations are missing.
- Multiple peaks behavior: Recovery must specify whether it follows the dominant peak or final qualifying peak.
- Multiple minima behavior: Inhibitory recovery must specify whether it follows the dominant trough.
- Noisy curve behavior: Requires smoothing or persistence to avoid false recovery.
- Validation strategy: Synthetic transient curves should recover at known time; sustained curves should return null.
- Dependencies: time ordering, baseline estimation, peak/trough detection, smoothing, QC.

### Peak Width

- Feature name: Peak Width
- Short name: `peak_width`
- Category: Temporal Features
- Priority: Experimental
- Scientific purpose: Measures the temporal breadth of a dominant induction event.
- Mathematical definition: Time between rising and falling threshold crossings around the dominant peak, typically at half peak prominence above baseline.
- Units: minutes.
- Expected range: `>= 0`; null when no dominant peak exists.
- Interpretation: Larger values indicate prolonged activation. Smaller values indicate sharp transient activation. No response returns null. Negative response requires a separate trough-width interpretation and should not be forced into this feature without a defined inhibitory analogue. Sustained response may have censored or large width if falling crossing is absent.
- Advantages: Captures response shape beyond maximum height.
- Limitations: Requires peak prominence and crossing interpolation rules.
- Missing data behavior: Null or low confidence when crossing points are missing.
- Multiple peaks behavior: Uses dominant peak by approved prominence rule.
- Multiple minima behavior: Not directly applicable to inhibitory troughs unless an explicit trough-width extension is defined.
- Noisy curve behavior: Very noise-sensitive without smoothing.
- Validation strategy: Triangular synthetic peaks should return known width; multi-peak curves should select the dominant peak reproducibly.
- Dependencies: time ordering, baseline estimation, peak detection, smoothing, QC.

### Half-Maximum Duration

- Feature name: Half-Maximum Duration
- Short name: `half_max_duration`
- Category: Temporal Features
- Priority: Recommended
- Scientific purpose: Measures how long the signal remains meaningfully elevated relative to its response amplitude.
- Mathematical definition: Total time for which `y(t) >= B + 0.5 * (P - B)` using interpolation between observed points for positive responses.
- Units: minutes.
- Expected range: `0` to observed duration.
- Interpretation: Larger values indicate sustained activation. Smaller values indicate brief transient activation or no response. No response should yield zero or null depending on the minimum amplitude threshold. Negative responses should be assessed by negative-area and minimum features unless an inhibitory half-duration is separately specified.
- Advantages: Distinguishes brief spikes from prolonged induction.
- Limitations: Depends on baseline, peak amplitude, interpolation, and duration.
- Missing data behavior: Null or low confidence when gaps prevent reliable threshold crossing.
- Multiple peaks behavior: Can sum durations across all above-threshold regions, but implementation must document whether total or dominant-region duration is reported.
- Multiple minima behavior: Not directly affected for positive-response definition.
- Noisy curve behavior: Smoothing or persistence reduces false crossings.
- Validation strategy: Square pulse and triangular pulse curves should return known duration.
- Dependencies: time ordering, baseline estimation, peak detection, smoothing, QC.

## Area Features

### Total Area Under Curve

- Feature name: Total Area Under Curve
- Short name: `auc_total`
- Category: Area Features
- Priority: Core
- Scientific purpose: Captures cumulative luminescence exposure over the observed time course.
- Mathematical definition: Trapezoidal integral of raw `y_i` over sorted `t_i`.
- Units: RLU * minutes.
- Expected range: finite numeric; can be negative only if raw signal includes negative values.
- Interpretation: Larger values indicate greater cumulative reporter output or sustained high signal. Smaller values indicate lower cumulative activity or suppression. No response produces approximately baseline times duration. Negative response reduces AUC relative to baseline. Transient response produces less AUC than sustained response with the same peak.
- Advantages: Integrates magnitude and duration.
- Limitations: Strongly duration-dependent and unsafe with unresolved duplicate timestamps.
- Missing data behavior: Null when fewer than two distinct time points exist; confidence lowered for gaps.
- Multiple peaks behavior: Integrates all peaks without distinguishing their count.
- Multiple minima behavior: Integrates all troughs through lower area.
- Noisy curve behavior: Moderately robust, but high-frequency noise can add area.
- Validation strategy: Constant, linear, and triangular curves should match analytic trapezoid expectations.
- Dependencies: time ordering, QC.

### Positive Area

- Feature name: Positive Area
- Short name: `auc_positive`
- Category: Area Features
- Priority: Recommended
- Scientific purpose: Quantifies cumulative induction above baseline.
- Mathematical definition: Trapezoidal integral of `max(y_i - B, 0)` over time.
- Units: RLU * minutes.
- Expected range: `>= 0`.
- Interpretation: Larger values indicate stronger or longer induction. Smaller values indicate no induction or dominant inhibition. No response has near-zero positive area. Negative response has low positive area unless it also includes rebound. Transient induction gives positive area limited by peak width; sustained induction gives larger positive area.
- Advantages: Separates induction from baseline and inhibition.
- Limitations: Depends on baseline and interpolation around crossings.
- Missing data behavior: Null when baseline is unavailable or fewer than two distinct time points exist.
- Multiple peaks behavior: Sums induction across all above-baseline regions.
- Multiple minima behavior: Minima do not contribute except by separating positive regions.
- Noisy curve behavior: Baseline noise can create false small positive area; thresholding may be needed.
- Validation strategy: Baseline-only curves should return zero; synthetic pulses should match known positive area.
- Dependencies: time ordering, baseline estimation, QC.

### Negative Area

- Feature name: Negative Area
- Short name: `auc_negative`
- Category: Area Features
- Priority: Recommended
- Scientific purpose: Quantifies cumulative suppression below baseline.
- Mathematical definition: Trapezoidal integral of `min(y_i - B, 0)` over time, reported as a signed value less than or equal to zero.
- Units: RLU * minutes.
- Expected range: `<= 0`; zero indicates no below-baseline area.
- Interpretation: More negative values indicate stronger or longer inhibition. Values near zero indicate no inhibition. No response has near-zero negative area. Transient inhibition gives a finite negative pulse; sustained inhibition gives larger magnitude negative area.
- Advantages: Preserves response direction rather than using absolute magnitude.
- Limitations: Depends on baseline and can be affected by baseline drift.
- Missing data behavior: Null when baseline is unavailable or fewer than two distinct time points exist.
- Multiple peaks behavior: Positive peaks do not contribute except by interrupting suppressed periods.
- Multiple minima behavior: Sums all below-baseline suppressions.
- Noisy curve behavior: Baseline noise can create small false negative area.
- Validation strategy: Synthetic below-baseline pulses should match known signed area.
- Dependencies: time ordering, baseline estimation, QC.

### Normalised Area

- Feature name: Normalised Area
- Short name: `auc_normalised`
- Category: Area Features
- Priority: Recommended
- Scientific purpose: Makes cumulative response more comparable across assay duration or baseline brightness.
- Mathematical definition: Preferred definition is `auc_total / (D * B)` when `D > 0` and `B > 0`, or the trapezoidal integral of an approved normalized signal divided by duration.
- Units: unitless.
- Expected range: non-negative for positive raw signals; interpretation depends on normalization method.
- Interpretation: Larger values indicate higher average response relative to baseline. Smaller values indicate lower relative response or inhibition. No response should be near one for baseline-ratio normalization or near zero for baseline-centered normalization. Negative, transient, and sustained interpretations depend on the chosen normalization.
- Advantages: Reduces comparability problems between bright and dim strains or different durations.
- Limitations: Requires a formally approved normalization method and valid baseline.
- Missing data behavior: Null when baseline, duration, or normalized signal is unavailable.
- Multiple peaks behavior: Integrates all normalized excursions.
- Multiple minima behavior: Integrates all normalized suppressions.
- Noisy curve behavior: Baseline noise can distort denominator-sensitive normalizations.
- Validation strategy: Scaled curves should yield similar normalized area under ratio normalization.
- Dependencies: time ordering, normalization, baseline estimation, QC.

## Slope Features

### Initial Slope

- Feature name: Initial Slope
- Short name: `initial_slope`
- Category: Slope Features
- Priority: Core
- Scientific purpose: Captures the earliest rate of change after the assay begins.
- Mathematical definition: `(y_2 - y_1) / (t_2 - t_1)` using the first two distinct valid time points.
- Units: RLU per minute.
- Expected range: finite numeric, positive, zero, or negative.
- Interpretation: Larger positive values indicate rapid induction. Values near zero indicate stable early signal. Negative values indicate early suppression. No response has slope near zero. Transient responses may have large initial slope followed by reversal. Sustained responses may maintain positive or negative direction beyond the initial slope.
- Advantages: Simple early kinetic descriptor.
- Limitations: Extremely sensitive to first two points and duplicate timestamps.
- Missing data behavior: Null if fewer than two distinct valid time points exist.
- Multiple peaks behavior: Does not capture later peaks.
- Multiple minima behavior: Does not capture later troughs.
- Noisy curve behavior: Early noise can dominate; confidence lowered when baseline variability is high.
- Validation strategy: Linear synthetic curves should return the known slope.
- Dependencies: time ordering, QC.

### Maximum Slope

- Feature name: Maximum Slope
- Short name: `max_slope`
- Category: Slope Features
- Priority: Core
- Scientific purpose: Captures the fastest observed induction rate.
- Mathematical definition: `max(s_i)` over all adjacent distinct time intervals.
- Units: RLU per minute.
- Expected range: finite numeric.
- Interpretation: Larger positive values indicate rapid activation or abrupt growth-linked increase. Smaller or negative values indicate absence of fast induction. No response has maximum slope near zero. Negative response may still have a small or late positive rebound. Transient responses often have high maximum slope before decay.
- Advantages: Sensitive to kinetic onset and sharp transitions.
- Limitations: Sensitive to noise, uneven time spacing, and duplicate timestamps.
- Missing data behavior: Null when fewer than two distinct valid time points exist.
- Multiple peaks behavior: Selects the steepest rise among all peaks.
- Multiple minima behavior: Not directly inhibitory except recovery from a trough can create high positive slope.
- Noisy curve behavior: High-frequency noise can inflate this feature; smoothing may be needed.
- Validation strategy: Piecewise-linear curves should return known steepest segment.
- Dependencies: time ordering, smoothing, QC.

### Minimum Slope

- Feature name: Minimum Slope
- Short name: `min_slope`
- Category: Slope Features
- Priority: Core
- Scientific purpose: Captures the fastest observed decline or inhibitory transition.
- Mathematical definition: `min(s_i)` over all adjacent distinct time intervals.
- Units: RLU per minute.
- Expected range: finite numeric.
- Interpretation: More negative values indicate rapid suppression, decay, toxicity, or reporter shutoff. Values near zero indicate stable signal. No response has minimum slope near zero. Transient induction followed by decline has negative minimum slope. Sustained induction may have minimum slope near zero after plateau.
- Advantages: Detects rapid inhibitory or decay phases.
- Limitations: Noise and duplicate timestamps can create extreme artifacts.
- Missing data behavior: Null when fewer than two distinct valid time points exist.
- Multiple peaks behavior: Declines after peaks can determine this feature.
- Multiple minima behavior: Fast descent into any trough can determine this feature.
- Noisy curve behavior: Downward noise spikes can dominate; confidence lowered without smoothing.
- Validation strategy: Piecewise-linear curves should return known steepest negative segment.
- Dependencies: time ordering, smoothing, QC.

### Average Slope

- Feature name: Average Slope
- Short name: `average_slope`
- Category: Slope Features
- Priority: Recommended
- Scientific purpose: Summarizes net change rate across the full observed duration.
- Mathematical definition: `(E - y_first) / D` when `D > 0`.
- Units: RLU per minute.
- Expected range: finite numeric.
- Interpretation: Positive values indicate net induction. Negative values indicate net suppression or decay. Values near zero indicate no net change or balanced transient behavior. Transient responses can have average slope near zero despite large extrema. Sustained responses usually keep the sign of their direction.
- Advantages: Stable and interpretable for sustained trends.
- Limitations: Masks transient responses and is duration-dependent.
- Missing data behavior: Null when fewer than two distinct time points or invalid duration.
- Multiple peaks behavior: Ignores internal peaks.
- Multiple minima behavior: Ignores internal troughs.
- Noisy curve behavior: Less noise-sensitive than adjacent slope extrema but endpoint-sensitive.
- Validation strategy: Linear curves should match true slope; transient return-to-baseline curves should approximate zero.
- Dependencies: time ordering, QC.

### Recovery Slope

- Feature name: Recovery Slope
- Short name: `recovery_slope`
- Category: Slope Features
- Priority: Recommended
- Scientific purpose: Measures the rate at which signal moves from an extremum back toward baseline or endpoint.
- Mathematical definition: For positive responses, `(E - P) / (t_endpoint - t_peak)` when peak precedes endpoint. For inhibitory responses, `(E - M) / (t_endpoint - t_minimum)`.
- Units: RLU per minute.
- Expected range: finite numeric; positive or negative depending on response direction.
- Interpretation: More negative values after a peak indicate faster decay from induction. More positive values after a trough indicate recovery from inhibition. Values near zero indicate sustained plateau or no recovery. No response gives near-zero or null. Transient responses show stronger recovery slopes than sustained responses.
- Advantages: Links transient dynamics to final state.
- Limitations: Requires selecting whether induction or inhibition is dominant.
- Missing data behavior: Null if the dominant extremum is at endpoint or late data are missing.
- Multiple peaks behavior: Must use approved dominant peak or global peak.
- Multiple minima behavior: Must use approved dominant trough for inhibitory recovery.
- Noisy curve behavior: Extremum noise can distort slope; smoothing recommended.
- Validation strategy: Synthetic peak-decay and trough-recovery curves should return known slopes.
- Dependencies: time ordering, baseline estimation, peak/trough detection, smoothing, QC.

## Relative Response Features

### Maximum Fold Change

- Feature name: Maximum Fold Change
- Short name: `max_fold_change`
- Category: Relative Response
- Priority: Core
- Scientific purpose: Quantifies strongest proportional induction relative to baseline.
- Mathematical definition: `(P - B) / B` or `P / B`, depending on final project convention; the implementation must choose and document one convention. This specification prefers `(P - B) / B` for signed fold change.
- Units: unitless.
- Expected range: `>= -1` when `B > 0` and signals are non-negative; can exceed zero without upper bound.
- Interpretation: Larger values indicate stronger induction relative to starting state. Values near zero indicate no induction. Negative values indicate peak below baseline, usually sustained suppression or baseline artifact. Transient induction can have high maximum fold change with low endpoint fold change. Sustained induction has high endpoint/baseline as well.
- Advantages: Compares responses across strains with different absolute brightness.
- Limitations: Undefined or unstable when baseline is zero or near zero.
- Missing data behavior: Null when baseline is unavailable or invalid.
- Multiple peaks behavior: Uses global maximum.
- Multiple minima behavior: Not directly inhibitory except if all values are below baseline.
- Noisy curve behavior: Spike-sensitive and denominator-sensitive.
- Validation strategy: Scaled curves with same proportional peak should produce the same fold change.
- Dependencies: baseline estimation, QC.

### Minimum Fold Change

- Feature name: Minimum Fold Change
- Short name: `min_fold_change`
- Category: Relative Response
- Priority: Recommended
- Scientific purpose: Quantifies strongest proportional suppression relative to baseline.
- Mathematical definition: `(M - B) / B` when `B > 0`.
- Units: unitless.
- Expected range: usually `>= -1` for non-negative signals, with zero meaning no suppression.
- Interpretation: More negative values indicate stronger inhibition or reporter suppression. Values near zero indicate no below-baseline response. Positive values mean even the minimum exceeded baseline. Transient inhibition has a negative minimum fold change followed by recovery. Sustained inhibition has negative endpoint/baseline as well.
- Advantages: Directional measure of inhibitory response.
- Limitations: Undefined or unstable when baseline is zero or near zero.
- Missing data behavior: Null when baseline is unavailable or invalid.
- Multiple peaks behavior: Not directly affected.
- Multiple minima behavior: Uses global minimum.
- Noisy curve behavior: Downward artifacts and low baseline distort the value.
- Validation strategy: Scaled inhibitory curves should retain proportional suppression values.
- Dependencies: baseline estimation, QC.

### Log2 Fold Change

- Feature name: Log2 Fold Change
- Short name: `log2_fold_change`
- Category: Relative Response
- Priority: Recommended
- Scientific purpose: Provides symmetric scale for proportional endpoint or peak response.
- Mathematical definition: Preferred endpoint definition is `log2(E / B)` when `E > 0` and `B > 0`; a peak variant may be documented separately.
- Units: log2 unitless ratio.
- Expected range: finite real number; zero means endpoint equals baseline.
- Interpretation: Positive values indicate induction. Negative values indicate suppression. Values near zero indicate no response. Transient responses with recovered endpoints may have near-zero endpoint log2 fold change despite high maximum fold change. Sustained responses keep large positive or negative endpoint log2 fold change.
- Advantages: Compresses large ratios and treats doubling/halving symmetrically.
- Limitations: Undefined for zero or negative denominator or numerator without an approved offset.
- Missing data behavior: Null when baseline or endpoint is missing, zero, or negative.
- Multiple peaks behavior: Endpoint version ignores earlier peaks.
- Multiple minima behavior: Endpoint version ignores earlier troughs unless sustained.
- Noisy curve behavior: Endpoint and baseline noise can distort ratios.
- Validation strategy: Twofold and half-fold synthetic endpoint curves should return `1` and `-1`.
- Dependencies: time ordering, baseline estimation, QC.

### Normalised Endpoint

- Feature name: Normalised Endpoint
- Short name: `normalised_endpoint`
- Category: Relative Response
- Priority: Optional
- Scientific purpose: Reports terminal state after an approved normalization method.
- Mathematical definition: Last valid value of `Luminescence_Normalized`, or endpoint raw signal transformed by an approved normalization method.
- Units: unitless or method-specific.
- Expected range: method-dependent.
- Interpretation: Larger values indicate stronger normalized terminal activation. Smaller values indicate normalized suppression. No response depends on method, typically near one for ratio normalization or zero for centered normalization. Transient responses may return toward no-response value; sustained responses remain displaced.
- Advantages: More comparable across experiments if normalization is valid.
- Limitations: Cannot be computed until normalization is specified and audited.
- Missing data behavior: Null when normalized signal or endpoint is unavailable.
- Multiple peaks behavior: Ignores earlier peaks unless they persist to endpoint.
- Multiple minima behavior: Ignores earlier minima unless sustained.
- Noisy curve behavior: Endpoint noise remains a limitation after normalization.
- Validation strategy: Known normalized curves should return known final normalized value.
- Dependencies: time ordering, normalization, baseline estimation, QC.

### Peak/Baseline Ratio

- Feature name: Peak/Baseline Ratio
- Short name: `peak_baseline_ratio`
- Category: Relative Response
- Priority: Optional
- Scientific purpose: Expresses peak signal as a direct ratio to baseline.
- Mathematical definition: `P / B` when `B > 0`.
- Units: unitless.
- Expected range: usually `>= 0`; one means peak equals baseline.
- Interpretation: Larger values indicate stronger peak induction. Values near one indicate no induction. Values below one indicate all observed signals below baseline. Transient responses can have high ratio despite recovery. Sustained responses usually also have high endpoint/baseline ratio.
- Advantages: Intuitive fold-ratio measure.
- Limitations: Less directionally explicit than signed fold change and unstable with low baseline.
- Missing data behavior: Null when baseline is invalid.
- Multiple peaks behavior: Uses global peak.
- Multiple minima behavior: Not directly inhibitory.
- Noisy curve behavior: Single-point spikes can inflate the ratio.
- Validation strategy: Scaled curves with same peak ratio should match.
- Dependencies: baseline estimation, peak detection, QC.

### Endpoint/Baseline Ratio

- Feature name: Endpoint/Baseline Ratio
- Short name: `endpoint_baseline_ratio`
- Category: Relative Response
- Priority: Core
- Scientific purpose: Expresses sustained terminal response relative to starting state.
- Mathematical definition: `E / B` when `B > 0`.
- Units: unitless.
- Expected range: usually `>= 0`; one means endpoint equals baseline.
- Interpretation: Larger values indicate sustained induction. Values near one indicate no net response or recovered transient response. Values below one indicate sustained suppression. Transient responses have endpoint ratio closer to one than peak ratio. Sustained responses remain far from one.
- Advantages: Simple sustained-response marker.
- Limitations: Undefined for zero baseline and sensitive to endpoint artifacts.
- Missing data behavior: Null when baseline or endpoint is unavailable or invalid.
- Multiple peaks behavior: Earlier peaks do not affect value unless endpoint remains elevated.
- Multiple minima behavior: Earlier troughs do not affect value unless endpoint remains suppressed.
- Noisy curve behavior: Last-point noise and low baseline reduce confidence.
- Validation strategy: Endpoint twice baseline should return two; endpoint half baseline should return 0.5.
- Dependencies: time ordering, baseline estimation, QC.

## Quality Metrics

### Signal-to-Noise Ratio

- Feature name: Signal-to-Noise Ratio
- Short name: `signal_to_noise_ratio`
- Category: Quality Metrics
- Priority: Optional
- Scientific purpose: Compares response amplitude to baseline noise.
- Mathematical definition: `dynamic_range / N_B`, or response amplitude above baseline divided by baseline noise, when `N_B > 0`.
- Units: unitless.
- Expected range: `>= 0`; null when noise denominator is unavailable or zero.
- Interpretation: Larger values indicate a response more clearly separated from noise. Smaller values indicate ambiguous or flat/noisy curves. No response has low SNR. Negative responses can have high SNR if suppression clearly exceeds noise. Transient and sustained responses both gain confidence when SNR is high.
- Advantages: Directly supports confidence scoring.
- Limitations: Requires a reliable baseline-noise estimate.
- Missing data behavior: Null when baseline window has insufficient points.
- Multiple peaks behavior: Multiple excursions can raise dynamic range but not necessarily SNR reliability.
- Multiple minima behavior: Strong troughs can raise dynamic range.
- Noisy curve behavior: Designed for noisy-curve interpretation but denominator choice matters.
- Validation strategy: Synthetic curves with known noise and amplitude should recover expected SNR.
- Dependencies: baseline estimation, QC.

### Valid Observation Count

- Feature name: Number of Valid Observations
- Short name: `valid_observation_count`
- Category: Quality Metrics
- Priority: Core
- Scientific purpose: Records how much usable evidence supports the feature vector.
- Mathematical definition: Count of rows in the series with valid finite time and raw luminescence and `Record_Valid` not false.
- Units: count.
- Expected range: integer `>= 0`.
- Interpretation: Larger values support richer kinetic characterization. Smaller values indicate sparse evidence. No, negative, transient, and sustained response interpretations all weaken when valid observations are few.
- Advantages: Transparent measure of feature support.
- Limitations: Count alone does not measure time coverage or spacing.
- Missing data behavior: Always computable from the series; zero means no usable signal features.
- Multiple peaks behavior: More observations improve peak resolution.
- Multiple minima behavior: More observations improve trough resolution.
- Noisy curve behavior: More observations can help distinguish noise from real dynamics.
- Validation strategy: Fixtures with known valid and invalid rows should return exact counts.
- Dependencies: QC.

### Missing Observation Count

- Feature name: Missing Observations
- Short name: `missing_observation_count`
- Category: Quality Metrics
- Priority: Core
- Scientific purpose: Records lost or unusable observations within the series.
- Mathematical definition: Count of rows with missing, non-finite, or invalid `Time_Minutes` or `Luminescence_Raw`, plus rows marked invalid if the implementation treats them as excluded from feature mathematics.
- Units: count.
- Expected range: integer `>= 0`.
- Interpretation: Larger values lower confidence and may bias every biological interpretation. Smaller values indicate more complete evidence. No-response calls are especially fragile with missing observations. Transient or brief negative responses may be missed if observations are missing.
- Advantages: Keeps missingness visible rather than silently dropping rows.
- Limitations: Requires an expected observation definition if absent time points are to be counted beyond present rows.
- Missing data behavior: This feature exists to report missingness and should always be emitted.
- Multiple peaks behavior: Missing observations can hide peaks.
- Multiple minima behavior: Missing observations can hide minima.
- Noisy curve behavior: Missingness can make noise harder to identify.
- Validation strategy: Fixtures with missing time and luminescence fields should return exact counts.
- Dependencies: QC.

### Duplicate Timestamp Count

- Feature name: Duplicate Timestamp Count
- Short name: `duplicate_timestamp_count`
- Category: Quality Metrics
- Priority: Core
- Scientific purpose: Counts unresolved repeated time values within a measurement series.
- Mathematical definition: Number of rows participating in duplicate `Time_Minutes` groups within the series, or number of duplicate groups; implementation must report which convention is used. This specification prefers row count plus an optional group count.
- Units: count.
- Expected range: integer `>= 0`.
- Interpretation: Larger values indicate unresolved identity, source layout, importer, or true repeated-measurement issues. Smaller values support clean time-series calculations. Any unresolved conflicting duplicate timestamp lowers confidence for AUC, slopes, peak timing, latency, and recovery.
- Advantages: Prevents accidental averaging or silent conflict suppression.
- Limitations: Does not by itself explain whether duplicates are legitimate repeats or source defects.
- Missing data behavior: Always computable when time values are present.
- Multiple peaks behavior: Duplicate peak times can make peak identity ambiguous.
- Multiple minima behavior: Duplicate trough times can make minimum identity ambiguous.
- Noisy curve behavior: Duplicate conflicting values can mimic noise but must be treated as identity/QC evidence.
- Validation strategy: Synthetic duplicate-time fixtures should return exact duplicate row and group counts.
- Dependencies: time ordering, QC.

### QC Flag Count

- Feature name: QC Flag Count
- Short name: `qc_flag_count`
- Category: Quality Metrics
- Priority: Core
- Scientific purpose: Summarizes row-level QC burden carried into the feature vector.
- Mathematical definition: Count of distinct or total parsed `QC_Flags` across rows in the series; implementation must choose and document total count and distinct flag set.
- Units: count.
- Expected range: integer `>= 0`.
- Interpretation: Larger values indicate more provenance, identity, numeric, or warning issues. Smaller values indicate cleaner evidence. Biological claims from any response class become weaker as QC burden increases.
- Advantages: Connects feature values to auditability.
- Limitations: Different QC flags have different severity and should not be treated as equal in confidence scoring.
- Missing data behavior: Zero if no flags are present; null only if `QC_Flags` column is absent, which should be a schema error.
- Multiple peaks behavior: Peak interpretation is lowered when flags include duplicate timestamps or noisy identity.
- Multiple minima behavior: Minimum interpretation is similarly lowered by relevant flags.
- Noisy curve behavior: Noise-related flags should lower confidence more than benign provenance notes.
- Validation strategy: Fixtures with multiple semicolon-delimited flags should return expected total and distinct counts.
- Dependencies: QC.

### Feature Confidence Score

- Feature name: Feature Confidence Score
- Short name: `feature_confidence_score`
- Category: Quality Metrics
- Priority: Core
- Scientific purpose: Provides a transparent confidence label for the feature vector and, where possible, individual features.
- Mathematical definition: A bounded score from 0 to 1 derived from QC status, valid observation coverage, duplicate timestamps, missingness, baseline adequacy, signal-to-noise evidence, and identifier provenance. Thresholds map to High, Medium, or Low confidence.
- Units: unitless score plus categorical label.
- Expected range: `0` to `1`; High, Medium, Low.
- Interpretation: Higher confidence supports stronger biological interpretation. Lower confidence means the measured pattern may be real but is not yet sufficiently supported for primary modeling or publication claims. No, negative, transient, and sustained response calls require confidence context.
- Advantages: Makes uncertainty explicit and comparable.
- Limitations: Depends on weighting choices that must be validated and versioned.
- Missing data behavior: Always emitted; severe missingness lowers the score.
- Multiple peaks behavior: Multiple peaks lower confidence only when dominance or threshold rules are ambiguous.
- Multiple minima behavior: Multiple minima lower confidence only when trough identity is ambiguous.
- Noisy curve behavior: High noise lowers confidence, especially for temporal and slope features.
- Validation strategy: Synthetic perfect, sparse, noisy, duplicate-timestamp, and QC-failed curves should map to expected confidence bands.
- Dependencies: time ordering, baseline estimation, smoothing, peak detection, normalization where applicable, QC.

## Dependency Matrix

| Dependency | Features |
|---|---|
| Time ordering | Endpoint Luminescence; Time to Peak; Time to Minimum; Response Latency; Recovery Time; Peak Width; Half-Maximum Duration; Total AUC; Positive Area; Negative Area; Normalised Area; Initial Slope; Maximum Slope; Minimum Slope; Average Slope; Recovery Slope; Log2 Fold Change; Normalised Endpoint; Endpoint/Baseline Ratio; Duplicate Timestamp Count |
| Baseline estimation | Baseline Luminescence; Baseline Variability; Positive Area; Negative Area; Normalised Area; Response Latency; Recovery Time; Peak Width; Half-Maximum Duration; Recovery Slope; Maximum Fold Change; Minimum Fold Change; Log2 Fold Change; Peak/Baseline Ratio; Endpoint/Baseline Ratio; Signal-to-Noise Ratio; Feature Confidence Score |
| Normalisation | Normalised Area; Normalised Endpoint; future normalized variants of intensity, slope, and AUC features |
| Peak detection | Time to Peak; Response Latency; Recovery Time; Peak Width; Half-Maximum Duration; Peak/Baseline Ratio; Recovery Slope |
| Trough detection | Time to Minimum; Response Latency for inhibitory response; Recovery Time for inhibitory response; Recovery Slope; Minimum Fold Change |
| Curve smoothing | Response Latency; Recovery Time; Peak Width; Half-Maximum Duration; Maximum Slope; Minimum Slope; Recovery Slope; Feature Confidence Score |
| QC | All features |

## Confidence Framework

Each feature vector must receive a confidence score and label. Individual feature-level confidence should also be supported when features have different requirements.

Recommended confidence bands:

- High: score `>= 0.80`. Series has no blocking QC errors, no unresolved conflicting duplicate timestamps, sufficient valid observations, stable baseline if required, and feature-specific dependencies are satisfied.
- Medium: score `>= 0.50` and `< 0.80`. Series is usable but has warnings such as synthetic identifiers, moderate missingness, baseline uncertainty, mild noise, irregular time spacing, or non-critical QC flags.
- Low: score `< 0.50`. Series has sparse observations, unresolved duplicate timestamps, missing baseline, severe noise, invalid rows affecting required calculations, or ambiguous measurement identity.

Confidence should be lowered by:

- `Record_Valid = False` rows in the series.
- Missing required canonical fields.
- Missing late or early observations relevant to a feature.
- Duplicate timestamps, especially conflicting values.
- Missing `Plate_ID`, `Well_ID`, or `Replicate_ID` when they create identity ambiguity.
- Synthetic `Measurement_Unit_ID` provenance.
- High baseline variability.
- Low signal-to-noise ratio.
- Isolated extrema unsupported by adjacent points.
- Normalization method unavailable for normalized features.

QC must influence confidence but must not silently remove scientific evidence. Rows may be excluded from numerical calculation only under a documented rule, and excluded counts must remain visible in quality features.

## Biological Interpretation Rules

Larger intensity or area features generally indicate stronger cumulative luminescence, induction, reporter activation, or growth-supported signal. However, very high baseline can reflect strain brightness rather than contaminant-specific response, so relative features are needed.

Smaller intensity or area features generally indicate suppression, inhibition, lower biomass, toxicity, reporter repression, weak activation, or technical low signal.

No response means the curve remains within baseline noise tolerance. It should be supported by low dynamic range, low positive and negative area, endpoint/baseline near one, low slopes, and sufficient observations.

Negative response means the signal moves below baseline or control-relative expectation. It is reflected by minimum luminescence, negative area, minimum fold change, negative average slope, or endpoint/baseline below one.

Transient response means the signal departs from baseline and later returns. It is reflected by high peak or trough features, finite recovery, endpoint near baseline, and moderate AUC relative to peak magnitude.

Sustained response means the endpoint remains far from baseline. It is reflected by endpoint luminescence, endpoint/baseline ratio, normalised endpoint, total AUC, positive or negative area, and missing recovery.

## Implementation Priority

Core implementation should start with features that require only clean time ordering, baseline from first valid observation, finite raw signal, and QC outputs:

- Baseline Luminescence
- Maximum Luminescence
- Minimum Luminescence
- Endpoint Luminescence
- Dynamic Range
- Mean Luminescence
- Signal Standard Deviation
- Signal Coefficient of Variation
- Time to Peak
- Total Area Under Curve
- Initial Slope
- Maximum Slope
- Minimum Slope
- Maximum Fold Change
- Endpoint/Baseline Ratio
- Number of Valid Observations
- Missing Observations
- Duplicate Timestamp Count
- QC Flag Count
- Feature Confidence Score

Recommended features should follow after baseline-window and duplicate-timestamp policies are stable:

- Baseline Variability
- Median Luminescence
- Time to Minimum
- Half-Maximum Duration
- Positive Area
- Negative Area
- Normalised Area
- Average Slope
- Recovery Slope
- Minimum Fold Change
- Log2 Fold Change

Optional features require stronger normalization or interpretive decisions:

- Signal-to-Noise Ratio
- Normalised Endpoint
- Peak/Baseline Ratio

Experimental features require validated thresholds, persistence rules, smoothing, and dominant-event detection:

- Response Latency
- Recovery Time
- Peak Width

## Stage 6B Readiness Constraints

Before implementing the feature engine, Stage 5D should resolve or explicitly quarantine the 26 conflicting duplicate-timepoint groups identified in Stage 5C. If Stage 6B begins before Stage 5D is complete, the engine must treat affected series as low confidence and must not average conflicting duplicated timestamps.

The first implementation should use the canonical schema directly and should not reuse legacy feature input preparation that averages by strain, chemical, concentration, experiment, replicate, and time.

The first implementation should emit both feature values and feature provenance, including series keys, row counts, duplicate timestamp counts, QC flags, confidence score, and confidence label.

