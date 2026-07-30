"""Feature-family metadata for Stage 8C Feature Engine V2."""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_ENGINE_V2_VERSION = "0.1.0"

TEMPORAL_KINETICS = "temporal_kinetics"
WINDOW_FEATURES = "window_features"
SHAPE_DESCRIPTORS = "shape_descriptors"
FREQUENCY_FEATURES = "frequency_features"
STRAIN_INTERACTION = "strain_interaction"
RESPONSE_DYNAMICS = "response_dynamics"
BASELINE_FEATURES = "baseline_features"
NORMALIZED_FEATURES = "normalized_features"

FEATURE_FAMILIES: tuple[str, ...] = (
    TEMPORAL_KINETICS,
    WINDOW_FEATURES,
    SHAPE_DESCRIPTORS,
    FREQUENCY_FEATURES,
    STRAIN_INTERACTION,
    RESPONSE_DYNAMICS,
    BASELINE_FEATURES,
    NORMALIZED_FEATURES,
)


@dataclass(frozen=True)
class FeatureDefinition:
    """Scientific metadata for one engineered feature."""

    feature_name: str
    feature_family: str
    mathematical_definition: str
    units: str
    dependencies: str


def feature_dictionary() -> list[FeatureDefinition]:
    """Return the Stage 8C feature dictionary in deterministic order."""

    definitions: list[FeatureDefinition] = []
    definitions.extend(_temporal_definitions())
    definitions.extend(_window_definitions())
    definitions.extend(_shape_definitions())
    definitions.extend(_frequency_definitions())
    definitions.extend(_strain_interaction_definitions())
    definitions.extend(_response_dynamics_definitions())
    definitions.extend(_baseline_definitions())
    definitions.extend(_normalized_definitions())
    return definitions


def feature_columns_by_family() -> dict[str, list[str]]:
    """Return feature names grouped by family."""

    grouped: dict[str, list[str]] = {family: [] for family in FEATURE_FAMILIES}
    for definition in feature_dictionary():
        grouped[definition.feature_family].append(definition.feature_name)
    return grouped


def _temporal_definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition("temporal_time_to_peak", TEMPORAL_KINETICS, "Time at first maximum signal.", "minutes", "time ordering; peak detection"),
        FeatureDefinition("temporal_time_to_half_peak", TEMPORAL_KINETICS, "First time signal reaches baseline + 0.5*(peak-baseline).", "minutes", "baseline estimation; peak detection; interpolation"),
        FeatureDefinition("temporal_rise_time", TEMPORAL_KINETICS, "time_to_peak - start_time.", "minutes", "time ordering; peak detection"),
        FeatureDefinition("temporal_decay_time", TEMPORAL_KINETICS, "end_time - time_to_peak.", "minutes", "time ordering; peak detection"),
        FeatureDefinition("temporal_recovery_time", TEMPORAL_KINETICS, "First post-peak time within 10% of baseline.", "minutes", "baseline estimation; peak detection"),
        FeatureDefinition("temporal_peak_width", TEMPORAL_KINETICS, "Duration signal remains above half-peak threshold.", "minutes", "baseline estimation; peak detection"),
        FeatureDefinition("temporal_peak_prominence", TEMPORAL_KINETICS, "peak - max(baseline, endpoint).", "luminescence", "baseline estimation; endpoint"),
        FeatureDefinition("temporal_maximum_derivative", TEMPORAL_KINETICS, "Maximum adjacent d(signal)/d(time).", "luminescence/minute", "time ordering"),
        FeatureDefinition("temporal_minimum_derivative", TEMPORAL_KINETICS, "Minimum adjacent d(signal)/d(time).", "luminescence/minute", "time ordering"),
        FeatureDefinition("temporal_derivative_variance", TEMPORAL_KINETICS, "Population variance of adjacent derivatives.", "(luminescence/minute)^2", "time ordering"),
        FeatureDefinition("temporal_derivative_entropy", TEMPORAL_KINETICS, "Shannon entropy of absolute derivative magnitudes.", "unitless", "time ordering; histogram binning"),
    ]


def _window_definitions() -> list[FeatureDefinition]:
    definitions: list[FeatureDefinition] = []
    windows = ("0_2h", "2_6h", "6_12h", "12_24h")
    stats = (
        ("mean", "Mean signal in window.", "luminescence"),
        ("median", "Median signal in window.", "luminescence"),
        ("maximum", "Maximum signal in window.", "luminescence"),
        ("minimum", "Minimum signal in window.", "luminescence"),
        ("variance", "Population signal variance in window.", "luminescence^2"),
        ("slope", "Endpoint slope from first to last observed point in window.", "luminescence/minute"),
        ("auc", "Trapezoidal area under the curve inside the window.", "luminescence*minute"),
        ("standard_deviation", "Population signal standard deviation in window.", "luminescence"),
    )
    for window in windows:
        for stat, definition, units in stats:
            definitions.append(
                FeatureDefinition(
                    f"window_{window}_{stat}",
                    WINDOW_FEATURES,
                    definition,
                    units,
                    "time windowing; time ordering",
                )
            )
    return definitions


def _shape_definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition("shape_skewness", SHAPE_DESCRIPTORS, "Third standardized moment of signal values.", "unitless", "distribution moments"),
        FeatureDefinition("shape_kurtosis", SHAPE_DESCRIPTORS, "Fourth standardized moment minus 3.", "unitless", "distribution moments"),
        FeatureDefinition("shape_entropy", SHAPE_DESCRIPTORS, "Shannon entropy of signal histogram.", "unitless", "histogram binning"),
        FeatureDefinition("shape_signal_energy", SHAPE_DESCRIPTORS, "Sum of squared signal values.", "luminescence^2", "complete signal"),
        FeatureDefinition("shape_roughness", SHAPE_DESCRIPTORS, "Sum of absolute adjacent signal differences.", "luminescence", "time ordering"),
        FeatureDefinition("shape_symmetry", SHAPE_DESCRIPTORS, "1 - normalized absolute difference between early and late AUC.", "unitless", "time ordering; AUC"),
        FeatureDefinition("shape_peak_count", SHAPE_DESCRIPTORS, "Number of interior local maxima.", "count", "time ordering; peak detection"),
        FeatureDefinition("shape_zero_crossings", SHAPE_DESCRIPTORS, "Number of baseline-centered sign changes.", "count", "baseline estimation; time ordering"),
        FeatureDefinition("shape_coefficient_of_variation", SHAPE_DESCRIPTORS, "Population signal SD divided by absolute mean.", "unitless", "distribution moments"),
    ]


def _frequency_definitions() -> list[FeatureDefinition]:
    definitions = [
        FeatureDefinition("frequency_dominant_frequency", FREQUENCY_FEATURES, "Frequency with maximum non-zero FFT magnitude.", "cycles/minute", "uniform-enough time ordering; FFT"),
        FeatureDefinition("frequency_spectral_entropy", FREQUENCY_FEATURES, "Shannon entropy of normalized FFT power spectrum.", "unitless", "FFT"),
        FeatureDefinition("frequency_spectral_energy", FREQUENCY_FEATURES, "Sum of FFT power values.", "luminescence^2", "FFT"),
    ]
    for index in range(1, 6):
        definitions.append(
            FeatureDefinition(
                f"frequency_fft_coefficient_{index}",
                FREQUENCY_FEATURES,
                f"Magnitude of FFT coefficient {index}.",
                "luminescence",
                "FFT",
            )
        )
    return definitions


def _strain_interaction_definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition("strain_interaction_difference", STRAIN_INTERACTION, "Response magnitude minus condition mean across strains.", "luminescence*minute", "matched condition; strain grouping"),
        FeatureDefinition("strain_interaction_ratio", STRAIN_INTERACTION, "Response magnitude divided by condition mean across strains.", "unitless", "matched condition; strain grouping"),
        FeatureDefinition("strain_interaction_mean", STRAIN_INTERACTION, "Mean response magnitude across strains under matched condition.", "luminescence*minute", "matched condition; strain grouping"),
        FeatureDefinition("strain_interaction_variance", STRAIN_INTERACTION, "Population variance of response magnitude across strains under matched condition.", "(luminescence*minute)^2", "matched condition; strain grouping"),
    ]


def _response_dynamics_definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition("response_induction_delay", RESPONSE_DYNAMICS, "First time signal rises above baseline + 10% dynamic range.", "minutes", "baseline estimation; threshold crossing"),
        FeatureDefinition("response_inhibition_delay", RESPONSE_DYNAMICS, "First time signal falls below baseline - 10% dynamic range.", "minutes", "baseline estimation; threshold crossing"),
        FeatureDefinition("response_duration", RESPONSE_DYNAMICS, "Duration outside baseline +/- 10% dynamic range.", "minutes", "baseline estimation; threshold crossing"),
        FeatureDefinition("response_recovery_fraction", RESPONSE_DYNAMICS, "(peak - endpoint) / (peak - baseline).", "unitless", "baseline estimation; peak detection; endpoint"),
        FeatureDefinition("response_sustained_response_score", RESPONSE_DYNAMICS, "Mean absolute baseline-centered late signal divided by dynamic range.", "unitless", "baseline estimation; late window"),
    ]


def _baseline_definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition("baseline_stability", BASELINE_FEATURES, "1 / (1 + baseline-window coefficient of variation).", "unitless", "baseline window"),
        FeatureDefinition("baseline_noise", BASELINE_FEATURES, "Population SD of earliest 10% of signal values.", "luminescence", "baseline window"),
        FeatureDefinition("baseline_drift", BASELINE_FEATURES, "Slope across earliest 10% of signal values.", "luminescence/minute", "baseline window; time ordering"),
    ]


def _normalized_definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition("normalized_peak_over_baseline", NORMALIZED_FEATURES, "peak / baseline.", "unitless", "baseline estimation; peak detection"),
        FeatureDefinition("normalized_endpoint_over_baseline", NORMALIZED_FEATURES, "endpoint / baseline.", "unitless", "baseline estimation; endpoint"),
        FeatureDefinition("normalized_auc_over_baseline_duration", NORMALIZED_FEATURES, "AUC / (baseline * duration).", "unitless", "baseline estimation; AUC"),
        FeatureDefinition("normalized_dynamic_range_over_baseline", NORMALIZED_FEATURES, "(peak - minimum) / baseline.", "unitless", "baseline estimation"),
        FeatureDefinition("normalized_positive_area_over_total_area", NORMALIZED_FEATURES, "Positive baseline-centered AUC divided by total absolute baseline-centered AUC.", "unitless", "baseline estimation; AUC"),
        FeatureDefinition("normalized_signal_zscore_auc", NORMALIZED_FEATURES, "AUC of within-series z-scored signal.", "minute", "within-series normalization; AUC"),
    ]
