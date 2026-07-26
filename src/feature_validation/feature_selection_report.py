"""Feature-retention recommendations and validation report outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


HIGH_MISSINGNESS_PERCENTAGE = 20.0


def recommend_features(
    feature_columns: Iterable[str],
    *,
    missing_value_summary: pd.DataFrame,
    nonfinite_value_summary: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    constant_feature_summary: pd.DataFrame,
    low_variance_feature_summary: pd.DataFrame,
    range_validation_summary: pd.DataFrame,
    highly_correlated_pairs: pd.DataFrame,
    replicate_reproducibility_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic unsupervised feature-retention recommendations."""

    constant_features = set(constant_feature_summary.get("feature", pd.Series(dtype=str)).astype(str))
    low_variance_features = set(low_variance_feature_summary.get("feature", pd.Series(dtype=str)).astype(str))
    correlated_features = _features_in_correlated_pairs(highly_correlated_pairs)
    rows: list[dict[str, Any]] = []

    for feature in feature_columns:
        reasons: list[str] = []
        missing_percentage = _lookup_float(missing_value_summary, feature, "missing_percentage")
        missing_count = _lookup_int(missing_value_summary, feature, "missing_count")
        serious_nonfinite_count = (
            _lookup_int(nonfinite_value_summary, feature, "positive_infinity_count")
            + _lookup_int(nonfinite_value_summary, feature, "negative_infinity_count")
            + _lookup_int(nonfinite_value_summary, feature, "non_numeric_count")
        )
        finite_count = _lookup_int(feature_statistics, feature, "finite_count")
        range_violation_count = _range_violation_count(range_validation_summary, feature)
        unstable_count = _lookup_int(
            replicate_reproducibility_summary,
            feature,
            "unstable_group_count",
        )
        insufficient_count = _lookup_int(
            replicate_reproducibility_summary,
            feature,
            "insufficient_data_group_count",
        )

        if finite_count == 0:
            recommendation = "Exclude"
            reasons.append("no_finite_numeric_values")
        elif feature in constant_features:
            recommendation = "Exclude"
            reasons.append("constant_feature")
        elif feature in low_variance_features:
            recommendation = "Review"
            reasons.append("low_variance_or_near_constant")
        elif serious_nonfinite_count > 0:
            recommendation = "Review"
            reasons.append("infinite_or_non_numeric_values")
        elif range_violation_count > 0:
            recommendation = "Review"
            reasons.append("range_validation_violations")
        elif missing_percentage >= HIGH_MISSINGNESS_PERCENTAGE:
            recommendation = "Review"
            reasons.append(f"missing_percentage>={HIGH_MISSINGNESS_PERCENTAGE}")
        elif missing_count > 0 or feature in correlated_features or unstable_count > 0 or insufficient_count > 0:
            recommendation = "Retain with caution"
            if missing_count > 0:
                reasons.append("missing_values_present")
            if feature in correlated_features:
                reasons.append("high_correlation_candidate")
            if unstable_count > 0:
                reasons.append("unstable_replicate_consistency_groups")
            if insufficient_count > 0:
                reasons.append("insufficient_replicate_consistency_groups")
        else:
            recommendation = "Retain"
            reasons.append("no_unsupervised_validation_concerns")

        rows.append(
            {
                "feature": feature,
                "recommendation": recommendation,
                "reason": ";".join(reasons),
                "missing_percentage": missing_percentage,
                "serious_nonfinite_count": serious_nonfinite_count,
                "finite_count": finite_count,
                "range_violation_count": range_violation_count,
                "high_correlation_pair_count": _high_correlation_count(highly_correlated_pairs, feature),
                "unstable_group_count": unstable_count,
                "insufficient_data_group_count": insufficient_count,
            }
        )
    return pd.DataFrame(rows)


def write_validation_outputs(result: Any, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write all Stage 6C validation artifacts."""

    target = Path(output_dir)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {target}")
        if target.is_file():
            raise FileExistsError(f"Output path is a file: {target}")
        for child in target.iterdir():
            if child.is_file():
                child.unlink()
            else:
                _remove_directory(child)
    else:
        target.mkdir(parents=True)

    outputs = {
        "feature_statistics.csv": result.feature_statistics,
        "feature_missingness.csv": result.missing_value_summary,
        "feature_nonfinite_values.csv": result.infinite_value_summary,
        "constant_features.csv": result.constant_feature_summary,
        "low_variance_features.csv": result.low_variance_feature_summary,
        "pearson_correlations.csv": result.correlation_summary.get("pearson", pd.DataFrame()),
        "spearman_correlations.csv": result.correlation_summary.get("spearman", pd.DataFrame()),
        "highly_correlated_pairs.csv": result.correlation_summary.get(
            "highly_correlated_pairs",
            pd.DataFrame(),
        ),
        "replicate_consistency.csv": result.replicate_reproducibility_summary,
        "feature_recommendations.csv": result.feature_recommendations,
    }

    created: list[Path] = []
    summary_path = target / "feature_validation_summary.json"
    summary_path.write_text(
        json.dumps(result.to_summary_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    created.append(summary_path)

    for filename, dataframe in outputs.items():
        path = target / filename
        dataframe.to_csv(path, index=False)
        created.append(path)

    report_path = target / "feature_validation_report.md"
    report_path.write_text(render_validation_report(result), encoding="utf-8")
    created.append(report_path)
    return created


def render_validation_report(result: Any) -> str:
    """Render a scientific Markdown validation report."""

    recommendations = result.feature_recommendations
    counts = recommendations["recommendation"].value_counts().to_dict() if not recommendations.empty else {}
    highly_correlated = result.correlation_summary.get("highly_correlated_pairs", pd.DataFrame())
    lines = [
        "# Stage 6C Feature Validation Report",
        "",
        "## Summary",
        f"- feature rows: {result.metadata.get('feature_rows', 0)}",
        f"- feature columns assessed: {len(result.metadata.get('feature_columns', []))}",
        f"- validation passed: {result.validation_passed}",
        f"- retained features: {counts.get('Retain', 0)}",
        f"- retained with caution: {counts.get('Retain with caution', 0)}",
        f"- review features: {counts.get('Review', 0)}",
        f"- excluded features: {counts.get('Exclude', 0)}",
        "",
        "## Scientific Interpretation",
        "This validation is unsupervised. It does not use chemical labels, feature importance, PCA, clustering, classification, or regression. Recommendations are based only on missingness, non-finite values, variance, range validity, correlation, and replicate consistency.",
        "",
        "## Thresholds",
        f"- low variance threshold: {result.metadata.get('low_variance_threshold')}",
        f"- dominant value threshold: {result.metadata.get('dominant_proportion_threshold')}",
        f"- high correlation threshold: {result.metadata.get('correlation_threshold')}",
        f"- stable replicate consistency CV threshold: {result.metadata.get('stable_cv_threshold')}",
        f"- acceptable replicate consistency CV threshold: {result.metadata.get('acceptable_cv_threshold')}",
        "",
        "## Missing And Non-Finite Values",
        f"- missing feature values: {int(result.missing_value_summary['missing_count'].sum()) if not result.missing_value_summary.empty else 0}",
        f"- non-finite or non-numeric values: {int(result.infinite_value_summary['nonfinite_count'].sum()) if not result.infinite_value_summary.empty else 0}",
        "",
        "## Constant And Low-Variance Features",
        f"- constant features: {_joined_features(result.constant_feature_summary)}",
        f"- low-variance features: {_joined_features(result.low_variance_feature_summary)}",
        "",
        "## Correlation",
        f"- highly correlated feature pairs: {len(highly_correlated)}",
        "",
        "## Replicate Consistency",
        "Replicate statistics are reported as replicate consistency only. Biological reproducibility is not claimed because replicate type is not established by the feature dataset.",
        "",
        "## Known Upstream QC Limitations",
        "Feature validation preserves known upstream problems, including source QC warnings, zero-baseline fold-change limitations, duplicate timestamp conflicts, and invalid retained source rows. These rows are reported rather than removed.",
    ]

    if result.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in result.errors)
    if result.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in result.warnings)

    return "\n".join(lines) + "\n"


def _features_in_correlated_pairs(pairs: pd.DataFrame) -> set[str]:
    if pairs.empty:
        return set()
    return set(pairs["feature_a"].astype(str)) | set(pairs["feature_b"].astype(str))


def _lookup_float(dataframe: pd.DataFrame, feature: str, column: str) -> float:
    if dataframe.empty or column not in dataframe.columns:
        return 0.0
    match = dataframe.loc[dataframe["feature"].astype(str).eq(feature), column]
    if match.empty or pd.isna(match.iloc[0]):
        return 0.0
    return float(match.iloc[0])


def _lookup_int(dataframe: pd.DataFrame, feature: str, column: str) -> int:
    return int(_lookup_float(dataframe, feature, column))


def _range_violation_count(range_validation_summary: pd.DataFrame, feature: str) -> int:
    if range_validation_summary.empty:
        return 0
    if feature == "peak":
        return int(range_validation_summary["feature"].isin(["peak_minimum"]).sum())
    if feature == "minimum":
        return int(range_validation_summary["feature"].isin(["peak_minimum"]).sum())
    if feature == "dynamic_range":
        return int(range_validation_summary["feature"].isin(["dynamic_range", "peak_minimum"]).sum())
    return int(range_validation_summary["feature"].astype(str).eq(feature).sum())


def _high_correlation_count(pairs: pd.DataFrame, feature: str) -> int:
    if pairs.empty:
        return 0
    mask = pairs["feature_a"].astype(str).eq(feature) | pairs["feature_b"].astype(str).eq(feature)
    return int(mask.sum())


def _joined_features(dataframe: pd.DataFrame) -> str:
    if dataframe.empty or "feature" not in dataframe.columns:
        return "None"
    return "; ".join(sorted(dataframe["feature"].astype(str).unique().tolist())) or "None"


def _remove_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            _remove_directory(child)
    path.rmdir()

