"""Replicate-consistency assessment for feature datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_STABLE_CV_THRESHOLD = 0.10
DEFAULT_ACCEPTABLE_CV_THRESHOLD = 0.25

REPLICATE_GROUP_COLUMNS: tuple[str, ...] = (
    "Experiment_ID",
    "Strain",
    "Chemical",
    "Concentration",
)


def calculate_replicate_consistency(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    stable_cv_threshold: float = DEFAULT_STABLE_CV_THRESHOLD,
    acceptable_cv_threshold: float = DEFAULT_ACCEPTABLE_CV_THRESHOLD,
) -> pd.DataFrame:
    """Calculate replicate-consistency statistics without biological claims."""

    if dataframe.empty:
        return pd.DataFrame(columns=_output_columns())

    group_columns = [column for column in REPLICATE_GROUP_COLUMNS if column in dataframe.columns]
    if not group_columns:
        return pd.DataFrame(columns=_output_columns())

    rows: list[dict[str, object]] = []
    for group_key, group in dataframe.groupby(group_columns, dropna=False, sort=True):
        group_values = _group_values(group_columns, group_key)
        for feature in feature_columns:
            if feature not in group.columns:
                continue
            values = pd.to_numeric(group[feature], errors="coerce").dropna().astype(float)
            replicate_count = int(len(values))
            unique_replicate_count = _unique_replicate_count(group)
            mean = float(values.mean()) if replicate_count else None
            standard_deviation = float(values.std(ddof=0)) if replicate_count else None
            coefficient_of_variation = _coefficient_of_variation(mean, standard_deviation)
            stability_flag = _stability_flag(
                replicate_count,
                coefficient_of_variation,
                stable_cv_threshold=stable_cv_threshold,
                acceptable_cv_threshold=acceptable_cv_threshold,
            )
            rows.append(
                {
                    **group_values,
                    "feature": feature,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "coefficient_of_variation": coefficient_of_variation,
                    "replicate_count": replicate_count,
                    "unique_replicate_id_count": unique_replicate_count,
                    "stability_flag": stability_flag,
                    "interpretation_label": "replicate consistency",
                    "biological_reproducibility_claimed": False,
                }
            )

    return pd.DataFrame(rows, columns=_output_columns(group_columns))


def summarize_replicate_consistency(consistency: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability flags per feature."""

    if consistency.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "eligible_group_count",
                "stable_group_count",
                "acceptable_group_count",
                "unstable_group_count",
                "insufficient_data_group_count",
                "dominant_stability_flag",
            ]
        )

    rows = []
    for feature, group in consistency.groupby("feature", dropna=False, sort=True):
        counts = group["stability_flag"].value_counts().to_dict()
        rows.append(
            {
                "feature": feature,
                "eligible_group_count": int(len(group)),
                "stable_group_count": int(counts.get("Stable", 0)),
                "acceptable_group_count": int(counts.get("Acceptable", 0)),
                "unstable_group_count": int(counts.get("Unstable", 0)),
                "insufficient_data_group_count": int(counts.get("Insufficient Data", 0)),
                "dominant_stability_flag": _dominant_flag(group["stability_flag"]),
            }
        )
    return pd.DataFrame(rows)


def _coefficient_of_variation(mean: float | None, standard_deviation: float | None) -> float | None:
    if mean is None or standard_deviation is None or mean == 0:
        return None
    return float(standard_deviation / abs(mean))


def _stability_flag(
    replicate_count: int,
    coefficient_of_variation: float | None,
    *,
    stable_cv_threshold: float,
    acceptable_cv_threshold: float,
) -> str:
    if replicate_count < 2 or coefficient_of_variation is None:
        return "Insufficient Data"
    if coefficient_of_variation <= stable_cv_threshold:
        return "Stable"
    if coefficient_of_variation <= acceptable_cv_threshold:
        return "Acceptable"
    return "Unstable"


def _group_values(group_columns: list[str], group_key) -> dict[str, object]:
    if len(group_columns) == 1:
        values = (group_key,)
    else:
        values = group_key
    return dict(zip(group_columns, values))


def _unique_replicate_count(group: pd.DataFrame) -> int:
    if "Replicate_ID" not in group.columns:
        return 0
    return int(group["Replicate_ID"].dropna().astype("string").nunique())


def _dominant_flag(flags: pd.Series) -> str:
    if flags.empty:
        return "Insufficient Data"
    order = ["Stable", "Acceptable", "Unstable", "Insufficient Data"]
    counts = flags.value_counts().to_dict()
    return sorted(counts, key=lambda flag: (-counts[flag], order.index(flag) if flag in order else 99))[0]


def _output_columns(group_columns: list[str] | None = None) -> list[str]:
    groups = list(group_columns or REPLICATE_GROUP_COLUMNS)
    return [
        *groups,
        "feature",
        "mean",
        "standard_deviation",
        "coefficient_of_variation",
        "replicate_count",
        "unique_replicate_id_count",
        "stability_flag",
        "interpretation_label",
        "biological_reproducibility_claimed",
    ]

