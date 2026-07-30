"""Publication-style plots for Stage 8D feature selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


FIGURE_DPI = 300


def write_feature_selection_plots(result, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write PNG and PDF feature-selection figures."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    plotters = [
        ("performance_vs_feature_count", _plot_performance_vs_feature_count),
        ("feature_importance", _plot_feature_importance),
        ("feature_ranking", _plot_feature_ranking),
    ]
    for stem, plotter in plotters:
        figure = plotter(result)
        for suffix in ("png", "pdf"):
            path = target / f"{stem}.{suffix}"
            if path.exists() and not overwrite:
                raise FileExistsError(f"Output file already exists: {path}")
            figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
            created.append(path)
        plt.close(figure)
    return created


def _plot_performance_vs_feature_count(result):
    data = result.performance_vs_feature_count.copy(deep=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=False)
    class_data = data.loc[data["task"].eq("classification")]
    reg_data = data.loc[data["task"].eq("regression")]

    for method, group in class_data.groupby("selector_method", sort=True):
        ordered = group.sort_values("feature_count")
        axes[0].plot(
            ordered["feature_count"],
            ordered["primary_metric"],
            marker="o",
            linewidth=1.2,
            label=method,
        )
    axes[0].set_title("Classification")
    axes[0].set_xlabel("Feature count")
    axes[0].set_ylabel("Macro F1")
    axes[0].grid(True, alpha=0.25)

    for method, group in reg_data.groupby("selector_method", sort=True):
        ordered = group.sort_values("feature_count")
        axes[1].plot(
            ordered["feature_count"],
            ordered["primary_metric"],
            marker="o",
            linewidth=1.2,
            label=method,
        )
    axes[1].set_title("Regression")
    axes[1].set_xlabel("Feature count")
    axes[1].set_ylabel("R2")
    axes[1].grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Performance vs Feature Count")
    figure.subplots_adjust(bottom=0.22)
    return figure


def _plot_feature_importance(result):
    ranking = result.feature_ranking.copy(deep=True)
    figure, axis = plt.subplots(figsize=(9.0, 6.2))
    if not ranking.empty and "score" in ranking.columns:
        scored = ranking.loc[ranking["selector_method"].isin(["permutation", "tree_importance"])].copy()
        scored["score"] = pd.to_numeric(scored["score"], errors="coerce").fillna(0.0)
        if not scored.empty:
            display = (
                scored.groupby("feature_name", as_index=False)["score"]
                .mean()
                .sort_values(["score", "feature_name"], ascending=[False, True])
                .head(20)
                .sort_values("score", ascending=True)
            )
            axis.barh(display["feature_name"], display["score"], color="#1f77b4", alpha=0.82)
    axis.set_title("Feature Importance")
    axis.set_xlabel("Mean selection score")
    axis.set_ylabel("Feature")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _plot_feature_ranking(result):
    ranking = result.feature_ranking.copy(deep=True)
    figure, axis = plt.subplots(figsize=(9.0, 6.2))
    if not ranking.empty:
        ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
        display = (
            ranking.groupby("feature_name", as_index=False)["rank"]
            .mean()
            .sort_values(["rank", "feature_name"], ascending=[True, True])
            .head(25)
            .sort_values("rank", ascending=False)
        )
        axis.barh(display["feature_name"], display["rank"], color="#2ca02c", alpha=0.82)
    axis.set_title("Feature Ranking")
    axis.set_xlabel("Mean rank; lower is better")
    axis.set_ylabel("Feature")
    axis.grid(True, axis="x", alpha=0.25)
    return figure
