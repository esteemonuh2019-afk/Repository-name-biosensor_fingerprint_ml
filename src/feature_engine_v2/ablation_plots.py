"""Plots for Stage 8C feature-family ablation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


FIGURE_DPI = 300


def write_ablation_plots(result, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write PNG and PDF ablation figures."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    plotters = [
        ("feature_family_comparison", _plot_family_comparison),
        ("feature_family_ablation", _plot_ablation),
        ("classification_improvement", _plot_classification_improvement),
        ("regression_improvement", _plot_regression_improvement),
        ("runtime_comparison", _plot_runtime),
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


def _plot_family_comparison(result):
    data = result.ablation_summary.copy(deep=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    if not data.empty:
        display = data.sort_values("classification_macro_f1_gain", ascending=True)
        axis.barh(display["feature_set"], display["classification_macro_f1_gain"], color="#1f77b4", alpha=0.82, label="Macro F1 gain")
        axis2 = axis.twiny()
        axis2.plot(display["regression_r2_gain"], display["feature_set"], color="#ff7f0e", marker="o", linewidth=1.2, label="R2 gain")
        axis2.set_xlabel("Regression R2 gain")
    axis.axvline(0, color="#222222", linewidth=1.0)
    axis.set_title("Feature-Family Comparison")
    axis.set_xlabel("Classification Macro F1 gain")
    axis.set_ylabel("Feature set")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _plot_ablation(result):
    data = result.ablation_summary.copy(deep=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    if not data.empty:
        display = data.sort_values("classification_macro_f1", ascending=True)
        axis.barh(display["feature_set"], display["classification_macro_f1"], color="#2ca02c", alpha=0.82)
    axis.set_title("Feature-Family Ablation")
    axis.set_xlabel("Macro F1")
    axis.set_ylabel("Feature set")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _plot_classification_improvement(result):
    data = result.classification_comparison.copy(deep=True)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    if not data.empty:
        display = data.sort_values("macro_f1_gain", ascending=True)
        axis.barh(display["feature_set"], display["macro_f1_gain"], color="#9467bd", alpha=0.82)
    axis.axvline(0, color="#222222", linewidth=1.0)
    axis.set_title("Classification Improvement")
    axis.set_xlabel("Macro F1 gain vs current features")
    axis.set_ylabel("Feature set")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _plot_regression_improvement(result):
    data = result.regression_r2_comparison.copy(deep=True)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    if not data.empty:
        display = data.sort_values("r2_gain", ascending=True)
        axis.barh(display["feature_set"], display["r2_gain"], color="#17becf", alpha=0.82)
    axis.axvline(0, color="#222222", linewidth=1.0)
    axis.set_title("Regression Improvement")
    axis.set_xlabel("R2 gain vs current features")
    axis.set_ylabel("Feature set")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _plot_runtime(result):
    data = result.runtime_comparison.copy(deep=True)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    if not data.empty:
        display = data.sort_values("total_runtime_seconds", ascending=True)
        axis.barh(display["feature_set"], display["total_runtime_seconds"], color="#8c564b", alpha=0.82)
    axis.set_title("Runtime Comparison")
    axis.set_xlabel("Runtime (seconds)")
    axis.set_ylabel("Feature set")
    axis.grid(True, axis="x", alpha=0.25)
    return figure
