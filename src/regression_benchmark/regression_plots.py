"""Publication-oriented plots for Stage 8B regression outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DPI = 300


def write_regression_plots(result, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write PNG and PDF regression benchmark figures."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    plotters = [
        ("prediction_vs_actual", _plot_prediction_vs_actual),
        ("residual_plot", _plot_residual_plot),
        ("residual_histogram", _plot_residual_histogram),
        ("feature_importance", _plot_feature_importance),
        ("fold_performance", _plot_fold_performance),
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


def _plot_prediction_vs_actual(result):
    data = result.prediction_vs_actual
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    if not data.empty:
        actual = pd.to_numeric(data["actual_concentration"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(data["predicted_concentration"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(actual) & np.isfinite(predicted)
        axis.scatter(actual[finite], predicted[finite], s=16, alpha=0.45, edgecolors="none", color="#1f77b4")
        if finite.any():
            minimum = float(min(actual[finite].min(), predicted[finite].min()))
            maximum = float(max(actual[finite].max(), predicted[finite].max()))
            axis.plot([minimum, maximum], [minimum, maximum], color="#222222", linewidth=1.2, linestyle="--")
    axis.set_title("Prediction vs Actual")
    axis.set_xlabel("Actual concentration (ug/mL)")
    axis.set_ylabel("Predicted concentration (ug/mL)")
    axis.grid(True, alpha=0.25)
    return figure


def _plot_residual_plot(result):
    data = result.residuals
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    if not data.empty:
        predicted = pd.to_numeric(data["predicted_concentration"], errors="coerce").to_numpy(dtype=float)
        residual = pd.to_numeric(data["residual"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(predicted) & np.isfinite(residual)
        axis.scatter(predicted[finite], residual[finite], s=16, alpha=0.45, edgecolors="none", color="#2ca02c")
    axis.axhline(0.0, color="#222222", linewidth=1.1, linestyle="--")
    axis.set_title("Residuals by Prediction")
    axis.set_xlabel("Predicted concentration (ug/mL)")
    axis.set_ylabel("Residual (actual - predicted)")
    axis.grid(True, alpha=0.25)
    return figure


def _plot_residual_histogram(result):
    data = result.residuals
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    if not data.empty:
        residual = pd.to_numeric(data["residual"], errors="coerce").dropna().to_numpy(dtype=float)
        residual = residual[np.isfinite(residual)]
        if len(residual):
            axis.hist(residual, bins=40, color="#9467bd", alpha=0.78, edgecolor="white", linewidth=0.5)
    axis.axvline(0.0, color="#222222", linewidth=1.1, linestyle="--")
    axis.set_title("Residual Distribution")
    axis.set_xlabel("Residual (actual - predicted)")
    axis.set_ylabel("Count")
    axis.grid(True, axis="y", alpha=0.25)
    return figure


def _plot_feature_importance(result):
    data = result.permutation_importance
    value_column = "importance_mean"
    title = "Permutation Importance"
    if data.empty:
        data = result.feature_importance.rename(columns={"importance": "importance_mean"})
        value_column = "importance_mean"
        title = "Model Feature Importance"
    figure, axis = plt.subplots(figsize=(7.0, 5.6))
    if not data.empty and value_column in data.columns:
        table = (
            data.groupby("feature", as_index=False)[value_column]
            .mean()
            .sort_values([value_column, "feature"], ascending=[True, False])
            .tail(12)
        )
        axis.barh(table["feature"], table[value_column], color="#ff7f0e", alpha=0.82)
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.grid(True, axis="x", alpha=0.25)
    return figure


def _plot_fold_performance(result):
    data = result.fold_metrics
    figure, axis = plt.subplots(figsize=(8.0, 5.6))
    if not data.empty:
        summary = (
            data.groupby("model_name", as_index=False)
            .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"))
            .sort_values(["r2_mean", "model_name"], ascending=[True, False])
        )
        axis.barh(
            summary["model_name"],
            summary["r2_mean"],
            xerr=summary["r2_std"].fillna(0.0),
            color="#17becf",
            alpha=0.82,
            ecolor="#333333",
            capsize=3,
        )
    axis.axvline(0.0, color="#222222", linewidth=1.0)
    axis.set_title("Cross-Validated R2 by Model")
    axis.set_xlabel("Mean fold R2")
    axis.set_ylabel("Model")
    axis.grid(True, axis="x", alpha=0.25)
    return figure
