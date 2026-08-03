from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import FIGURES_DIR


def _name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _save(fig, name, save):
    if not save:
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def _finish(fig, show):
    plt.show() if show else plt.close(fig)


def create_model_evaluation_plots(y_true, y_pred, model_name, save=True, show=False):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residuals = y_true - y_pred
    paths = {}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true, y_pred, alpha=0.65)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set(
        xlabel="Actual Heating Load",
        ylabel="Predicted Heating Load",
        title=f"{model_name}: Actual vs Predicted",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["actual_vs_predicted"] = _save(
        fig, f"{_name(model_name)}_actual_vs_predicted.png", save
    )
    _finish(fig, show)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_pred, residuals, alpha=0.65)
    ax.axhline(0, ls="--")
    ax.set(xlabel="Predicted", ylabel="Residual", title=f"{model_name}: Residuals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["residuals"] = _save(fig, f"{_name(model_name)}_residuals.png", save)
    _finish(fig, show)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(residuals, bins=30)
    ax.axvline(0, ls="--")
    ax.set(
        xlabel="Residual",
        ylabel="Frequency",
        title=f"{model_name}: Residual Distribution",
    )
    fig.tight_layout()
    paths["distribution"] = _save(
        fig, f"{_name(model_name)}_residual_distribution.png", save
    )
    _finish(fig, show)
    fig, ax = plt.subplots(figsize=(8, 7))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{model_name}: Q-Q Plot")
    fig.tight_layout()
    paths["qq"] = _save(fig, f"{_name(model_name)}_qq.png", save)
    _finish(fig, show)
    return paths


def plot_feature_importance(
    importance, feature_names, model_name, top_n=15, save=True, show=False
):
    df = pd.DataFrame(
        {"Feature": list(feature_names), "Importance": np.asarray(importance, float)}
    ).sort_values("Importance", ascending=False)
    top = df.head(top_n).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"], top["Importance"])
    ax.set(xlabel="Importance", title=f"{model_name}: Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, f"{_name(model_name)}_feature_importance.png", save)
    _finish(fig, show)
    return df


def create_comparison_chart(results_df, save=True, show=False):
    """
    Save labelled model-comparison charts used by the report and presentation.

    Charts:
    - Test R²
    - RMSE and MAE
    - Cross-validation R²
    - Train-test R² gap
    - Training time
    - Balanced deployment score
    """
    aliases = {
        "Algorithm Name": "Algorithm",
        "R2 Score": "R2",
        "Test R2": "R2",
        "Test RMSE": "RMSE",
        "Test MAE": "MAE",
        "Cross Validation R2 Mean": "CV R2 Mean",
        "Cross Validation R2 Std": "CV R2 Std",
        "Train-Test Gap": "Train-Test R2 Gap",
    }
    df = results_df.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in results_df.columns and new not in results_df.columns
        }
    ).copy()

    required = {"Algorithm", "R2", "RMSE", "MAE"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    paths = {}

    def horizontal_chart(metric, title, filename, higher_is_better):
        ordered = df.sort_values(metric, ascending=higher_is_better)
        fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.5)))
        bars = ax.barh(ordered["Algorithm"], ordered[metric])
        ax.set(xlabel=metric, title=title)
        ax.grid(axis="x", alpha=0.3)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        fig.tight_layout()
        paths[metric] = _save(fig, filename, save)
        _finish(fig, show)

    horizontal_chart(
        "R2",
        "Figure: Held-out Test R² by Model",
        "model_comparison_test_r2.png",
        True,
    )
    horizontal_chart(
        "RMSE",
        "Figure: Test RMSE by Model",
        "model_comparison_rmse.png",
        False,
    )
    horizontal_chart(
        "MAE",
        "Figure: Test MAE by Model",
        "model_comparison_mae.png",
        False,
    )

    optional_charts = [
        (
            "CV R2 Mean",
            "Figure: Cross-validation R² by Model",
            "model_comparison_cv_r2.png",
            True,
        ),
        (
            "Train-Test R2 Gap",
            "Figure: Train-Test R² Gap by Model",
            "model_comparison_train_test_gap.png",
            False,
        ),
        (
            "Training Time (s)",
            "Figure: Training Time by Model",
            "model_comparison_training_time.png",
            False,
        ),
        (
            "Deployment Score",
            "Figure: Balanced Deployment Score",
            "model_comparison_deployment_score.png",
            True,
        ),
    ]
    for metric, title, filename, higher_is_better in optional_charts:
        if metric in df.columns:
            horizontal_chart(
                metric,
                title,
                filename,
                higher_is_better,
            )

    # Combined error chart.
    error_long = df[["Algorithm", "RMSE", "MAE"]].melt(
        id_vars="Algorithm",
        var_name="Metric",
        value_name="Error",
    )
    pivot = error_long.pivot(index="Algorithm", columns="Metric", values="Error")
    pivot = pivot.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind="bar", ax=ax)
    ax.set(
        xlabel="Algorithm",
        ylabel="Error",
        title="Figure: MAE and RMSE Comparison",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    paths["error_comparison"] = _save(
        fig,
        "model_comparison_errors.png",
        save,
    )
    _finish(fig, show)

    return paths
