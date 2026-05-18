"""
visualization.py
=================
Centralized plotting functions for EDA, classification, and regression results.
All plots are saved to the outputs/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

# ── Style configuration ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1d23",
    "axes.edgecolor": "#2d3139",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
    "grid.color": "#2d3139",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0e1117",
    "font.family": "sans-serif",
    "font.size": 10,
})

# Color palette
COLORS = [
    "#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E",
    "#55EFC4", "#E17055", "#74B9FF", "#A29BFE",
]

OUTPUT_DIR = "outputs"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_co_distribution(df: pd.DataFrame):
    """Histogram of CO concentration values."""
    _ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution
    axes[0].hist(df["CO (ppm)"], bins=50, color=COLORS[0], alpha=0.85, edgecolor="none")
    axes[0].set_xlabel("CO Concentration (ppm)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("CO Concentration Distribution")
    axes[0].grid(True, alpha=0.3)

    # Class balance (present / absent)
    counts = [(df["CO (ppm)"] == 0).sum(), (df["CO (ppm)"] > 0).sum()]
    labels = ["Absent (0 ppm)", "Present (>0 ppm)"]
    bars = axes[1].bar(labels, counts, color=[COLORS[1], COLORS[2]], edgecolor="none")
    axes[1].set_ylabel("Count")
    axes[1].set_title("CO Presence Class Balance")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f"{count:,}", ha="center", va="bottom", fontsize=9, color="#e0e0e0",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_co_distribution.png"))
    plt.close()
    print("  ✓ Saved eda_co_distribution.png")


def plot_sensor_distributions(df: pd.DataFrame):
    """Box plots of all 14 sensor resistances."""
    _ensure_output_dir()
    sensor_cols = [f"R{i} (MOhm)" for i in range(1, 15)]
    fig, ax = plt.subplots(figsize=(16, 6))

    data_to_plot = [df[col].values for col in sensor_cols]
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        labels=[f"R{i}" for i in range(1, 15)],
        showfliers=False,
    )
    for i, patch in enumerate(bp["boxes"]):
        color = COLORS[0] if i < 7 else COLORS[1]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#e0e0e0")
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("#e0e0e0")

    ax.set_xlabel("Sensor")
    ax.set_ylabel("Resistance (MOhm)")
    ax.set_title("Sensor Resistance Distributions  (Purple=Figaro, Teal=FIS)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_sensor_distributions.png"))
    plt.close()
    print("  ✓ Saved eda_sensor_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    """Correlation heatmap for feature columns."""
    _ensure_output_dir()
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        square=True, linewidths=0.5, linecolor="#2d3139",
        cbar_kws={"shrink": 0.7, "label": "Correlation"},
        annot=False, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png"))
    plt.close()
    print("  ✓ Saved eda_correlation_heatmap.png")


def plot_sensor_vs_co(df: pd.DataFrame):
    """Scatter plot of mean sensor resistance vs CO concentration."""
    _ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Figaro sensors mean
    if "R_figaro_mean" in df.columns:
        axes[0].scatter(
            df["CO (ppm)"], df["R_figaro_mean"],
            alpha=0.1, s=2, color=COLORS[0], rasterized=True,
        )
        axes[0].set_xlabel("CO Concentration (ppm)")
        axes[0].set_ylabel("Mean Resistance (MOhm)")
        axes[0].set_title("Figaro Sensors vs CO")
        axes[0].grid(True, alpha=0.3)

    # FIS sensors mean
    if "R_fis_mean" in df.columns:
        axes[1].scatter(
            df["CO (ppm)"], df["R_fis_mean"],
            alpha=0.1, s=2, color=COLORS[1], rasterized=True,
        )
        axes[1].set_xlabel("CO Concentration (ppm)")
        axes[1].set_ylabel("Mean Resistance (MOhm)")
        axes[1].set_title("FIS Sensors vs CO")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_sensor_vs_co.png"))
    plt.close()
    print("  ✓ Saved eda_sensor_vs_co.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(results: dict, y_test):
    """Grid of confusion matrices for each classifier."""
    _ensure_output_dir()
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Absent", "Present"])
        disp.plot(ax=axes[idx], cmap="BuPu", colorbar=False)
        axes[idx].set_title(name, fontsize=11, fontweight="bold")
        # Style the text
        for text in disp.text_.ravel():
            text.set_color("#e0e0e0")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Classification — Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clf_confusion_matrices.png"))
    plt.close()
    print("  ✓ Saved clf_confusion_matrices.png")


def plot_roc_curves(results: dict, y_test):
    """ROC curves for all classifiers on a single plot."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, res) in enumerate(results.items()):
        if "y_proba" in res and res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[idx % len(COLORS)], lw=2,
                    label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Classification — ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clf_roc_curves.png"))
    plt.close()
    print("  ✓ Saved clf_roc_curves.png")


def plot_classification_comparison(results: dict):
    """Bar chart comparing classification metrics across models."""
    _ensure_output_dir()
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.15

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [results[n]["metrics"].get(metric, 0) for n in names]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor="none")
        # Add value labels on top
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="#e0e0e0")

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Classification — Model Comparison")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clf_model_comparison.png"))
    plt.close()
    print("  ✓ Saved clf_model_comparison.png")


def plot_feature_importance(importances: dict, title: str, filename: str, top_n: int = 15):
    """Horizontal bar chart of feature importances."""
    _ensure_output_dir()
    # Sort and take top_n
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, values = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, values, color=COLORS[0], alpha=0.85, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved {filename}")


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESSION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(results: dict, y_test):
    """Scatter plots of actual vs predicted CO concentration."""
    _ensure_output_dir()
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        ax.scatter(
            y_test, res["y_pred"],
            alpha=0.15, s=4, color=COLORS[idx % len(COLORS)], rasterized=True,
        )
        # Perfect prediction line
        lims = [min(y_test.min(), res["y_pred"].min()),
                max(y_test.max(), res["y_pred"].max())]
        ax.plot(lims, lims, "w--", alpha=0.4, lw=1)
        ax.set_xlabel("Actual CO (ppm)")
        ax.set_ylabel("Predicted CO (ppm)")
        r2 = res["metrics"]["r2"]
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Regression — Actual vs Predicted", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reg_actual_vs_predicted.png"))
    plt.close()
    print("  ✓ Saved reg_actual_vs_predicted.png")


def plot_residuals(results: dict, y_test):
    """Residual plots for each regression model."""
    _ensure_output_dir()
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        residuals = y_test.values - res["y_pred"]
        ax.scatter(
            res["y_pred"], residuals,
            alpha=0.15, s=4, color=COLORS[idx % len(COLORS)], rasterized=True,
        )
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.4, lw=1)
        ax.set_xlabel("Predicted CO (ppm)")
        ax.set_ylabel("Residual (ppm)")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Regression — Residual Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reg_residual_plots.png"))
    plt.close()
    print("  ✓ Saved reg_residual_plots.png")


def plot_regression_comparison(results: dict):
    """Bar chart comparing regression metrics across models."""
    _ensure_output_dir()
    names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² Score
    r2_vals = [results[n]["metrics"]["r2"] for n in names]
    bars = axes[0].bar(names, r2_vals, color=COLORS[:len(names)], alpha=0.85, edgecolor="none")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("R² Score Comparison")
    axes[0].grid(True, alpha=0.2, axis="y")
    axes[0].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#e0e0e0")

    # RMSE
    rmse_vals = [results[n]["metrics"]["rmse"] for n in names]
    bars = axes[1].bar(names, rmse_vals, color=COLORS[:len(names)], alpha=0.85, edgecolor="none")
    axes[1].set_ylabel("RMSE (ppm)")
    axes[1].set_title("RMSE Comparison")
    axes[1].grid(True, alpha=0.2, axis="y")
    axes[1].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#e0e0e0")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reg_model_comparison.png"))
    plt.close()
    print("  ✓ Saved reg_model_comparison.png")
