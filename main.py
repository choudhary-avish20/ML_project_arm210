#!/usr/bin/env python3
"""
main.py
=======
Entry point for the Gas Sensor Array Temperature Modulation ML pipeline.
Orchestrates data loading, feature engineering, EDA, classification, and regression.

Usage:
    python main.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_data, get_data_summary
from src.feature_engineering import (
    add_derived_features,
    create_classification_target,
    get_feature_columns,
)
from src.preprocessing import clean_data, split_data, scale_features
from src.classification import (
    run_classification_pipeline,
    get_feature_importances as clf_feature_importances,
)
from src.regression import (
    run_regression_pipeline,
    get_feature_importances as reg_feature_importances,
)
from src.visualization import (
    plot_co_distribution,
    plot_sensor_distributions,
    plot_correlation_heatmap,
    plot_sensor_vs_co,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_classification_comparison,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_regression_comparison,
)

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "datapoints"
SUBSAMPLE_STEP = 10        # Keep every Nth row
TEST_SIZE = 0.2
RANDOM_STATE = 42
SVM_MAX_SAMPLES = 50000    # Limit SVM/KNN training size for speed


def main():
    start_time = time.time()

    print("=" * 70)
    print("  GAS SENSOR ARRAY TEMPERATURE MODULATION")
    print("  Classification & Regression Pipeline")
    print("=" * 70)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    df = load_all_data(DATA_DIR, subsample_step=SUBSAMPLE_STEP)

    summary = get_data_summary(df)
    print(f"  Shape: {summary['shape']}")
    print(f"  CO range: {summary['co_range']}")
    print(f"  Days: {summary['days']}")

    # ── 2. Feature Engineering ────────────────────────────────────────────
    print("\n[2/7] Feature engineering ...")
    df = add_derived_features(df)
    y_clf = create_classification_target(df)
    y_reg = df["CO (ppm)"].copy()
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Classification target — Absent: {(y_clf == 0).sum():,}, Present: {(y_clf == 1).sum():,}")

    # ── 3. EDA Visualizations ────────────────────────────────────────────
    print("\n[3/7] Generating EDA plots ...")
    os.makedirs("outputs", exist_ok=True)
    plot_co_distribution(df)
    plot_sensor_distributions(df)
    plot_correlation_heatmap(df, feature_cols)
    plot_sensor_vs_co(df)

    # ── 4. Preprocessing ─────────────────────────────────────────────────
    print("\n[4/7] Preprocessing ...")
    df = clean_data(df)
    X = df[feature_cols]
    y_clf = create_classification_target(df)
    y_reg = df["CO (ppm)"].copy()

    # Split for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(
        X, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf,
    )
    # Split for regression (same indices for consistency)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    # Scale
    X_train_clf_s, X_test_clf_s, scaler_clf = scale_features(X_train_clf, X_test_clf)
    X_train_reg_s, X_test_reg_s, scaler_reg = scale_features(X_train_reg, X_test_reg)

    print(f"  Train size: {len(X_train_clf):,}  |  Test size: {len(X_test_clf):,}")

    # ── 5. Classification ────────────────────────────────────────────────
    print("\n[5/7] Running classification pipeline ...")
    clf_results = run_classification_pipeline(
        X_train_clf_s, X_test_clf_s, y_train_clf, y_test_clf,
    )

    # Classification plots
    print("\n  Generating classification plots ...")
    plot_confusion_matrices(clf_results, y_test_clf)
    plot_roc_curves(clf_results, y_test_clf)
    plot_classification_comparison(clf_results)

    # Feature importance
    clf_importances = clf_feature_importances(clf_results, feature_cols)
    for model_name, imp in clf_importances.items():
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plot_feature_importance(
            imp,
            f"Feature Importance — {model_name} (Classification)",
            f"clf_feature_importance_{safe_name}.png",
        )

    # ── 6. Regression ────────────────────────────────────────────────────
    print("\n[6/7] Running regression pipeline ...")
    reg_results = run_regression_pipeline(
        X_train_reg_s, X_test_reg_s, y_train_reg, y_test_reg,
    )

    # Regression plots
    print("\n  Generating regression plots ...")
    plot_actual_vs_predicted(reg_results, y_test_reg)
    plot_residuals(reg_results, y_test_reg)
    plot_regression_comparison(reg_results)

    # Feature importance
    reg_importances = reg_feature_importances(reg_results, feature_cols)
    for model_name, imp in reg_importances.items():
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plot_feature_importance(
            imp,
            f"Feature Importance — {model_name} (Regression)",
            f"reg_feature_importance_{safe_name}.png",
        )

    # ── 7. Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  All plots saved to: outputs/")
    print("=" * 70)

    # Save summary report
    _save_summary_report(clf_results, reg_results, summary, elapsed)


def _save_summary_report(clf_results, reg_results, data_summary, elapsed):
    """Save a text summary report to outputs/."""
    lines = []
    lines.append("=" * 70)
    lines.append("  GAS SENSOR ARRAY - RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\nDataset shape: {data_summary['shape']}")
    lines.append(f"CO range: {data_summary['co_range']} ppm")
    lines.append(f"Total runtime: {elapsed:.1f}s\n")

    lines.append("\n--- CLASSIFICATION RESULTS ---")
    lines.append(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    lines.append("-" * 70)
    for name, res in clf_results.items():
        m = res["metrics"]
        lines.append(f"{name:<25} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                      f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f}")

    lines.append("\n\n--- REGRESSION RESULTS ---")
    lines.append(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    lines.append("-" * 70)
    for name, res in reg_results.items():
        m = res["metrics"]
        lines.append(f"{name:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

    report_path = os.path.join("outputs", "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary report saved to {report_path}")


if __name__ == "__main__":
    main()
