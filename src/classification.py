"""
classification.py
==================
Classification pipeline: train multiple classifiers to detect CO presence
(binary classification) and evaluate their performance.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)


def get_classifiers() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="saga", n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = 0.0

    return {"y_pred": y_pred, "y_proba": y_proba, "metrics": metrics}


SLOW_MODELS = {"SVM (RBF)", "KNN"}
MAX_TRAIN_SAMPLES_SLOW = 50000  # Subsample for O(n²+) models


def _subsample(X, y, max_n, random_state=42):
    """Stratified subsample for slow models."""
    if len(X) <= max_n:
        return X, y
    from sklearn.model_selection import train_test_split
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=max_n, random_state=random_state, stratify=y,
    )
    return X_sub, y_sub


def run_classification_pipeline(X_train, X_test, y_train, y_test) -> dict:
    classifiers = get_classifiers()
    results = {}

    print("\n" + "=" * 70)
    print("  CLASSIFICATION PIPELINE - CO Presence Detection")
    print("=" * 70)

    for name, model in classifiers.items():
        # Subsample for slow models
        if name in SLOW_MODELS and len(X_train) > MAX_TRAIN_SAMPLES_SLOW:
            X_tr, y_tr = _subsample(X_train, y_train, MAX_TRAIN_SAMPLES_SLOW)
            print(f"\n  > Training {name} (subsampled to {len(X_tr):,} rows) ...")
        else:
            X_tr, y_tr = X_train, y_train
            print(f"\n  > Training {name} ...")

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        res = evaluate_classifier(model, X_test, y_test)
        res["model"] = model
        res["train_time"] = train_time
        results[name] = res

        m = res["metrics"]
        print(f"    Accuracy:  {m['accuracy']:.4f}  |  Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}  |  F1:        {m['f1']:.4f}")
        print(f"    ROC-AUC:   {m['roc_auc']:.4f}  |  Time:      {train_time:.2f}s")

    # Summary table
    print("\n" + "-" * 70)
    header = f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Time':>7}"
    print(header)
    print("-" * 70)
    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:<25} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f} "
              f"{res['train_time']:>6.2f}s")
    print("-" * 70)

    return results


def get_feature_importances(results: dict, feature_names: list) -> dict:
    importances = {}
    for name, res in results.items():
        model = res["model"]
        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, "coef_"):
            importances[name] = dict(zip(feature_names, np.abs(model.coef_[0])))
    return importances
