"""
regression.py
=============
Regression pipeline: train multiple regressors to predict CO concentration
(ppm) from sensor readings and evaluate their performance.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)


def get_regressors() -> dict:
    return {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, random_state=42,
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, gamma="scale"),
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    return {"y_pred": y_pred, "metrics": metrics}


SLOW_MODELS = {"SVR (RBF)"}
MAX_TRAIN_SAMPLES_SLOW = 50000


def _subsample_reg(X, y, max_n, random_state=42):
    """Random subsample for slow regression models."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def run_regression_pipeline(X_train, X_test, y_train, y_test) -> dict:
    regressors = get_regressors()
    results = {}

    print("\n" + "=" * 70)
    print("  REGRESSION PIPELINE - CO Concentration Prediction")
    print("=" * 70)

    for name, model in regressors.items():
        # Subsample for slow models
        if name in SLOW_MODELS and len(X_train) > MAX_TRAIN_SAMPLES_SLOW:
            X_tr, y_tr = _subsample_reg(X_train, y_train, MAX_TRAIN_SAMPLES_SLOW)
            print(f"\n  > Training {name} (subsampled to {len(X_tr):,} rows) ...")
        else:
            X_tr, y_tr = X_train, y_train
            print(f"\n  > Training {name} ...")

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        res = evaluate_regressor(model, X_test, y_test)
        res["model"] = model
        res["train_time"] = train_time
        results[name] = res

        m = res["metrics"]
        print(f"    MAE:  {m['mae']:.4f}  |  RMSE: {m['rmse']:.4f}")
        print(f"    MSE:  {m['mse']:.4f}  |  R2:   {m['r2']:.4f}")
        print(f"    Time: {train_time:.2f}s")

    # Summary table
    print("\n" + "-" * 70)
    header = f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Time':>8}"
    print(header)
    print("-" * 70)
    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['r2']:>8.4f} {res['train_time']:>7.2f}s")
    print("-" * 70)

    return results


def get_feature_importances(results: dict, feature_names: list) -> dict:
    importances = {}
    for name, res in results.items():
        model = res["model"]
        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, "coef_"):
            importances[name] = dict(zip(feature_names, np.abs(model.coef_)))
    return importances
