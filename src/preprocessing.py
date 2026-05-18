"""
preprocessing.py
================
Data cleaning, train/test splitting, and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning:
    - Drop rows with NaN values
    - Clip negative resistance values to zero
    - Remove infinite values
    """
    df = df.copy()

    # Replace inf with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN/Inf values")

    # Clip negative resistances
    sensor_cols = [c for c in df.columns if c.startswith("R") and "(MOhm)" in c]
    for col in sensor_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  Clipped {neg_count} negative values in {col}")
            df[col] = df[col].clip(lower=0)

    return df.reset_index(drop=True)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: pd.Series = None,
):
    """
    Perform stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on train data and transform both train and test.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
