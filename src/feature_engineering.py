"""
feature_engineering.py
======================
Derive meaningful features from raw sensor readings and create
target variables for classification and regression.
"""

import pandas as pd
import numpy as np
from .data_loader import SENSOR_COLS, FIGARO_COLS, FIS_COLS


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame.

    New columns created:
        - R_mean          : Mean resistance across all 14 sensors
        - R_std           : Std dev across all 14 sensors
        - R_figaro_mean   : Mean of Figaro sensors (R1–R7)
        - R_fis_mean      : Mean of FIS sensors (R8–R14)
        - R_ratio         : Figaro mean / FIS mean
        - R_min           : Min resistance across all 14 sensors
        - R_max           : Max resistance across all 14 sensors
        - R_range         : R_max - R_min
        - Heater_high     : Binary flag (1 if heater voltage > 0.5 V)
    """
    df = df.copy()

    # Aggregate sensor statistics
    sensor_vals = df[SENSOR_COLS].values
    df["R_mean"] = sensor_vals.mean(axis=1)
    df["R_std"] = sensor_vals.std(axis=1)
    df["R_min"] = sensor_vals.min(axis=1)
    df["R_max"] = sensor_vals.max(axis=1)
    df["R_range"] = df["R_max"] - df["R_min"]

    # Sensor group means
    df["R_figaro_mean"] = df[FIGARO_COLS].mean(axis=1)
    df["R_fis_mean"] = df[FIS_COLS].mean(axis=1)

    # Ratio (add small epsilon to avoid division by zero)
    df["R_ratio"] = df["R_figaro_mean"] / (df["R_fis_mean"] + 1e-9)

    # Heater state (high voltage phase vs low voltage phase)
    df["Heater_high"] = (df["Heater voltage (V)"] > 0.5).astype(int)

    return df


def create_classification_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """
    Create a binary classification target: CO present (1) vs absent (0).

    Parameters
    ----------
    threshold : float
        CO concentration threshold in ppm. Values > threshold are labeled 1.
    """
    return (df["CO (ppm)"] > threshold).astype(int)


def create_co_concentration_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    """
    Bin CO concentrations for stratified splitting.
    Returns bin labels as integers.
    """
    return pd.cut(df["CO (ppm)"], bins=n_bins, labels=False, duplicates="drop")


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns to use for modelling.
    Excludes target, time, and metadata columns.
    """
    exclude = {
        "Time (s)",
        "CO (ppm)",
        "Flow rate (mL/min)",  # Constant (~240 mL/min)
        "Day",
    }
    return [c for c in df.columns if c not in exclude]
