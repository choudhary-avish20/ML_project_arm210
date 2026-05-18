"""
data_loader.py
==============
Load and combine all 13 CSV files from the Gas Sensor Array Temperature
Modulation dataset. Supports subsampling to manage memory.
"""

import os
import glob
import pandas as pd
import numpy as np


# Column names as specified in the README
COLUMN_NAMES = [
    "Time (s)",
    "CO (ppm)",
    "Humidity (%r.h.)",
    "Temperature (C)",
    "Flow rate (mL/min)",
    "Heater voltage (V)",
    "R1 (MOhm)", "R2 (MOhm)", "R3 (MOhm)", "R4 (MOhm)",
    "R5 (MOhm)", "R6 (MOhm)", "R7 (MOhm)",
    "R8 (MOhm)", "R9 (MOhm)", "R10 (MOhm)", "R11 (MOhm)",
    "R12 (MOhm)", "R13 (MOhm)", "R14 (MOhm)",
]

SENSOR_COLS = [f"R{i} (MOhm)" for i in range(1, 15)]
FIGARO_COLS = [f"R{i} (MOhm)" for i in range(1, 8)]   # TGS 3870 A-04
FIS_COLS    = [f"R{i} (MOhm)" for i in range(8, 15)]   # SB-500-12


def load_all_data(data_dir: str = "datapoints", subsample_step: int = 10) -> pd.DataFrame:
    """
    Load all CSV files from *data_dir*, concatenate, and subsample.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the CSV files.
    subsample_step : int
        Keep every Nth row to reduce dataset size. Set to 1 for full data.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with an added 'Day' column indicating the
        experiment day (1-13).
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for day_idx, fpath in enumerate(csv_files, start=1):
        print(f"  Loading day {day_idx}/13: {os.path.basename(fpath)} ...")
        df = pd.read_csv(fpath)
        df.columns = [c.strip() for c in df.columns]
        df["Day"] = day_idx
        # Subsample
        if subsample_step > 1:
            df = df.iloc[::subsample_step].reset_index(drop=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total rows after subsampling (step={subsample_step}): {len(combined):,}")
    return combined


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict with basic dataset statistics."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "co_range": (df["CO (ppm)"].min(), df["CO (ppm)"].max()),
        "humidity_range": (df["Humidity (%r.h.)"].min(), df["Humidity (%r.h.)"].max()),
        "days": sorted(df["Day"].unique()),
        "rows_per_day": df.groupby("Day").size().to_dict(),
    }
