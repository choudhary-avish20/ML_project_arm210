# Gas Sensor Array Temperature Modulation — ML Pipeline

## Overview

This project implements **Classification** and **Regression** pipelines on the
[Gas Sensor Array Temperature Modulation](https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation)
dataset from the UCI Machine Learning Repository.

A chemical detection platform composed of **14 temperature-modulated MOX gas sensors**
was exposed to dynamic mixtures of **carbon monoxide (CO)** and humid synthetic air over
13 experiment days. The dataset contains ~3.8M time-series samples at 3.5 Hz.

## Tasks

| Task | Target | Type |
|---|---|---|
| **Classification** | CO present vs absent | Binary (CO > 0 ppm → 1) |
| **Regression** | CO concentration (ppm) | Continuous (0–20 ppm) |

## Models

### Classification
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting (HistGradientBoosting)
- SVM (RBF kernel)
- K-Nearest Neighbors

### Regression
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR (RBF kernel)

## Project Structure

```
itml/
├── datapoints/               # Raw CSV data (13 files, one per experiment day)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load & combine CSVs with subsampling
│   ├── feature_engineering.py # Derived features from sensor readings
│   ├── preprocessing.py      # Cleaning, splitting, scaling
│   ├── classification.py     # Classification models & evaluation
│   ├── regression.py         # Regression models & evaluation
│   └── visualization.py     # All plotting functions (dark theme)
├── outputs/                  # Generated plots and reports
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

## Dataset

- **Source**: [UCI ML Repository #487](https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation)
- **Features**: 20 columns (time, CO concentration, humidity, temperature, flow rate, heater voltage, 14 sensor resistances)
- **Instances**: ~3.8M rows across 13 CSV files
- **Sensors**: 7× Figaro TGS 3870-A04 + 7× FIS SB-500-12

## Citation

```
Burgués, J., Jiménez-Soto, J.M., and Marco, S. "Estimation of the limit of detection
in semiconductor gas sensors through linearized calibration models."
Analytica chimica acta 1013 (2018): 13-25.
```
# ML_project_arm210
