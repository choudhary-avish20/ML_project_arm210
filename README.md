# Gas Sensor Array Temperature Modulation — ML Pipeline

## Overview

This project implements **Classification** and **Regression** pipelines on the
[Gas Sensor Array Temperature Modulation](https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation)
dataset from the UCI Machine Learning Repository.

A chemical detection platform composed of **14 temperature-modulated MOX gas sensors**
was exposed to dynamic mixtures of **carbon monoxide (CO)** and humid synthetic air over
13 experiment days. The dataset contains approximately 3.8M time-series samples at 3.5 Hz.

## Tasks

| Task | Target | Type |
| --- | --- | --- |
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
│   ├── data_loader.py        # Load & combine CSVs with sub-sampling
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
- **Features**: 20 columns (time, CO concentration, humidity, temperature, flow rate, heater voltage, and 14 sensor resistances)
- **Instances**: approximately 3.8M rows across 13 CSV files
- **Sensors**: 7× Figaro TGS 3870-A04 and 7× FIS SB-500-12

## Summary Report

| Metric | Value |
| --- | --- |
| Dataset shape | (384321, 21) |
| CO range | (0.0, 20.0) ppm |
| Total runtime | 443.2s |

### Classification Results

| Model | Acc | Prec | Rec | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9379 | 0.9492 | 0.9829 | 0.9658 | 0.9296 |
| Random Forest | 0.9736 | 0.9792 | 0.9914 | 0.9853 | 0.9870 |
| Gradient Boosting | 0.9745 | 0.9804 | 0.9913 | 0.9858 | 0.9869 |
| SVM (RBF) | 0.9424 | 0.9502 | 0.9871 | 0.9683 | 0.9238 |
| KNN | 0.9439 | 0.9556 | 0.9828 | 0.9690 | 0.9201 |

### Regression Results

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Linear Regression | 2.9054 | 4.1161 | 0.5893 |
| Ridge Regression | 2.9054 | 4.1161 | 0.5893 |
| Random Forest | 0.8798 | 1.9012 | 0.9124 |
| Gradient Boosting | 1.1679 | 2.0902 | 0.8941 |
| SVR (RBF) | 2.2397 | 3.7389 | 0.6611 |