"""
Configuration file for the ML pipeline
Centralized parameters and paths
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URL = "https://archive.ics.uci.edu/ml/datasets/appliances+energy+prediction"
RAW_DATA_FILE = RAW_DATA_DIR / "energy_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "energy_data_processed.pkl"
TRAIN_TEST_SPLIT_FILE = PROCESSED_DATA_DIR / "train_test_split.pkl"

# Data preprocessing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
VALIDATION_SIZE = 0.1

# Model parameters
MODELS_CONFIG = {
    "Linear Regression": {
        "model": "LinearRegression",
        "params": {}
    },
    "Random Forest": {
        "model": "RandomForestRegressor",
        "params": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
    },
    "Gradient Boosting": {
        "model": "GradientBoostingRegressor",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": RANDOM_STATE
        }
    },
    "XGBoost": {
        "model": "XGBRegressor",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
    }
}

# Feature engineering parameters
TEMPORAL_FEATURES = True
ROLLING_WINDOW_SIZES = [3, 7, 24]  # Hours
OUTLIER_THRESHOLD = 3  # Standard deviations

# Evaluation metrics
METRICS = ["MAE", "MSE", "RMSE", "R2", "MAPE"]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs.txt"
