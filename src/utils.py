"""
Utility functions for the ML pipeline
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from .config import LOG_LEVEL, LOG_FILE


def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Create handlers
    file_handler = logging.FileHandler(LOG_FILE)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_object(obj: Any, filepath: Path) -> None:
    """
    Save Python object using pickle
    
    Args:
        obj: Object to save
        filepath: Destination path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filepath: Path) -> Any:
    """
    Load Python object from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE with handling for zero values
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4)
    }


def print_metrics(metrics: dict, model_name: str) -> None:
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:.<30} {value}")
    print(f"{'='*50}\n")


def remove_outliers(data: pd.DataFrame, columns: list, threshold: float = 3) -> pd.DataFrame:
    """
    Remove outliers using Z-score method
    
    Args:
        data: Input dataframe
        columns: Columns to check for outliers
        threshold: Z-score threshold (default 3)
        
    Returns:
        DataFrame without outliers
    """
    z_scores = np.abs((data[columns] - data[columns].mean()) / data[columns].std())
    return data[(z_scores < threshold).all(axis=1)]
