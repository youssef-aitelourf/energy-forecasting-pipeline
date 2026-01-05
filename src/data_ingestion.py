"""
Data ingestion module
Loads and explores the energy consumption dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import urllib.request
import os

from .config import RAW_DATA_FILE, RAW_DATA_DIR
from .utils import setup_logger

logger = setup_logger(__name__)


def download_energy_data() -> pd.DataFrame:
    """
    Download the energy consumption dataset from UCI ML Repository
    If download fails, generates synthetic data
    
    Returns:
        DataFrame with energy consumption data
    """
    try:
        # Try to download from UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
        
        logger.info(f"Attempting to download data from: {url}")
        
        # Create the raw data file if it doesn't exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        if not RAW_DATA_FILE.exists():
            urllib.request.urlretrieve(url, RAW_DATA_FILE)
            logger.info(f"Data downloaded successfully to {RAW_DATA_FILE}")
        else:
            logger.info(f"Data already exists at {RAW_DATA_FILE}")
            
        df = pd.read_csv(RAW_DATA_FILE)
        return df
        
    except Exception as e:
        logger.warning(f"Download failed: {e}. Generating synthetic data...")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples: int = 8760) -> pd.DataFrame:
    """
    Generate synthetic energy consumption data
    Simulates hourly data for 1 year
    
    Args:
        n_samples: Number of samples (default: 8760 for 1 year hourly)
        
    Returns:
        DataFrame with synthetic energy data
    """
    np.random.seed(42)
    
    # Time index
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="H")
    hour_of_day = dates.hour.values
    day_of_week = dates.dayofweek.values
    month = dates.month.values
    
    # Energy consumption with patterns
    base_energy = 100
    
    # Daily pattern (higher during day/evening)
    daily_pattern = 50 * np.sin(2 * np.pi * hour_of_day / 24)
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = 20 * (day_of_week < 5).astype(float)
    
    # Seasonal pattern (higher in winter and summer)
    seasonal_pattern = 30 * np.cos(2 * np.pi * month / 12)
    
    # Noise
    noise = np.random.normal(0, 15, n_samples)
    
    # Total energy consumption
    energy = base_energy + daily_pattern + weekly_pattern + seasonal_pattern + noise
    energy = np.maximum(energy, 10)  # Ensure positive values
    
    # Create dataframe (NO Datetime column - keep data purely numerical)
    df = pd.DataFrame({
        'Appliances': energy,
        'lights': np.random.uniform(0, 80, n_samples),
        'T1': 20 + 5 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_1': 40 + 20 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T2': 19 + 5 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_2': 42 + 18 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T3': 18 + 6 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_3': 44 + 16 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T4': 17 + 7 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_4': 45 + 15 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T5': 19 + 5 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_5': 43 + 17 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T6': 18 + 6 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_6': 46 + 14 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T7': 16 + 8 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_7': 48 + 12 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T8': 20 + 4 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_8': 41 + 19 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
        'T9': 19 + 5 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 1, n_samples),
        'RH_9': 42 + 18 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5, n_samples),
    })
    
    # Clamp humidity values between 0 and 100
    humidity_cols = [col for col in df.columns if col.startswith('RH_')]
    for col in humidity_cols:
        df[col] = df[col].clip(0, 100)
    
    logger.info(f"Generated synthetic data with {len(df)} samples")
    return df


def load_data() -> pd.DataFrame:
    """
    Load energy consumption data
    
    Returns:
        DataFrame with energy data (purely numeric)
    """
    logger.info("Loading data...")
    
    try:
        if RAW_DATA_FILE.exists():
            df = pd.read_csv(RAW_DATA_FILE)
            logger.info(f"Loaded data from {RAW_DATA_FILE}")
        else:
            df = download_energy_data()
            df.to_csv(RAW_DATA_FILE, index=False)
            
    except Exception as e:
        logger.warning(f"Error loading data: {e}. Generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv(RAW_DATA_FILE, index=False)
    
    # Drop date/time columns to keep data purely numeric for ML
    date_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'time']]
    if date_cols:
        logger.info(f"Dropping non-numeric columns: {date_cols}")
        df = df.drop(columns=date_cols)
    
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Explore and display data statistics
    
    Args:
        df: Input DataFrame
    """
    logger.info("\n--- Dataset Overview ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nFirst few rows:\n{df.head()}")
    logger.info(f"\nDataset Info:\n{df.dtypes}")
    logger.info(f"\nBasic Statistics:\n{df.describe()}")
    logger.info(f"\nMissing Values:\n{df.isnull().sum()}")
