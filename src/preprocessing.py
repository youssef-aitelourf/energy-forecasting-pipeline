import warnings
"""
Preprocessing module
Handles data cleaning, normalization, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

from .config import (
    TEST_SIZE, RANDOM_STATE, VALIDATION_SIZE, 
    PROCESSED_DATA_FILE, TRAIN_TEST_SPLIT_FILE,
    OUTLIER_THRESHOLD
)
from .utils import setup_logger, save_object, remove_outliers

logger = setup_logger(__name__)


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('mean', 'forward_fill', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    logger.info(f"Handling missing values using '{strategy}' strategy...")
    
    missing_before = df.isnull().sum().sum()
    
    if strategy == "mean":
        df = df.fillna(df.mean())
    elif strategy == "forward_fill":
        df = df.fillna(method='ffill').fillna(df.mean())
    elif strategy == "drop":
        df = df.dropna()
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values before: {missing_before}, after: {missing_after}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame without duplicates
    """
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_after = len(df)
    
    logger.info(f"Removed {duplicates_before - duplicates_after} duplicate rows")
    return df


def handle_outliers(df: pd.DataFrame, columns: list = None, threshold: float = OUTLIER_THRESHOLD) -> pd.DataFrame:
    """
    Remove outliers using Z-score method
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: all numeric)
        threshold: Z-score threshold
        
    Returns:
        DataFrame without outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Removing outliers with Z-score threshold: {threshold}")
    rows_before = len(df)
    
    df = remove_outliers(df, columns, threshold)
    
    rows_after = len(df)
    logger.info(f"Removed {rows_before - rows_after} outlier rows")
    
    return df


def normalize_features(df: pd.DataFrame, scaler: StandardScaler = None, 
                      fit: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler
    
    Args:
        df: Input DataFrame
        scaler: Existing scaler object (for inference)
        fit: Whether to fit the scaler (True for train, False for test)
        
    Returns:
        Tuple of normalized DataFrame and scaler object
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        logger.info("Fitting scaler on training data...")
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        logger.info("Applying scaler on test data...")
        df[numeric_columns] = scaler.transform(df[numeric_columns])
    
    return df, scaler


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of processed DataFrame and preprocessing metadata
    """
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("="*80)
    
    original_shape = df.shape
    logger.info(f"Original shape: {original_shape}")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df, strategy="mean")
    
    # Step 2: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 3: Handle outliers
    df = handle_outliers(df, threshold=OUTLIER_THRESHOLD)
    
    # Step 4: Normalize features
    df, scaler = normalize_features(df, fit=True)
    
    logger.info(f"Processed shape: {df.shape}")
    logger.info("="*80 + "\n")
    
    metadata = {
        "original_shape": original_shape,
        "processed_shape": df.shape,
        "scaler": scaler
    }
    
    return df, metadata


def split_data(df: pd.DataFrame, target_column: str = "Appliances") -> Dict[str, Any]:
    """
    Split data into train and test sets
    
    Args:
        df: Processed DataFrame
        target_column: Name of target column
        
    Returns:
        Dictionary with train/test split information
    """
    logger.info("\n" + "="*80)
    logger.info("DATA SPLITTING")
    logger.info("="*80)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Train set size: {len(X_train)} samples ({(1-TEST_SIZE)*100:.0f}%)")
    logger.info(f"Test set size: {len(X_test)} samples ({TEST_SIZE*100:.0f}%)")
    
    # Further split train into train/validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Train final size: {len(X_train_final)} samples")
    logger.info(f"Validation size: {len(X_val)} samples")
    logger.info("="*80 + "\n")
    
    split_data = {
        "X_train": X_train_final,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_final,
        "y_val": y_val,
        "y_test": y_test
    }
    
    return split_data


if __name__ == "__main__":
    from .data_ingestion import load_data
    
    df = load_data()
    df_processed, metadata = prepare_data(df)
    split_dict = split_data(df_processed)
    
    print("\nPreprocessing completed successfully!")
