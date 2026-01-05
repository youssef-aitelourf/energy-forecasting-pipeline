"""
Feature engineering module
Creates new features and selects relevant ones
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Tuple, List

from .config import ROLLING_WINDOW_SIZES, TEMPORAL_FEATURES
from .utils import setup_logger

logger = setup_logger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features if datetime column exists
    
    Args:
        df: Input DataFrame (should have Datetime or index as datetime)
        
    Returns:
        DataFrame with temporal features
    """
    if not TEMPORAL_FEATURES:
        return df
    
    logger.info("Creating temporal features...")
    
    # Try to extract from Datetime column or index
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        datetime_col = df['Datetime']
    elif isinstance(df.index, pd.DatetimeIndex):
        datetime_col = df.index
    else:
        logger.warning("No datetime information found. Skipping temporal features.")
        return df
    
    # Create features
    df['hour'] = datetime_col.dt.hour
    df['day_of_week'] = datetime_col.dt.dayofweek
    df['day_of_month'] = datetime_col.dt.day
    df['month'] = datetime_col.dt.month
    df['quarter'] = datetime_col.dt.quarter
    df['is_weekend'] = (datetime_col.dt.dayofweek >= 5).astype(int)
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    logger.info(f"Created {11} temporal features")
    return df


def create_rolling_features(df: pd.DataFrame, columns: List[str] = None, 
                           window_sizes: List[int] = None) -> pd.DataFrame:
    """
    Create rolling window features (lag-based)
    
    Args:
        df: Input DataFrame
        columns: Columns to create rolling features for
        window_sizes: Window sizes for rolling statistics
        
    Returns:
        DataFrame with rolling features
    """
    if window_sizes is None:
        window_sizes = ROLLING_WINDOW_SIZES
    
    if columns is None:
        # Select numeric columns, exclude temporal features
        temporal_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
                        'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                  if col not in temporal_cols]
    
    logger.info(f"Creating rolling features with window sizes: {window_sizes}")
    
    rolling_features_count = 0
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in window_sizes:
            # Mean
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            # Std
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
            rolling_features_count += 2
    
    # Drop NaN rows created by rolling windows
    df = df.dropna()
    
    logger.info(f"Created {rolling_features_count} rolling features")
    return df


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int] = None) -> pd.DataFrame:
    """
    Create lag features for time-series data
    
    Args:
        df: Input DataFrame
        column: Column to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    if lags is None:
        lags = [1, 2, 3, 24]  # 1h, 2h, 3h, 1day lags
    
    logger.info(f"Creating lag features for {column}: {lags}")
    
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def select_features(X_train: pd.DataFrame, y_train: pd.Series, 
                   n_features: int = None, method: str = "f_regression") -> Tuple[pd.DataFrame, List[str]]:
    """
    Select most important features
    
    Args:
        X_train: Training features
        y_train: Training target
        n_features: Number of features to select (default: sqrt of total)
        method: Selection method ('f_regression' or 'mutual_info')
        
    Returns:
        Tuple of selected features DataFrame and list of selected feature names
    """
    if n_features is None:
        n_features = int(np.sqrt(X_train.shape[1]))
    
    logger.info(f"\nSelecting top {n_features} features using {method}...")
    
    if method == "f_regression":
        selector = SelectKBest(f_regression, k=min(n_features, X_train.shape[1]))
    else:
        selector = SelectKBest(mutual_info_regression, k=min(n_features, X_train.shape[1]))
    
    X_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    logger.info(f"Selected features: {selected_features}")
    
    return pd.DataFrame(X_selected, columns=selected_features), selected_features


def feature_importance_analysis(feature_names: List[str], importance_scores: np.ndarray) -> pd.DataFrame:
    """
    Analyze and display feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        
    Returns:
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 10 important features:")
    logger.info(importance_df.head(10).to_string(index=False))
    
    return importance_df


if __name__ == "__main__":
    from .data_ingestion import load_data
    from .preprocessing import prepare_data, split_data
    
    df = load_data()
    df_processed, metadata = prepare_data(df)
    df_features = create_temporal_features(df_processed)
    
    print(f"\nDataFrame with features:\n{df_features.head()}")
    print(f"\nFeature columns: {df_features.columns.tolist()}")
