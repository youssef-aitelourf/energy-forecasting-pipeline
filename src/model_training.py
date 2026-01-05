"""
Model training module
Trains multiple models and compares their performance
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import warnings

from .config import MODELS_CONFIG, RANDOM_STATE
from .utils import setup_logger, calculate_metrics, print_metrics

logger = setup_logger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train Linear Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    logger.info("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression trained successfully")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, **params) -> RandomForestRegressor:
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        **params: Additional hyperparameters
        
    Returns:
        Trained model
    """
    logger.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("Random Forest trained successfully")
    return model


def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, **params) -> GradientBoostingRegressor:
    """
    Train Gradient Boosting model
    
    Args:
        X_train: Training features
        y_train: Training target
        **params: Additional hyperparameters
        
    Returns:
        Trained model
    """
    logger.info("Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting trained successfully")
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, **params) -> XGBRegressor:
    """
    Train XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        **params: Additional hyperparameters
        
    Returns:
        Trained model
    """
    logger.info("Training XGBoost Regressor...")
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("XGBoost trained successfully")
    return model


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train all models specified in config
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dictionary with trained models and their names
    """
    logger.info("\n" + "="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80 + "\n")
    
    trained_models = {}
    
    for model_name, config in MODELS_CONFIG.items():
        try:
            if model_name == "Linear Regression":
                model = train_linear_regression(X_train, y_train)
            elif model_name == "Random Forest":
                model = train_random_forest(X_train, y_train, **config["params"])
            elif model_name == "Gradient Boosting":
                model = train_gradient_boosting(X_train, y_train, **config["params"])
            elif model_name == "XGBoost":
                model = train_xgboost(X_train, y_train, **config["params"])
            else:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            trained_models[model_name] = model
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    logger.info(f"\nSuccessfully trained {len(trained_models)} models")
    logger.info("="*80 + "\n")
    
    return trained_models


def make_predictions(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a trained model
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        Array of predictions
    """
    return model.predict(X)


def evaluate_models(trained_models: Dict[str, Any], 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Dict]]:
    """
    Evaluate all trained models
    
    Args:
        trained_models: Dictionary of trained models
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with evaluation results for each model
    """
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80 + "\n")
    
    results = {}
    
    for model_name, model in trained_models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Predictions on all sets
        y_train_pred = make_predictions(model, X_train)
        y_val_pred = make_predictions(model, X_val)
        y_test_pred = make_predictions(model, X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        results[model_name] = {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "model": model
        }
        
        print_metrics(train_metrics, f"{model_name} (Train)")
        print_metrics(val_metrics, f"{model_name} (Validation)")
        print_metrics(test_metrics, f"{model_name} (Test)")
    
    logger.info("="*80 + "\n")
    return results


def select_best_model(results: Dict[str, Dict[str, Dict]]) -> Tuple[str, Any]:
    """
    Select the best model based on test R2 score
    
    Args:
        results: Evaluation results for all models
        
    Returns:
        Tuple of best model name and model object
    """
    logger.info("\n" + "="*80)
    logger.info("BEST MODEL SELECTION")
    logger.info("="*80)
    
    best_model_name = None
    best_r2 = -np.inf
    
    for model_name, metrics_dict in results.items():
        test_r2 = metrics_dict["test"]["R2"]
        logger.info(f"{model_name}: Test R² = {test_r2}")
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_name = model_name
    
    best_model = results[best_model_name]["model"]
    
    logger.info(f"\n✓ Best model: {best_model_name} (R² = {best_r2})")
    logger.info("="*80 + "\n")
    
    return best_model_name, best_model


if __name__ == "__main__":
    from .data_ingestion import load_data
    from .preprocessing import prepare_data, split_data
    from .feature_engineering import create_temporal_features
    
    df = load_data()
    df_processed, metadata = prepare_data(df)
    df_features = create_temporal_features(df_processed)
    split_dict = split_data(df_features)
    
    trained_models = train_all_models(split_dict["X_train"], split_dict["y_train"])
    results = evaluate_models(
        trained_models,
        split_dict["X_train"], split_dict["y_train"],
        split_dict["X_val"], split_dict["y_val"],
        split_dict["X_test"], split_dict["y_test"]
    )
    
    best_model_name, best_model = select_best_model(results)
