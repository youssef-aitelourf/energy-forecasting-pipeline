"""
Inference script
Makes predictions on new data using the trained best model
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logger, load_object
from src.preprocessing import normalize_features
from src.feature_engineering import create_temporal_features, create_lag_features
from src.config import MODELS_DIR

logger = setup_logger(__name__)


def load_trained_artifacts():
    """
    Load trained model, scaler, and feature names
    
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    logger.info("Loading trained artifacts...")
    
    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    
    if not all([model_path.exists(), scaler_path.exists(), feature_names_path.exists()]):
        logger.error("Required model artifacts not found. Please train the model first.")
        raise FileNotFoundError("Model artifacts not found. Run train_pipeline.py first.")
    
    model = load_object(model_path)
    scaler = load_object(scaler_path)
    feature_names = load_object(feature_names_path)
    
    logger.info("✓ Model loaded successfully")
    logger.info("✓ Scaler loaded successfully")
    logger.info(f"✓ Feature names loaded ({len(feature_names)} features)")
    
    return model, scaler, feature_names


def prepare_inference_data(df: pd.DataFrame, scaler, feature_names: list) -> pd.DataFrame:
    """
    Prepare new data for inference
    
    Args:
        df: Raw input DataFrame
        scaler: Fitted scaler
        feature_names: List of expected feature names
        
    Returns:
        Prepared DataFrame ready for prediction
    """
    logger.info("Preparing data for inference...")
    logger.info("Preparing data for inference...")
    
    # Create temporal features
    df = create_temporal_features(df)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Get columns that exist in both df and scaler's training features
    if hasattr(scaler, 'feature_names_in_'):
        valid_cols = [col for col in scaler.feature_names_in_ if col in df.columns]
        if valid_cols:
            logger.info(f"Scaling {len(valid_cols)} matching columns")
            # Create a temporary dataframe with only valid columns
            df_to_scale = df[valid_cols].copy()
            df_scaled = scaler.transform(df_to_scale)
            # Update original df with scaled values
            for i, col in enumerate(valid_cols):
                df[col] = df_scaled[:, i]
        else:
            logger.warning("No columns matched between input and scaler")
    
    # Select only the required features
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}. Will use zeros for missing features.")
        for feat in missing_features:
            df[feat] = 0
    
    # Ensure feature order matches training data
    df = df[feature_names]
    
    return df
def predict_energy_consumption(data: pd.DataFrame) -> np.ndarray:
    """
    Predict energy consumption for given data
    
    Args:
        data: Prepared DataFrame with features
        
    Returns:
        Array of predictions
    """
    logger.info("Making predictions...")
    
    model, scaler, feature_names = load_trained_artifacts()
    
    # Prepare data
    data_prepared = prepare_inference_data(data, scaler, feature_names)
    
    # Make predictions
    predictions = model.predict(data_prepared)
    
    logger.info(f"✓ Predictions made. Shape: {predictions.shape}")
    return predictions


def example_inference():
    """
    Example inference on synthetic data
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE INFERENCE: ENERGY CONSUMPTION PREDICTION")
    logger.info("="*80 + "\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range("2024-06-01", periods=n_samples, freq="H")
    
    sample_data = pd.DataFrame({
        'Appliances': 100 + np.random.uniform(-20, 20, n_samples),
        'lights': np.random.uniform(0, 80, n_samples),
        'T1': 20 + np.random.normal(0, 2, n_samples),
        'RH_1': 40 + np.random.normal(0, 5, n_samples),
        'T2': 19 + np.random.normal(0, 2, n_samples),
        'RH_2': 42 + np.random.normal(0, 5, n_samples),
        'T3': 18 + np.random.normal(0, 2, n_samples),
        'RH_3': 44 + np.random.normal(0, 5, n_samples),
        'T4': 17 + np.random.normal(0, 2, n_samples),
        'RH_4': 45 + np.random.normal(0, 5, n_samples),
        'T5': 19 + np.random.normal(0, 2, n_samples),
        'RH_5': 43 + np.random.normal(0, 5, n_samples),
        'T6': 18 + np.random.normal(0, 2, n_samples),
        'RH_6': 46 + np.random.normal(0, 5, n_samples),
        'T7': 16 + np.random.normal(0, 2, n_samples),
        'RH_7': 48 + np.random.normal(0, 5, n_samples),
        'T8': 20 + np.random.normal(0, 2, n_samples),
        'RH_8': 41 + np.random.normal(0, 5, n_samples),
        'T9': 19 + np.random.normal(0, 2, n_samples),
        'RH_9': 42 + np.random.normal(0, 5, n_samples),
        'T_out': 15 + np.random.normal(0, 3, n_samples),
        'Press_mm_hg': 730 + np.random.normal(0, 2, n_samples),
        'RH_out': 60 + np.random.normal(0, 5, n_samples),
        'Windspeed': 5 + np.random.normal(0, 2, n_samples),
        'Visibility': 40 + np.random.normal(0, 5, n_samples),
        'Tdewpoint': 5 + np.random.normal(0, 2, n_samples),
        'rv1': 50 + np.random.normal(0, 5, n_samples),
        'rv2': 50 + np.random.normal(0, 5, n_samples),
    })
    
    logger.info(f"Generated sample data with shape: {sample_data.shape}")
    logger.info(f"Generated sample data with 28 energy features")
    
    # Make predictions
    try:
        predictions = predict_energy_consumption(sample_data)
        
        # Add predictions to data
        sample_data['predicted_energy'] = predictions
        
        # Display results
        logger.info("\nPrediction Results (first 10 samples):")
        logger.info(sample_data[['Appliances', 'predicted_energy']].head(10).to_string(index=False))
        
        # Statistics
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Mean: {predictions.mean():.2f} Wh")
        logger.info(f"  Min: {predictions.min():.2f} Wh")
        logger.info(f"  Max: {predictions.max():.2f} Wh")
        logger.info(f"  Std: {predictions.std():.2f} Wh")
        
        logger.info("\n" + "="*80)
        logger.info("✓ Inference completed successfully!")
        logger.info("="*80 + "\n")
        
        return sample_data
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return None


def predict_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Make predictions on data from CSV file
    
    Args:
        csv_path: Path to CSV file with features
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    predictions = predict_energy_consumption(df)
    df['predicted_energy'] = predictions
    
    # Save predictions
    output_path = Path(csv_path).parent / f"{Path(csv_path).stem}_predictions.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Predictions saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Run example inference
    results = example_inference()
    
    # Uncomment to predict from CSV
    # df_with_predictions = predict_from_csv("path/to/data.csv")
