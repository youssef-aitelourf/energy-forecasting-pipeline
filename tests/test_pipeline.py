"""
Unit tests for the ML pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_ingestion import load_data, generate_synthetic_data
from preprocessing import prepare_data, split_data
from feature_engineering import create_temporal_features
from model_training import train_all_models
from utils import calculate_metrics


def test_data_loading():
    """Test data loading and basic validation"""
    print("Testing data loading...")
    df = load_data()
    
    assert df is not None, "Data loading failed"
    assert len(df) > 0, "No data loaded"
    assert df.shape[1] > 0, "No features in data"
    
    print(f"✓ Data loaded successfully: {df.shape}")


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("Testing synthetic data generation...")
    df = generate_synthetic_data(n_samples=100)
    
    assert df is not None, "Synthetic data generation failed"
    assert len(df) == 100, f"Expected 100 samples, got {len(df)}"
    assert 'Appliances' in df.columns, "Target column missing"
    
    print(f"✓ Synthetic data generated: {df.shape}")


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("Testing preprocessing...")
    df = generate_synthetic_data(n_samples=500)
    df_processed, metadata = prepare_data(df)
    
    assert df_processed is not None, "Preprocessing failed"
    assert len(df_processed) > 0, "No data after preprocessing"
    assert df_processed.isnull().sum().sum() == 0, "Still has missing values"
    
    print(f"✓ Preprocessing completed: {df_processed.shape}")


def test_feature_engineering():
    """Test feature engineering"""
    print("Testing feature engineering...")
    df = generate_synthetic_data(n_samples=500)
    df_processed, _ = prepare_data(df)
    df_features = create_temporal_features(df_processed)
    
    assert df_features is not None, "Feature engineering failed"
    assert df_features.shape[1] > df_processed.shape[1], "No new features created"
    
    print(f"✓ Features engineered: {df_features.shape}")


def test_data_splitting():
    """Test train/test splitting"""
    print("Testing data splitting...")
    df = generate_synthetic_data(n_samples=500)
    df_processed, _ = prepare_data(df)
    split_dict = split_data(df_processed)
    
    assert 'X_train' in split_dict, "Missing X_train"
    assert 'y_train' in split_dict, "Missing y_train"
    assert 'X_test' in split_dict, "Missing X_test"
    assert 'y_test' in split_dict, "Missing y_test"
    
    total = len(split_dict['X_train']) + len(split_dict['X_val']) + len(split_dict['X_test'])
    assert total == len(df_processed), "Data lost during splitting"
    
    print(f"✓ Data split correctly")
    print(f"  - Train: {len(split_dict['X_train'])}")
    print(f"  - Val: {len(split_dict['X_val'])}")
    print(f"  - Test: {len(split_dict['X_test'])}")


def test_model_training():
    """Test model training"""
    print("Testing model training...")
    df = generate_synthetic_data(n_samples=500)
    df_processed, _ = prepare_data(df)
    split_dict = split_data(df_processed)
    
    trained_models = train_all_models(split_dict["X_train"], split_dict["y_train"])
    
    assert len(trained_models) > 0, "No models trained"
    assert all(hasattr(m, 'predict') for m in trained_models.values()), "Models don't have predict method"
    
    print(f"✓ Models trained: {list(trained_models.keys())}")


def test_metrics_calculation():
    """Test metrics calculation"""
    print("Testing metrics calculation...")
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'MAE' in metrics, "MAE not calculated"
    assert 'RMSE' in metrics, "RMSE not calculated"
    assert 'R2' in metrics, "R2 not calculated"
    assert metrics['MAE'] > 0, "Invalid MAE"
    
    print(f"✓ Metrics calculated:")
    for name, value in metrics.items():
        print(f"  - {name}: {value}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RUNNING PIPELINE TESTS")
    print("="*80 + "\n")
    
    tests = [
        test_data_loading,
        test_synthetic_data_generation,
        test_preprocessing,
        test_feature_engineering,
        test_data_splitting,
        test_model_training,
        test_metrics_calculation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ FAILED: {str(e)}\n")
            failed += 1
    
    print("="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
