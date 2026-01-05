"""
Complete ML pipeline orchestration script
Handles the entire workflow from data loading to model evaluation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion import load_data, explore_data
from src.preprocessing import prepare_data, split_data
from src.feature_engineering import create_temporal_features, create_lag_features
from src.model_training import train_all_models, evaluate_models, select_best_model
from src.evaluation import create_evaluation_report, plot_all_metrics, plot_feature_importance, plot_predictions_vs_actual
from src.utils import setup_logger, save_object, load_object

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = setup_logger(__name__)


def main():
    """
    Execute the complete ML pipeline
    """
    
    logger.info("\n")
    logger.info("#" * 100)
    logger.info("# " + "ML END-TO-END PIPELINE FOR ENERGY CONSUMPTION FORECASTING".center(96) + " #")
    logger.info("#" * 100)
    logger.info("\n")
    
    # ============ STEP 1: DATA INGESTION ============
    logger.info("STEP 1: DATA INGESTION")
    logger.info("-" * 100)
    df = load_data()
    explore_data(df)
    
    # ============ STEP 2: PREPROCESSING ============
    logger.info("\nSTEP 2: DATA PREPROCESSING")
    logger.info("-" * 100)
    df_processed, metadata = prepare_data(df)
    
    # ============ STEP 3: FEATURE ENGINEERING ============
    logger.info("\nSTEP 3: FEATURE ENGINEERING")
    logger.info("-" * 100)
    df_features = create_temporal_features(df_processed)
    
    # Create lag features for target variable (Appliances)
    df_features = create_lag_features(df_features, column="Appliances", lags=[1, 2, 3, 24])
    
    logger.info(f"Final dataset shape: {df_features.shape}")
    logger.info(f"Features: {list(df_features.columns)}")
    
    # ============ STEP 4: DATA SPLITTING ============
    logger.info("\nSTEP 4: DATA SPLITTING")
    logger.info("-" * 100)
    split_dict = split_data(df_features, target_column="Appliances")
    
    # ============ STEP 5: MODEL TRAINING ============
    logger.info("\nSTEP 5: MODEL TRAINING")
    logger.info("-" * 100)
    trained_models = train_all_models(split_dict["X_train"], split_dict["y_train"])
    
    # ============ STEP 6: MODEL EVALUATION ============
    logger.info("\nSTEP 6: MODEL EVALUATION")
    logger.info("-" * 100)
    results = evaluate_models(
        trained_models,
        split_dict["X_train"], split_dict["y_train"],
        split_dict["X_val"], split_dict["y_val"],
        split_dict["X_test"], split_dict["y_test"]
    )
    
    # ============ STEP 7: SELECT BEST MODEL ============
    logger.info("\nSTEP 7: SELECT BEST MODEL")
    logger.info("-" * 100)
    best_model_name, best_model = select_best_model(results)
    
    # ============ STEP 8: CREATE EVALUATION REPORT ============
    logger.info("\nSTEP 8: CREATE EVALUATION REPORT")
    logger.info("-" * 100)
    report = create_evaluation_report(results)
    
    # Save report
    report_path = PROCESSED_DATA_DIR / "evaluation_report.csv"
    report.to_csv(report_path, index=False)
    logger.info(f"Saved evaluation report to {report_path}")
    
    # ============ STEP 9: SAVE ARTIFACTS ============
    logger.info("\nSTEP 9: SAVE ARTIFACTS")
    logger.info("-" * 100)
    
    # Save best model
    best_model_path = MODELS_DIR / "best_model.pkl"
    save_object(best_model, best_model_path)
    logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
    
    # Save scaler
    scaler = metadata["scaler"]
    scaler_path = MODELS_DIR / "scaler.pkl"
    save_object(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    feature_names = split_dict["X_train"].columns.tolist()
    save_object(feature_names, feature_names_path)
    logger.info(f"Saved feature names to {feature_names_path}")
    
    # Save split info (for later analysis)
    split_info_path = PROCESSED_DATA_DIR / "split_info.pkl"
    save_object(split_dict, split_info_path)
    logger.info(f"Saved split data to {split_info_path}")
    
    # ============ STEP 10: VISUALIZATIONS ============
    logger.info("\nSTEP 10: VISUALIZATIONS")
    logger.info("-" * 100)
    
    # Create visualizations directory
    viz_dir = Path(__file__).parent / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    logger.info("Creating visualizations...")
    
    # Predictions vs Actual for best model
    y_test_pred = best_model.predict(split_dict["X_test"])
    plot_predictions_vs_actual(
        split_dict["y_test"].values, y_test_pred, 
        best_model_name, "Test",
        save_path=viz_dir / f"{best_model_name}_predictions.png"
    )
    
    # Feature importance (if model supports it)
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(
            best_model, feature_names, best_model_name, top_n=20,
            save_path=viz_dir / f"{best_model_name}_feature_importance.png"
        )
    
    logger.info(f"Visualizations saved to {viz_dir}")
    
    # ============ FINAL SUMMARY ============
    logger.info("\n" + "#" * 100)
    logger.info("# " + "PIPELINE COMPLETED SUCCESSFULLY".center(96) + " #")
    logger.info("#" * 100)
    logger.info(f"\n✓ Best Model: {best_model_name}")
    logger.info(f"✓ Test R²: {results[best_model_name]['test']['R2']}")
    logger.info(f"✓ Test RMSE: {results[best_model_name]['test']['RMSE']}")
    logger.info(f"✓ Test MAE: {results[best_model_name]['test']['MAE']}")
    logger.info(f"\n✓ All artifacts saved in {MODELS_DIR}")
    logger.info(f"✓ Reports saved in {PROCESSED_DATA_DIR}")
    logger.info(f"✓ Visualizations saved in {viz_dir}\n")
    
    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "results": results,
        "report": report,
        "split_dict": split_dict,
        "metadata": metadata,
        "feature_names": feature_names
    }


if __name__ == "__main__":
    pipeline_output = main()
