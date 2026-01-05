"""
Evaluation module
Visualizes and analyzes model performance
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .utils import setup_logger

logger = setup_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, dataset_name: str = "Test",
                               save_path: Path = None) -> None:
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        dataset_name: Name of the dataset (Train/Val/Test)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Actual vs Predicted ({dataset_name} Set)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Residual Plot ({dataset_name} Set)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_metrics_comparison(results: Dict[str, Dict[str, Dict]], 
                            metric_name: str = "R2",
                            save_path: Path = None) -> None:
    """
    Compare metrics across all models
    
    Args:
        results: Dictionary with evaluation results
        metric_name: Metric to compare
        save_path: Path to save the plot
    """
    models = list(results.keys())
    train_scores = [results[m]["train"][metric_name] for m in models]
    val_scores = [results[m]["validation"][metric_name] for m in models]
    test_scores = [results[m]["test"][metric_name] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, train_scores, width, label='Train', alpha=0.8)
    ax.bar(x, val_scores, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_scores, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison Across Models', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_all_metrics(results: Dict[str, Dict[str, Dict]], 
                    save_dir: Path = None) -> None:
    """
    Plot all metrics for all models
    
    Args:
        results: Dictionary with evaluation results
        save_dir: Directory to save plots
    """
    metrics = list(results[list(results.keys())[0]]["test"].keys())
    
    for metric in metrics:
        plot_metrics_comparison(results, metric_name=metric, 
                               save_path=save_dir / f"{metric}_comparison.png" if save_dir else None)


def plot_feature_importance(model: Any, feature_names: list, 
                            model_name: str = "Model",
                            top_n: int = 20,
                            save_path: Path = None) -> None:
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"{model_name} does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.barh(range(top_n), importances[indices], alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "Model",
                           save_path: Path = None) -> None:
    """
    Plot error distribution
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the plot
    """
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Error Distribution - {model_name}', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot - {model_name}', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def create_evaluation_report(results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """
    Create comprehensive evaluation report
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        DataFrame with comprehensive results
    """
    report_data = []
    
    for model_name, metrics_dict in results.items():
        for dataset in ["train", "validation", "test"]:
            metrics = metrics_dict[dataset]
            row = {
                "Model": model_name,
                "Dataset": dataset.capitalize(),
                **metrics
            }
            report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    
    logger.info("\n" + "="*100)
    logger.info("COMPREHENSIVE EVALUATION REPORT")
    logger.info("="*100)
    logger.info(report_df.to_string(index=False))
    logger.info("="*100 + "\n")
    
    return report_df


if __name__ == "__main__":
    from .data_ingestion import load_data
    from .preprocessing import prepare_data, split_data
    from .feature_engineering import create_temporal_features
    from .model_training import train_all_models, evaluate_models, select_best_model
    
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
    
    report = create_evaluation_report(results)
