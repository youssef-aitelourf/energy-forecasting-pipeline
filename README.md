# 🔋 Energy Forecasting Pipeline: Advanced ML for Consumption Prediction

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-success)]()

*A professional machine learning pipeline demonstrating production-grade ML engineering practices*

</div>

---

## 📋 Overview

This project showcases a **complete, production-ready ML pipeline** for energy consumption forecasting. It demonstrates:

- ✅ **Professional code structure** with modularized components
- ✅ **End-to-end ML workflow** from data ingestion to inference
- ✅ **Multiple models** trained and evaluated systematically
- ✅ **Comprehensive feature engineering** with temporal features and lag variables
- ✅ **Rigorous evaluation** with detailed metrics and visualizations
- ✅ **Model versioning** and artifact management
- ✅ **Inference pipeline** for production predictions

### Target Use Case

**Energy Consumption Forecasting** for:
- 🏢 Smart grid optimization
- 💡 Peak demand prediction
- 📊 Load balancing planning
- 🌍 Renewable energy integration
- 💰 Cost optimization

---

## 🎯 Key Challenges Addressed

| Challenge | Solution |
|-----------|----------|
| **Data Quality** | Outlier detection, missing value imputation, normalization |
| **Temporal Dependencies** | Rolling features, lag features, cyclical encoding |
| **Model Selection** | Multiple models trained, systematic comparison |
| **Overfitting** | Train/Validation/Test split, early stopping |
| **Production Deployment** | Model serialization, inference pipeline, scalability |
| **Reproducibility** | Fixed random state, configuration management |

---

## 🏗️ Architecture

### Project Structure

```
energy-forecasting-pipeline/
├── 📂 data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Preprocessed data & artifacts
│
├── 📂 src/                     # Core ML modules
│   ├── __init__.py
│   ├── config.py              # Configuration & hyperparameters
│   ├── data_ingestion.py      # Data loading & exploration
│   ├── preprocessing.py       # Cleaning & normalization
│   ├── feature_engineering.py # Feature creation & selection
│   ├── model_training.py      # Model training
│   ├── evaluation.py          # Evaluation & visualization
│   └── utils.py               # Utility functions
│
├── 📂 models/                  # Serialized models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── 📂 scripts/
│   ├── train_pipeline.py      # Main training orchestration
│   └── inference.py           # Prediction pipeline
│
├── 📂 notebooks/
│   └── exploratory_analysis.ipynb  # EDA (optional)
│
├── 📂 tests/                   # Unit tests
│   └── test_pipeline.py
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
└── .gitignore
```

### Pipeline Flow

```
1. DATA INGESTION
   ↓
2. PREPROCESSING (Cleaning, Normalization, Outlier Detection)
   ↓
3. FEATURE ENGINEERING (Temporal Features, Rolling Stats, Lag Features)
   ↓
4. DATA SPLITTING (Train 70% / Validation 10% / Test 20%)
   ↓
5. MODEL TRAINING (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
   ↓
6. MODEL EVALUATION (MAE, RMSE, R², MAPE)
   ↓
7. MODEL SELECTION (Best model based on test R²)
   ↓
8. ARTIFACT SAVING (Model, Scaler, Feature Names)
   ↓
9. INFERENCE (Predictions on new data)
```

---

## 📊 Dataset

### Energy Consumption Data
- **Source**: Synthetic + UCI ML Repository
- **Records**: ~8,700 hourly observations
- **Time Period**: 1 year of data
- **Features**: 19 environmental sensors (temperature, humidity)
- **Target**: Energy consumption (Appliances) in Wh

### Features Used
```
- Appliances (TARGET): Energy consumption in Wh
- lights: Lighting usage
- T1-T9: Temperature sensors (Celsius)
- RH_1-RH_9: Relative humidity sensors (%)
```

---

## 🛠️ Models Implemented

| Model | Type | Hyperparameters | Best For |
|-------|------|-----------------|----------|
| **Linear Regression** | Parametric | - | Baseline, interpretability |
| **Random Forest** | Tree Ensemble | n_estimators=100, max_depth=15 | Feature importance, robustness |
| **Gradient Boosting** | Sequential Ensemble | n_estimators=100, lr=0.1 | Performance, complex patterns |
| **XGBoost** | Optimized Ensemble | n_estimators=100, lr=0.1 | Speed, scalability |

---

## 📈 Feature Engineering

### Temporal Features
- **Hour of day**: When electricity is consumed
- **Day of week**: Weekday vs weekend patterns
- **Month**: Seasonal effects
- **Cyclical encoding**: sin/cos transformation for periodic features

### Statistical Features
- **Rolling mean** (3h, 7h, 24h windows): Trend information
- **Rolling std** (3h, 7h, 24h windows): Volatility
- **Lag features** (1h, 2h, 3h, 24h): Historical dependencies

### Total Features: 40+

---

## 📊 Evaluation Metrics

```
MAE (Mean Absolute Error)    → Average prediction error in absolute terms
RMSE (Root Mean Squared)     → Penalizes large errors
R² Score                      → Proportion of variance explained
MAPE (Mean Absolute %)       → Percentage error
```

### Typical Results
```
Best Model: Gradient Boosting
├── Train R²: 0.85-0.90
├── Validation R²: 0.78-0.85
├── Test R²: 0.75-0.82
└── Test RMSE: 20-30 Wh
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/energy-forecasting-pipeline.git
   cd energy-forecasting-pipeline
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training

**Run the complete pipeline**
```bash
python scripts/train_pipeline.py
```

This will:
- Load and explore data
- Preprocess and engineer features
- Train all models
- Evaluate and compare performance
- Save the best model and artifacts
- Generate visualizations

### Inference

**Make predictions on new data**
```bash
python scripts/inference.py
```

Or from Python:
```python
from scripts.inference import predict_energy_consumption
import pandas as pd

# Load your data
df_new = pd.read_csv("new_data.csv")

# Get predictions
predictions = predict_energy_consumption(df_new)
```

---

## 💻 Code Quality

### Best Practices Implemented
- ✅ **Modular design**: Each step is an independent module
- ✅ **Configuration management**: Centralized `config.py`
- ✅ **Logging**: Comprehensive logging throughout pipeline
- ✅ **Error handling**: Graceful error management
- ✅ **Type hints**: Full type annotations for clarity
- ✅ **Documentation**: Docstrings for all functions
- ✅ **Reproducibility**: Fixed random seeds
- ✅ **Scalability**: Support for large datasets

### Code Structure Example
```python
# config.py: Central configuration
MODELS_CONFIG = {
    "Linear Regression": {...},
    "Random Forest": {...},
    "Gradient Boosting": {...},
    "XGBoost": {...}
}

# data_ingestion.py: Modular data loading
def load_data() -> pd.DataFrame:
    """Load and validate data"""
    
# preprocessing.py: Clear preprocessing pipeline
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Handle missing values, outliers, normalization"""
    
# model_training.py: Systematic model training
def train_all_models(X_train, y_train) -> Dict:
    """Train multiple models in parallel"""
```

---

## 📊 Performance Analysis

### Model Comparison
The pipeline generates comprehensive evaluation reports including:
- Individual model metrics
- Train/Validation/Test performance
- Feature importance rankings
- Residual analysis
- Prediction error distributions

### Visualizations Generated
- ✅ Actual vs Predicted plots
- ✅ Residual analysis
- ✅ Feature importance charts
- ✅ Model performance comparison
- ✅ Error distribution histograms
- ✅ Q-Q plots for residuals

---

## 🔧 Configuration

### Modifying Hyperparameters
Edit `src/config.py`:

```python
# Model parameters
MODELS_CONFIG = {
    "Random Forest": {
        "params": {
            "n_estimators": 150,      # Increase for better accuracy
            "max_depth": 20,          # Increase for complexity
            "random_state": RANDOM_STATE
        }
    }
}

# Data split
TEST_SIZE = 0.2                       # 20% test set
VALIDATION_SIZE = 0.1                # 10% validation set

# Features
ROLLING_WINDOW_SIZES = [3, 7, 24]   # Hour windows
```

---

## 🧪 Testing

**Run tests**
```bash
python -m pytest tests/
```

**Manual testing**
```python
from src.preprocessing import prepare_data
from src.data_ingestion import load_data

df = load_data()
df_clean, metadata = prepare_data(df)
assert df_clean.shape[0] > 0
print("✓ Pipeline works correctly")
```

---

## 📁 Artifacts Generated

After running `train_pipeline.py`, the following files are created:

```
models/
├── best_model.pkl              # Serialized best model
├── scaler.pkl                  # Fitted StandardScaler
└── feature_names.pkl           # Expected feature names

data/processed/
├── evaluation_report.csv       # Comprehensive metrics
└── split_info.pkl              # Train/val/test splits

visualizations/
├── *_predictions.png           # Actual vs Predicted
├── *_feature_importance.png    # Top 20 features
└── *_residuals.png             # Residual analysis
```

---

## 🎓 Learning Outcomes

This project demonstrates:

### Technical Skills
- 🎯 End-to-end ML pipeline development
- 📊 Data preprocessing and feature engineering
- 🤖 Model training and hyperparameter tuning
- 📈 Comprehensive evaluation and comparison
- 💾 Model serialization and deployment

### Best Practices
- 📋 Clean, modular code architecture
- 🔍 Logging and monitoring
- 🛡️ Error handling and validation
- 📚 Documentation and type hints
- 🔄 Reproducibility and versioning

### Production Concepts
- 🚀 Inference pipeline design
- 🎯 Model evaluation for production
- 💡 Feature management
- ⚙️ Configuration management
- 📊 Performance tracking

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add cross-validation
- [ ] Implement hyperparameter optimization (Optuna, Ray Tune)
- [ ] Add deep learning models (LSTM for time-series)
- [ ] Implement MLflow for experiment tracking
- [ ] Add unit tests
- [ ] Create Docker containerization
- [ ] Add REST API endpoint
- [ ] Implement data validation with Great Expectations

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Youssef AIT ELOURF**
- 🔗 GitHub: [@youssefaitelourf](https://github.com/youssefaitelourf)
- 💼 LinkedIn: [youssef-aitelourf](https://linkedin.com/in/youssef-aitelourf)
- 📧 Email: youssefaitelourf@gmail.com | youssef.aitelourf.pro@gmail.com

---

## 🙏 Acknowledgments

- Inspiration from production ML systems at leading tech companies
- Data sourced from UCI Machine Learning Repository
- Built with Python ML stack: scikit-learn, pandas, matplotlib

---

## 📖 Additional Resources

### Recommended Reading
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Feature Engineering Best Practices](https://www.kaggle.com/learn/feature-engineering)
- [ML Engineering Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)

### Similar Projects to Explore
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Fast.ai Course](https://www.fast.ai/)
- [MLOps.community](https://mlops.community/)

---

## ⭐ If this helps you, please consider giving it a star!

<div align="center">

**Made with ❤️ for the ML community**

[⬆ Back to top](#-energy-forecasting-pipeline-advanced-ml-for-consumption-prediction)

</div>
