"""
Training and Evaluation Pipeline for PandemicWatch AI
Trains XGBoost and Random Forest models on multi-region epidemiological time-series,
evaluates goodness-of-fit metrics (RMSE, MAE, R², MAPE), and persists the production model.
"""

import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.generator import get_or_create_benchmark_dataset
from models.ml_forecaster import EpidemicForecaster

def run_training():
    print("==================================================")
    print("      PandemicWatch AI - ML Model Training       ")
    print("==================================================")

    print("\n[1/3] Loading benchmark epidemiological surveillance data...")
    df = get_or_create_benchmark_dataset()
    print(f"Dataset loaded: {len(df)} records across regions {df['region'].unique().tolist()}")

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Train XGBoost
    print("\n[2/3] Training XGBoost Time-Series Regressor...")
    xgb_forecaster = EpidemicForecaster(model_type="xgboost")
    xgb_metrics = xgb_forecaster.train(df)
    print("XGBoost Evaluation:")
    for k, v in xgb_metrics.items():
        print(f"  - {k}: {v}")
    
    xgb_save_path = os.path.join(models_dir, "xgboost_forecaster.joblib")
    xgb_forecaster.save(xgb_save_path)
    print(f"  -> Saved model to: {xgb_save_path}")

    # 2. Train Random Forest for comparison/benchmarking
    print("\n[3/3] Training Random Forest Regressor (Benchmark Model)...")
    rf_forecaster = EpidemicForecaster(model_type="random_forest")
    rf_metrics = rf_forecaster.train(df)
    print("Random Forest Evaluation:")
    for k, v in rf_metrics.items():
        print(f"  - {k}: {v}")
    
    rf_save_path = os.path.join(models_dir, "random_forest_forecaster.joblib")
    rf_forecaster.save(rf_save_path)
    print(f"  -> Saved model to: {rf_save_path}")

    print("\n==================================================")
    print("Model Training & Evaluation Completed Successfully!")
    print("==================================================")
    return {
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics
    }

if __name__ == "__main__":
    run_training()
