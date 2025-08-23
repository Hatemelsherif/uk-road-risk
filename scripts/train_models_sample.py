#!/usr/bin/env python3
"""
Train models using sample data
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering import RiskFeatureEngineer
from src.model_training import RiskModelTrainer
from config.settings import LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"], 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Training models using sample data...")
    
    try:
        # Load the sample data with features
        sample_data_path = project_root / "data" / "processed" / "sample_with_features.csv"
        
        if not sample_data_path.exists():
            logger.error(f"Sample data not found at {sample_data_path}")
            logger.info("Please run: python scripts/generate_sample_data.py --with-features")
            return False
        
        logger.info(f"Loading sample data from {sample_data_path}")
        df = pd.read_csv(sample_data_path)
        
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Initialize components
        feature_engineer = RiskFeatureEngineer()
        model_trainer = RiskModelTrainer()
        
        # Prepare for modeling
        X, y, df_processed = feature_engineer.prepare_for_modeling(df)
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Risk distribution: {pd.Series(y).value_counts()}")
        
        # Clustering analysis
        logger.info("Performing clustering analysis...")
        clustering_results, cluster_scaler = model_trainer.perform_clustering(X, n_clusters=3)
        
        for method, results in clustering_results.items():
            logger.info(f"{method} - Silhouette Score: {results['silhouette']:.3f}")
        
        # Classification
        logger.info("Training classification models...")
        results, scaler, encoder, X_test, y_test, importance = model_trainer.train_classification_models(X, y)
        
        # Save best model
        best_model_name = model_trainer.save_best_model(results, scaler, encoder, importance)
        
        logger.info(f"Training completed! Best model: {best_model_name}")
        logger.info("Model artifacts saved to: data/models/")
        
        # Display results summary
        print("\n" + "="*50)
        print("TRAINING RESULTS SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  F1-Score: {result['f1_score']:.3f}")
            print(f"  CV Score: {result['cv_mean']:.3f} (+/- {result['cv_std']:.3f})")
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1-Score: {results[best_model_name]['f1_score']:.3f}")
        
        print("\nTop 5 Feature Importances:")
        print(importance.head().to_string(index=False))
        
        print("\n✅ Models trained successfully!")
        print("You can now:")
        print("1. Run the Streamlit app to see model performance")
        print("2. Use the API for predictions")
        print("3. View the Model Performance page for detailed metrics")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)