#!/usr/bin/env python3
"""
Script to train and evaluate machine learning models for UK road risk classification
"""

import sys
from pathlib import Path
import logging
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import UKRoadDataLoader
from src.feature_engineering import RiskFeatureEngineer
from src.model_training import RiskModelTrainer
from config.settings import MODEL_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"], 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train UK road risk classification models')
    parser.add_argument('--limit-rows', type=int, default=None, help='Limit number of rows (default: use full dataset)')
    parser.add_argument('--skip-clustering', action='store_true', help='Skip clustering analysis')
    parser.add_argument('--model-dir', type=str, default='data/models', help='Directory to save models')
    
    args = parser.parse_args()
    
    logger.info("Starting model training pipeline...")
    
    try:
        # Initialize components
        data_loader = UKRoadDataLoader()
        feature_engineer = RiskFeatureEngineer()
        model_trainer = RiskModelTrainer(args.model_dir)
        
        # Load data
        logger.info("Loading data...")
        accidents_df, vehicles_df, _ = data_loader.load_uk_road_data(args.limit_rows)
        
        # Merge datasets
        df = data_loader.merge_datasets(accidents_df, vehicles_df)
        
        # Feature engineering
        logger.info("Creating features...")
        df = feature_engineer.create_all_features(df)
        
        # Prepare for modeling
        X, y, df_processed = feature_engineer.prepare_for_modeling(df)
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Clustering analysis
        if not args.skip_clustering:
            logger.info("Performing clustering analysis...")
            clustering_results, cluster_scaler = model_trainer.perform_clustering(X)
            
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
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()