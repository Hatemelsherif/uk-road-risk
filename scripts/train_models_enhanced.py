#!/usr/bin/env python3
"""
Enhanced script to train and evaluate machine learning models for UK road risk classification
with advanced techniques for improved performance
"""

import sys
from pathlib import Path
import logging
import argparse
import warnings
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import UKRoadDataLoader
from src.feature_engineering import RiskFeatureEngineer
from src.model_training_improved import ImprovedRiskModelTrainer
from src.training_monitor import update_training_status
from src.process_manager import training_session, emergency_cleanup
from config.settings import MODEL_CONFIG, LOGGING_CONFIG

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"], 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print a nice banner for the training session"""
    print("\n" + "="*70)
    print(" "*20 + "UK ROAD RISK MODEL TRAINING")
    print(" "*20 + "Enhanced Version with ML Optimizations")
    print("="*70 + "\n")

def print_results_summary(results: dict, best_model_name: str):
    """Print formatted results summary"""
    print("\n" + "="*70)
    print(" "*25 + "TRAINING RESULTS SUMMARY")
    print("="*70)
    
    # Sort models by F1 score
    sorted_models = sorted(results.items(), 
                          key=lambda x: x[1]['f1_score'], 
                          reverse=True)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Accuracy", "F1-Score", "Balanced Acc", "ROC-AUC"
    ))
    print("-"*70)
    
    for name, result in sorted_models:
        is_best = "★" if name == best_model_name else " "
        roc_auc = result.get('roc_auc', 0) or 0
        print(f"{is_best} {name:<23} {result['accuracy']:>11.4f} "
              f"{result['f1_score']:>11.4f} {result['balanced_accuracy']:>11.4f} "
              f"{roc_auc:>11.4f}")
    
    print("\n" + "="*70)
    print(f"★ Best Model: {best_model_name}")
    print(f"  Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"  Best Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.4f}")
    
    # Print cross-validation details for best model
    if 'f1_weighted_mean' in results[best_model_name]:
        print(f"\n  Cross-Validation (10-fold):")
        print(f"    F1 Score: {results[best_model_name]['f1_weighted_mean']:.4f} "
              f"(+/- {results[best_model_name]['f1_weighted_std']:.4f})")
        print(f"    Accuracy: {results[best_model_name]['accuracy_mean']:.4f} "
              f"(+/- {results[best_model_name]['accuracy_std']:.4f})")
    
    # Print best hyperparameters if available
    if 'best_params' in results[best_model_name] and results[best_model_name]['best_params']:
        print(f"\n  Best Hyperparameters:")
        for param, value in results[best_model_name]['best_params'].items():
            print(f"    {param}: {value}")
    
    print("="*70 + "\n")

def print_feature_importance(importance_df, top_n=10):
    """Print top feature importances"""
    if importance_df.empty:
        return
    
    print("\n" + "="*70)
    print(f" "*20 + f"TOP {top_n} FEATURE IMPORTANCES")
    print("="*70)
    print("\n{:<30} {:>15}".format("Feature", "Importance"))
    print("-"*45)
    
    for _, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:<30} {row['importance']:>14.4f}")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced training for UK road risk classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with basic settings
  python train_models_enhanced.py --quick
  
  # Full training with all enhancements
  python train_models_enhanced.py --full
  
  # Custom configuration
  python train_models_enhanced.py --balance-method smote --tune-hyperparameters \\
                                  --use-ensemble --feature-selection mutual_info
        """
    )
    
    # Data arguments
    parser.add_argument('--limit-rows', type=int, default=None, 
                       help='Limit number of rows (default: use full dataset)')
    
    # Model training arguments
    parser.add_argument('--balance-method', type=str, 
                       choices=['smote', 'adasyn', 'random_under', 'smote_tomek', 'smote_enn', 'none'],
                       default='smote', help='Method for handling class imbalance')
    parser.add_argument('--feature-selection', type=str,
                       choices=['mutual_info', 'f_classif', 'rfe', 'rfecv', 'none'],
                       default='mutual_info', help='Feature selection method')
    parser.add_argument('--tune-hyperparameters', action='store_true', default=False,
                       help='Perform hyperparameter tuning (slower but better results)')
    parser.add_argument('--use-ensemble', action='store_true', default=False,
                       help='Train ensemble models (voting, stacking)')
    parser.add_argument('--use-deep-learning', action='store_true', default=False,
                       help='Include deep learning models (neural networks)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for training if available (XGBoost, LightGBM, CatBoost)')
    
    # Quick presets
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with minimal enhancements')
    parser.add_argument('--full', action='store_true',
                       help='Full training with all enhancements (slower but best results)')
    
    # Output arguments
    parser.add_argument('--model-dir', type=str, default='data/models',
                       help='Directory to save models')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed training report')
    
    args = parser.parse_args()
    
    # Apply presets
    if args.quick:
        args.balance_method = 'none'
        args.feature_selection = 'none'
        args.tune_hyperparameters = False
        args.use_ensemble = False
        logger.info("Using QUICK preset: minimal enhancements for faster training")
    elif args.full:
        args.balance_method = 'smote'
        args.feature_selection = 'mutual_info'
        args.tune_hyperparameters = True
        args.use_ensemble = True
        args.use_deep_learning = True
        logger.info("Using FULL preset: all enhancements including deep learning for best performance")
    
    # Convert 'none' strings to None
    if args.balance_method == 'none':
        args.balance_method = None
    if args.feature_selection == 'none':
        args.feature_selection = None
    
    print_banner()
    
    logger.info("Starting enhanced model training pipeline...")
    logger.info(f"Configuration:")
    logger.info(f"  - Class balancing: {args.balance_method or 'None'}")
    logger.info(f"  - Feature selection: {args.feature_selection or 'None'}")
    logger.info(f"  - Hyperparameter tuning: {args.tune_hyperparameters}")
    logger.info(f"  - Ensemble models: {args.use_ensemble}")
    logger.info(f"  - Deep learning: {args.use_deep_learning}")
    logger.info(f"  - GPU acceleration: {args.use_gpu}")
    
    try:
        # Emergency cleanup any existing orphaned processes
        emergency_cleanup()
        
        # Use training session context manager for process safety
        with training_session() as process_mgr:
            # Update status: Starting
            update_training_status("starting", 0, "Initializing training pipeline...")
            
            # Initialize components
            data_loader = UKRoadDataLoader()
            feature_engineer = RiskFeatureEngineer()
            model_trainer = ImprovedRiskModelTrainer(
                args.model_dir, 
                use_gpu=args.use_gpu
            )
            
            # Load data
            logger.info("\n📊 Loading data...")
            update_training_status("data_loading", 10, "Loading accident and vehicle data...")
            accidents_df, vehicles_df, _ = data_loader.load_uk_road_data(args.limit_rows)
        
        # Merge datasets
        df = data_loader.merge_datasets(accidents_df, vehicles_df)
        
        # Feature engineering
        logger.info("🔧 Creating features...")
        update_training_status("data_loading", 20, "Creating engineered features...")
        df = feature_engineer.create_all_features(df)
        
        # Prepare for modeling
        X, y, df_processed = feature_engineer.prepare_for_modeling(df)
        
        logger.info(f"📈 Dataset prepared: {X.shape[0]:,} samples, {X.shape[1]} features")
        update_training_status("data_loading", 30, f"Dataset prepared: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Check class distribution
        class_dist = y.value_counts()
        logger.info(f"📊 Class distribution:")
        for class_name, count in class_dist.items():
            logger.info(f"    {class_name}: {count:,} ({count/len(y)*100:.1f}%)")
        
        # Train models with all enhancements
        logger.info("\n🚀 Training models with enhanced pipeline...")
        update_training_status("training", 40, "Starting model training with enhanced pipeline...")
        training_results = model_trainer.train_all_models(
            X, y,
            balance_method=args.balance_method,
            feature_selection_method=args.feature_selection,
            tune_hyperparameters=args.tune_hyperparameters,
            use_ensemble=args.use_ensemble,
            use_deep_learning=args.use_deep_learning
        )
        
        # Save enhanced model
        logger.info("\n💾 Saving model artifacts...")
        update_training_status("training", 90, "Saving trained models and artifacts...")
        model_trainer.save_enhanced_model(training_results)
        
        # Display results
        print_results_summary(
            training_results['results'], 
            training_results['best_model_name']
        )
        
        # Display feature importance
        if not training_results['feature_importance'].empty:
            print_feature_importance(training_results['feature_importance'])
        
        # Save detailed report if requested
        if args.save_report:
            report_path = Path(args.model_dir) / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            report = {
                'configuration': {
                    'balance_method': args.balance_method,
                    'feature_selection': args.feature_selection,
                    'tune_hyperparameters': args.tune_hyperparameters,
                    'use_ensemble': args.use_ensemble,
                    'use_gpu': args.use_gpu,
                    'data_samples': X.shape[0],
                    'num_features': X.shape[1]
                },
                'results': {
                    name: {
                        'accuracy': float(res['accuracy']),
                        'f1_score': float(res['f1_score']),
                        'balanced_accuracy': float(res['balanced_accuracy']),
                        'precision': float(res['precision']),
                        'recall': float(res['recall']),
                        'cohen_kappa': float(res.get('cohen_kappa', 0)),
                        'matthews_corrcoef': float(res.get('matthews_corrcoef', 0)),
                        'roc_auc': float(res.get('roc_auc', 0)) if res.get('roc_auc') else None
                    }
                    for name, res in training_results['results'].items()
                },
                'best_model': training_results['best_model_name'],
                'feature_importance': training_results['feature_importance'].to_dict('records') 
                                     if not training_results['feature_importance'].empty else []
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📄 Detailed report saved to: {report_path}")
        
        # Mark as completed
        update_training_status("completed", 100, f"Training completed! Best model: {training_results['best_model_name']}")
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model artifacts saved to: {args.model_dir}/")
        print(f"🏆 Best model achieved {model_trainer.best_score:.1%} F1-score")
        
        # Provide recommendations
        print("\n💡 Recommendations for further improvement:")
        if not args.tune_hyperparameters:
            print("  • Enable hyperparameter tuning with --tune-hyperparameters")
        if not args.use_ensemble:
            print("  • Try ensemble methods with --use-ensemble")
        if args.balance_method is None:
            print("  • Handle class imbalance with --balance-method smote")
        if model_trainer.best_score < 0.85:
            print("  • Consider collecting more data or engineering additional features")
            print("  • Try deep learning models for complex patterns")
        
    except Exception as e:
        # Emergency cleanup of orphaned processes
        emergency_cleanup()
        
        # Mark as failed
        update_training_status("failed", 0, f"Training failed: {str(e)}")
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure cleanup even if exception occurs
        emergency_cleanup()

if __name__ == "__main__":
    main()