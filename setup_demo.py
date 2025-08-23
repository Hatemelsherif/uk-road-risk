#!/usr/bin/env python3
"""
One-click setup script to bypass Kaggle issues and get the demo running
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"], 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def main():
    """Setup demo data and models, bypassing Kaggle entirely"""
    
    print("🚀 UK Road Risk Classification System - Demo Setup")
    print("=" * 60)
    print("This script will set up the demo using sample data (no Kaggle required)")
    print()
    
    try:
        # Step 1: Generate sample data
        print("📊 Step 1: Generating sample data...")
        from src.data_loader import UKRoadDataLoader
        from src.feature_engineering import RiskFeatureEngineer
        
        data_loader = UKRoadDataLoader()
        feature_engineer = RiskFeatureEngineer()
        
        # Generate sample data
        logger.info("Generating 5000 sample records...")
        accidents_df, vehicles_df, sample_path = data_loader.generate_sample_data(50000)
        
        # Merge datasets
        df_merged = data_loader.merge_datasets(accidents_df, vehicles_df)
        
        # Create features
        df_features = feature_engineer.create_all_features(df_merged)
        
        # Save processed data
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_features.to_csv(output_dir / "demo_data.csv", index=False)
        print(f"✅ Sample data generated: {len(df_features):,} records")
        
        # Step 2: Train models
        print("\n🤖 Step 2: Training machine learning models...")
        from src.model_training import RiskModelTrainer
        
        model_trainer = RiskModelTrainer()
        
        # Prepare data for modeling
        X, y, _ = feature_engineer.prepare_for_modeling(df_features)
        logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Train models
        results, scaler, encoder, X_test, y_test, importance = model_trainer.train_classification_models(X, y)
        
        # Save best model
        best_model_name = model_trainer.save_best_model(results, scaler, encoder, importance)
        
        print(f"✅ Models trained successfully! Best: {best_model_name}")
        print(f"   Accuracy: {results[best_model_name]['accuracy']:.3f}")
        print(f"   F1-Score: {results[best_model_name]['f1_score']:.3f}")
        
        # Step 3: Test system
        print("\n🧪 Step 3: Testing system components...")
        from src.risk_predictor import RiskPredictor
        
        predictor = RiskPredictor()
        if predictor.model is not None:
            print("✅ Trained model loaded successfully")
        
        # Test prediction
        test_conditions = {
            'weather': 2, 'road_surface': 2, 'light_condition': 4,
            'junction_type': 3, 'is_rush_hour': True, 'vehicle_age': 10
        }
        risk, score, recs = predictor.calculate_simple_risk_score(test_conditions)
        print(f"✅ Risk prediction working: {risk}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        print(f"📊 Data: {len(df_features):,} records with {len(df_features.columns)} features")
        print(f"🤖 Model: {best_model_name} ({results[best_model_name]['accuracy']:.1%} accuracy)")
        print(f"📁 Files: Saved in data/processed/ and data/models/")
        
        print("\n🚀 Ready to run:")
        print("   streamlit run app/streamlit_app.py")
        print("   uvicorn api.main:app --reload")
        
        print("\n🌐 Access points:")
        print("   Dashboard: http://localhost:8501")
        print("   API Docs:  http://localhost:8000/docs")
        
        print("\n✨ Demo features available:")
        print("   📊 Interactive data visualization")
        print("   🔮 Real-time risk prediction")
        print("   🗺️ Geographic analysis")
        print("   📈 Temporal pattern analysis") 
        print("   📉 Model performance metrics")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        print(f"\n❌ Setup failed: {str(e)}")
        print("\nPlease check the error above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next: Run 'streamlit run app/streamlit_app.py' to start!")
    sys.exit(0 if success else 1)