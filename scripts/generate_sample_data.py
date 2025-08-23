#!/usr/bin/env python3
"""
Generate sample data for testing when Kaggle download fails
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
from config.settings import LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"], 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate sample UK road safety data for testing')
    parser.add_argument('--num-rows', type=int, default=5000, help='Number of sample rows to generate')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--with-features', action='store_true', help='Include feature engineering')
    
    args = parser.parse_args()
    
    logger.info(f"Generating {args.num_rows} sample rows...")
    
    try:
        # Initialize data loader
        data_loader = UKRoadDataLoader()
        
        # Generate sample data
        accidents_df, vehicles_df, sample_path = data_loader.generate_sample_data(args.num_rows)
        
        logger.info(f"Sample data generated in: {sample_path}")
        
        # Merge datasets
        df_merged = data_loader.merge_datasets(accidents_df, vehicles_df)
        
        # Save merged data
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "sample_merged_data.csv"
        df_merged.to_csv(output_file, index=False)
        logger.info(f"Merged sample data saved to: {output_file}")
        
        # Apply feature engineering if requested
        if args.with_features:
            logger.info("Applying feature engineering...")
            feature_engineer = RiskFeatureEngineer()
            df_features = feature_engineer.create_all_features(df_merged)
            
            features_file = output_dir / "sample_with_features.csv"
            df_features.to_csv(features_file, index=False)
            logger.info(f"Featured data saved to: {features_file}")
            
            # Display statistics
            print("\n" + "="*60)
            print("SAMPLE DATA GENERATION COMPLETE")
            print("="*60)
            print(f"Total Records: {len(df_features):,}")
            print(f"Total Features: {len(df_features.columns)}")
            print("\nRisk Level Distribution:")
            for risk, count in df_features['Risk_Level'].value_counts().items():
                pct = (count / len(df_features)) * 100
                print(f"  {risk}: {count:,} ({pct:.1f}%)")
        else:
            print("\n" + "="*60)
            print("SAMPLE DATA GENERATION COMPLETE")
            print("="*60)
            print(f"Total Records: {len(df_merged):,}")
            print(f"Accidents Shape: {accidents_df.shape}")
            print(f"Vehicles Shape: {vehicles_df.shape}")
        
        print(f"\nData saved to: {output_dir}")
        print("\nYou can now:")
        print("1. Run Streamlit app to visualize the sample data")
        print("2. Train models using the sample data")
        print("3. Test API endpoints with the sample data")
        
        return True
        
    except Exception as e:
        logger.error(f"Sample data generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)