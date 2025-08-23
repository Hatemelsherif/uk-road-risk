#!/usr/bin/env python3
"""
Script to load and preprocess UK road safety data from local CSV files
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
    parser = argparse.ArgumentParser(description='Load and preprocess UK road safety data from local CSV files')
    parser.add_argument('--limit-rows', type=int, default=None, help='Limit number of rows (default: use full dataset)')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory for processed data')
    parser.add_argument('--no-features', action='store_true', help='Skip feature engineering')
    
    args = parser.parse_args()
    
    logger.info("Starting data loading and preprocessing...")
    
    try:
        # Initialize components
        data_loader = UKRoadDataLoader()
        
        # Load data from local CSV files
        logger.info("Loading UK road safety dataset from local files...")
        accidents_df, vehicles_df, source_path = data_loader.load_uk_road_data(args.limit_rows)
        logger.info("Successfully loaded data from local CSV files")
        
        logger.info(f"Downloaded from: {source_path}")
        logger.info(f"Accidents data: {accidents_df.shape}")
        logger.info(f"Vehicles data: {vehicles_df.shape}")
        
        # Merge datasets
        logger.info("Merging datasets...")
        df = data_loader.merge_datasets(accidents_df, vehicles_df)
        
        # Save raw merged data
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        raw_output = output_dir / "merged_raw_data.csv"
        df.to_csv(raw_output, index=False)
        logger.info(f"Raw merged data saved to: {raw_output}")
        
        # Feature engineering
        if not args.no_features:
            logger.info("Creating engineered features...")
            feature_engineer = RiskFeatureEngineer()
            df_processed = feature_engineer.create_all_features(df)
            
            # Save processed data
            processed_output = output_dir / "processed_data.csv"
            df_processed.to_csv(processed_output, index=False)
            logger.info(f"Processed data saved to: {processed_output}")
            
            # Save feature summary
            feature_summary = {
                'total_records': len(df_processed),
                'total_features': len(df_processed.columns),
                'risk_distribution': df_processed['Risk_Level'].value_counts().to_dict(),
                'feature_columns': feature_engineer.get_feature_columns()
            }
            
            import json
            summary_output = output_dir / "feature_summary.json"
            with open(summary_output, 'w') as f:
                json.dump(feature_summary, f, indent=2)
            
            logger.info(f"Feature summary saved to: {summary_output}")
            
            # Print summary
            print("\n" + "="*50)
            print("DATA PROCESSING SUMMARY")
            print("="*50)
            print(f"Total Records: {len(df_processed):,}")
            print(f"Total Features: {len(df_processed.columns)}")
            print("\nRisk Level Distribution:")
            for risk, count in df_processed['Risk_Level'].value_counts().items():
                pct = (count / len(df_processed)) * 100
                print(f"  {risk}: {count:,} ({pct:.1f}%)")
            
            print(f"\nFiles saved to: {output_dir}")
            print("- merged_raw_data.csv (raw merged data)")
            print("- processed_data.csv (with engineered features)")
            print("- feature_summary.json (processing metadata)")
        
        else:
            print("\n" + "="*50)
            print("DATA DOWNLOAD SUMMARY")
            print("="*50)
            print(f"Total Records: {len(df):,}")
            print(f"Raw data saved to: {raw_output}")
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()