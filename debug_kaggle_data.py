#!/usr/bin/env python3
"""
Debug script to understand Kaggle data structure
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import UKRoadDataLoader

def debug_kaggle_data():
    """Debug the Kaggle data loading and merging"""
    
    print("🔍 Debugging Kaggle Data Structure...")
    print("=" * 50)
    
    try:
        data_loader = UKRoadDataLoader()
        
        # Load small sample
        accidents_df, vehicles_df, path = data_loader.load_uk_road_data(limit_rows=10)
        
        print(f"📁 Data loaded from: {path}")
        print(f"📊 Accidents shape: {accidents_df.shape}")
        print(f"🚗 Vehicles shape: {vehicles_df.shape}")
        
        print("\n📋 Accidents columns:")
        print(accidents_df.columns.tolist())
        
        print("\n🚗 Vehicles columns:")
        print(vehicles_df.columns.tolist())
        
        print("\n🔑 Sample Accident_Index values from accidents:")
        print(accidents_df['Accident_Index'].head().tolist() if 'Accident_Index' in accidents_df.columns else "No Accident_Index column")
        
        print("\n🔑 Sample Accident_Index values from vehicles:")
        print(vehicles_df['Accident_Index'].head().tolist() if 'Accident_Index' in vehicles_df.columns else "No Accident_Index column")
        
        # Check for common keys
        if 'Accident_Index' in accidents_df.columns and 'Accident_Index' in vehicles_df.columns:
            acc_indices = set(accidents_df['Accident_Index'].dropna())
            veh_indices = set(vehicles_df['Accident_Index'].dropna())
            common_indices = acc_indices.intersection(veh_indices)
            
            print(f"\n📊 Unique accident indices in accidents: {len(acc_indices)}")
            print(f"🚗 Unique accident indices in vehicles: {len(veh_indices)}")
            print(f"🔗 Common indices: {len(common_indices)}")
            
            if common_indices:
                print("✅ Merge should work")
                # Try merge
                merged = data_loader.merge_datasets(accidents_df, vehicles_df)
                print(f"✅ Merged result: {merged.shape}")
            else:
                print("❌ No common indices - merge will fail")
                print("Sample accident indices:", list(acc_indices)[:5])
                print("Sample vehicle indices:", list(veh_indices)[:5])
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_kaggle_data()