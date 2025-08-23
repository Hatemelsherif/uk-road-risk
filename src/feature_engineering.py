"""
Feature engineering module for UK road risk classification
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RiskFeatureEngineer:
    """Handles comprehensive feature engineering for road risk assessment"""
    
    def __init__(self):
        self.weather_risk_map = {
            1: 0,  # Fine no high winds
            2: 1,  # Raining no high winds
            3: 2,  # Snowing no high winds
            4: 1,  # Fine + high winds
            5: 2,  # Raining + high winds
            6: 3,  # Snowing + high winds
            7: 2,  # Fog or mist
            8: 2,  # Other
            9: 0   # Unknown
        }
        
        self.road_risk_map = {
            1: 0,  # Dry
            2: 2,  # Wet or damp
            3: 3,  # Snow
            4: 3,  # Frost or ice
            5: 2,  # Flood over 3cm deep
            6: 1,  # Oil or diesel
            7: 1   # Mud
        }
        
        self.light_risk_map = {
            1: 0,  # Daylight
            4: 2,  # Darkness - lights lit
            5: 3,  # Darkness - lights unlit
            6: 2,  # Darkness - no lighting
            7: 3   # Darkness - lighting unknown
        }
        
        self.junction_risk_map = {
            0: 0,  # Not at junction
            1: 1,  # Roundabout
            2: 1,  # Mini-roundabout
            3: 2,  # T or staggered junction
            5: 2,  # Slip road
            6: 3,  # Crossroads
            7: 2,  # More than 4 arms
            8: 2,  # Private drive or entrance
            9: 1   # Other junction
        }
    
    def categorize_risk(self, severity) -> str:
        """Categorize accident severity into risk levels"""
        # Handle both numeric and string severity values
        if isinstance(severity, str):
            if severity.lower() == 'fatal':
                return 'High Risk'
            elif severity.lower() == 'serious':
                return 'Medium Risk'
            elif severity.lower() == 'slight':
                return 'Low Risk'
            else:
                return 'Low Risk'  # Default for unknown strings
        else:
            # Handle numeric values (legacy support)
            if severity == 1:  # Fatal
                return 'High Risk'
            elif severity == 2:  # Serious
                return 'Medium Risk'
            else:  # Slight or unknown
                return 'Low Risk'
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        df = df.copy()
        
        # Parse datetime columns
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        df['Is_Rush_Hour'] = df['Hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        return df
    
    def create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create environmental risk features"""
        logger.info("Creating environmental features...")
        df = df.copy()
        
        # Map conditions to risk scores
        df['Weather_Risk_Score'] = df['Weather_Conditions'].map(self.weather_risk_map).fillna(0)
        df['Road_Risk_Score'] = df['Road_Surface_Conditions'].map(self.road_risk_map).fillna(0)
        df['Light_Risk_Score'] = df['Light_Conditions'].map(self.light_risk_map).fillna(0)
        
        # Aggregate environmental risk
        df['Environmental_Risk'] = (df['Weather_Risk_Score'] + 
                                   df['Road_Risk_Score'] + 
                                   df['Light_Risk_Score']) / 3
        
        return df
    
    def create_vehicle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create vehicle-related features"""
        logger.info("Creating vehicle features...")
        df = df.copy()
        
        # Vehicle age features
        current_year = pd.Timestamp.now().year
        df['Vehicle_Age'] = current_year - df['Age_of_Vehicle']
        df['Is_Old_Vehicle'] = (df['Vehicle_Age'] > 10).astype(int)
        
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        logger.info("Creating location features...")
        df = df.copy()
        
        # Location-based risk (number of accidents in same location)
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            location_risk = df.groupby(['Latitude', 'Longitude']).size().reset_index(name='Location_Accident_Count')
            df = pd.merge(df, location_risk, on=['Latitude', 'Longitude'], how='left')
        else:
            df['Location_Accident_Count'] = 1
        
        # Junction complexity
        df['Junction_Risk'] = df['Junction_Detail'].map(self.junction_risk_map).fillna(0)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all risk features and target variable"""
        logger.info("Creating comprehensive risk features...")
        
        # Create risk level target
        df['Risk_Level'] = df['Accident_Severity'].apply(self.categorize_risk)
        
        # Create all feature groups
        df = self.create_temporal_features(df)
        df = self.create_environmental_features(df)
        df = self.create_vehicle_features(df)
        df = self.create_location_features(df)
        
        logger.info("Feature engineering completed")
        return df
    
    def get_feature_columns(self) -> list:
        """Return list of engineered feature columns for modeling"""
        return [
            'Number_of_Vehicles', 'Number_of_Casualties',
            'Speed_limit', 'Hour', 'Day_of_Week', 'Month',
            'Is_Weekend', 'Is_Rush_Hour', 'Weather_Risk_Score',
            'Road_Risk_Score', 'Light_Risk_Score', 'Environmental_Risk',
            'Vehicle_Age', 'Is_Old_Vehicle', 'Location_Accident_Count',
            'Junction_Risk', 'Urban_or_Rural_Area'
        ]
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for modeling"""
        logger.info("Preparing data for modeling...")
        
        feature_columns = self.get_feature_columns()
        
        # Handle missing values
        for col in feature_columns:
            if col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        # Ensure all features exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_columns]
        y = df['Risk_Level']
        
        return X, y, df