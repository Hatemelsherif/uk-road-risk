"""
Data loading and preprocessing utilities for UK road safety data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UKRoadDataLoader:
    """Handles loading and initial processing of UK road safety data"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_uk_road_data(self, limit_rows: int = None):
        """Load UK road safety dataset from local CSV files"""
        # Use local CSV files directly
        accidents_file = self.data_dir / "Accident_Information.csv"
        vehicles_file = self.data_dir / "Vehicle_Information.csv"
        
        # Check if local files exist
        if accidents_file.exists() and vehicles_file.exists():
            logger.info("Loading data from local CSV files...")
            try:
                logger.info(f"Loading accidents from: {accidents_file}")
                logger.info(f"Loading vehicles from: {vehicles_file}")
                
                if limit_rows:
                    # For limited rows, use sampling to ensure we get matching data
                    logger.info(f"Using sampling strategy for {limit_rows} rows to ensure data matching...")
                    
                    # Load full accidents data first to get date range
                    accidents_sample = pd.read_csv(accidents_file, nrows=50000, low_memory=False, dtype=str)
                    accidents_sample = self._convert_data_types(accidents_sample)
                    
                    # Get unique years and sample from each
                    if 'Year' in accidents_sample.columns:
                        years = sorted(accidents_sample['Year'].unique())
                        logger.info(f"Found years: {years}")
                        
                        # Take proportional sample from each year
                        rows_per_year = limit_rows // len(years)
                        sampled_accidents = []
                        
                        for year in years:
                            year_data = accidents_sample[accidents_sample['Year'] == year]
                            if len(year_data) > rows_per_year:
                                year_sample = year_data.sample(n=rows_per_year, random_state=42)
                            else:
                                year_sample = year_data
                            sampled_accidents.append(year_sample)
                        
                        accidents_df = pd.concat(sampled_accidents, ignore_index=True)
                        logger.info(f"Sampled accidents data: {accidents_df.shape}")
                        
                        # Get accident indices for matching vehicles
                        accident_indices = accidents_df['Accident_Index'].unique()
                        
                        # Load vehicles data in chunks and filter by matching indices
                        vehicles_chunks = []
                        chunksize = 10000
                        
                        for chunk in pd.read_csv(vehicles_file, chunksize=chunksize, low_memory=False, dtype=str, encoding='latin-1'):
                            chunk = self._convert_data_types(chunk)
                            matching_vehicles = chunk[chunk['Accident_Index'].isin(accident_indices)]
                            if len(matching_vehicles) > 0:
                                vehicles_chunks.append(matching_vehicles)
                        
                        vehicles_df = pd.concat(vehicles_chunks, ignore_index=True) if vehicles_chunks else pd.DataFrame()
                        logger.info(f"Matching vehicles data: {vehicles_df.shape}")
                    else:
                        # Fallback to simple row limiting
                        accidents_df = accidents_sample.head(limit_rows)
                        vehicles_df = pd.read_csv(vehicles_file, nrows=limit_rows, low_memory=False, dtype=str, encoding='latin-1')
                        vehicles_df = self._convert_data_types(vehicles_df)
                else:
                    # Load full datasets
                    logger.info("Loading full datasets...")
                    accidents_df = pd.read_csv(accidents_file, low_memory=False, dtype=str)
                    vehicles_df = pd.read_csv(vehicles_file, low_memory=False, dtype=str, encoding='latin-1')
                    
                    # Convert data types
                    accidents_df = self._convert_data_types(accidents_df)
                    vehicles_df = self._convert_data_types(vehicles_df)
                
                logger.info(f"Accidents data shape: {accidents_df.shape}")
                logger.info(f"Vehicles data shape: {vehicles_df.shape}")
                
                return accidents_df, vehicles_df, str(self.data_dir)
            except Exception as e:
                logger.error(f"Failed to load local data files: {e}")
                raise
        
        # If local files don't exist, generate sample data for testing
        logger.warning(f"Local data files not found at {self.data_dir}")
        logger.info("Generating sample data for testing...")
        return self.generate_sample_data(limit_rows)
    
    def merge_datasets(self, accidents_df: pd.DataFrame, vehicles_df: pd.DataFrame) -> pd.DataFrame:
        """Merge accident and vehicle datasets"""
        logger.info("Merging datasets...")
        df = pd.merge(accidents_df, vehicles_df, on='Accident_Index', how='inner')
        logger.info(f"Merged dataset shape: {df.shape}")
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "merged_data.csv"):
        """Save raw merged data to disk"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Raw data saved to {filepath}")
        
    def load_raw_data(self, filename: str = "merged_data.csv") -> pd.DataFrame:
        """Load previously saved raw data"""
        filepath = self.data_dir / filename
        if filepath.exists():
            logger.info(f"Loading data from {filepath}")
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Data file not found: {filepath}")
    
    def generate_sample_data(self, num_rows: int = None):
        """Generate sample data for testing when real data is unavailable"""
        import random
        from datetime import datetime, timedelta
        
        logger.info("Generating sample data for testing...")
        
        if num_rows is None:
            num_rows = 1000
        
        # Generate sample accidents data
        base_date = datetime(2023, 1, 1)
        
        accidents_data = {
            'Accident_Index': [f'A{i:06d}' for i in range(num_rows)],
            'Date': [(base_date + timedelta(days=random.randint(0, 365))).strftime('%d/%m/%Y') for _ in range(num_rows)],
            'Time': [f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}' for _ in range(num_rows)],
            'Accident_Severity': [random.choices([1, 2, 3], weights=[5, 20, 75])[0] for _ in range(num_rows)],
            'Number_of_Vehicles': [random.randint(1, 5) for _ in range(num_rows)],
            'Number_of_Casualties': [random.randint(0, 3) for _ in range(num_rows)],
            'Weather_Conditions': [random.randint(1, 9) for _ in range(num_rows)],
            'Road_Surface_Conditions': [random.randint(1, 7) for _ in range(num_rows)],
            'Light_Conditions': [random.choice([1, 4, 5, 6, 7]) for _ in range(num_rows)],
            'Speed_limit': [random.choice([20, 30, 40, 50, 60, 70]) for _ in range(num_rows)],
            'Junction_Detail': [random.choice([0, 1, 2, 3, 5, 6, 7, 8, 9]) for _ in range(num_rows)],
            'Urban_or_Rural_Area': [random.choice([1, 2]) for _ in range(num_rows)],
            'Latitude': [51.5 + random.uniform(-2, 2) for _ in range(num_rows)],
            'Longitude': [-0.1 + random.uniform(-3, 3) for _ in range(num_rows)]
        }
        
        accidents_df = pd.DataFrame(accidents_data)
        
        # Generate sample vehicles data
        vehicles_data = {
            'Accident_Index': [f'A{i:06d}' for i in range(num_rows)],
            'Vehicle_Reference': [1 for _ in range(num_rows)],
            'Vehicle_Type': [random.randint(1, 20) for _ in range(num_rows)],
            'Age_of_Vehicle': [2024 - random.randint(1, 20) for _ in range(num_rows)],
            'Engine_Capacity': [random.choice([1000, 1400, 1600, 2000, 2500]) for _ in range(num_rows)],
            'Age_of_Driver': [random.randint(18, 80) for _ in range(num_rows)],
            'Sex_of_Driver': [random.choice([1, 2]) for _ in range(num_rows)]
        }
        
        vehicles_df = pd.DataFrame(vehicles_data)
        
        logger.info(f"Generated sample accidents data: {accidents_df.shape}")
        logger.info(f"Generated sample vehicles data: {vehicles_df.shape}")
        
        # Save sample data for future use
        sample_path = self.data_dir / "sample"
        sample_path.mkdir(exist_ok=True)
        
        accidents_df.to_csv(sample_path / "sample_accidents.csv", index=False)
        vehicles_df.to_csv(sample_path / "sample_vehicles.csv", index=False)
        
        return accidents_df, vehicles_df, str(sample_path)
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for common UK road safety dataset columns"""
        df = df.copy()
        
        # Common numeric columns that should be converted (excluding Accident_Severity and Urban_or_Rural_Area)
        numeric_columns = [
            'Number_of_Vehicles', 'Number_of_Casualties',
            'Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions',
            'Speed_limit', 'Junction_Detail',
            'Vehicle_Reference', 'Vehicle_Type', 'Age_of_Vehicle', 'Engine_Capacity',
            'Age_of_Driver', 'Sex_of_Driver'
        ]
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle Accident_Severity specially (keep as string if already strings)
        if 'Accident_Severity' in df.columns:
            # Check if values are already strings like 'Fatal', 'Serious', 'Slight'
            sample_values = df['Accident_Severity'].dropna().head(10).astype(str)
            if any(val.lower() in ['fatal', 'serious', 'slight'] for val in sample_values):
                # Keep as strings
                df['Accident_Severity'] = df['Accident_Severity'].astype(str)
            else:
                # Try to convert to numeric
                df['Accident_Severity'] = pd.to_numeric(df['Accident_Severity'], errors='coerce')
        
        # Handle Urban_or_Rural_Area specially (keep as string)
        if 'Urban_or_Rural_Area' in df.columns:
            # This column contains 'Urban' or 'Rural' strings, or numeric codes (1=Urban, 2=Rural)
            sample_values = df['Urban_or_Rural_Area'].dropna().head(10).astype(str)
            if any(val.lower() in ['urban', 'rural'] for val in sample_values):
                # Already strings, keep as is
                df['Urban_or_Rural_Area'] = df['Urban_or_Rural_Area'].astype(str)
            else:
                # Try to convert numeric codes to strings
                df['Urban_or_Rural_Area'] = pd.to_numeric(df['Urban_or_Rural_Area'], errors='coerce')
                df['Urban_or_Rural_Area'] = df['Urban_or_Rural_Area'].map({1: 'Urban', 2: 'Rural'})
        
        # Convert coordinate columns
        coordinate_columns = ['Latitude', 'Longitude']
        for col in coordinate_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep text columns as strings (Accident_Index, Date, Time, etc.)
        
        return df