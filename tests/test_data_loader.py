"""
Tests for data loading functionality
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import UKRoadDataLoader

class TestUKRoadDataLoader:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = UKRoadDataLoader(self.temp_dir)
    
    def test_initialization(self):
        """Test data loader initialization"""
        assert self.data_loader.data_dir == Path(self.temp_dir)
        assert self.data_loader.data_dir.exists()
    
    @patch('kagglehub.dataset_download')
    @patch('pandas.read_csv')
    def test_load_uk_road_data_success(self, mock_read_csv, mock_download):
        """Test successful data loading"""
        # Mock kagglehub download
        mock_download.return_value = "/mock/path"
        
        # Create mock dataframes
        accidents_mock = pd.DataFrame({
            'Accident_Index': ['A1', 'A2'],
            'Date': ['2023-01-01', '2023-01-02'],
            'Accident_Severity': [1, 2]
        })
        vehicles_mock = pd.DataFrame({
            'Accident_Index': ['A1', 'A2'],
            'Vehicle_Type': [1, 2],
            'Age_of_Vehicle': [5, 10]
        })
        
        mock_read_csv.side_effect = [accidents_mock, vehicles_mock]
        
        # Test method
        accidents_df, vehicles_df, path = self.data_loader.load_uk_road_data()
        
        # Assertions
        assert path == "/mock/path"
        assert len(accidents_df) == 2
        assert len(vehicles_df) == 2
        assert 'Accident_Index' in accidents_df.columns
        assert 'Accident_Index' in vehicles_df.columns
        
        mock_download.assert_called_once()
        assert mock_read_csv.call_count == 2
    
    def test_merge_datasets(self):
        """Test dataset merging"""
        accidents_df = pd.DataFrame({
            'Accident_Index': ['A1', 'A2', 'A3'],
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Accident_Severity': [1, 2, 3]
        })
        vehicles_df = pd.DataFrame({
            'Accident_Index': ['A1', 'A2'],
            'Vehicle_Type': [1, 2],
            'Age_of_Vehicle': [5, 10]
        })
        
        merged_df = self.data_loader.merge_datasets(accidents_df, vehicles_df)
        
        assert len(merged_df) == 2  # Inner join should result in 2 rows
        assert 'Accident_Index' in merged_df.columns
        assert 'Date' in merged_df.columns
        assert 'Vehicle_Type' in merged_df.columns
    
    def test_save_and_load_raw_data(self):
        """Test saving and loading raw data"""
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Save data
        self.data_loader.save_raw_data(test_df, "test_data.csv")
        
        # Check file exists
        file_path = self.data_loader.data_dir / "test_data.csv"
        assert file_path.exists()
        
        # Load data back
        loaded_df = self.data_loader.load_raw_data("test_data.csv")
        
        # Verify data integrity
        pd.testing.assert_frame_equal(test_df, loaded_df)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            self.data_loader.load_raw_data("nonexistent.csv")
    
    @patch('kagglehub.dataset_download')
    def test_load_uk_road_data_with_limit(self, mock_download):
        """Test data loading with row limit"""
        mock_download.return_value = "/mock/path"
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({'col': [1, 2, 3]})
            
            self.data_loader.load_uk_road_data(limit_rows=100)
            
            # Verify nrows parameter was passed
            calls = mock_read_csv.call_args_list
            for call in calls:
                assert call[1]['nrows'] == 100