"""
Utility functions for the Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from src.data_loader import UKRoadDataLoader
from src.feature_engineering import RiskFeatureEngineer
from src.risk_predictor import RiskPredictor
from config.settings import STREAMLIT_CONFIG, VIZ_CONFIG

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)  # Cache for 1 hour, allows refresh
def load_and_process_data(limit_rows: Optional[int] = None, force_refresh: bool = False):
    """Load and process the UK road safety data with caching"""
    # force_refresh parameter helps invalidate cache when needed
    
    with st.spinner("Loading data... This may take a few minutes on first run."):
        try:
            data_loader = UKRoadDataLoader()
            feature_engineer = RiskFeatureEngineer()
            
            # Load data
            accidents_df, vehicles_df, path = data_loader.load_uk_road_data(limit_rows)
            
            # Merge datasets
            df = data_loader.merge_datasets(accidents_df, vehicles_df)
            
            # Create features
            df = feature_engineer.create_all_features(df)
            
            logger.info(f"Processed dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
            max-width: 100%;
        }
        .block-container {
            max-width: 100%;
            padding: 1rem;
        }
        .stAlert {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
        }
        h1, h2, h3 {
            color: #1f77b4;
        }
        .risk-high {
            background-color: #ffcccc;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            color: #cc0000;
        }
        .risk-medium {
            background-color: #fff4cc;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            color: #cc6600;
        }
        .risk-low {
            background-color: #ccffcc;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            color: #006600;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
        }
        /* Responsive adjustments */
        .element-container {
            width: 100% !important;
        }
        .stPlotlyChart {
            width: 100% !important;
            height: auto !important;
        }
        /* Make dataframes responsive */
        .stDataFrame {
            width: 100% !important;
        }
        /* Button styling */
        .stButton > button {
            width: 100%;
        }
        /* Column gap adjustments */
        [data-testid="column"] {
            padding: 0.5rem;
        }
        /* Mobile responsive */
        @media (max-width: 768px) {
            .main {
                padding: 0.5rem;
            }
            [data-testid="column"] {
                margin-bottom: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'risk_predictor' not in st.session_state:
        st.session_state.risk_predictor = RiskPredictor()

def display_metrics_cards(df: pd.DataFrame):
    """Display key metrics in cards"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accidents", f"{len(df):,}")
    
    with col2:
        total_casualties = df['Number_of_Casualties'].sum()
        st.metric("Total Casualties", f"{total_casualties:,}")
    
    with col3:
        # Check for both string 'Fatal' and numeric 1
        if 'Accident_Severity' in df.columns:
            fatal_accidents = ((df['Accident_Severity'] == 'Fatal') | (df['Accident_Severity'] == 1)).sum()
        else:
            fatal_accidents = 0
        st.metric("Fatal Accidents", f"{fatal_accidents:,}")
    
    with col4:
        if 'Year' in df.columns:
            date_range = f"{df['Year'].min()}-{df['Year'].max()}"
        else:
            date_range = "N/A"
        st.metric("Date Range", date_range)

def get_risk_color_and_emoji(risk_level: str) -> tuple:
    """Get color class and emoji for risk level"""
    
    if risk_level == "High Risk":
        return "risk-high", "🔴"
    elif risk_level == "Medium Risk":
        return "risk-medium", "🟡"
    else:
        return "risk-low", "🟢"

def create_download_button(data: pd.DataFrame, filename: str, label: str):
    """Create a download button for data"""
    
    csv = data.to_csv(index=False)
    st.download_button(
        label=f"📥 {label}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def format_prediction_display(risk_level: str, probabilities: Dict[str, float], 
                            recommendations: Dict[str, str]):
    """Format and display prediction results"""
    
    risk_color, emoji = get_risk_color_and_emoji(risk_level)
    
    # Display main prediction
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown(f"""
        <div class="{risk_color}" style="text-align: center; font-size: 24px;">
            {emoji} Predicted Risk Level: {risk_level}
        </div>
        """, unsafe_allow_html=True)
    
    # Display probabilities
    st.subheader("Prediction Confidence")
    prob_cols = st.columns(len(probabilities))
    for i, (level, prob) in enumerate(probabilities.items()):
        with prob_cols[i]:
            st.metric(level, f"{prob:.2%}")
    
    # Display recommendations
    if recommendations:
        st.subheader("Safety Recommendations")
        for category, recommendation in recommendations.items():
            st.write(recommendation)

def validate_prediction_input(input_data: Dict[str, Any]) -> bool:
    """Validate prediction input data"""
    
    required_fields = [
        'hour', 'day_of_week', 'month', 'weather', 'road_surface',
        'light_condition', 'speed_limit', 'num_vehicles', 'num_casualties',
        'vehicle_age', 'junction_type', 'urban_rural'
    ]
    
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            st.error(f"Missing required field: {field}")
            return False
    
    # Validate ranges
    validations = {
        'hour': (0, 23),
        'day_of_week': (0, 6),
        'month': (1, 12),
        'speed_limit': (10, 100),
        'num_vehicles': (1, 10),
        'num_casualties': (0, 20),
        'vehicle_age': (0, 50)
    }
    
    for field, (min_val, max_val) in validations.items():
        if not (min_val <= input_data[field] <= max_val):
            st.error(f"{field} must be between {min_val} and {max_val}")
            return False
    
    return True

def get_weather_conditions_map():
    """Get weather conditions mapping for display"""
    return {
        1: 'Fine',
        2: 'Rain',
        3: 'Snow',
        4: 'Fine + Wind',
        5: 'Rain + Wind',
        6: 'Snow + Wind',
        7: 'Fog',
        8: 'Other',
        9: 'Unknown'
    }

def get_road_surface_map():
    """Get road surface conditions mapping for display"""
    return {
        1: 'Dry',
        2: 'Wet/Damp',
        3: 'Snow',
        4: 'Frost/Ice',
        5: 'Flood',
        6: 'Oil/Diesel',
        7: 'Mud'
    }

def get_light_conditions_map():
    """Get light conditions mapping for display"""
    return {
        1: 'Daylight',
        4: 'Dark - Lit',
        5: 'Dark - Unlit',
        6: 'Dark - No Light',
        7: 'Dark - Unknown'
    }

def get_junction_type_map():
    """Get junction type mapping for display"""
    return {
        0: 'Not at junction',
        1: 'Roundabout',
        2: 'Mini-roundabout',
        3: 'T junction',
        5: 'Slip road',
        6: 'Crossroads',
        7: 'Multi-arm',
        8: 'Private drive',
        9: 'Other'
    }

def display_data_quality_info(df: pd.DataFrame):
    """Display data quality information"""
    
    st.subheader("Data Quality Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values by Column:**")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            st.dataframe(missing_data.head(10))
        else:
            st.success("No missing values found!")
    
    with col2:
        st.write("**Data Types:**")
        # Ensure all arrays have same length
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).tolist()
        non_null_counts = df.count().tolist()
        memory_usage = df.memory_usage(deep=True).tolist()[1:]  # Skip index memory
        
        # Ensure all lists have same length
        min_length = min(len(columns), len(dtypes), len(non_null_counts), len(memory_usage))
        
        dtype_info = pd.DataFrame({
            'Column': columns[:min_length],
            'Data Type': dtypes[:min_length],
            'Non-Null Count': non_null_counts[:min_length],
            'Memory Usage': memory_usage[:min_length]
        })
        st.dataframe(dtype_info.head(10))