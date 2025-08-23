"""
Main Streamlit application for UK Road Risk Assessment Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils import apply_custom_css, initialize_session_state
from config.settings import STREAMLIT_CONFIG

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout="wide",  # Use wide layout for better responsiveness
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"],
    menu_items={
        'Get Help': 'https://github.com/anthropics/claude-code/issues',
        'Report a bug': 'https://github.com/anthropics/claude-code/issues',
        'About': "UK Road Risk Assessment Dashboard - Built with Claude Code"
    }
)

# Apply custom styling
apply_custom_css()

# Initialize session state
initialize_session_state()

# Main page content
st.title("🚗 UK Road Risk Assessment Dashboard")
st.markdown("""
Welcome to the UK Road Risk Assessment Dashboard! This comprehensive system provides 
data-driven insights into road safety using machine learning and interactive visualizations.

## Features
- **📊 Data Overview**: Explore accident statistics and risk distributions
- **📈 Risk Analysis**: Analyze temporal patterns and environmental factors
- **🗺️ Geographic Analysis**: View accident hotspots and regional statistics  
- **🤖 Risk Prediction**: Get real-time risk assessments for current conditions
- **📉 Model Performance**: Review machine learning model accuracy and metrics

## Getting Started
Use the navigation panel on the left to explore different sections of the dashboard.
Start with "Data Overview" to load and explore the dataset.

## About the Data
This system analyzes UK road safety data from the Department for Transport, 
covering accident records with detailed information about:
- Environmental conditions (weather, road surface, lighting)
- Temporal factors (time of day, day of week, seasonality)
- Vehicle information (age, type, number involved)
- Geographic location and road characteristics
- Casualty outcomes and accident severity

## Risk Classification
Accidents are classified into three risk levels:
- **🔴 High Risk**: Fatal accidents requiring immediate attention
- **🟡 Medium Risk**: Serious injury accidents needing monitoring
- **🟢 Low Risk**: Minor accidents with slight injuries

Navigate to the specific pages using the sidebar to begin your analysis!
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>UK Road Risk Assessment Dashboard v1.0 | Built with Streamlit & Machine Learning</p>
    <p>Data from UK Department for Transport | For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)