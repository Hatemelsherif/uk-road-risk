"""
Risk Analysis page for the UK Road Risk Assessment Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils import apply_custom_css, initialize_session_state
from src.visualization import RiskVisualizer
from config.settings import VIZ_CONFIG

# Apply styling and initialize state
apply_custom_css()
initialize_session_state()

st.header("📈 Risk Analysis")
st.markdown("""
Analyze temporal patterns, environmental factors, and other risk contributors 
that influence accident severity and frequency.
""")

if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    visualizer = RiskVisualizer()
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Temporal Analysis", "Environmental Factors", "Vehicle Analysis", "Road Conditions"]
    )
    
    if analysis_type == "Temporal Analysis":
        st.subheader("⏰ Temporal Patterns in Accidents")
        
        # Time period selection
        time_period = st.radio("Select Time Period", ["Hour", "Day of Week", "Month", "Year"])
        
        if time_period == "Hour" and 'Hour' in df.columns:
            hourly_data = df.groupby(['Hour', 'Risk_Level']).size().reset_index(name='count')
            fig = px.line(
                hourly_data,
                x='Hour',
                y='count',
                color='Risk_Level',
                title="Accidents by Hour of Day",
                markers=True,
                color_discrete_map=VIZ_CONFIG["color_scheme"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rush hour analysis
            if 'Is_Rush_Hour' in df.columns:
                rush_hours = df[df['Is_Rush_Hour'] == 1]
                non_rush = df[df['Is_Rush_Hour'] == 0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rush Hour Accidents", f"{len(rush_hours):,}")
                with col2:
                    st.metric("Non-Rush Hour Accidents", f"{len(non_rush):,}")
        
        elif time_period == "Day of Week" and 'Day_of_Week' in df.columns:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_data = df.groupby(['Day_of_Week', 'Risk_Level']).size().reset_index(name='count')
            daily_data['Day_Name'] = daily_data['Day_of_Week'].map(dict(enumerate(day_names)))
            
            fig = px.bar(
                daily_data,
                x='Day_Name',
                y='count',
                color='Risk_Level',
                title="Accidents by Day of Week",
                barmode='group',
                color_discrete_map=VIZ_CONFIG["color_scheme"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info(f"Data for {time_period} analysis not available in current dataset")
    
    elif analysis_type == "Environmental Factors":
        st.subheader("🌦️ Environmental Impact on Accidents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Weather_Risk_Score' in df.columns:
                weather_data = df.groupby(['Weather_Risk_Score', 'Risk_Level']).size().reset_index(name='count')
                fig_weather = px.bar(
                    weather_data,
                    x='Weather_Risk_Score',
                    y='count',
                    color='Risk_Level',
                    title="Accidents by Weather Risk Score",
                    barmode='stack',
                    color_discrete_map=VIZ_CONFIG["color_scheme"]
                )
                st.plotly_chart(fig_weather, use_container_width=True)
        
        with col2:
            if 'Light_Risk_Score' in df.columns:
                light_data = df.groupby(['Light_Risk_Score', 'Risk_Level']).size().reset_index(name='count')
                fig_light = px.bar(
                    light_data,
                    x='Light_Risk_Score',
                    y='count',
                    color='Risk_Level',
                    title="Accidents by Light Risk Score",
                    barmode='stack',
                    color_discrete_map=VIZ_CONFIG["color_scheme"]
                )
                st.plotly_chart(fig_light, use_container_width=True)
    
    elif analysis_type == "Vehicle Analysis":
        st.subheader("🚗 Vehicle-Related Risk Factors")
        
        if 'Vehicle_Age' in df.columns:
            # Vehicle age analysis
            df['Age_Category'] = pd.cut(
                df['Vehicle_Age'],
                bins=[0, 3, 7, 10, 15, 100],
                labels=['0-3 years', '4-7 years', '8-10 years', '11-15 years', '15+ years'],
                include_lowest=True
            )
            
            age_data = df.groupby(['Age_Category', 'Risk_Level']).size().reset_index(name='count')
            
            fig_age = px.bar(
                age_data,
                x='Age_Category',
                y='count',
                color='Risk_Level',
                title="Accidents by Vehicle Age",
                barmode='group',
                color_discrete_map=VIZ_CONFIG["color_scheme"]
            )
            st.plotly_chart(fig_age, use_container_width=True)
    
    else:  # Road Conditions
        st.subheader("🛣️ Road Condition Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Road_Risk_Score' in df.columns:
                road_data = df.groupby(['Road_Risk_Score', 'Risk_Level']).size().reset_index(name='count')
                fig_road = px.bar(
                    road_data,
                    x='Road_Risk_Score',
                    y='count',
                    color='Risk_Level',
                    title="Accidents by Road Risk Score",
                    barmode='stack',
                    color_discrete_map=VIZ_CONFIG["color_scheme"]
                )
                st.plotly_chart(fig_road, use_container_width=True)
        
        with col2:
            if 'Speed_limit' in df.columns:
                speed_data = df.groupby(['Speed_limit', 'Risk_Level']).size().reset_index(name='count')
                fig_speed = px.bar(
                    speed_data,
                    x='Speed_limit',
                    y='count',
                    color='Risk_Level',
                    title="Accidents by Speed Limit",
                    barmode='group',
                    color_discrete_map=VIZ_CONFIG["color_scheme"]
                )
                st.plotly_chart(fig_speed, use_container_width=True)

else:
    st.warning("⚠️ Please load the data first from the Data Overview page")
    st.markdown("""
    ### Available Analysis Types:
    
    **📊 Temporal Analysis**
    - Hourly accident patterns and rush hour impacts
    - Day of week and weekend vs weekday comparisons  
    - Monthly and seasonal trends
    - Year-over-year accident statistics
    
    **🌦️ Environmental Factors**
    - Weather condition impacts on accident severity
    - Road surface condition analysis
    - Light condition effects (daylight vs darkness)
    - Combined environmental risk scoring
    
    **🚗 Vehicle Analysis**
    - Vehicle age impact on accident risk
    - Vehicle type distribution in accidents
    - Correlation between vehicle condition and severity
    
    **🛣️ Road Conditions**
    - Speed limit analysis across different road types
    - Junction type complexity and accident patterns
    - Urban vs rural accident distributions
    - Infrastructure quality impact assessment
    """)