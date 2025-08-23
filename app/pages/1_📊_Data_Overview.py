"""
Data Overview page for the UK Road Risk Assessment Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils import (
    load_and_process_data, apply_custom_css, initialize_session_state,
    display_metrics_cards, create_download_button, display_data_quality_info
)
from src.visualization import RiskVisualizer
from config.settings import VIZ_CONFIG

# Apply styling and initialize state
apply_custom_css()
initialize_session_state()

# Add option to clear cache on page load (for debugging)
if st.sidebar.checkbox("🔧 Debug Mode - Auto Clear Cache", value=False):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared on page load")

# Page header
st.header("📊 Data Overview")
st.markdown("""
This page provides an overview of the UK road safety dataset, including key statistics,
risk level distributions, and data quality metrics.
""")

# Check if we have potentially old cached data
if st.session_state.get('data_loaded', False) and st.session_state.get('df') is not None:
    df = st.session_state.df
    if 'Risk_Level' in df.columns:
        risk_dist = df['Risk_Level'].value_counts(normalize=True) * 100
        # If 100% Low Risk, likely old cached data
        if risk_dist.get('Low Risk', 0) >= 99:
            st.warning("⚠️ **Old cached data detected!** Click 'Clear Cache & Reload' below to see the corrected risk distribution.")
            st.info("🔧 We recently fixed the risk classification logic. The data should show ~86% Low, ~13% Medium, ~1% High risk.")

# Data loading section
st.subheader("Dataset Loading")
st.info("Click the button below to load and process the UK road safety dataset. This may take a few minutes on the first run.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Load Dataset", type="primary", use_container_width=True):
        # Clear any cached data to force fresh load
        if 'df' in st.session_state:
            del st.session_state.df
        if 'data_loaded' in st.session_state:
            del st.session_state.data_loaded
        
        df = load_and_process_data(force_refresh=False)  # Load full dataset
        if not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
            st.info(f"📊 Loaded {len(df):,} records with proper risk classification")

with col2:
    if st.button("Clear Cache & Reload", type="secondary", use_container_width=True):
        # Clear all session state and force cache refresh
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()  # Clear Streamlit's cache
        st.success("🗑️ Cache cleared! Click 'Load Dataset' to reload with latest changes.")
        st.info("💡 The page will now reload with fresh data.")

# Show data if loaded
if st.session_state.get('data_loaded', False) and st.session_state.get('df') is not None:
    df = st.session_state.df
    visualizer = RiskVisualizer()
    
    # Diagnostic information
    st.subheader("🔍 Data Diagnostic")
    if 'Risk_Level' in df.columns:
        risk_counts = df['Risk_Level'].value_counts()
        risk_percentages = df['Risk_Level'].value_counts(normalize=True) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Risk", f"{risk_counts.get('Low Risk', 0):,}", 
                     f"{risk_percentages.get('Low Risk', 0):.1f}%")
        with col2:
            st.metric("Medium Risk", f"{risk_counts.get('Medium Risk', 0):,}", 
                     f"{risk_percentages.get('Medium Risk', 0):.1f}%")
        with col3:
            st.metric("High Risk", f"{risk_counts.get('High Risk', 0):,}", 
                     f"{risk_percentages.get('High Risk', 0):.1f}%")
    else:
        st.error("❌ Risk_Level column not found! There may be an issue with feature engineering.")
        st.write("Available columns:", list(df.columns))
    
    # Display key metrics
    st.subheader("Key Statistics")
    display_metrics_cards(df)
    
    # Risk distribution
    st.subheader("Risk Level Distribution")
    
    # Create responsive columns based on screen size
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        # Pie chart
        risk_counts = df['Risk_Level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Accident Risk Distribution",
            color_discrete_map=VIZ_CONFIG["color_scheme"]
        )
        fig_pie.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Level Counts",
            labels={'x': 'Risk Level', 'y': 'Number of Accidents'},
            color=risk_counts.index,
            color_discrete_map=VIZ_CONFIG["color_scheme"]
        )
        fig_bar.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk percentages
    st.markdown("### Risk Distribution Percentages")
    risk_percentages = df['Risk_Level'].value_counts(normalize=True) * 100
    
    col1, col2, col3 = st.columns(3)
    for i, (risk_level, percentage) in enumerate(risk_percentages.items()):
        with [col1, col2, col3][i]:
            if risk_level == "High Risk":
                st.metric(f"🔴 {risk_level}", f"{percentage:.1f}%")
            elif risk_level == "Medium Risk":
                st.metric(f"🟡 {risk_level}", f"{percentage:.1f}%")
            else:
                st.metric(f"🟢 {risk_level}", f"{percentage:.1f}%")
    
    # Temporal overview
    st.subheader("Temporal Overview")
    
    if 'Year' in df.columns:
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            # Yearly trends
            yearly_data = df.groupby('Year').size().reset_index(name='count')
            fig_yearly = px.line(
                yearly_data,
                x='Year',
                y='count',
                title="Accidents by Year",
                markers=True
            )
            fig_yearly.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
        
        with col2:
            # Monthly distribution
            monthly_data = df.groupby('Month').size().reset_index(name='count')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_data['Month_Name'] = monthly_data['Month'].map(
                dict(enumerate(month_names, 1))
            )
            
            fig_monthly = px.bar(
                monthly_data,
                x='Month_Name',
                y='count',
                title="Accidents by Month"
            )
            fig_monthly.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig_monthly.update_xaxes(tickangle=45)
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Geographic overview
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.subheader("Geographic Distribution")
        
        # Simple geographic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            # Check for both string 'Urban' and numeric 1
            urban_accidents = ((df['Urban_or_Rural_Area'] == 'Urban') | (df['Urban_or_Rural_Area'] == 1)).sum()
            st.metric("Urban Accidents", f"{urban_accidents:,}")
        with col2:
            # Check for both string 'Rural' and numeric 2
            rural_accidents = ((df['Urban_or_Rural_Area'] == 'Rural') | (df['Urban_or_Rural_Area'] == 2)).sum()
            st.metric("Rural Accidents", f"{rural_accidents:,}")
        with col3:
            if 'Location_Accident_Count' in df.columns:
                avg_location_density = df['Location_Accident_Count'].mean()
                st.metric("Avg. Location Density", f"{avg_location_density:.1f}")
    
    # Data quality section
    display_data_quality_info(df)
    
    # Data sample
    st.subheader("Data Sample")
    
    # Column selection for display
    all_columns = df.columns.tolist()
    default_columns = [
        'Accident_Index', 'Date', 'Time', 'Risk_Level', 'Accident_Severity',
        'Number_of_Vehicles', 'Number_of_Casualties', 'Weather_Conditions',
        'Road_Surface_Conditions', 'Light_Conditions', 'Speed_limit'
    ]
    
    # Filter to only existing columns
    default_display = [col for col in default_columns if col in all_columns]
    if not default_display:
        default_display = all_columns[:10]  # Fallback to first 10 columns
    
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=all_columns,
        default=default_display
    )
    
    if selected_columns:
        st.dataframe(
            df[selected_columns].head(100),
            use_container_width=True,
            height=400
        )
    
    # Download section
    st.subheader("Download Data")
    
    # Use responsive columns for download buttons
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    
    with col1:
        create_download_button(
            df.head(1000), 
            "uk_road_risk_sample.csv", 
            "Sample Data (1000 rows)"
        )
    
    with col2:
        if len(df) <= 10000:
            create_download_button(
                df, 
                "uk_road_risk_full.csv", 
                "Full Dataset"
            )
        else:
            st.info("Full dataset too large. Use sample download.")
    
    with col3:
        # Summary statistics
        summary_stats = df.describe()
        create_download_button(
            summary_stats, 
            "uk_road_risk_summary.csv", 
            "Summary Statistics"
        )
    
    # Dataset information
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
    with col2:
        if 'Date' in df.columns:
            st.write(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
        st.write(f"**Risk Categories:** {df['Risk_Level'].nunique()}")
        if 'Location_Accident_Count' in df.columns:
            st.write(f"**Unique Locations:** {df[['Latitude', 'Longitude']].drop_duplicates().shape[0]:,}")

else:
    st.info("👆 Please click 'Load Dataset' to begin exploring the data.")
    
    st.markdown("""
    ### What you'll see after loading:
    - **Key Statistics**: Total accidents, casualties, and date range
    - **Risk Distribution**: Visual breakdown of High/Medium/Low risk accidents  
    - **Temporal Patterns**: Yearly and monthly accident trends
    - **Geographic Overview**: Urban vs rural accident distribution
    - **Data Quality**: Missing values and data type information
    - **Interactive Data Table**: Browse and filter the actual dataset
    - **Download Options**: Export processed data for further analysis
    
    The dataset contains detailed information about UK road accidents including:
    - Environmental conditions (weather, road surface, lighting)
    - Temporal information (date, time, day of week)
    - Vehicle details (age, type, number involved)  
    - Geographic coordinates and road characteristics
    - Casualty information and accident severity
    """)