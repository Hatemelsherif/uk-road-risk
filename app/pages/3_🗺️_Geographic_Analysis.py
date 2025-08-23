"""
Geographic Analysis page for the UK Road Risk Assessment Dashboard
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

st.header("🗺️ Geographic Analysis")
st.markdown("""
Explore the geographical distribution of accidents, identify hotspots, 
and analyze regional patterns in road safety data.
""")

if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    visualizer = RiskVisualizer()
    
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.subheader("🌍 Accident Distribution Map")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.multiselect(
                "Select Risk Levels",
                options=['High Risk', 'Medium Risk', 'Low Risk'],
                default=['High Risk', 'Medium Risk', 'Low Risk']
            )
        with col2:
            sample_size = st.slider(
                "Sample Size (for performance)",
                min_value=100,
                max_value=min(5000, len(df)),
                value=min(1000, len(df)),
                step=100
            )
        with col3:
            map_style = st.selectbox(
                "Map Style",
                ["open-street-map", "carto-positron", "carto-darkmatter"]
            )
        
        # Filter and sample data
        df_filtered = df[df['Risk_Level'].isin(risk_filter)]
        df_sample = df_filtered.sample(min(sample_size, len(df_filtered)))
        
        # Create scatter map
        fig_map = visualizer.create_geographical_plot(df_sample, sample_size, map_style)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Create density heatmap
        st.subheader("🔥 Accident Density Heatmap")
        fig_density = visualizer.create_density_heatmap(df_sample, sample_size, map_style)
        if fig_density:
            st.plotly_chart(fig_density, use_container_width=True)
        
        # Regional statistics
        st.subheader("📊 Regional Statistics")
        
        # Urban vs Rural comparison
        if 'Urban_or_Rural_Area' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                urban_accidents = (df['Urban_or_Rural_Area'] == 1).sum()
                st.metric("Urban Accidents", f"{urban_accidents:,}")
            with col2:
                rural_accidents = (df['Urban_or_Rural_Area'] == 2).sum()
                st.metric("Rural Accidents", f"{rural_accidents:,}")
            
            # Urban vs Rural risk distribution
            urban_rural_data = df.groupby(['Urban_or_Rural_Area', 'Risk_Level']).size().reset_index(name='count')
            urban_rural_data['Area_Type'] = urban_rural_data['Urban_or_Rural_Area'].map({1: 'Urban', 2: 'Rural'})
            
            fig_urban_rural = px.bar(
                urban_rural_data,
                x='Area_Type',
                y='count',
                color='Risk_Level',
                title="Urban vs Rural Accident Distribution",
                barmode='group',
                color_discrete_map=VIZ_CONFIG["color_scheme"]
            )
            st.plotly_chart(fig_urban_rural, use_container_width=True)
        
        # Hotspot analysis
        if 'Location_Accident_Count' in df.columns:
            st.subheader("🔥 Accident Hotspots")
            
            hotspots = df[df['Location_Accident_Count'] > df['Location_Accident_Count'].quantile(0.9)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Hotspots", f"{len(hotspots):,}")
            with col2:
                avg_hotspot_accidents = hotspots['Location_Accident_Count'].mean()
                st.metric("Avg Accidents per Hotspot", f"{avg_hotspot_accidents:.1f}")
            with col3:
                hotspot_casualties = hotspots['Number_of_Casualties'].sum()
                st.metric("Total Casualties at Hotspots", f"{hotspot_casualties:,}")
            
            # Show top hotspots
            st.markdown("**Top 10 Accident Locations:**")
            top_hotspots = (hotspots.groupby(['Latitude', 'Longitude'])
                           .agg({
                               'Location_Accident_Count': 'first',
                               'Number_of_Casualties': 'sum',
                               'Risk_Level': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
                           })
                           .sort_values('Location_Accident_Count', ascending=False)
                           .head(10)
                           .reset_index())
            
            st.dataframe(top_hotspots, use_container_width=True)
        
        # Geographic patterns by time
        st.subheader("⏰ Temporal Geographic Patterns")
        
        if 'Hour' in df.columns:
            time_geo = df.groupby(['Hour', 'Urban_or_Rural_Area']).size().reset_index(name='count')
            time_geo['Area_Type'] = time_geo['Urban_or_Rural_Area'].map({1: 'Urban', 2: 'Rural'})
            
            fig_time_geo = px.line(
                time_geo,
                x='Hour',
                y='count',
                color='Area_Type',
                title="Hourly Accident Patterns: Urban vs Rural",
                markers=True
            )
            st.plotly_chart(fig_time_geo, use_container_width=True)
        
    else:
        st.warning("⚠️ Geographic coordinates not available in the current dataset")
        st.info("The geographic analysis requires latitude and longitude data to create maps and spatial analysis.")

else:
    st.warning("⚠️ Please load the data first from the Data Overview page")
    st.markdown("""
    ### Geographic Analysis Features:
    
    **🗺️ Interactive Maps**
    - Scatter plot showing individual accident locations
    - Color-coded by risk level (High/Medium/Low)
    - Size indicates number of casualties
    - Multiple map styles available
    
    **🔥 Density Heatmaps**
    - Visual representation of accident concentration
    - Identify high-risk areas and corridors
    - Adjustable sample sizes for performance
    
    **📊 Regional Comparisons**
    - Urban vs Rural accident patterns
    - Speed limit correlation with location type
    - Junction type distribution by area
    
    **🎯 Hotspot Analysis**
    - Locations with highest accident frequency
    - Casualty impact assessment
    - Temporal patterns at high-risk locations
    
    **⏰ Temporal Geographic Patterns**
    - How accident patterns vary by time and location
    - Rush hour impacts in urban vs rural areas
    - Seasonal geographic variations
    """)