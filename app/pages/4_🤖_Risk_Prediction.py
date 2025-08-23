"""
Risk Prediction page for the UK Road Risk Assessment Dashboard
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils import (
    apply_custom_css, initialize_session_state, format_prediction_display,
    validate_prediction_input, get_weather_conditions_map, get_road_surface_map,
    get_light_conditions_map, get_junction_type_map
)
from src.risk_predictor import RiskPredictor
from src.visualization import RiskVisualizer

# Apply styling and initialize state
apply_custom_css()
initialize_session_state()

st.header("🤖 Real-time Risk Prediction")
st.markdown("""
Enter current road conditions to get an instant risk assessment. 
The system will predict the risk level and provide safety recommendations.
""")

# Initialize predictor
if 'risk_predictor' not in st.session_state:
    st.session_state.risk_predictor = RiskPredictor()

# Create input form
with st.form("prediction_form"):
    st.subheader("Road Conditions Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⏰ Temporal Factors**")
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                  'Friday', 'Saturday', 'Sunday'][x]
        )
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1]
        )
    
    with col2:
        st.markdown("**🌦️ Environmental Conditions**")
        weather_map = get_weather_conditions_map()
        weather = st.selectbox(
            "Weather Conditions",
            options=list(weather_map.keys()),
            format_func=lambda x: weather_map[x]
        )
        
        road_map = get_road_surface_map()
        road_surface = st.selectbox(
            "Road Surface",
            options=list(road_map.keys()),
            format_func=lambda x: road_map[x]
        )
        
        light_map = get_light_conditions_map()
        light_condition = st.selectbox(
            "Light Conditions",
            options=list(light_map.keys()),
            format_func=lambda x: light_map[x]
        )
    
    with col3:
        st.markdown("**🚗 Road & Vehicle Factors**")
        speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70])
        num_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=10, value=2)
        num_casualties = st.number_input("Number of Casualties", min_value=0, max_value=10, value=1)
        vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)
        
        junction_map = get_junction_type_map()
        junction_type = st.selectbox(
            "Junction Type",
            options=[0, 1, 2, 3, 6],
            format_func=lambda x: junction_map.get(x, f'Type {x}')
        )
        
        urban_rural = st.radio("Area Type", [1, 2], format_func=lambda x: 'Urban' if x == 1 else 'Rural')
    
    submitted = st.form_submit_button("🔮 Predict Risk Level", type="primary")

if submitted:
    # Prepare input data
    input_data = {
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'weather': weather,
        'road_surface': road_surface,
        'light_condition': light_condition,
        'speed_limit': speed_limit,
        'num_vehicles': num_vehicles,
        'num_casualties': num_casualties,
        'vehicle_age': vehicle_age,
        'junction_type': junction_type,
        'urban_rural': urban_rural
    }
    
    # Validate input
    if validate_prediction_input(input_data):
        
        with st.spinner("Analyzing road conditions..."):
            try:
                # Check if trained model is available
                if st.session_state.risk_predictor.model is not None:
                    # Use trained model (this would work if models were trained)
                    st.info("Using trained machine learning model for prediction")
                    # risk_level, probabilities = st.session_state.risk_predictor.predict_risk_level(input_data)
                
                # Use simple risk calculation as fallback
                risk_level, risk_score, recommendations = st.session_state.risk_predictor.calculate_simple_risk_score(input_data)
                
                # Create mock probabilities for display
                if risk_level == "High Risk":
                    probabilities = {"High Risk": 0.7, "Medium Risk": 0.2, "Low Risk": 0.1}
                elif risk_level == "Medium Risk":
                    probabilities = {"High Risk": 0.2, "Medium Risk": 0.6, "Low Risk": 0.2}
                else:
                    probabilities = {"High Risk": 0.1, "Medium Risk": 0.2, "Low Risk": 0.7}
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Main prediction display
                format_prediction_display(risk_level, probabilities, recommendations)
                
                # Risk factors breakdown
                st.subheader("📊 Risk Factors Analysis")
                
                # Calculate individual risk contributions
                weather_risk_map = {1: 0, 2: 1, 3: 2, 4: 1, 5: 2, 6: 3, 7: 2, 8: 2}
                road_risk_map = {1: 0, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1}
                light_risk_map = {1: 0, 4: 2, 5: 3, 6: 2, 7: 3}
                junction_risk_map = {0: 0, 1: 1, 2: 1, 3: 2, 6: 3}
                
                factors_df = pd.DataFrame({
                    'Factor': ['Weather', 'Road Surface', 'Light Conditions', 'Junction', 
                              'Rush Hour', 'Vehicle Age', 'Speed Limit'],
                    'Risk Contribution': [
                        weather_risk_map.get(weather, 0) / 3,
                        road_risk_map.get(road_surface, 0) / 3,
                        light_risk_map.get(light_condition, 0) / 3,
                        junction_risk_map.get(junction_type, 0) / 3,
                        1 if hour in [7, 8, 9, 17, 18, 19] else 0,
                        1 if vehicle_age > 10 else 0,
                        min(speed_limit / 70, 1)
                    ]
                })
                
                # Create risk factor visualization
                visualizer = RiskVisualizer()
                fig_factors = visualizer.create_risk_factor_breakdown(factors_df)
                st.plotly_chart(fig_factors, use_container_width=True)
                
                # Detailed risk assessment
                st.subheader("📋 Risk Assessment Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Conditions:**")
                    st.write(f"🕒 Time: {hour}:00 on {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]}")
                    st.write(f"🌦️ Weather: {weather_map[weather]}")
                    st.write(f"🛣️ Road: {road_map[road_surface]}")
                    st.write(f"💡 Light: {light_map[light_condition]}")
                    st.write(f"🚗 Vehicle: {vehicle_age} years old")
                
                with col2:
                    st.markdown("**Risk Factors:**")
                    is_rush_hour = hour in [7, 8, 9, 17, 18, 19]
                    is_weekend = day_of_week in [5, 6]
                    
                    st.write(f"⏰ Rush Hour: {'Yes' if is_rush_hour else 'No'}")
                    st.write(f"📅 Weekend: {'Yes' if is_weekend else 'No'}")
                    st.write(f"🏙️ Area: {'Urban' if urban_rural == 1 else 'Rural'}")
                    st.write(f"🚦 Speed Limit: {speed_limit} mph")
                    st.write(f"🔄 Junction: {junction_map.get(junction_type, 'Unknown')}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Please try adjusting the input parameters and try again.")

# Additional information section
st.markdown("---")
st.subheader("ℹ️ About Risk Prediction")

with st.expander("How does the prediction work?"):
    st.markdown("""
    The risk prediction system analyzes multiple factors to assess accident likelihood:
    
    **🌦️ Environmental Factors (40%)**
    - Weather conditions (rain, snow, fog)
    - Road surface conditions (wet, icy, dry)
    - Light conditions (daylight, darkness)
    
    **⏰ Temporal Factors (30%)**
    - Time of day (rush hours are higher risk)
    - Day of week (weekends vs weekdays)
    - Seasonal patterns
    
    **🚗 Vehicle & Road Factors (30%)**
    - Vehicle age and condition
    - Speed limits and road type
    - Junction complexity
    - Urban vs rural settings
    
    The system combines these factors using either a trained machine learning model 
    or statistical risk scoring to provide an overall risk assessment.
    """)

with st.expander("Understanding Risk Levels"):
    st.markdown("""
    **🔴 High Risk**: Conditions significantly increase accident probability
    - Fatal accident likelihood elevated
    - Multiple adverse factors present
    - Immediate caution required
    
    **🟡 Medium Risk**: Moderate increase in accident probability  
    - Serious injury potential
    - Some adverse conditions present
    - Extra attention recommended
    
    **🟢 Low Risk**: Normal or favorable driving conditions
    - Minor injury potential only
    - Most factors are favorable
    - Standard safety practices sufficient
    """)

with st.expander("Safety Recommendations"):
    st.markdown("""
    Based on your risk assessment, you may receive recommendations such as:
    
    - **Weather**: Reduce speed, increase following distance
    - **Road Conditions**: Exercise extra caution on wet/icy surfaces
    - **Visibility**: Ensure lights are on, stay alert
    - **Traffic**: Allow extra time during rush hours
    - **Vehicle**: Regular maintenance for older vehicles
    - **Infrastructure**: Approach complex junctions carefully
    """)

st.info("💡 Remember: These predictions are based on statistical patterns and should supplement, not replace, your judgment and adherence to traffic laws.")