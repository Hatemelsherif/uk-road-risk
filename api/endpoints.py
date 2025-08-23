"""
API endpoints for UK Road Risk Assessment System
"""

import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import logging

from api.models import (
    RoadConditionInput, RiskPredictionResponse, BatchPredictionInput, 
    BatchPredictionResponse, ModelPerformanceResponse, HealthCheckResponse,
    DataOverviewResponse, FeatureImportanceResponse, ErrorResponse
)
from src.risk_predictor import RiskPredictor
from src.feature_engineering import RiskFeatureEngineer
from config.settings import MODEL_CONFIG, VIZ_CONFIG

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize components
risk_predictor = RiskPredictor()
feature_engineer = RiskFeatureEngineer()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        model_loaded=risk_predictor.model is not None,
        uptime_seconds=time.time()
    )

@router.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(conditions: RoadConditionInput):
    """Predict risk level for given road conditions"""
    
    start_time = time.time()
    
    try:
        # Convert input to feature format
        input_data = _convert_input_to_features(conditions)
        
        if risk_predictor.model is not None:
            # Use trained model
            risk_level, probabilities = risk_predictor.predict_risk_level(input_data)
            confidence = max(probabilities.values())
        else:
            # Use simple risk calculation
            simple_conditions = {
                'weather': conditions.weather_condition.value,
                'road_surface': conditions.road_surface.value,
                'light_condition': conditions.light_condition.value,
                'junction_type': conditions.junction_type.value,
                'is_rush_hour': conditions.hour in [7, 8, 9, 17, 18, 19],
                'vehicle_age': conditions.vehicle_age
            }
            risk_level, risk_score, recommendations = risk_predictor.calculate_simple_risk_score(simple_conditions)
            
            # Create mock probabilities
            if risk_level == "High Risk":
                probabilities = {"High Risk": 0.7, "Medium Risk": 0.2, "Low Risk": 0.1}
            elif risk_level == "Medium Risk":
                probabilities = {"High Risk": 0.2, "Medium Risk": 0.6, "Low Risk": 0.2}
            else:
                probabilities = {"High Risk": 0.1, "Medium Risk": 0.2, "Low Risk": 0.7}
            
            confidence = max(probabilities.values())
        
        # Generate recommendations
        recommendations = _generate_recommendations(conditions)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RiskPredictionResponse(
            predicted_risk_level=risk_level,
            risk_score=max(probabilities.values()),
            probabilities=probabilities,
            confidence=confidence,
            recommendations=recommendations,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchPredictionInput):
    """Perform batch predictions on multiple road conditions"""
    
    start_time = time.time()
    
    try:
        predictions = []
        
        for conditions in batch_input.conditions:
            # Convert to individual prediction
            prediction_response = await predict_risk(conditions)
            predictions.append(prediction_response)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/model/performance", response_model=ModelPerformanceResponse)
async def get_model_performance():
    """Get current model performance metrics"""
    
    try:
        # Mock performance data (in production, load from model metadata)
        return ModelPerformanceResponse(
            model_name="Random Forest",
            accuracy=0.89,
            precision=0.88,
            recall=0.89,
            f1_score=0.88,
            cross_validation_score=0.87,
            last_trained=datetime.now(),
            feature_importance={
                "Environmental_Risk": 0.18,
                "Junction_Risk": 0.15,
                "Speed_limit": 0.12,
                "Hour": 0.11,
                "Weather_Risk_Score": 0.10
            }
        )
        
    except Exception as e:
        logger.error(f"Performance retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance: {str(e)}")

@router.get("/data/overview", response_model=DataOverviewResponse)
async def get_data_overview():
    """Get overview statistics of the dataset"""
    
    try:
        # Mock data overview (in production, query actual database)
        return DataOverviewResponse(
            total_accidents=150000,
            total_casualties=200000,
            fatal_accidents=2500,
            date_range="2019-2023",
            risk_distribution={
                "High Risk": 2500,
                "Medium Risk": 25000,
                "Low Risk": 122500
            },
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Data overview error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data overview: {str(e)}")

@router.get("/features/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Get feature importance from the trained model"""
    
    try:
        # Mock feature importance (in production, load from saved model)
        features = {
            "Environmental_Risk": 0.18,
            "Junction_Risk": 0.15,
            "Speed_limit": 0.12,
            "Hour": 0.11,
            "Weather_Risk_Score": 0.10,
            "Number_of_Casualties": 0.09,
            "Road_Risk_Score": 0.08,
            "Light_Risk_Score": 0.07,
            "Vehicle_Age": 0.06,
            "Is_Weekend": 0.04
        }
        
        top_features = [{"feature": k, "importance": v} for k, v in list(features.items())[:5]]
        
        return FeatureImportanceResponse(
            features=features,
            model_name="Random Forest",
            top_features=top_features
        )
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feature importance: {str(e)}")

def _convert_input_to_features(conditions: RoadConditionInput) -> Dict[str, Any]:
    """Convert API input to model feature format"""
    
    # Calculate derived features
    is_weekend = 1 if conditions.day_of_week in [5, 6] else 0
    is_rush_hour = 1 if conditions.hour in [7, 8, 9, 17, 18, 19] else 0
    
    # Map conditions to risk scores using the same logic as feature engineering
    weather_risk_map = {1: 0, 2: 1, 3: 2, 4: 1, 5: 2, 6: 3, 7: 2, 8: 2, 9: 0}
    road_risk_map = {1: 0, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1}
    light_risk_map = {1: 0, 4: 2, 5: 3, 6: 2, 7: 3}
    junction_risk_map = {0: 0, 1: 1, 2: 1, 3: 2, 5: 2, 6: 3, 7: 2, 8: 2, 9: 1}
    
    weather_risk = weather_risk_map.get(conditions.weather_condition.value, 0)
    road_risk = road_risk_map.get(conditions.road_surface.value, 0)
    light_risk = light_risk_map.get(conditions.light_condition.value, 0)
    junction_risk = junction_risk_map.get(conditions.junction_type.value, 0)
    
    environmental_risk = (weather_risk + road_risk + light_risk) / 3
    is_old_vehicle = 1 if conditions.vehicle_age > 10 else 0
    
    return {
        'Number_of_Vehicles': conditions.num_vehicles,
        'Number_of_Casualties': conditions.num_casualties,
        'Speed_limit': conditions.speed_limit,
        'Hour': conditions.hour,
        'Day_of_Week': conditions.day_of_week,
        'Month': conditions.month,
        'Is_Weekend': is_weekend,
        'Is_Rush_Hour': is_rush_hour,
        'Weather_Risk_Score': weather_risk,
        'Road_Risk_Score': road_risk,
        'Light_Risk_Score': light_risk,
        'Environmental_Risk': environmental_risk,
        'Vehicle_Age': conditions.vehicle_age,
        'Is_Old_Vehicle': is_old_vehicle,
        'Location_Accident_Count': 1,  # Default value
        'Junction_Risk': junction_risk,
        'Urban_or_Rural_Area': conditions.urban_rural
    }

def _generate_recommendations(conditions: RoadConditionInput) -> Dict[str, str]:
    """Generate safety recommendations based on conditions"""
    
    recommendations = {}
    
    # Weather-based recommendations
    if conditions.weather_condition in [2, 3, 5, 6, 7]:  # Adverse weather
        recommendations['weather'] = "⚠️ Adverse weather conditions - reduce speed and increase following distance"
    
    # Road surface recommendations
    if conditions.road_surface in [2, 3, 4, 5]:  # Poor road conditions
        recommendations['road'] = "⚠️ Poor road surface - exercise extra caution and reduce speed"
    
    # Light condition recommendations
    if conditions.light_condition in [5, 6, 7]:  # Dark conditions
        recommendations['light'] = "⚠️ Limited visibility - ensure lights are on and remain alert"
    
    # Time-based recommendations
    if conditions.hour in [7, 8, 9, 17, 18, 19]:  # Rush hour
        recommendations['traffic'] = "⚠️ Rush hour traffic - expect congestion and allow extra time"
    
    # Junction recommendations
    if conditions.junction_type in [3, 6, 7]:  # Complex junctions
        recommendations['junction'] = "⚠️ Complex junction ahead - approach with caution"
    
    # Vehicle age recommendations
    if conditions.vehicle_age > 10:
        recommendations['vehicle'] = "⚠️ Older vehicle - ensure regular maintenance and safety checks"
    
    if not recommendations:
        recommendations['general'] = "✅ Conditions appear favorable - maintain safe driving practices"
    
    return recommendations