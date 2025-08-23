"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    HIGH = "High Risk"
    MEDIUM = "Medium Risk"
    LOW = "Low Risk"

class WeatherCondition(int, Enum):
    FINE = 1
    RAIN = 2
    SNOW = 3
    FINE_WINDY = 4
    RAIN_WINDY = 5
    SNOW_WINDY = 6
    FOG = 7
    OTHER = 8
    UNKNOWN = 9

class RoadSurface(int, Enum):
    DRY = 1
    WET_DAMP = 2
    SNOW = 3
    FROST_ICE = 4
    FLOOD = 5
    OIL_DIESEL = 6
    MUD = 7

class LightCondition(int, Enum):
    DAYLIGHT = 1
    DARK_LIT = 4
    DARK_UNLIT = 5
    DARK_NO_LIGHT = 6
    DARK_UNKNOWN = 7

class JunctionType(int, Enum):
    NOT_JUNCTION = 0
    ROUNDABOUT = 1
    MINI_ROUNDABOUT = 2
    T_JUNCTION = 3
    SLIP_ROAD = 5
    CROSSROADS = 6
    MULTI_ARM = 7
    PRIVATE_DRIVE = 8
    OTHER = 9

class RoadConditionInput(BaseModel):
    """Input schema for risk prediction"""
    
    # Temporal factors
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    
    # Environmental conditions
    weather_condition: WeatherCondition = Field(..., description="Weather condition")
    road_surface: RoadSurface = Field(..., description="Road surface condition")
    light_condition: LightCondition = Field(..., description="Light condition")
    
    # Road and vehicle factors
    speed_limit: int = Field(..., ge=10, le=100, description="Speed limit in mph")
    num_vehicles: int = Field(..., ge=1, le=10, description="Number of vehicles involved")
    num_casualties: int = Field(..., ge=0, le=20, description="Number of casualties")
    vehicle_age: int = Field(..., ge=0, le=50, description="Age of vehicle in years")
    junction_type: JunctionType = Field(..., description="Type of junction")
    urban_rural: int = Field(..., ge=1, le=2, description="1=Urban, 2=Rural")
    
    # Optional location data
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")

class RiskPredictionResponse(BaseModel):
    """Response schema for risk prediction"""
    
    predicted_risk_level: RiskLevel
    risk_score: float = Field(..., ge=0, le=1, description="Risk score between 0 and 1")
    probabilities: Dict[str, float] = Field(..., description="Probability for each risk level")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendations: Dict[str, str] = Field(..., description="Safety recommendations")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    
    conditions: List[RoadConditionInput] = Field(..., max_items=100, description="List of road conditions")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    predictions: List[RiskPredictionResponse]
    total_processed: int
    processing_time_ms: float

class ModelPerformanceResponse(BaseModel):
    """Response schema for model performance metrics"""
    
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_validation_score: float
    last_trained: datetime
    feature_importance: Dict[str, float]

class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    
    status: str
    timestamp: datetime
    version: str
    model_loaded: bool
    uptime_seconds: float

class ErrorResponse(BaseModel):
    """Standard error response schema"""
    
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None

class DataOverviewResponse(BaseModel):
    """Response schema for data overview statistics"""
    
    total_accidents: int
    total_casualties: int
    fatal_accidents: int
    date_range: str
    risk_distribution: Dict[str, int]
    last_updated: datetime

class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance"""
    
    features: Dict[str, float] = Field(..., description="Feature importance scores")
    model_name: str
    top_features: List[Dict[str, float]] = Field(..., description="Top 10 most important features")