"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app
from api.models import RoadConditionInput, WeatherCondition, RoadSurface, LightCondition, JunctionType

client = TestClient(app)

class TestAPIEndpoints:
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "UK Road Risk Assessment API" in data["message"]
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_status_endpoint(self):
        """Test status endpoint"""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "endpoints" in data
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction endpoint with valid input"""
        valid_input = {
            "hour": 10,
            "day_of_week": 1,
            "month": 6,
            "weather_condition": WeatherCondition.FINE.value,
            "road_surface": RoadSurface.DRY.value,
            "light_condition": LightCondition.DAYLIGHT.value,
            "speed_limit": 30,
            "num_vehicles": 2,
            "num_casualties": 1,
            "vehicle_age": 5,
            "junction_type": JunctionType.NOT_JUNCTION.value,
            "urban_rural": 1
        }
        
        response = client.post("/api/v1/predict", json=valid_input)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_risk_level" in data
        assert "probabilities" in data
        assert "recommendations" in data
        assert "processing_time_ms" in data
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        invalid_input = {
            "hour": 25,  # Invalid hour
            "day_of_week": 1,
            "month": 6,
            "weather_condition": 1,
            "road_surface": 1,
            "light_condition": 1,
            "speed_limit": 30,
            "num_vehicles": 2,
            "num_casualties": 1,
            "vehicle_age": 5,
            "junction_type": 0,
            "urban_rural": 1
        }
        
        response = client.post("/api/v1/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        batch_input = {
            "conditions": [
                {
                    "hour": 10,
                    "day_of_week": 1,
                    "month": 6,
                    "weather_condition": WeatherCondition.FINE.value,
                    "road_surface": RoadSurface.DRY.value,
                    "light_condition": LightCondition.DAYLIGHT.value,
                    "speed_limit": 30,
                    "num_vehicles": 2,
                    "num_casualties": 1,
                    "vehicle_age": 5,
                    "junction_type": JunctionType.NOT_JUNCTION.value,
                    "urban_rural": 1
                },
                {
                    "hour": 18,
                    "day_of_week": 5,
                    "month": 12,
                    "weather_condition": WeatherCondition.RAIN.value,
                    "road_surface": RoadSurface.WET_DAMP.value,
                    "light_condition": LightCondition.DARK_LIT.value,
                    "speed_limit": 60,
                    "num_vehicles": 1,
                    "num_casualties": 0,
                    "vehicle_age": 15,
                    "junction_type": JunctionType.ROUNDABOUT.value,
                    "urban_rural": 2
                }
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=batch_input)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_processed" in data
        assert len(data["predictions"]) == 2
    
    def test_model_performance_endpoint(self):
        """Test model performance endpoint"""
        response = client.get("/api/v1/model/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
    
    def test_data_overview_endpoint(self):
        """Test data overview endpoint"""
        response = client.get("/api/v1/data/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_accidents" in data
        assert "total_casualties" in data
        assert "risk_distribution" in data
    
    def test_feature_importance_endpoint(self):
        """Test feature importance endpoint"""
        response = client.get("/api/v1/features/importance")
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert "model_name" in data
        assert "top_features" in data
    
    def test_predict_with_optional_coordinates(self):
        """Test prediction with optional latitude/longitude"""
        input_with_coords = {
            "hour": 10,
            "day_of_week": 1,
            "month": 6,
            "weather_condition": WeatherCondition.FINE.value,
            "road_surface": RoadSurface.DRY.value,
            "light_condition": LightCondition.DAYLIGHT.value,
            "speed_limit": 30,
            "num_vehicles": 2,
            "num_casualties": 1,
            "vehicle_age": 5,
            "junction_type": JunctionType.NOT_JUNCTION.value,
            "urban_rural": 1,
            "latitude": 51.5074,
            "longitude": -0.1278
        }
        
        response = client.post("/api/v1/predict", json=input_with_coords)
        assert response.status_code == 200
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty conditions list"""
        empty_batch = {"conditions": []}
        
        response = client.post("/api/v1/predict/batch", json=empty_batch)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_processed"] == 0
        assert len(data["predictions"]) == 0
    
    @patch('api.endpoints.risk_predictor')
    def test_predict_with_model_error(self, mock_predictor):
        """Test prediction when model throws error"""
        mock_predictor.predict_risk_level.side_effect = Exception("Model error")
        
        valid_input = {
            "hour": 10,
            "day_of_week": 1,
            "month": 6,
            "weather_condition": WeatherCondition.FINE.value,
            "road_surface": RoadSurface.DRY.value,
            "light_condition": LightCondition.DAYLIGHT.value,
            "speed_limit": 30,
            "num_vehicles": 2,
            "num_casualties": 1,
            "vehicle_age": 5,
            "junction_type": JunctionType.NOT_JUNCTION.value,
            "urban_rural": 1
        }
        
        response = client.post("/api/v1/predict", json=valid_input)
        assert response.status_code == 500