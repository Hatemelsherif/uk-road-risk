"""
Configuration settings for the UK Road Risk Classification System
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" 
MODEL_DIR = DATA_DIR / "models"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "kaggle_dataset": "tsiaras/uk-road-safety-accidents-and-vehicles",
    "accident_file": "Accident_Information.csv",
    "vehicle_file": "Vehicle_Information.csv",
    "sample_size_limit": 50000,  # For demo/testing purposes
    "full_dataset": False  # Set to True for production
}

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_clusters": 3,
    "models": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "logistic_regression": {
            "random_state": 42,
            "max_iter": 1000,
            "multi_class": "multinomial"
        }
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "risk_maps": {
        "weather": {
            1: 0, 2: 1, 3: 2, 4: 1, 5: 2, 6: 3, 7: 2, 8: 2, 9: 0
        },
        "road_surface": {
            1: 0, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1
        },
        "light_conditions": {
            1: 0, 4: 2, 5: 3, 6: 2, 7: 3
        },
        "junction_detail": {
            0: 0, 1: 1, 2: 1, 3: 2, 5: 2, 6: 3, 7: 2, 8: 2, 9: 1
        }
    },
    "feature_columns": [
        'Number_of_Vehicles', 'Number_of_Casualties',
        'Speed_limit', 'Hour', 'Day_of_Week', 'Month',
        'Is_Weekend', 'Is_Rush_Hour', 'Weather_Risk_Score',
        'Road_Risk_Score', 'Light_Risk_Score', 'Environmental_Risk',
        'Vehicle_Age', 'Is_Old_Vehicle', 'Location_Accident_Count',
        'Junction_Risk', 'Urban_or_Rural_Area'
    ]
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "UK Road Risk Assessment Dashboard",
    "page_icon": "🚗",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "pages": [
        "📊 Data Overview",
        "📈 Risk Analysis", 
        "🗺️ Geographic Analysis",
        "🤖 Risk Prediction",
        "📉 Model Performance",
        "🚀 Model Training"
    ]
}

# API configuration
API_CONFIG = {
    "title": "UK Road Risk Assessment API",
    "description": "REST API for road risk prediction and analysis",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "rate_limit": {
        "calls": 100,
        "period": 60  # seconds
    }
}

# Visualization configuration
VIZ_CONFIG = {
    "color_scheme": {
        "High Risk": "#FF4444",
        "Medium Risk": "#FFA500",
        "Low Risk": "#44FF44"
    },
    "map_config": {
        "default_zoom": 5,
        "sample_size": 5000,
        "map_style": "open-street-map"
    },
    "plot_height": 600,
    "plot_width": 800
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
        {
            "class": "logging.StreamHandler",
        },
        {
            "class": "logging.FileHandler",
            "filename": str(PROJECT_ROOT / "app.log"),
        }
    ],
    "root": {
        "level": "INFO"
    }
}

# Environment-specific overrides
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    MODEL_CONFIG["models"]["random_forest"]["n_estimators"] = 200
    DATASET_CONFIG["full_dataset"] = True
    DATASET_CONFIG["sample_size_limit"] = None
    API_CONFIG["reload"] = False
    LOGGING_CONFIG["root"]["level"] = "WARNING"

elif ENVIRONMENT == "testing":
    DATASET_CONFIG["sample_size_limit"] = 1000
    MODEL_CONFIG["cv_folds"] = 3
    MODEL_CONFIG["models"]["random_forest"]["n_estimators"] = 50

# Database configuration (for future use)
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///uk_road_risk.db"),
    "echo": False,
    "pool_size": 10,
    "max_overflow": 20
}

# External API keys (for future integrations)
EXTERNAL_APIS = {
    "weather_api_key": os.getenv("WEATHER_API_KEY", ""),
    "mapbox_token": os.getenv("MAPBOX_TOKEN", ""),
    "google_maps_key": os.getenv("GOOGLE_MAPS_KEY", "")
}