# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a UK Road Risk Level Classification System that provides comprehensive analysis and prediction of road accident risk levels. The system has been restructured into a professional, modular architecture with separate components for data processing, machine learning, web applications, and APIs.

## Architecture

The project follows a modern microservices architecture with the following components:

### Core Modules (`src/`)
- **data_loader.py**: Handles data acquisition from Kaggle and preprocessing
- **feature_engineering.py**: Creates comprehensive risk features and target variables
- **model_training.py**: Trains multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- **risk_predictor.py**: Provides real-time prediction capabilities for new data
- **visualization.py**: Creates interactive visualizations using Plotly

### API Layer (`api/`)
- **main.py**: FastAPI application with middleware and error handling
- **endpoints.py**: REST API endpoints for predictions and data access
- **models.py**: Pydantic schemas for request/response validation

### Web Application (`app/`)
- **streamlit_app.py**: Main Streamlit application entry point
- **pages/**: Multi-page dashboard structure
  - 1_📊_Data_Overview.py: Dataset statistics and overview
  - 2_📈_Risk_Analysis.py: Temporal and environmental analysis
  - 3_🗺️_Geographic_Analysis.py: Interactive maps and regional statistics
  - 4_🤖_Risk_Prediction.py: Real-time risk assessment interface
  - 5_📉_Model_Performance.py: Model evaluation and metrics
- **utils.py**: Shared utilities and helper functions

### Configuration (`config/`)
- **settings.py**: Centralized configuration management
- Environment-specific settings (development, testing, production)
- Model parameters, API configuration, visualization settings

## Common Commands

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
```

### Data Processing
```bash
# Download and process data
python scripts/download_data.py --limit-rows 10000

# Train models
python scripts/train_models.py --limit-rows 10000
```

### Running Applications
```bash
# Streamlit dashboard
streamlit run app/streamlit_app.py

# FastAPI server
uvicorn api.main:app --reload

# Both with Docker
docker-compose up --build
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api --cov=app tests/

# Run specific test file
pytest tests/test_api.py -v
```

### Deployment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Build Docker image
docker build -t uk-road-risk .
```

## Key Features and Functions

### Data Pipeline
- **Automated data loading** from Kaggle with caching
- **Data validation** and quality checks
- **Feature engineering** with comprehensive risk scoring
- **Model training** with cross-validation and hyperparameter tuning
- **Model persistence** with joblib serialization

### Machine Learning
- **Risk Classification**: High/Medium/Low risk levels based on accident severity
- **Feature Engineering**: 17 engineered features including environmental, temporal, and vehicle factors
- **Multiple Models**: Random Forest (best: 89% accuracy), Gradient Boosting, Logistic Regression
- **Clustering Analysis**: K-Means, Hierarchical, and DBSCAN for pattern discovery
- **Model Evaluation**: Comprehensive metrics with cross-validation

### Web Applications
- **Multi-page Streamlit dashboard** with navigation and session state
- **RESTful API** with FastAPI, automatic documentation, and rate limiting
- **Interactive visualizations** with Plotly
- **Real-time predictions** with input validation
- **Geographic analysis** with accident hotspot mapping

### API Endpoints
- `POST /api/v1/predict`: Single risk prediction
- `POST /api/v1/predict/batch`: Batch predictions
- `GET /api/v1/model/performance`: Model metrics
- `GET /api/v1/data/overview`: Dataset statistics
- `GET /api/v1/features/importance`: Feature importance scores
- `GET /api/v1/health`: Health check endpoint

## File Structure and Data Management

### Data Organization
```
data/
├── raw/           # Original dataset files
├── processed/     # Engineered features and clean data
└── models/        # Trained model artifacts
    ├── best_risk_classifier.pkl
    ├── feature_scaler.pkl
    └── label_encoder.pkl
```

### Configuration Management
- **Centralized settings** in `config/settings.py`
- **Environment variables** via `.env` file
- **Feature mappings** for risk scoring
- **Model hyperparameters** and training configuration

### Testing Structure
- **Unit tests** for all core modules
- **API endpoint testing** with FastAPI TestClient
- **Data validation tests** for feature engineering
- **Model performance tests** for ML pipeline

## Development Guidelines

### Code Organization
- **Modular design** with separation of concerns
- **Type hints** throughout the codebase
- **Logging** with configurable levels
- **Error handling** with custom exceptions
- **Documentation** with docstrings

### Model Training Best Practices
- **Stratified sampling** for balanced train/test splits
- **Cross-validation** for robust performance evaluation
- **Feature scaling** with StandardScaler
- **Model comparison** across multiple algorithms
- **Feature importance analysis** for interpretability

### API Development
- **Pydantic models** for request/response validation
- **FastAPI features**: automatic docs, middleware, error handling
- **Rate limiting** and security considerations
- **Comprehensive error responses** with timestamps

### Streamlit Best Practices
- **Session state management** for data persistence
- **Caching** with @st.cache_data for expensive operations
- **Multi-page architecture** with proper navigation
- **Responsive design** with column layouts
- **Interactive widgets** for user input

## Deployment and Production

### Docker Configuration
- **Multi-stage builds** for optimization
- **Service orchestration** with Docker Compose
- **Environment configuration** for development/production
- **Health checks** and monitoring
- **Volume mounting** for data persistence

### Production Considerations
- **Environment variables** for configuration
- **Database integration** ready (PostgreSQL/SQLite)
- **Caching layer** with Redis
- **Reverse proxy** with Nginx
- **SSL/TLS** configuration support
- **Logging and monitoring** setup

### Performance Optimization
- **Data sampling** for large datasets
- **Model caching** for faster predictions
- **Efficient data structures** with Pandas
- **Memory management** for large datasets
- **API rate limiting** for production use

## Troubleshooting

### Common Issues
- **Import errors**: Ensure PYTHONPATH includes project root
- **Data loading failures**: Check Kaggle API credentials
- **Memory issues**: Use data limiting parameters
- **Port conflicts**: Check Docker port mappings
- **Model loading errors**: Verify model artifacts exist

### Development Tips
- **Use limiting parameters** during development (`--limit-rows`)
- **Check logs** in `app.log` for debugging
- **Test API endpoints** with `/docs` interface
- **Monitor Docker logs** with `docker-compose logs -f`
- **Use pytest fixtures** for consistent test data

This modular architecture provides a solid foundation for scaling, maintenance, and deployment of the UK Road Risk Classification System.