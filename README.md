# UK Road Risk Classification System

A comprehensive machine learning system for analyzing and predicting road accident risk levels using UK government data. Features advanced data processing, multiple ML algorithms including PyTorch deep learning, ensemble methods, and real-time web interfaces.

## 🎯 Key Features

- **🤖 Advanced ML Pipeline**: Multiple algorithms including Random Forest, XGBoost, LightGBM, CatBoost, and ensemble methods
- **🧠 Deep Learning Integration**: PyTorch-based neural networks with MPS acceleration for Apple Silicon
- **📊 Interactive Dashboard**: Streamlit-based web interface with real-time training monitoring
- **🔄 RESTful API**: FastAPI-powered prediction service with automatic documentation
- **🛡️ Process Management**: Comprehensive orphaned process prevention and monitoring
- **📈 Feature Engineering**: 17+ engineered features for optimal model performance
- **🍎 Apple Silicon Optimized**: Native performance on M1/M2/M3 Macs with MPS acceleration
- **🔧 Process Safety**: Advanced worker process management prevents system overload

## 🏆 Model Performance

**Best Model: Stacking Ensemble**
- **F1-Score**: 84.53%
- **Accuracy**: 87.72%
- **Architecture**: Combines Random Forest, Extra Trees, and Gradient Boosting

**PyTorch Deep Learning Results:**
- **Simple Network**: 87.93% accuracy, 2.08s training time
- **Attention Network**: 87.93% accuracy, 1.48s training time  
- **Deep Network**: 82.76% accuracy, 2.51s training time
- **Wide Network**: 86.21% accuracy, 1.33s training time
- **Residual Network**: 74.14% accuracy, 1.67s training time

## 📊 Project Structure

```
uk-road-risk-classification/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── raw/              # Original dataset files
│   ├── processed/        # Processed data files
│   └── models/          # Saved model files
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_training_improved.py    # Enhanced ML training
│   ├── pytorch_deep_learning.py     # PyTorch neural networks  
│   ├── process_manager.py           # Orphaned process prevention
│   ├── training_monitor.py          # Real-time training monitoring
│   ├── risk_predictor.py
│   └── visualization.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── endpoints.py
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_📊_Data_Overview.py
│   │   ├── 2_📈_Risk_Analysis.py
│   │   ├── 3_🗺️_Geographic_Analysis.py
│   │   ├── 4_🤖_Risk_Prediction.py
│   │   ├── 5_📉_Model_Performance.py
│   │   └── 6_🚀_Model_Training.py    # Real-time training interface
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_api.py
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
│
└── scripts/
    ├── download_data.py
    ├── train_models.py
    ├── train_models_enhanced.py     # Enhanced training with all features
    ├── monitor_processes.py         # Process monitoring utility
    └── deploy.sh
    
├── PROCESS_PREVENTION_GUIDE.md      # Process management documentation
└── CLAUDE.md                        # Project development guidelines
```

## 🛠️ Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/your-username/uk-road-risk-classification.git
cd uk-road-risk-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **Access the applications**
- Streamlit Dashboard: http://localhost:8501
- FastAPI Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000/api/v1/health

## 🚀 Quick Start

### 1. Download and Process Data
```bash
python scripts/download_data.py --limit-rows 10000
```

### 2. Train Models

**Quick Training (5-10 minutes):**
```bash
python scripts/train_models_enhanced.py --quick --limit-rows 5000
```

**Enhanced Training with Deep Learning (15-30 minutes):**
```bash
python scripts/train_models_enhanced.py --use-deep-learning --balance-method smote --limit-rows 10000
```

**Best Performance (1-3 hours):**
```bash
python scripts/train_models_enhanced.py --full --tune-hyperparameters --use-ensemble --use-deep-learning
```

### 3. Run Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. Run FastAPI Server
```bash
uvicorn api.main:app --reload
```

## 📈 Usage

### Streamlit Dashboard

Navigate through the multi-page dashboard:

1. **📊 Data Overview**: Load dataset and view statistics
2. **📈 Risk Analysis**: Explore temporal and environmental patterns
3. **🗺️ Geographic Analysis**: Interactive maps and regional statistics
4. **🤖 Risk Prediction**: Real-time risk assessment tool
5. **📉 Model Performance**: Model metrics and evaluation
6. **🚀 Model Training**: Interactive training with real-time monitoring, process management, and deep learning options

### API Usage

#### Predict Risk Level
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 18,
    "day_of_week": 1,
    "month": 12,
    "weather_condition": 2,
    "road_surface": 2,
    "light_condition": 4,
    "speed_limit": 30,
    "num_vehicles": 2,
    "num_casualties": 1,
    "vehicle_age": 10,
    "junction_type": 3,
    "urban_rural": 1
  }'
```

#### Batch Predictions
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "conditions": [
      {...condition1...},
      {...condition2...}
    ]
  }'
```

## 🧮 Model Performance

Comprehensive model performance metrics:

| Model | Accuracy | F1-Score | Balanced Acc | Training Time | Notes |
|-------|----------|----------|---------------|---------------|-------|
| **Stacking Ensemble** | **87.72%** | **84.53%** | **38.00%** | 15min | Best overall |
| LightGBM | 87.72% | 83.49% | 35.67% | 5min | Fast training |
| Extra Trees | 86.84% | 84.08% | 37.67% | 8min | Good balance |
| CatBoost | 86.84% | 83.58% | 36.50% | 12min | Handles categories well |
| PyTorch Deep Network | 80.26% | 80.52% | 80.26% | 18s | Deep learning |
| PyTorch Simple Network | 76.75% | 78.02% | 76.75% | 14s | Fast neural net |
| PyTorch Attention | 66.67% | 71.65% | 66.67% | 13s | Attention mechanism |

*Benchmarks on 2,000 samples with full feature engineering and hyperparameter tuning*

## 🔧 Configuration

Key configuration files:

- `config/settings.py`: Application settings and model parameters
- `.env`: Environment variables and API keys
- `docker-compose.yml`: Container orchestration
- `requirements.txt`: Python dependencies

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api --cov=app tests/

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 Data Schema

### Risk Features
- **Environmental Risk**: Weather, road surface, and lighting conditions
- **Temporal Features**: Hour, day of week, month, rush hour indicators
- **Vehicle Features**: Age, type, and condition information
- **Location Features**: Geographic coordinates and junction complexity
- **Infrastructure**: Speed limits, urban/rural classification

### Risk Levels
- **🔴 High Risk**: Fatal accidents (Severity = 1)
- **🟡 Medium Risk**: Serious injury accidents (Severity = 2)
- **🟢 Low Risk**: Minor accidents with slight injuries (Severity = 3)

## 🛡️ Process Management & Monitoring

Advanced process management system prevents orphaned worker processes:

### Process Safety Features
- **Automatic cleanup** on script termination (atexit handlers)
- **Signal handling** for graceful interruption (SIGINT, SIGTERM)
- **Background monitoring** for orphaned process detection
- **Limited workers** (4 max) to prevent system overload
- **Emergency cleanup** utilities for stuck processes

### Process Monitoring Commands
```bash
# Monitor training processes in real-time
python scripts/monitor_processes.py --monitor --duration 60

# Clean up orphaned processes immediately  
python scripts/monitor_processes.py --cleanup

# Monitor continuously (until Ctrl+C)
python scripts/monitor_processes.py --monitor --duration 0
```

### Streamlit Process Management
The Model Training page includes:
- **🧹 Cleanup Processes** button for orphaned process cleanup
- **🔄 Refresh Status** for real-time status updates
- **⏹️ Stop Training** to force session reset
- Real-time progress monitoring with process status

### Troubleshooting
**Training appears stuck:**
```bash
python scripts/monitor_processes.py --cleanup
```

**High CPU usage from orphaned processes:**
```bash
# Check for orphaned processes
ps aux | grep joblib
python scripts/monitor_processes.py --cleanup
```

**Frontend shows wrong status:**
- Click "🧹 Cleanup Processes" in Streamlit Model Training page
- Use "🔄 Refresh Status" button to update display
- Restart Streamlit if needed: `streamlit run app/streamlit_app.py`

## 🔮 API Endpoints

### Core Endpoints
- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/model/performance` - Model metrics
- `GET /api/v1/data/overview` - Dataset statistics
- `GET /api/v1/features/importance` - Feature importance

### Documentation
- Interactive API docs: `/docs`
- ReDoc documentation: `/redoc`
- OpenAPI schema: `/openapi.json`

## 🚢 Deployment

### Production Deployment

1. **Set production environment**
```bash
export ENVIRONMENT=production
```

2. **Use production Docker Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Set up monitoring**
```bash
# Enable health checks and logging
docker-compose logs -f web
```

### Cloud Deployment

The application is ready for deployment on:
- AWS (ECS, EC2, Lambda)
- Google Cloud Platform (Cloud Run, GKE)
- Azure (Container Instances, AKS)
- Heroku
- Railway

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Dataset**: [UK Road Safety Data](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles)
- **Documentation**: [Project Wiki](https://github.com/your-username/uk-road-risk-classification/wiki)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/your-username/uk-road-risk-classification/issues)

## ⚠️ Disclaimer

This system provides risk assessments based on historical data patterns. Always follow official traffic regulations and exercise caution while driving regardless of predicted risk levels. The predictions are for educational and research purposes only.

## 📞 Support

For questions, issues, or contributions:
- 📧 Email: support@ukroadrisk.com
- 💬 Discord: [Join our community](https://discord.gg/ukroadrisk)
- 📖 Documentation: [Full docs](https://docs.ukroadrisk.com)

---

<div align="center">
Built with ❤️ using Python, Streamlit, FastAPI, and Machine Learning
</div>