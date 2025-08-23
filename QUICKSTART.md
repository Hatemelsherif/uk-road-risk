# 🚀 Quick Start Guide

## ✅ System Status: READY TO USE!

All core components have been tested and are working correctly. The project has been successfully restructured into a professional, modular architecture.

## 🏃‍♂️ Quick Commands

### 1. **Install Dependencies** (if not already done)
```bash
pip install -r requirements.txt
```

### 2. **Generate Sample Data** (recommended - avoids Kaggle issues)
```bash
python scripts/generate_sample_data.py --num-rows 5000 --with-features
```

### 3. **Train Models** (Enhanced with PyTorch Deep Learning)
```bash
# Quick training with sample data
python scripts/train_models_sample.py

# Enhanced training with PyTorch deep learning
python scripts/train_models_enhanced.py --use-deep-learning --quick

# Full training with all features (Apple Silicon optimized)
python scripts/train_models_enhanced.py --full --use-deep-learning --use-ensemble
```

### 4. **Run Streamlit Dashboard**
```bash
streamlit run app/streamlit_app.py
```
🌐 Access at: http://localhost:8501

### 5. **Run FastAPI Server**
```bash
uvicorn api.main:app --reload
```
🌐 API docs at: http://localhost:8000/docs

### 6. **Run Both with Docker**
```bash
docker-compose up --build
```

## 📱 What's Available Now

### ✅ **Working Components**

**🎨 Streamlit Dashboard** (6 pages)
- 📊 Data Overview: Load and explore datasets
- 📈 Risk Analysis: Temporal and environmental patterns  
- 🗺️ Geographic Analysis: Interactive maps and hotspots
- 🤖 Risk Prediction: Real-time risk assessment
- 📉 Model Performance: ML metrics and evaluation
- 🚀 Model Training: Real-time training with PyTorch deep learning

**🧠 PyTorch Deep Learning Integration**
- Apple Silicon optimized with MPS acceleration
- 5 neural network architectures (Simple, Deep, Wide, Residual, Attention)
- Process management to prevent orphaned workers
- Real-time training monitoring with progress tracking

**🔌 FastAPI Backend**
- `/api/v1/predict` - Single risk prediction
- `/api/v1/predict/batch` - Batch predictions
- `/api/v1/model/performance` - Model metrics
- `/api/v1/health` - Health check
- `/docs` - Interactive API documentation

**🛠️ Scripts & Tools**
- Data download and processing
- Enhanced ML training pipeline with PyTorch
- Process monitoring and orphaned worker cleanup
- Comprehensive test suite
- Docker deployment ready

**🛡️ Process Management System**
- Automatic cleanup of orphaned training processes
- Background monitoring with CPU/memory tracking  
- Emergency cleanup utilities
- Apple Silicon performance optimization

### ⚠️ **Expected Behavior**
- **"Model artifacts not found"** warning is normal - system works without trained models
- **Simple risk scoring** used when ML models aren't trained
- **All pages and APIs functional** with sample/mock data

## 🎯 Next Steps

1. **Explore the Dashboard**: Start with Data Overview page to load sample data
2. **Test Risk Prediction**: Try the real-time prediction interface  
3. **Check API Docs**: Visit `/docs` endpoint for interactive API testing
4. **Train Models**: Run training scripts for ML-powered predictions
5. **Deploy**: Use Docker for production deployment

## 🐛 Troubleshooting

### Common Issues & Solutions

**Port already in use:**
```bash
# Kill processes on port 8501/8000
lsof -ti:8501 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

**Import errors:**
```bash
# Ensure you're in project root directory
cd /Users/hatemelsherif/Dropbox/pyprojects/uk-road-risk
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Memory issues:**
```bash
# Use smaller data samples
python scripts/download_data.py --limit-rows 100
```

**Orphaned training processes (high CPU usage):**
```bash
# Clean up orphaned processes immediately
python scripts/monitor_processes.py --cleanup

# Monitor training processes in real-time
python scripts/monitor_processes.py --monitor --duration 60

# Use Streamlit cleanup button in Model Training page
# Click "🧹 Cleanup Processes" button
```

**Training appears stuck:**
```bash
# Emergency cleanup and status reset
python scripts/monitor_processes.py --cleanup
streamlit run app/streamlit_app.py  # Restart interface
```

## 🏆 Success Indicators

When everything is working, you should see:
- ✅ Streamlit dashboard loads without errors
- ✅ All 5 pages accessible via sidebar
- ✅ API returns JSON responses at `/api/v1/health`
- ✅ Interactive API docs at `/docs`
- ✅ Risk prediction form accepts input and returns results

## 📞 Need Help?

- Check the full README.md for detailed documentation
- Review CLAUDE.md for development guidance  
- All core functionality is working and ready to use!

---

**🎉 Congratulations!** Your UK Road Risk Classification System is successfully restructured and ready for development or deployment.