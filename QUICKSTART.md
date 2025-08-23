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

### 3. **Train Models** (using sample data)
```bash
python scripts/train_models_sample.py
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

**🎨 Streamlit Dashboard** (5 pages)
- 📊 Data Overview: Load and explore datasets
- 📈 Risk Analysis: Temporal and environmental patterns  
- 🗺️ Geographic Analysis: Interactive maps and hotspots
- 🤖 Risk Prediction: Real-time risk assessment
- 📉 Model Performance: ML metrics and evaluation

**🔌 FastAPI Backend**
- `/api/v1/predict` - Single risk prediction
- `/api/v1/predict/batch` - Batch predictions
- `/api/v1/model/performance` - Model metrics
- `/api/v1/health` - Health check
- `/docs` - Interactive API documentation

**🛠️ Scripts & Tools**
- Data download and processing
- Model training pipeline
- Comprehensive test suite
- Docker deployment ready

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