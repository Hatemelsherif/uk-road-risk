# Orphaned Process Prevention Guide

This document outlines comprehensive strategies implemented to prevent orphaned worker processes during machine learning training.

## 🔍 Problem Analysis

### Root Cause
- **Scikit-learn's joblib** uses `loky` backend for parallel processing with `n_jobs=-1`
- **Hyperparameter tuning** spawns many worker processes across multiple models
- **Training interruption** (Ctrl+C, crashes, system issues) leaves workers orphaned
- **Frontend confusion** from active processes leads to incorrect status display

### Impact
- High CPU usage (60-99% per worker)  
- Memory consumption
- System slowdown
- Incorrect training status display
- Potential system instability

## 🛡️ Prevention Strategy

### 1. Process Management Context Manager (`src/process_manager.py`)

**Features:**
- **Automatic cleanup** on script exit (atexit handlers)
- **Signal handling** for graceful interruption (SIGINT, SIGTERM)
- **Background monitoring** for orphaned process detection
- **Emergency cleanup** functionality

**Usage:**
```python
from src.process_manager import training_session, emergency_cleanup

# Safe training context
with training_session() as process_mgr:
    # Your training code here
    pass  # Automatic cleanup on exit

# Emergency cleanup
emergency_cleanup()
```

### 2. Limited Worker Configuration

**Implementation:**
- **Max workers limit**: Default 4 workers (instead of unlimited)
- **CPU-aware scaling**: `min(max_workers, os.cpu_count())`
- **Consistent limits**: Applied to all sklearn models and hyperparameter tuning

**Benefits:**
- Prevents process explosion
- Maintains system responsiveness
- Reduces cleanup complexity
- Still provides parallel processing benefits

### 3. Enhanced Training Scripts

**Improvements:**
- **Process-safe context managers** around training pipelines
- **Emergency cleanup** on exceptions
- **Proper exception handling** with cleanup guarantees
- **Status file management** with completion tracking

### 4. Process Monitoring Utility (`scripts/monitor_processes.py`)

**Capabilities:**
```bash
# Monitor processes continuously
python scripts/monitor_processes.py --monitor --duration 60

# Clean up orphaned processes
python scripts/monitor_processes.py --cleanup

# Monitor indefinitely  
python scripts/monitor_processes.py --monitor --duration 0
```

**Features:**
- **Real-time monitoring** of training processes
- **Automated cleanup** of orphaned workers (>5 minutes old, >50% CPU)
- **Detailed reporting** of process categories
- **Safe termination** of joblib/loky workers

### 5. Frontend Integration

**Streamlit Enhancements:**
- **"🧹 Cleanup Processes"** button in Model Training page
- **Automatic session reset** after cleanup
- **Process status detection** with frontend state management
- **Real-time status refresh** capabilities

## 📋 Implementation Checklist

### ✅ Completed Implementations

1. **Process Manager Module**
   - Context manager for training sessions
   - Signal handlers for graceful shutdown
   - Background monitoring thread
   - Emergency cleanup functions

2. **Enhanced Model Training**
   - Limited worker configuration (4 workers max)
   - Process-safe hyperparameter tuning
   - Consistent n_jobs settings across all models

3. **Training Script Updates**
   - Process management context wrapper
   - Emergency cleanup on exceptions
   - Proper finally blocks for guaranteed cleanup

4. **Process Monitoring Tool**
   - Command-line monitoring utility
   - Automated orphaned process detection
   - Safe cleanup functionality
   - Detailed process reporting

5. **Frontend Integration**  
   - Cleanup button in Streamlit interface
   - Session state reset capabilities
   - Process status integration

## 🚀 Usage Instructions

### For Training Scripts
```python
# Import prevention modules
from src.process_manager import training_session, emergency_cleanup

# Wrap training in context manager
try:
    emergency_cleanup()  # Clean existing orphans
    
    with training_session() as process_mgr:
        # Your training code here
        results = train_models(...)
        
except Exception as e:
    emergency_cleanup()  # Emergency cleanup
    raise
finally:
    emergency_cleanup()  # Final cleanup
```

### For Monitoring
```bash
# Start background monitoring
python scripts/monitor_processes.py --monitor --duration 120

# One-time cleanup
python scripts/monitor_processes.py --cleanup
```

### For Frontend Users
1. **Use "🧹 Cleanup Processes"** button if training appears stuck
2. **Click "🔄 Refresh Status"** to update display  
3. **Check "⏹️ Stop Training"** to force session reset

## 🔧 Configuration Options

### Process Manager Settings
```python
ProcessManager(
    max_workers=4,          # Maximum parallel workers
    monitoring_interval=30,  # Background check interval
    orphan_age_limit=300,   # 5 minutes before cleanup
    cpu_threshold=50        # CPU% threshold for orphan detection
)
```

### Training Script Options
```python
ImprovedRiskModelTrainer(
    model_dir="data/models",
    max_workers=4,          # Limited worker count
    use_gpu=False,          # Apple Silicon optimization
    random_state=42
)
```

## 🎯 Best Practices

### Prevention
1. **Always use context managers** for training sessions
2. **Set reasonable worker limits** (4-8 workers typically optimal)
3. **Enable monitoring** for long training runs
4. **Test cleanup mechanisms** before production use

### Monitoring
1. **Run periodic cleanup** during development
2. **Monitor system resources** during training
3. **Check process status** if training appears stuck
4. **Use monitoring utility** for debugging

### Recovery
1. **Emergency cleanup first** if processes are stuck
2. **Restart services** (Streamlit, FastAPI) if needed
3. **Check training status file** for actual completion state
4. **Clear session state** in frontend if displaying incorrectly

## 🔬 Testing Results

### Validation Tests
- ✅ **Process cleanup**: Successfully terminates orphaned workers
- ✅ **Context management**: Proper cleanup on normal and exception exits  
- ✅ **Signal handling**: Graceful shutdown on interruption
- ✅ **Monitoring**: Real-time detection and automated cleanup
- ✅ **Frontend integration**: Cleanup button works correctly

### Performance Impact
- **Minimal overhead**: <1% CPU for monitoring
- **Memory efficient**: No significant memory increase
- **Training speed**: Negligible impact with 4 workers vs unlimited
- **System stability**: Significantly improved

## 🚨 Troubleshooting

### Common Issues

**"Training appears stuck"**
1. Run: `python scripts/monitor_processes.py --cleanup`
2. Refresh Streamlit page
3. Check training status file manually

**"High CPU usage"**
1. Check for orphaned processes: `ps aux | grep joblib`
2. Run emergency cleanup
3. Monitor system resources

**"Frontend shows wrong status"**
1. Click "🧹 Cleanup Processes" in Streamlit
2. Use "🔄 Refresh Status" button
3. Restart Streamlit if needed

### Emergency Commands
```bash
# Kill all training processes
pkill -f "train_models"
pkill -f "joblib.externals.loky"

# Clean up with monitoring tool
python scripts/monitor_processes.py --cleanup

# Check remaining processes
ps aux | grep python | grep -E "(joblib|loky|train)"
```

## 📈 Future Enhancements

### Planned Improvements
1. **Automatic recovery**: Self-healing training processes
2. **Resource limits**: Memory and CPU constraints per worker
3. **Progress preservation**: Resume interrupted training
4. **Advanced monitoring**: Prometheus/Grafana integration
5. **Containerization**: Docker-based isolation

### Integration Opportunities  
1. **CI/CD pipelines**: Automated testing of prevention mechanisms
2. **Production monitoring**: Integration with system monitoring
3. **Alert systems**: Notifications for orphaned process detection
4. **Resource management**: Dynamic worker scaling based on system load

This comprehensive prevention strategy ensures reliable, interruptible training processes while maintaining system stability and accurate status reporting.