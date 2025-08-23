"""
Model Training Management page for the UK Road Risk Assessment Dashboard
Real-time training with progress monitoring and results display
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import subprocess
import threading
import time
import json
import os
from pathlib import Path
from datetime import datetime
import queue

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils import apply_custom_css, initialize_session_state
from src.training_monitor import get_training_status

# Apply styling and initialize state
apply_custom_css()
initialize_session_state()

st.header("🚀 Model Training Management")
st.markdown("""
Train and manage machine learning models for road risk classification with real-time monitoring,
configuration options, and comprehensive results tracking.
""")

# Initialize session state for training
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'training_process' not in st.session_state:
    st.session_state.training_process = None

# Training Configuration Section
st.subheader("⚙️ Training Configuration")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("**Data Configuration**")
    limit_rows = st.number_input(
        "Limit Rows (0 = use full dataset)", 
        min_value=0, 
        max_value=100000, 
        value=10000, 
        step=1000,
        help="Limit the number of rows for faster training. Set to 0 for full dataset."
    )
    
    balance_method = st.selectbox(
        "Class Balancing Method",
        options=["none", "smote", "adasyn", "random_under", "smote_tomek", "smote_enn"],
        index=1,  # Default to SMOTE
        help="Method to handle class imbalance in the dataset"
    )
    
    feature_selection = st.selectbox(
        "Feature Selection Method",
        options=["none", "mutual_info", "f_classif", "rfe", "rfecv"],
        index=1,  # Default to mutual_info
        help="Method for selecting the most important features"
    )

with col2:
    st.markdown("**Model Configuration**")
    tune_hyperparameters = st.checkbox(
        "Enable Hyperparameter Tuning",
        value=False,
        help="Perform automated hyperparameter optimization (slower but better results)"
    )
    
    use_ensemble = st.checkbox(
        "Use Ensemble Methods",
        value=False,
        help="Train ensemble models (voting, stacking) for improved performance"
    )
    
    use_gpu = st.checkbox(
        "GPU Acceleration",
        value=True,  # Enable by default for better performance
        help="Use GPU for training if available (XGBoost, LightGBM, CatBoost) - Recommended for faster training"
    )
    
    use_deep_learning = st.checkbox(
        "🧠 Deep Learning Models",
        value=False,
        help="Include advanced neural network architectures (feedforward, residual, transformer) - May take longer but can achieve better performance"
    )

# Training presets
st.markdown("**Quick Presets**")
preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 1])

with preset_col1:
    if st.button("🚀 Quick Training", use_container_width=True):
        limit_rows = 5000
        balance_method = "none"
        feature_selection = "none"
        tune_hyperparameters = False
        use_ensemble = False
        use_gpu = False
        st.success("Quick training preset selected!")

with preset_col2:
    if st.button("⚡ Balanced Training", use_container_width=True):
        limit_rows = 10000
        balance_method = "smote"
        feature_selection = "mutual_info"
        tune_hyperparameters = False
        use_ensemble = False
        use_gpu = False
        st.success("Balanced training preset selected!")

with preset_col3:
    if st.button("🎯 Best Performance", use_container_width=True):
        limit_rows = 0  # Full dataset
        balance_method = "smote"
        feature_selection = "mutual_info"
        tune_hyperparameters = True
        use_ensemble = True
        use_deep_learning = True
        use_gpu = True
        st.success("Best performance preset selected (includes deep learning)!")

st.divider()

# Training Control Section
st.subheader("🎮 Training Controls")

def build_training_command():
    """Build the training command based on configuration"""
    cmd = ["python", "scripts/train_models_enhanced.py"]
    
    if limit_rows > 0:
        cmd.extend(["--limit-rows", str(limit_rows)])
    
    if balance_method != "none":
        cmd.extend(["--balance-method", balance_method])
    
    if feature_selection != "none":
        cmd.extend(["--feature-selection", feature_selection])
    
    if tune_hyperparameters:
        cmd.append("--tune-hyperparameters")
    
    if use_ensemble:
        cmd.append("--use-ensemble")
    
    if use_deep_learning:
        cmd.append("--use-deep-learning")
    
    if use_gpu:
        cmd.append("--use-gpu")
    
    cmd.append("--save-report")
    
    return cmd

def start_training():
    """Start the training process in background"""
    if st.session_state.training_running:
        return
    
    st.session_state.training_running = True
    st.session_state.training_logs = []
    st.session_state.training_progress = 0
    st.session_state.training_results = None
    
    # Build command
    cmd = build_training_command()
    
    # Show configuration summary
    config_msg = f"""
    🚀 **Training Configuration:**
    - Dataset: {limit_rows if limit_rows else 'Full'} rows
    - Balance Method: {balance_method.title() if balance_method else 'None'}
    - Feature Selection: {feature_selection.title() if feature_selection else 'None'}
    - Hyperparameter Tuning: {'✅' if tune_hyperparameters else '❌'}
    - Ensemble Methods: {'✅' if use_ensemble else '❌'}
    - 🧠 Deep Learning: {'✅' if use_deep_learning else '❌'}
    - GPU Acceleration: {'✅' if use_gpu else '❌'}
    """
    st.info(config_msg)
    
    st.code(' '.join(cmd), language='bash')
    
    # Start process
    try:
        import os
        # Set environment variables for better subprocess handling
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Separate stderr to capture errors
            universal_newlines=True,
            bufsize=1,
            cwd=str(project_root),
            env=env
        )
        st.session_state.training_process = process
        st.success("✅ Training started successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to start training: {str(e)}")
        st.session_state.training_running = False

def stop_training():
    """Stop the training process"""
    if st.session_state.get('training_process'):
        try:
            st.session_state.training_process.terminate()
        except:
            pass
        st.session_state.training_process = None
    st.session_state.training_running = False
    st.warning("⚠️ Training stopped by user")

# Training control buttons
control_col1, control_col2, control_col3 = st.columns([1, 1, 2])

with control_col1:
    if st.button("▶️ Start Training", 
                type="primary", 
                disabled=st.session_state.training_running,
                use_container_width=True):
        start_training()

with control_col2:
    if st.button("⏹️ Stop Training", 
                disabled=not st.session_state.training_running,
                use_container_width=True):
        stop_training()

with control_col3:
    # Display training status
    if st.session_state.training_running:
        st.success("🔄 Training in progress...")
    else:
        st.info("⭐ Ready to start training")

st.divider()

# Real-time Progress Monitoring
training_status = get_training_status()

# Always show training status
if training_status.get('status') in ['starting', 'data_loading', 'training'] or \
   (st.session_state.training_running and st.session_state.get('training_process')):
    st.subheader("📊 Real-time Training Progress")
    
    # Check if process is still running
    if st.session_state.get('training_process') and st.session_state.training_process.poll() is None:
        # Process is still running
        progress_container = st.container()
        log_container = st.container()
        
        with progress_container:
            # Display progress bar
            progress_value = training_status.get('progress', 0) / 100
            st.progress(progress_value)
            
            # Create progress indicators
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                status_emoji = {
                    'starting': '🚀',
                    'data_loading': '📊', 
                    'training': '🔄',
                    'completed': '✅',
                    'failed': '❌'
                }.get(training_status.get('status', 'unknown'), '❓')
                
                st.metric("Status", f"{status_emoji} {training_status.get('status', 'Unknown').title()}", 
                         delta=f"{training_status.get('progress', 0):.0f}%")
            
            with col2:
                # Calculate elapsed time
                try:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(training_status.get('timestamp', datetime.now().isoformat()))
                    elapsed = datetime.now() - start_time
                    elapsed_str = f"{elapsed.seconds//60}m {elapsed.seconds%60}s"
                except:
                    elapsed_str = "calculating..."
                st.metric("Elapsed Time", elapsed_str)
            
            with col3:
                current_message = training_status.get('message', 'Processing...')
                st.metric("Current Step", current_message[:30] + "..." if len(current_message) > 30 else current_message, 
                         delta="Processing")
        
        with log_container:
            st.markdown("**Training Logs:**")
            
            # Read process output (cross-platform compatible)
            try:
                # Check if there's output available
                if st.session_state.get('training_process') and st.session_state.training_process.stdout:
                    try:
                        # Try to read available output without blocking
                        import queue
                        import threading
                        
                        def read_output(process, q):
                            for line in iter(process.stdout.readline, ''):
                                if line:
                                    q.put(line.strip())
                                if process.poll() is not None:
                                    break
                        
                        # Use a queue to collect output
                        if not hasattr(st.session_state, 'output_queue'):
                            st.session_state.output_queue = queue.Queue()
                            st.session_state.output_thread = threading.Thread(
                                target=read_output, 
                                args=(st.session_state.training_process, st.session_state.output_queue)
                            )
                            st.session_state.output_thread.daemon = True
                            st.session_state.output_thread.start()
                        
                        # Get new output from queue
                        while not st.session_state.output_queue.empty():
                            try:
                                line = st.session_state.output_queue.get_nowait()
                                if line and line not in st.session_state.training_logs:
                                    st.session_state.training_logs.append(line)
                            except queue.Empty:
                                break
                    except Exception as e:
                        # Fallback: just check process status
                        if st.session_state.get('training_process'):
                            st.session_state.training_logs.append(f"Monitoring process... (Status: {st.session_state.training_process.poll()})")
                
                # Display recent logs
                if st.session_state.training_logs:
                    log_text = "\n".join(st.session_state.training_logs[-15:])  # Last 15 lines
                    st.text_area(
                        "Recent Output", 
                        value=log_text, 
                        height=200,
                        disabled=True
                    )
                else:
                    st.info("Training started... waiting for output...")
                
                # Auto-refresh every 3 seconds
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.warning(f"Monitoring training process... (Error: {str(e)})")
                # Still show basic status
                st.info("Training is running in background. Check console for detailed output.")
    
    else:
        # Process finished
        st.session_state.training_running = False
        return_code = st.session_state.training_process.returncode if st.session_state.get('training_process') else None
        
        if return_code == 0:
            st.success("🎉 Training completed successfully!")
            
            # Try to load results
            try:
                # Look for latest model metadata
                latest_model_dir = Path("data/models/latest")
                if latest_model_dir.exists():
                    metadata_file = latest_model_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            st.session_state.training_results = json.load(f)
            except Exception as e:
                st.warning(f"Could not load training results: {str(e)}")
        else:
            st.error(f"❌ Training failed with return code: {return_code}")
            
            # Try to get error output
            try:
                if st.session_state.get('training_process') and st.session_state.training_process.stderr:
                    stderr_output = st.session_state.training_process.stderr.read()
                    if stderr_output:
                        st.error("**Error Output:**")
                        st.code(stderr_output)
            except Exception as e:
                st.warning("Could not retrieve error details")
            
            # Also show recent logs for debugging
            if st.session_state.training_logs:
                st.error("**Recent Training Logs:**")
                st.code("\n".join(st.session_state.training_logs[-10:]))

# Training Results Display
if st.session_state.training_results:
    st.subheader("📈 Training Results")
    
    results = st.session_state.training_results
    
    # Best model summary
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            "Best Model", 
            results.get('best_model_name', 'Unknown'),
            delta="Selected"
        )
    
    with col2:
        best_score = results.get('best_score', 0)
        st.metric(
            "Best F1-Score", 
            f"{best_score:.4f}",
            delta=f"{best_score*100:.1f}%"
        )
    
    with col3:
        timestamp = results.get('timestamp', 'Unknown')
        st.metric(
            "Training Time", 
            timestamp,
            delta="Completed"
        )
    
    # Model comparison
    if 'results_summary' in results:
        st.markdown("**Model Performance Comparison**")
        
        results_df = pd.DataFrame(results['results_summary']).T
        results_df = results_df.round(4)
        
        # Create comparison chart
        fig_comparison = go.Figure()
        
        metrics = ['accuracy', 'f1_score', 'balanced_accuracy']
        for metric in metrics:
            if metric in results_df.columns:
                fig_comparison.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=results_df.index,
                    y=results_df[metric]
                ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Results table
        st.markdown("**Detailed Results**")
        st.dataframe(results_df, use_container_width=True)

# Training History
st.subheader("📚 Training History")

# Look for historical training reports
reports_dir = Path("data/models")
if reports_dir.exists():
    # Find training report files
    report_files = list(reports_dir.glob("training_report_*.json"))
    
    if report_files:
        # Sort by creation time
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        history_data = []
        for report_file in report_files[:10]:  # Show last 10 runs
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                history_data.append({
                    'Timestamp': report_file.stem.replace('training_report_', ''),
                    'Best Model': report.get('best_model', 'Unknown'),
                    'Data Samples': report.get('configuration', {}).get('data_samples', 'Unknown'),
                    'Features': report.get('configuration', {}).get('num_features', 'Unknown'),
                    'Balance Method': report.get('configuration', {}).get('balance_method', 'None'),
                    'Hyperparameter Tuning': report.get('configuration', {}).get('tune_hyperparameters', False),
                    'Best F1-Score': report.get('results', {}).get(report.get('best_model', ''), {}).get('f1_score', 0)
                })
            except Exception as e:
                continue
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Training trends
            if len(history_data) > 1:
                fig_trends = px.line(
                    history_df,
                    x='Timestamp',
                    y='Best F1-Score',
                    title='Training Performance Over Time',
                    markers=True
                )
                fig_trends.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No training history found.")
    else:
        st.info("No previous training reports found.")
else:
    st.info("Models directory not found. Run training to create history.")

# Model Management Section
st.subheader("🔧 Model Management")

model_mgmt_col1, model_mgmt_col2, model_mgmt_col3, model_mgmt_col4 = st.columns([1, 1, 1, 1])

with model_mgmt_col1:
    if st.button("🗑️ Clear Training Logs", use_container_width=True):
        st.session_state.training_logs = []
        st.success("Training logs cleared!")

with model_mgmt_col2:
    if st.button("🧹 Cleanup Processes", use_container_width=True):
        try:
            import subprocess
            result = subprocess.run([
                "python", "scripts/monitor_processes.py", "--cleanup"
            ], capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                st.success("✅ Orphaned processes cleaned up!")
                st.session_state.training_running = False
            else:
                st.error("❌ Cleanup failed")
        except Exception as e:
            st.error(f"❌ Cleanup error: {e}")

with model_mgmt_col3:
    if st.button("📁 View Model Files", use_container_width=True):
        models_dir = Path("data/models")
        if models_dir.exists():
            files = list(models_dir.glob("*"))
            if files:
                st.write("**Model Files:**")
                for file in sorted(files):
                    if file.is_file():
                        size = file.stat().st_size / 1024 / 1024  # MB
                        st.write(f"📄 {file.name} ({size:.2f} MB)")
                    elif file.is_dir():
                        st.write(f"📁 {file.name}/")
            else:
                st.info("No model files found.")
        else:
            st.info("Models directory not found.")

with model_mgmt_col4:
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

# Tips and Help
with st.expander("💡 Training Tips & Help"):
    st.markdown("""
    **Training Configuration Guide:**
    
    🚀 **Quick Training** (5-10 minutes):
    - Use 5,000-10,000 rows
    - No class balancing
    - No hyperparameter tuning
    - Good for testing and prototyping
    
    ⚡ **Balanced Training** (15-30 minutes):
    - Use 10,000-50,000 rows
    - Enable SMOTE for class balancing
    - Use mutual information for feature selection
    - Good balance of speed and performance
    
    🎯 **Best Performance** (1-3 hours):
    - Use full dataset (all rows)
    - Enable all enhancements
    - Hyperparameter tuning
    - Ensemble methods
    - 🧠 Deep learning models
    - Best possible accuracy
    
    **Troubleshooting:**
    - If training fails, try reducing the number of rows
    - Check that CSV files exist in `data/raw/`
    - Ensure sufficient disk space (>1GB recommended)
    - GPU acceleration requires compatible hardware
    """)

# Footer
st.markdown("---")
st.markdown("*Real-time model training with progress monitoring and comprehensive results tracking*")