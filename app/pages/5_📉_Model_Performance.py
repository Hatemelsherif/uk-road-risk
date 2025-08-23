"""
Model Performance page for the UK Road Risk Assessment Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils import apply_custom_css, initialize_session_state
from src.visualization import RiskVisualizer

# Apply styling and initialize state
apply_custom_css()
initialize_session_state()

st.header("📉 Model Performance Metrics")
st.markdown("""
Comprehensive evaluation of machine learning models used for road risk classification,
including accuracy metrics, feature importance, and model comparison.
""")

# Check if models are trained and available
model_artifacts_exist = False
try:
    from pathlib import Path
    model_dir = Path("data/models")
    model_artifacts_exist = (
        (model_dir / "best_risk_classifier.pkl").exists() and
        (model_dir / "feature_scaler.pkl").exists() and
        (model_dir / "label_encoder.pkl").exists()
    )
except:
    pass

if model_artifacts_exist:
    st.success("✅ Trained models detected!")
    
    # Load feature importance if available
    feature_importance_file = Path("data/models/feature_importance.csv")
    if feature_importance_file.exists():
        feature_importance = pd.read_csv(feature_importance_file)
        
        st.subheader("🎯 Feature Importance Analysis")
        visualizer = RiskVisualizer()
        fig_importance = visualizer.create_feature_importance_plot(feature_importance)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("**Top 10 Most Important Features:**")
        st.dataframe(feature_importance.head(10), use_container_width=True)
    
    # Load model metadata if available
    metadata_file = Path("data/models/latest/metadata.json")
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        st.subheader("🏆 Model Performance Results")
        st.markdown(f"**Best Model:** {metadata.get('best_model_name', 'Unknown')}")
        st.markdown(f"**Best F1-Score:** {metadata.get('best_score', 'Unknown'):.4f}")
        
        # Display results summary
        if 'results_summary' in metadata:
            results_df = pd.DataFrame(metadata['results_summary']).T
            results_df = results_df.round(4)
            st.dataframe(results_df, use_container_width=True)
            
            # Create performance comparison chart
            if len(results_df) > 1:
                fig_metrics = go.Figure()
                metrics = ['accuracy', 'f1_score', 'balanced_accuracy']
                
                for metric in metrics:
                    if metric in results_df.columns:
                        fig_metrics.add_trace(go.Bar(
                            name=metric.replace('_', ' ').title(),
                            x=results_df.index,
                            y=results_df[metric]
                        ))
                
                fig_metrics.update_layout(
                    title='Model Performance Comparison',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    barmode='group',
                    yaxis=dict(range=[0, 1]),
                    height=500
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
    
else:
    st.warning("⚠️ No trained models found!")
    st.markdown("""
    ## How to Generate Model Performance Data
    
    To see real model performance metrics, you need to train models first:
    
    ### Option 1: Quick Training (Recommended for testing)
    ```bash
    python scripts/train_models_enhanced.py --quick --limit-rows 10000
    ```
    
    ### Option 2: Full Training (Best performance)
    ```bash
    python scripts/train_models_enhanced.py --full
    ```
    
    ### Option 3: Custom Training
    ```bash
    python scripts/train_models_enhanced.py --balance-method smote --tune-hyperparameters --use-ensemble
    ```
    
    After training completes, refresh this page to see:
    - ✅ Real model accuracy metrics
    - ✅ Feature importance rankings
    - ✅ Model comparison charts
    - ✅ Training performance details
    
    **Current Status:** No model artifacts found in `data/models/`
    """)
    
    st.stop()  # Stop execution here if no models exist

