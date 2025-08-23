"""
Visualization utilities for UK road risk analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RiskVisualizer:
    """Handles creation of comprehensive visualizations for risk analysis"""
    
    def __init__(self):
        self.risk_color_map = {
            'High Risk': '#FF4444',
            'Medium Risk': '#FFA500', 
            'Low Risk': '#44FF44'
        }
    
    def create_risk_distribution_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create risk level distribution visualization"""
        risk_counts = df['Risk_Level'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Distribution of Risk Levels',
            color_discrete_map=self.risk_color_map
        )
        
        return fig
    
    def create_temporal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive temporal analysis visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accidents by Hour', 'Accidents by Day of Week', 
                           'Accidents by Month', 'Weekend vs Weekday')
        )
        
        # Hour distribution
        hour_data = df.groupby(['Hour', 'Risk_Level']).size().reset_index(name='count')
        for risk in df['Risk_Level'].unique():
            data = hour_data[hour_data['Risk_Level'] == risk]
            fig.add_trace(
                go.Scatter(x=data['Hour'], y=data['count'], 
                          name=risk, mode='lines+markers',
                          line=dict(color=self.risk_color_map[risk])),
                row=1, col=1
            )
        
        # Day of week
        day_data = df.groupby(['Day_of_Week', 'Risk_Level']).size().reset_index(name='count')
        for risk in df['Risk_Level'].unique():
            data = day_data[day_data['Risk_Level'] == risk]
            fig.add_trace(
                go.Bar(x=data['Day_of_Week'], y=data['count'], 
                      name=risk, marker_color=self.risk_color_map[risk]),
                row=1, col=2
            )
        
        # Monthly distribution
        month_data = df.groupby(['Month', 'Risk_Level']).size().reset_index(name='count')
        for risk in df['Risk_Level'].unique():
            data = month_data[month_data['Risk_Level'] == risk]
            fig.add_trace(
                go.Scatter(x=data['Month'], y=data['count'], 
                          name=risk, mode='lines+markers',
                          line=dict(color=self.risk_color_map[risk])),
                row=2, col=1
            )
        
        # Weekend comparison
        weekend_data = df.groupby(['Is_Weekend', 'Risk_Level']).size().reset_index(name='count')
        weekend_data['Type'] = weekend_data['Is_Weekend'].map({0: 'Weekday', 1: 'Weekend'})
        for risk in df['Risk_Level'].unique():
            data = weekend_data[weekend_data['Risk_Level'] == risk]
            fig.add_trace(
                go.Bar(x=data['Type'], y=data['count'], 
                      name=risk, marker_color=self.risk_color_map[risk]),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Temporal Analysis of Accidents")
        return fig
    
    def create_environmental_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create environmental factors analysis"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weather Conditions', 'Road Conditions', 
                           'Light Conditions', 'Environmental Risk Score')
        )
        
        # Weather impact
        weather_risk = df.groupby(['Weather_Risk_Score', 'Risk_Level']).size().reset_index(name='count')
        for risk in df['Risk_Level'].unique():
            data = weather_risk[weather_risk['Risk_Level'] == risk]
            fig.add_trace(
                go.Bar(x=data['Weather_Risk_Score'], y=data['count'], 
                      name=risk, marker_color=self.risk_color_map[risk]),
                row=1, col=1
            )
        
        # Road conditions
        road_risk = df.groupby(['Road_Risk_Score', 'Risk_Level']).size().reset_index(name='count')
        for risk in df['Risk_Level'].unique():
            data = road_risk[road_risk['Risk_Level'] == risk]
            fig.add_trace(
                go.Bar(x=data['Road_Risk_Score'], y=data['count'], 
                      name=risk, marker_color=self.risk_color_map[risk]),
                row=1, col=2
            )
        
        # Light conditions
        light_risk = df.groupby(['Light_Risk_Score', 'Risk_Level']).size().reset_index(name='count')
        for risk in df['Risk_Level'].unique():
            data = light_risk[light_risk['Risk_Level'] == risk]
            fig.add_trace(
                go.Bar(x=data['Light_Risk_Score'], y=data['count'], 
                      name=risk, marker_color=self.risk_color_map[risk]),
                row=2, col=1
            )
        
        # Environmental risk distribution
        fig.add_trace(
            go.Histogram(x=df['Environmental_Risk'], name='Environmental Risk', nbinsx=20),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Environmental Factors Analysis")
        return fig
    
    def create_feature_importance_plot(self, feature_importance: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Create feature importance visualization"""
        
        top_features = feature_importance.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Most Important Features for Risk Classification',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        return fig
    
    def create_model_performance_plot(self, classification_results: Dict[str, Any]) -> go.Figure:
        """Create model performance comparison"""
        
        model_names = list(classification_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        for metric in metrics:
            values = [classification_results[model][metric] for model in model_names]
            fig.add_trace(go.Bar(name=metric.capitalize(), x=model_names, y=values))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_geographical_plot(self, df: pd.DataFrame, sample_size: int = 5000, 
                                map_style: str = 'open-street-map') -> Optional[go.Figure]:
        """Create geographical distribution plot"""
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            logger.warning("Latitude/Longitude columns not found")
            return None
        
        # Sample for performance
        df_sample = df.sample(min(sample_size, len(df)))
        
        fig = px.scatter_mapbox(
            df_sample,
            lat='Latitude',
            lon='Longitude',
            color='Risk_Level',
            size='Number_of_Casualties',
            hover_data=['Weather_Conditions', 'Road_Surface_Conditions'],
            title=f'Geographical Distribution of Accidents ({len(df_sample)} points)',
            mapbox_style=map_style,
            zoom=5,
            height=600,
            color_discrete_map=self.risk_color_map
        )
        
        return fig
    
    def create_density_heatmap(self, df: pd.DataFrame, sample_size: int = 5000,
                              map_style: str = 'open-street-map') -> Optional[go.Figure]:
        """Create accident density heatmap"""
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            logger.warning("Latitude/Longitude columns not found")
            return None
        
        df_sample = df.sample(min(sample_size, len(df)))
        
        fig = px.density_mapbox(
            df_sample,
            lat='Latitude',
            lon='Longitude',
            radius=20,
            title="Accident Density Heatmap",
            mapbox_style=map_style,
            zoom=5,
            height=600
        )
        
        return fig
    
    def create_confusion_matrix(self, confusion_matrix: np.ndarray, 
                               class_names: list) -> go.Figure:
        """Create confusion matrix visualization"""
        
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_names,
            y=class_names,
            title="Confusion Matrix",
            color_continuous_scale='Blues',
            text_auto=True
        )
        
        return fig
    
    def create_risk_factor_breakdown(self, factors_df: pd.DataFrame) -> go.Figure:
        """Create risk factor contribution breakdown"""
        
        fig = px.bar(
            factors_df,
            x='Risk Contribution',
            y='Factor',
            orientation='h',
            title="Individual Risk Factor Contributions",
            color='Risk Contribution',
            color_continuous_scale=['green', 'yellow', 'red']
        )
        
        return fig