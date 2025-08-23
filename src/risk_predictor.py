"""
Risk prediction utilities for new data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskPredictor:
    """Handles real-time risk prediction for new road conditions"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load trained model artifacts"""
        try:
            # Try to load from latest model directory first
            latest_path = self.model_dir / 'latest'
            if latest_path.exists() and latest_path.is_symlink():
                latest_model_dir = latest_path.resolve()
                model_path = latest_model_dir / 'best_model.pkl'
                scaler_path = latest_model_dir / 'scaler.pkl'
                encoder_path = latest_model_dir / 'label_encoder.pkl'
                
                if all(p.exists() for p in [model_path, scaler_path, encoder_path]):
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    self.label_encoder = joblib.load(encoder_path)
                    logger.info(f"Latest model artifacts loaded successfully from {latest_model_dir}")
                    return
            
            # Fallback to old model structure
            model_path = self.model_dir / 'best_risk_classifier.pkl'
            scaler_path = self.model_dir / 'feature_scaler.pkl'
            encoder_path = self.model_dir / 'label_encoder.pkl'
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            logger.info("Model artifacts loaded successfully")
            
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Model artifacts loading failed: {e}. Please train models first.")
    
    def predict_risk_level(self, new_data: Union[Dict, pd.DataFrame]) -> Tuple[str, Dict[str, float]]:
        """
        Predict risk level for new road conditions
        
        Parameters:
        -----------
        new_data : dict or DataFrame
            Features for prediction
        
        Returns:
        --------
        risk_level : str
            Predicted risk level
        probability : dict
            Probability for each risk level
        """
        
        if self.model is None:
            raise ValueError("Model not loaded. Please ensure model artifacts are available.")
        
        # Prepare input data
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        
        # Ensure all required features are present
        required_features = [
            'Number_of_Vehicles', 'Number_of_Casualties',
            'Speed_limit', 'Hour', 'Day_of_Week', 'Month',
            'Is_Weekend', 'Is_Rush_Hour', 'Weather_Risk_Score',
            'Road_Risk_Score', 'Light_Risk_Score', 'Environmental_Risk',
            'Vehicle_Age', 'Is_Old_Vehicle', 'Location_Accident_Count',
            'Junction_Risk', 'Urban_or_Rural_Area'
        ]
        
        for feature in required_features:
            if feature not in new_data.columns:
                new_data[feature] = 0
        
        # Select and order features
        X = new_data[required_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        risk_level = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {
            self.label_encoder.inverse_transform([i])[0]: prob 
            for i, prob in enumerate(probabilities)
        }
        
        return risk_level, prob_dict
    
    def calculate_simple_risk_score(self, conditions: Dict[str, Any]) -> Tuple[str, float, Dict[str, str]]:
        """
        Calculate a simple risk score based on input conditions
        Used when trained model is not available
        """
        
        # Map conditions to risk scores
        weather_risk_map = {1: 0, 2: 1, 3: 2, 4: 1, 5: 2, 6: 3, 7: 2, 8: 2}
        road_risk_map = {1: 0, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1}
        light_risk_map = {1: 0, 4: 2, 5: 3, 6: 2, 7: 3}
        junction_risk_map = {0: 0, 1: 1, 2: 1, 3: 2, 6: 3}
        
        # Extract risk scores
        weather_risk = weather_risk_map.get(conditions.get('weather', 1), 0)
        road_risk = road_risk_map.get(conditions.get('road_surface', 1), 0)
        light_risk = light_risk_map.get(conditions.get('light_condition', 1), 0)
        junction_risk = junction_risk_map.get(conditions.get('junction_type', 0), 0)
        
        # Calculate risk score
        risk_score = (
            weather_risk * 0.25 +
            road_risk * 0.25 +
            light_risk * 0.2 +
            junction_risk * 0.15 +
            (1 if conditions.get('is_rush_hour', False) else 0) * 0.1 +
            (1 if conditions.get('vehicle_age', 0) > 10 else 0) * 0.05
        )
        
        # Determine risk level
        if risk_score > 1.5:
            risk_level = "High Risk"
            emoji = "🔴"
        elif risk_score > 0.7:
            risk_level = "Medium Risk"
            emoji = "🟡"
        else:
            risk_level = "Low Risk"
            emoji = "🟢"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            weather_risk, road_risk, light_risk, junction_risk,
            conditions.get('is_rush_hour', False),
            conditions.get('vehicle_age', 0)
        )
        
        return risk_level, risk_score, recommendations
    
    def _generate_recommendations(self, weather_risk: float, road_risk: float, 
                                 light_risk: float, junction_risk: float,
                                 is_rush_hour: bool, vehicle_age: int) -> Dict[str, str]:
        """Generate safety recommendations based on risk factors"""
        
        recommendations = {}
        
        if weather_risk > 1:
            recommendations['weather'] = "⚠️ Adverse weather conditions - reduce speed and increase following distance"
        if road_risk > 1:
            recommendations['road'] = "⚠️ Poor road surface - exercise extra caution and reduce speed"
        if light_risk > 1:
            recommendations['light'] = "⚠️ Limited visibility - ensure lights are on and remain alert"
        if is_rush_hour:
            recommendations['traffic'] = "⚠️ Rush hour traffic - expect congestion and allow extra time"
        if junction_risk > 1:
            recommendations['junction'] = "⚠️ Complex junction ahead - approach with caution"
        if vehicle_age > 10:
            recommendations['vehicle'] = "⚠️ Older vehicle - ensure regular maintenance and safety checks"
        
        if not recommendations:
            recommendations['general'] = "✅ Conditions appear favorable - maintain safe driving practices"
        
        return recommendations
    
    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform batch predictions on multiple records"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Please ensure model artifacts are available.")
        
        predictions = []
        probabilities = []
        
        for idx, row in data.iterrows():
            risk_level, prob_dict = self.predict_risk_level(row.to_dict())
            predictions.append(risk_level)
            probabilities.append(prob_dict)
        
        result_df = data.copy()
        result_df['Predicted_Risk_Level'] = predictions
        
        # Add probability columns
        for risk_class in self.label_encoder.classes_:
            result_df[f'Prob_{risk_class.replace(" ", "_")}'] = [
                prob[risk_class] for prob in probabilities
            ]
        
        return result_df