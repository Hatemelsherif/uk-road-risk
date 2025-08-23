"""
Model training and evaluation module for risk classification
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

class RiskModelTrainer:
    """Handles training and evaluation of risk classification models"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='multinomial'
            )
        }
        
    def perform_clustering(self, X: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform multiple clustering algorithms"""
        logger.info("Performing clustering analysis...")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clustering_results = {}
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        clustering_results['KMeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X_scaled, kmeans_labels),
            'model': kmeans
        }
        
        # Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hier_labels = hierarchical.fit_predict(X_scaled)
        clustering_results['Hierarchical'] = {
            'labels': hier_labels,
            'silhouette': silhouette_score(X_scaled, hier_labels),
            'model': hierarchical
        }
        
        # DBSCAN (density-based)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        if len(set(dbscan_labels)) > 1:  # Check if clusters were found
            clustering_results['DBSCAN'] = {
                'labels': dbscan_labels,
                'silhouette': silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1,
                'model': dbscan
            }
        
        logger.info("Clustering analysis completed")
        return clustering_results, scaler
    
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple classification models"""
        logger.info("Training classification models...")
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': results['Random Forest']['model'].feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame()
        
        return results, scaler, label_encoder, X_test, y_test, feature_importance
    
    def save_best_model(self, results: Dict[str, Any], scaler: Any, label_encoder: Any, 
                       feature_importance: pd.DataFrame) -> str:
        """Save the best performing model and associated artifacts"""
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"Best performing model: {best_model_name}")
        logger.info(f"Accuracy: {results[best_model_name]['accuracy']:.3f}")
        logger.info(f"F1 Score: {results[best_model_name]['f1_score']:.3f}")
        
        # Save model artifacts
        model_path = self.model_dir / 'best_risk_classifier.pkl'
        scaler_path = self.model_dir / 'feature_scaler.pkl'
        encoder_path = self.model_dir / 'label_encoder.pkl'
        importance_path = self.model_dir / 'feature_importance.csv'
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)
        feature_importance.to_csv(importance_path, index=False)
        
        logger.info("Models and artifacts saved successfully!")
        
        return best_model_name
    
    def load_model_artifacts(self) -> Tuple[Any, Any, Any]:
        """Load saved model artifacts"""
        model_path = self.model_dir / 'best_risk_classifier.pkl'
        scaler_path = self.model_dir / 'feature_scaler.pkl'
        encoder_path = self.model_dir / 'label_encoder.pkl'
        
        if not all([model_path.exists(), scaler_path.exists(), encoder_path.exists()]):
            raise FileNotFoundError("Model artifacts not found. Please train models first.")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        
        return model, scaler, encoder