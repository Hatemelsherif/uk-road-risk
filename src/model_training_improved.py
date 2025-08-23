"""
Enhanced model training and evaluation module for risk classification
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging

# Deep learning imports (optional) - loaded on demand
DEEP_LEARNING_AVAILABLE = False
PyTorchDeepLearning = None

def _load_deep_learning():
    """Load PyTorch deep learning module on demand (Apple Silicon compatible)"""
    global DEEP_LEARNING_AVAILABLE, PyTorchDeepLearning
    if PyTorchDeepLearning is None:
        try:
            from src.pytorch_deep_learning import PyTorchDeepLearningIntegration
            PyTorchDeepLearning = PyTorchDeepLearningIntegration
            DEEP_LEARNING_AVAILABLE = True
            logger = logging.getLogger(__name__)
            logger.info("🧠 PyTorch deep learning module loaded successfully (Apple Silicon compatible)")
        except Exception as e:
            DEEP_LEARNING_AVAILABLE = False
            logger = logging.getLogger(__name__)
            logger.warning(f"PyTorch deep learning not available: {e}")
    return DEEP_LEARNING_AVAILABLE

# Machine Learning imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, silhouette_score,
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    make_scorer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
)
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Some features will be unavailable.")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Some features will be unavailable.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    warnings.warn("CatBoost not installed. Some features will be unavailable.")

logger = logging.getLogger(__name__)


class ImprovedRiskModelTrainer:
    """Enhanced training and evaluation of risk classification models"""
    
    def __init__(self, model_dir: str = "data/models", 
                 use_gpu: bool = False,
                 random_state: int = 42,
                 max_workers: int = 4):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit number of workers to prevent process explosion
        import os
        self.max_workers = min(max_workers, os.cpu_count() or 4)
        logger.info(f"🔧 Using {self.max_workers} workers for parallel processing")
        
        # Check platform and adjust GPU settings for Apple Silicon
        import platform
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            if use_gpu:
                logger.info("🍎 Apple Silicon detected - using optimized CPU training (faster than GPU for these libraries)")
            self.use_gpu = False  # Apple Silicon works better with CPU for XGBoost/LightGBM/CatBoost
        else:
            self.use_gpu = use_gpu
            
        self.random_state = random_state
        
        # Initialize base models with better defaults
        self.base_models = self._initialize_base_models()
        
        # Hyperparameter grids for tuning
        self.param_grids = self._get_param_grids()
        
        # Best model storage
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        
    def _robust_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle imputation for mixed data types (numeric and categorical)"""
        import numpy as np
        logger.info("Performing robust imputation for mixed data types...")
        
        X_imputed = X.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = []
        categorical_cols = []
        
        for col in X.columns:
            # Check if column contains numeric data (excluding NaN values)
            non_null_values = X[col].dropna()
            if len(non_null_values) == 0:
                # All NaN - treat as categorical and fill with 'Unknown'
                categorical_cols.append(col)
                continue
                
            # Check if all non-null values can be converted to numeric
            sample_size = min(100, len(non_null_values))
            sample_values = non_null_values.iloc[:sample_size]
            
            try:
                # Try to convert all sample values to numeric
                numeric_values = pd.to_numeric(sample_values, errors='raise')
                # If successful, check if at least 80% are actually numeric
                numeric_count = sum(pd.notna(pd.to_numeric(sample_values, errors='coerce')))
                if numeric_count / len(sample_values) >= 0.8:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except (ValueError, TypeError):
                # Contains non-numeric values, treat as categorical
                categorical_cols.append(col)
        
        logger.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
        logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        # Impute numeric columns with median
        if numeric_cols:
            from sklearn.impute import SimpleImputer
            numeric_imputer = SimpleImputer(strategy='median')
            X_imputed[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        # Impute categorical columns with mode (most frequent)
        if categorical_cols:
            from sklearn.impute import SimpleImputer
            # Convert categorical columns to string first to handle mixed types
            for col in categorical_cols:
                X_imputed[col] = X_imputed[col].astype(str)
                X_imputed[col] = X_imputed[col].replace('nan', np.nan)  # Restore NaN values
            
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X_imputed[categorical_cols] = categorical_imputer.fit_transform(X_imputed[categorical_cols])
        
        logger.info(f"Imputation completed. Shape: {X_imputed.shape}")
        return X_imputed
        
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Encode categorical features to numeric for machine learning models"""
        logger.info("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_cols = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                categorical_cols.append(col)
        
        logger.info(f"Found categorical columns: {categorical_cols}")
        
        if not categorical_cols:
            # No categorical columns, return as is
            return X_train.values, X_test.values, None
        
        # Use LabelEncoder for categorical columns
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for col in categorical_cols:
            encoder = LabelEncoder()
            
            # Fit on training data
            X_train_encoded[col] = encoder.fit_transform(X_train[col].astype(str))
            
            # Transform test data, handling unseen labels
            X_test_col = X_test[col].astype(str)
            unseen_mask = ~X_test_col.isin(encoder.classes_)
            
            if unseen_mask.any():
                logger.warning(f"Found {unseen_mask.sum()} unseen labels in column {col}, replacing with most frequent")
                most_frequent = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else encoder.classes_[0]
                X_test_col.loc[unseen_mask] = most_frequent
            
            X_test_encoded[col] = encoder.transform(X_test_col)
            encoders[col] = encoder
        
        logger.info(f"Categorical encoding completed. Encoded {len(categorical_cols)} columns")
        return X_train_encoded.values, X_test_encoded.values, encoders
        
    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize all available models with optimized defaults"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=self.max_workers,
                class_weight='balanced'
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=self.max_workers,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                random_state=self.random_state,
                validation_fraction=0.2,
                n_iter_no_change=5
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=self.random_state,
                algorithm='SAMME'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced',
                solver='saga',
                n_jobs=self.max_workers
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced',
                kernel='rbf'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2
            )
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
                tree_method='gpu_hist' if self.use_gpu else 'auto',
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models['LightGBM'] = LGBMClassifier(
                n_estimators=200,
                random_state=self.random_state,
                device='gpu' if self.use_gpu else 'cpu',
                n_jobs=-1,
                verbose=-1
            )
        
        # Add CatBoost if available
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostClassifier(
                iterations=200,
                random_state=self.random_state,
                task_type='GPU' if self.use_gpu else 'CPU',
                verbose=False
            )
        
        return models
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for each model"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'Extra Trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
            },
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [0.2, 0.5, 0.8]  # Only for elasticnet
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
        
        if HAS_XGBOOST:
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0, 0.01, 0.1, 1]
            }
        
        if HAS_LIGHTGBM:
            param_grids['LightGBM'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [-1, 10, 20, 30],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'num_leaves': [31, 50, 100, 200],
                'feature_fraction': [0.7, 0.8, 0.9, 1.0],
                'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [5, 10, 20, 30]
            }
        
        if HAS_CATBOOST:
            param_grids['CatBoost'] = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        
        return param_grids
    
    def perform_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 method: str = 'mutual_info',
                                 n_features: Optional[int] = None) -> pd.DataFrame:
        """Perform feature selection using various methods"""
        logger.info(f"Performing feature selection using {method}...")
        
        # Handle NaN values before feature selection using robust imputation
        X_imputed_df = self._robust_imputation(X)
        
        # Encode categorical features for feature selection
        X_dummy = X_imputed_df.copy()
        X_encoded, _, encoders = self._encode_categorical_features(X_dummy, X_dummy)
        X_encoded_df = pd.DataFrame(X_encoded, columns=X_imputed_df.columns, index=X.index)
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, 
                                  k=n_features or min(20, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, 
                                  k=n_features or min(20, X.shape[1]))
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFE(estimator, n_features_to_select=n_features or 15)
        elif method == 'rfecv':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFECV(estimator, step=1, cv=5, n_jobs=-1)
        else:
            logger.warning(f"Unknown method {method}, returning original features")
            return X_imputed_df, None
        
        X_selected = selector.fit_transform(X_encoded_df, y)
        selected_features = X_encoded_df.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Return the original imputed dataframe with selected features (will be encoded later in main pipeline)
        X_selected_original = X_imputed_df[selected_features]
        return X_selected_original, selector
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                              method: str = 'smote', skip_encoding: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using various sampling techniques"""
        logger.info(f"Handling class imbalance using {method}...")
        
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        # Encode categorical features before applying SMOTE (unless already done)
        if not skip_encoding:
            X_encoded, _, encoders = self._encode_categorical_features(X, X)
        else:
            X_encoded = X.values if hasattr(X, 'values') else X
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif method == 'random_under':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=self.random_state)
        else:
            logger.warning(f"Unknown method {method}, returning original data")
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X_encoded, y)
        
        new_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"Resampled class distribution: {new_counts.to_dict()}")
        
        # Convert back to DataFrame with encoded values
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def create_ensemble_models(self, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Create various ensemble models"""
        ensemble_models = {}
        
        # Voting Classifier (hard and soft)
        estimators = [(name, model) for name, model in base_models.items() 
                     if name not in ['Neural Network', 'SVM']][:3]  # Top 3 models
        
        ensemble_models['Voting (Hard)'] = VotingClassifier(
            estimators=estimators,
            voting='hard',
            n_jobs=-1
        )
        
        ensemble_models['Voting (Soft)'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Stacking Classifier
        base_estimators = estimators[:2]  # Use top 2 as base
        meta_estimator = LogisticRegression(random_state=self.random_state)
        
        ensemble_models['Stacking'] = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=5,
            n_jobs=-1
        )
        
        return ensemble_models
    
    def hyperparameter_tuning(self, model: Any, param_grid: Dict, 
                            X_train: np.ndarray, y_train: np.ndarray,
                            search_type: str = 'grid', n_iter: int = 50) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning"""
        logger.info(f"Performing {search_type} search for hyperparameter tuning...")
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(lambda y_true, y_pred: 
                                      precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]),
            'balanced_accuracy': make_scorer(balanced_accuracy_score)
        }
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1_weighted',
                n_jobs=self.max_workers, verbose=1, refit=True
            )
        else:  # RandomizedSearch
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, 
                scoring='f1_weighted', n_jobs=self.max_workers, verbose=1, 
                random_state=self.random_state, refit=True
            )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
    
    def train_with_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                  cv_folds: int = 10) -> Dict[str, Any]:
        """Perform comprehensive cross-validation"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'roc_auc_ovr': 'roc_auc_ovr'
        }
        
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=True
        )
        
        results = {}
        for metric in scoring.keys():
            results[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            results[f'{metric}_std'] = cv_results[f'test_{metric}'].std()
            results[f'{metric}_train_mean'] = cv_results[f'train_{metric}'].mean()
        
        return results
    
    def evaluate_model_comprehensive(self, model: Any, X_test: np.ndarray, 
                                    y_test: np.ndarray, y_pred: np.ndarray,
                                    label_encoder: Optional[LabelEncoder] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation with multiple metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Advanced metrics
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        matthews_corr = matthews_corrcoef(y_test, y_pred)
        
        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC if model supports probability predictions
        roc_auc = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': cohen_kappa,
            'matthews_corrcoef': matthews_corr,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'support': support
        }
        
        return results
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series,
                        balance_method: str = 'smote',
                        feature_selection_method: Optional[str] = 'mutual_info',
                        tune_hyperparameters: bool = True,
                        use_ensemble: bool = True,
                        use_deep_learning: bool = False,
                        test_size: float = 0.2) -> Dict[str, Any]:
        """Main training pipeline with all enhancements"""
        
        logger.info("Starting enhanced model training pipeline...")
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Handle NaN values with mixed data types
        X = self._robust_imputation(X)
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            random_state=self.random_state, stratify=y_encoded
        )
        
        # Feature selection
        selector = None
        if feature_selection_method:
            X_train_df = pd.DataFrame(X_train, columns=X.columns)
            X_test_df = pd.DataFrame(X_test, columns=X.columns)
            
            X_train, selector = self.perform_feature_selection(
                X_train_df, 
                y_train, 
                method=feature_selection_method
            )
            # Apply same feature selection to test set (features already selected in perform_feature_selection)
            X_test = X_test_df[X_train.columns]
        
        # Handle NaN values and categorical encoding before class balancing
        X_train_df = pd.DataFrame(X_train, columns=X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]))
        X_test_df = pd.DataFrame(X_test, columns=X_train_df.columns)
        
        X_train_imputed_df = self._robust_imputation(X_train_df)
        X_test_imputed_df = self._robust_imputation(X_test_df)
        
        # Encode categorical features
        X_train_encoded, X_test_encoded, encoders = self._encode_categorical_features(
            X_train_imputed_df, X_test_imputed_df
        )
        
        # Handle class imbalance (now on properly encoded data)
        if balance_method:
            X_train_encoded, y_train = self.handle_class_imbalance(
                pd.DataFrame(X_train_encoded, columns=X_train_imputed_df.columns),
                y_train,
                method=balance_method,
                skip_encoding=True  # Data already encoded
            )
        
        # Feature scaling
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # Ensure data is in correct format for scaling
        if isinstance(X_train_encoded, pd.DataFrame):
            X_train_for_scaling = X_train_encoded.values
        else:
            X_train_for_scaling = X_train_encoded
            
        if isinstance(X_test_encoded, pd.DataFrame):
            X_test_for_scaling = X_test_encoded.values
        else:
            X_test_for_scaling = X_test_encoded
            
        X_train_scaled = scaler.fit_transform(X_train_for_scaling)
        X_test_scaled = scaler.transform(X_test_for_scaling)
        
        results = {}
        models_trained = {}
        
        # Train base models
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Hyperparameter tuning
                if tune_hyperparameters and name in self.param_grids:
                    model, best_params = self.hyperparameter_tuning(
                        model, self.param_grids[name],
                        X_train_scaled, y_train,
                        search_type='random'  # Faster than grid search
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                    best_params = {}
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Comprehensive evaluation
                eval_results = self.evaluate_model_comprehensive(
                    model, X_test_scaled, y_test, y_pred, label_encoder
                )
                
                # Cross-validation
                cv_results = self.train_with_cross_validation(
                    model, X_train_scaled, y_train
                )
                
                results[name] = {
                    **eval_results,
                    **cv_results,
                    'best_params': best_params,
                    'model': model
                }
                
                models_trained[name] = model
                
                logger.info(f"{name} - Accuracy: {eval_results['accuracy']:.4f}, "
                          f"F1: {eval_results['f1_score']:.4f}, "
                          f"Balanced Acc: {eval_results['balanced_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                continue
        
        # Train ensemble models
        if use_ensemble and len(models_trained) >= 3:
            logger.info("Training ensemble models...")
            ensemble_models = self.create_ensemble_models(models_trained)
            
            for name, model in ensemble_models.items():
                try:
                    logger.info(f"Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    eval_results = self.evaluate_model_comprehensive(
                        model, X_test_scaled, y_test, y_pred, label_encoder
                    )
                    
                    cv_results = self.train_with_cross_validation(
                        model, X_train_scaled, y_train, cv_folds=5  # Fewer folds for ensemble
                    )
                    
                    results[name] = {
                        **eval_results,
                        **cv_results,
                        'model': model
                    }
                    
                    logger.info(f"{name} - Accuracy: {eval_results['accuracy']:.4f}, "
                              f"F1: {eval_results['f1_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train ensemble {name}: {str(e)}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['f1_score']
        
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Best F1 Score: {self.best_score:.4f}")
        
        # Feature importance
        feature_importance = self.get_feature_importance(
            self.best_model, 
            X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        )
        
        # Deep learning integration (PyTorch on Apple Silicon)
        deep_learning_results = None
        if use_deep_learning:
            # Load PyTorch deep learning module on demand
            if _load_deep_learning():
                try:
                    logger.info("\n🧠 Training PyTorch deep learning models...")
                    pytorch_dl = PyTorchDeepLearning()
                    
                    # Train all PyTorch architectures with the processed data
                    pytorch_results = pytorch_dl.train_all_models(
                        X_train_scaled, X_test_scaled, y_train, y_test, epochs=50
                    )
                    
                    # Add PyTorch results to main results with proper formatting
                    for arch_name, pytorch_result in pytorch_results.items():
                        if 'error' not in pytorch_result:
                            results[f"PyTorch_{arch_name}"] = {
                                'accuracy': pytorch_result['accuracy'],
                                'f1_score': pytorch_result['f1_score'],
                                'balanced_accuracy': pytorch_result.get('balanced_accuracy', pytorch_result['accuracy']),
                                'precision': pytorch_result.get('precision', pytorch_result['f1_score']),
                                'recall': pytorch_result.get('recall', pytorch_result['f1_score']),
                                'cohen_kappa': 0.0,  # Not implemented in PyTorch module
                                'matthews_corrcoef': 0.0,  # Not implemented in PyTorch module
                                'roc_auc': None,
                                'confusion_matrix': None,
                                'classification_report': None,
                                'support': None,
                                'model': None,  # PyTorch models saved separately
                                'best_params': {}
                            }
                            
                            logger.info(f"PyTorch {arch_name} - Accuracy: {pytorch_result['accuracy']:.4f}, "
                                      f"F1: {pytorch_result['f1_score']:.4f}, "
                                      f"Training time: {pytorch_result['training_time']:.2f}s")
                    
                    # Update best model if PyTorch performed better
                    best_pytorch_f1 = max((result['f1_score'] for result in pytorch_results.values() 
                                         if 'error' not in result), default=0)
                    
                    if best_pytorch_f1 > self.best_score:
                        best_pytorch_name = max(pytorch_results.keys(), 
                                              key=lambda x: pytorch_results[x].get('f1_score', 0))
                        logger.info(f"🏆 PyTorch {best_pytorch_name} achieved best performance: {best_pytorch_f1:.4f}")
                        self.best_score = best_pytorch_f1
                        best_model_name = f"PyTorch_{best_pytorch_name}"
                    
                except Exception as e:
                    logger.error(f"PyTorch deep learning training failed: {str(e)}")
                    logger.info("Continuing with traditional ML models only...")
        
            else:
                logger.warning("Deep learning requested but PyTorch not available.")
        
        return {
            'results': results,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_selector': selector,
            'X_test': X_test,
            'y_test': y_test,
            'feature_importance': feature_importance,
            'best_model_name': best_model_name,
            'deep_learning_results': deep_learning_results
        }
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from various model types"""
        importance_values = None
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
        # Linear models
        elif hasattr(model, 'coef_'):
            importance_values = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        # Ensemble models
        elif hasattr(model, 'estimators_'):
            if hasattr(model.estimators_[0], 'feature_importances_'):
                importance_values = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        
        if importance_values is not None:
            # Ensure arrays have the same length
            min_length = min(len(feature_names), len(importance_values))
            feature_names_trimmed = feature_names[:min_length]
            importance_values_trimmed = importance_values[:min_length]
            
            importance_df = pd.DataFrame({
                'feature': feature_names_trimmed,
                'importance': importance_values_trimmed
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        # Return empty DataFrame if no importance available
        return pd.DataFrame(columns=['feature', 'importance'])
    
    def save_enhanced_model(self, training_results: Dict[str, Any]) -> None:
        """Save all model artifacts with versioning"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Create versioned directory
        version_dir = self.model_dir / f'model_v_{timestamp}'
        version_dir.mkdir(exist_ok=True)
        
        # Save best model
        model_path = version_dir / 'best_model.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Save preprocessing artifacts
        joblib.dump(training_results['scaler'], version_dir / 'scaler.pkl')
        joblib.dump(training_results['label_encoder'], version_dir / 'label_encoder.pkl')
        
        if training_results['feature_selector']:
            joblib.dump(training_results['feature_selector'], version_dir / 'feature_selector.pkl')
        
        # Save feature importance
        if not training_results['feature_importance'].empty:
            training_results['feature_importance'].to_csv(
                version_dir / 'feature_importance.csv', index=False
            )
        
        # Save training metadata
        metadata = {
            'best_model_name': training_results['best_model_name'],
            'best_score': self.best_score,
            'timestamp': timestamp,
            'results_summary': {
                name: {
                    'accuracy': res['accuracy'],
                    'f1_score': res['f1_score'],
                    'balanced_accuracy': res['balanced_accuracy']
                }
                for name, res in training_results['results'].items()
            }
        }
        
        import json
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlink to latest model
        latest_link = self.model_dir / 'latest'
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version_dir.name)
        
        logger.info(f"Model artifacts saved to {version_dir}")
        logger.info(f"Latest model symlinked at {latest_link}")