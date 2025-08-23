"""
Advanced Deep Learning module for UK Road Risk Classification
Provides sophisticated neural network architectures for enhanced risk prediction
"""

import os
import platform

# Configure environment for Apple Silicon stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Apple Silicon specific configuration
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_METAL'] = '1'

import tensorflow as tf
import keras

# Configure TensorFlow for Apple Silicon stability
try:
    # Use conservative threading to avoid mutex issues
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Disable eager execution for stability
    tf.compat.v1.disable_eager_execution()
except Exception:
    # Fallback configuration if the above fails
    pass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, Tuple, Optional, Any, List
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Enable Apple Silicon GPU if available
if tf.config.list_physical_devices('GPU'):
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPU devices available: {len(gpus)}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU setup failed: {e}")

class DeepRiskClassifier:
    """Advanced deep learning classifier for road risk prediction"""
    
    def __init__(self, model_dir: str = "data/models/deep_learning"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.history = {}
        
    def create_feedforward_model(self, input_dim: int, num_classes: int, 
                                architecture: str = "deep") -> keras.Model:
        """Create feedforward neural network model"""
        
        if architecture == "simple":
            # Simple 2-layer network
            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            
        elif architecture == "deep":
            # Deep 4-layer network
            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.4),
                
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(64, activation='relu'),
                keras.layers.BatchNormalization(), 
                keras.layers.Dropout(0.2),
                
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.1),
                
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            
        elif architecture == "wide":
            # Wide network with more neurons per layer
            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                
                keras.layers.Dense(256, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.4),
                
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        return model
    
    def create_tabular_transformer(self, input_dim: int, num_classes: int) -> keras.Model:
        """Create TabNet-inspired transformer model for tabular data"""
        
        def attention_block(inputs, d_model, num_heads):
            attention_layer = keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )
            attn_output = attention_layer(inputs, inputs)
            
            # Add & Norm
            out1 = keras.layers.Add()([inputs, attn_output])
            out1 = keras.layers.LayerNormalization()(out1)
            
            # Feed Forward
            ffn = keras.Sequential([
                keras.layers.Dense(d_model * 2, activation='relu'),
                keras.layers.Dense(d_model)
            ])
            ffn_output = ffn(out1)
            
            # Add & Norm
            out2 = keras.layers.Add()([out1, ffn_output])
            out2 = keras.layers.LayerNormalization()(out2)
            
            return out2
        
        inputs = keras.layers.Input(shape=(input_dim,))
        
        # Feature embedding
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
        
        # Reshape for attention
        x = keras.layers.Reshape((-1, 128))(x)
        
        # Attention blocks
        x = attention_block(x, 128, 4)
        x = attention_block(x, 128, 4)
        
        # Global pooling
        x = keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_residual_model(self, input_dim: int, num_classes: int) -> keras.Model:
        """Create ResNet-inspired model with residual connections"""
        
        def residual_block(x, units, dropout_rate=0.2):
            # First layer
            out = keras.layers.Dense(units, activation='relu')(x)
            out = keras.layers.BatchNormalization()(out)
            out = keras.layers.Dropout(dropout_rate)(out)
            
            # Second layer  
            out = keras.layers.Dense(units, activation='relu')(out)
            out = keras.layers.BatchNormalization()(out)
            
            # Residual connection (if dimensions match)
            if x.shape[-1] == units:
                out = keras.layers.Add()([x, out])
            else:
                # Project input to match output dimension
                shortcut = keras.layers.Dense(units)(x)
                out = keras.layers.Add()([shortcut, out])
                
            out = keras.layers.Dropout(dropout_rate)(out)
            return out
        
        inputs = keras.layers.Input(shape=(input_dim,))
        
        # Initial dense layer
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        
        # Residual blocks
        x = residual_block(x, 128, 0.3)
        x = residual_block(x, 128, 0.3)
        x = residual_block(x, 64, 0.2)
        x = residual_block(x, 64, 0.2)
        
        # Final classification
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   architecture: str = "deep",
                   epochs: int = 100,
                   batch_size: int = 128,
                   validation_split: float = 0.2,
                   early_stopping: bool = True,
                   learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train deep learning model"""
        
        logger.info(f"Training {architecture} neural network...")
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        
        # Convert to categorical
        y_categorical = keras.utils.to_categorical(y_encoded, num_classes)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_categorical, test_size=validation_split, 
            random_state=42, stratify=y_encoded
        )
        
        # Create model
        input_dim = X_scaled.shape[1]
        
        if architecture == "transformer":
            model = self.create_tabular_transformer(input_dim, num_classes)
        elif architecture == "residual":
            model = self.create_residual_model(input_dim, num_classes)
        else:
            model = self.create_feedforward_model(input_dim, num_classes, architecture)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Model architecture: {architecture}")
        logger.info(f"Input dimensions: {input_dim}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        # Callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Learning rate reduction
        lr_reduce = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7
        )
        callbacks.append(lr_reduce)
        
        # Model checkpoint
        checkpoint_path = self.model_dir / f"{architecture}_best.keras"
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor='val_accuracy'
        )
        callbacks.append(checkpoint)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(
            X_val, y_val, verbose=0
        )
        
        # Calculate F1 score
        y_pred = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
        
        f1 = f1_score(y_val_classes, y_pred_classes, average='weighted')
        balanced_acc = balanced_accuracy_score(y_val_classes, y_pred_classes)
        
        # ROC AUC for multiclass
        try:
            roc_auc = roc_auc_score(y_val, y_pred, multi_class='ovr')
        except:
            roc_auc = None
        
        # Store model and scalers
        self.models[architecture] = model
        self.scalers[architecture] = scaler
        self.encoders[architecture] = label_encoder
        self.history[architecture] = history
        
        # Save model artifacts
        model.save(self.model_dir / f"{architecture}_model.keras")
        joblib.dump(scaler, self.model_dir / f"{architecture}_scaler.pkl")
        joblib.dump(label_encoder, self.model_dir / f"{architecture}_encoder.pkl")
        
        results = {
            'model_name': f"Deep {architecture.title()}",
            'architecture': architecture,
            'accuracy': float(val_accuracy),
            'f1_score': float(f1),
            'balanced_accuracy': float(balanced_acc),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'val_loss': float(val_loss),
            'epochs_trained': len(history.history['loss']),
            'total_params': int(model.count_params())
        }
        
        logger.info(f"✅ {architecture} model trained successfully!")
        logger.info(f"   Final validation accuracy: {val_accuracy:.4f}")
        logger.info(f"   F1 score: {f1:.4f}")
        logger.info(f"   Balanced accuracy: {balanced_acc:.4f}")
        
        return results
    
    def train_all_architectures(self, X: pd.DataFrame, y: pd.Series, 
                              epochs: int = 100) -> Dict[str, Dict[str, Any]]:
        """Train multiple deep learning architectures and compare"""
        
        architectures = ["simple", "deep", "wide", "residual", "transformer"]
        results = {}
        
        logger.info(f"🧠 Training {len(architectures)} deep learning architectures...")
        
        for arch in architectures:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {arch.upper()} architecture")
                logger.info(f"{'='*50}")
                
                result = self.train_model(X, y, architecture=arch, epochs=epochs)
                results[arch] = result
                
            except Exception as e:
                logger.error(f"Failed to train {arch}: {str(e)}")
                continue
        
        # Find best model
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
            logger.info(f"\n🏆 Best model: {best_model[0]} (F1: {best_model[1]['f1_score']:.4f})")
        
        return results
    
    def predict(self, X: pd.DataFrame, architecture: str = "deep") -> np.ndarray:
        """Make predictions with trained model"""
        
        if architecture not in self.models:
            raise ValueError(f"Model {architecture} not trained yet")
        
        model = self.models[architecture]
        scaler = self.scalers[architecture]
        encoder = self.encoders[architecture]
        
        X_scaled = scaler.transform(X)
        y_pred_proba = model.predict(X_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Convert back to original labels
        y_pred_labels = encoder.inverse_transform(y_pred_classes)
        
        return y_pred_labels
    
    def plot_training_history(self, architecture: str = "deep", save_plot: bool = True):
        """Plot training history"""
        
        if architecture not in self.history:
            logger.warning(f"No training history found for {architecture}")
            return
        
        history = self.history[architecture]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {architecture.title()} Architecture', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.model_dir / f"{architecture}_training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {plot_path}")
        
        plt.show()
    
    def get_model_summary(self, architecture: str = "deep") -> str:
        """Get model architecture summary"""
        
        if architecture not in self.models:
            return f"Model {architecture} not trained yet"
        
        model = self.models[architecture]
        
        # Capture model summary
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            model.summary()
        summary = f.getvalue()
        
        return summary

class DeepLearningIntegration:
    """Integration layer for adding deep learning to existing ML pipeline"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.deep_classifier = DeepRiskClassifier(model_dir / "deep_learning")
        
    def enhance_training_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                                epochs: int = 50) -> Dict[str, Any]:
        """Add deep learning models to existing training results"""
        
        logger.info("🧠 Adding deep learning models to training pipeline...")
        
        # Train deep learning models
        dl_results = self.deep_classifier.train_all_architectures(X, y, epochs=epochs)
        
        # Format results to match existing ML pipeline format
        formatted_results = {}
        for arch, result in dl_results.items():
            model_name = f"Deep {arch.title()}"
            formatted_results[model_name] = {
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'balanced_accuracy': result['balanced_accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'roc_auc': result.get('roc_auc'),
                'architecture': result['architecture'],
                'total_params': result['total_params']
            }
        
        # Find best deep learning model
        best_dl_model = max(dl_results.items(), key=lambda x: x[1]['f1_score'])
        
        return {
            'results': formatted_results,
            'best_model': best_dl_model[0],
            'best_score': best_dl_model[1]['f1_score'],
            'classifier': self.deep_classifier
        }