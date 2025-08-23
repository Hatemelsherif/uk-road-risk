"""
PyTorch-based Deep Learning module for Apple Silicon
Alternative to TensorFlow that works reliably on M1/M2/M3 Macs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from typing import Dict, Tuple, Optional, Any, List
import logging
from pathlib import Path
import joblib
import time

logger = logging.getLogger(__name__)

class PyTorchRiskDataset(Dataset):
    """Custom Dataset for road risk classification"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleRiskNet(nn.Module):
    """Simple feedforward neural network"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super(SimpleRiskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepRiskNet(nn.Module):
    """Deep neural network with multiple layers"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(DeepRiskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class WideRiskNet(nn.Module):
    """Wide neural network with large hidden layers"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(WideRiskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture"""
    
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual  # Residual connection
        return self.relu(out)

class ResidualRiskNet(nn.Module):
    """Residual neural network for risk classification"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(ResidualRiskNet, self).__init__()
        hidden_dim = 128
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        return self.output_layer(x)

class AttentionRiskNet(nn.Module):
    """Attention-based neural network"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(AttentionRiskNet, self).__init__()
        self.input_dim = input_dim
        hidden_dim = 128
        
        # Feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Embed features
        x = self.feature_embedding(x)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Remove sequence dimension
        x = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # Classify
        return self.classifier(x)

class PyTorchDeepLearningIntegration:
    """PyTorch-based deep learning integration for Apple Silicon"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.device = self._setup_device()
        self.models = {}
        self.architectures = {
            'Simple_Network': SimpleRiskNet,
            'Deep_Network': DeepRiskNet,
            'Wide_Network': WideRiskNet,
            'Residual_Network': ResidualRiskNet,
            'Attention_Network': AttentionRiskNet
        }
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        logger.info(f"PyTorch Deep Learning initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup the best available device for PyTorch"""
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("🍎 Using Apple Silicon GPU (MPS) for PyTorch")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("🔥 Using CUDA GPU for PyTorch")
        else:
            device = torch.device('cpu')
            logger.info("💻 Using CPU for PyTorch")
        
        return device
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray,
                   architecture: str = 'Deep_Network',
                   epochs: int = 100, batch_size: int = 32,
                   learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train a PyTorch neural network model"""
        
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        # Create model
        model_class = self.architectures[architecture]
        model = model_class(input_dim, num_classes).to(self.device)
        
        # Create datasets and data loaders
        train_dataset = PyTorchRiskDataset(X_train, y_train)
        test_dataset = PyTorchRiskDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        train_losses = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"{architecture} - Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Store model
        self.models[architecture] = {
            'model': model.cpu(),  # Move to CPU for storage
            'train_losses': train_losses,
            'training_time': training_time
        }
        
        logger.info(f"{architecture} - Training completed in {training_time:.2f}s")
        logger.info(f"{architecture} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        epochs: int = 50) -> Dict[str, Dict[str, Any]]:
        """Train all neural network architectures"""
        
        logger.info("Training all PyTorch neural network architectures...")
        results = {}
        
        for arch_name in self.architectures.keys():
            logger.info(f"Training {arch_name}...")
            try:
                result = self.train_model(
                    X_train, X_test, y_train, y_test,
                    architecture=arch_name,
                    epochs=epochs
                )
                results[arch_name] = result
            except Exception as e:
                logger.error(f"Failed to train {arch_name}: {e}")
                results[arch_name] = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def save_models(self, save_dir: Path):
        """Save trained PyTorch models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for arch_name, model_data in self.models.items():
            model_path = save_dir / f"{arch_name.lower()}_model.pth"
            torch.save(model_data['model'].state_dict(), model_path)
            logger.info(f"Saved {arch_name} to {model_path}")
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str],
                             architecture: str = 'Deep_Network') -> pd.DataFrame:
        """Get feature importance using permutation importance"""
        if architecture not in self.models:
            logger.warning(f"Model {architecture} not trained yet")
            return pd.DataFrame()
        
        model = self.models[architecture]['model'].to(self.device)
        model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get baseline predictions
        with torch.no_grad():
            baseline_outputs = model(X_tensor)
            baseline_probs = F.softmax(baseline_outputs, dim=1)
            baseline_entropy = -torch.sum(baseline_probs * torch.log(baseline_probs + 1e-8), dim=1)
            baseline_score = torch.mean(baseline_entropy).item()
        
        # Calculate permutation importance
        importances = []
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])  # Permute feature i
            
            X_perm_tensor = torch.FloatTensor(X_permuted).to(self.device)
            with torch.no_grad():
                perm_outputs = model(X_perm_tensor)
                perm_probs = F.softmax(perm_outputs, dim=1)
                perm_entropy = -torch.sum(perm_probs * torch.log(perm_probs + 1e-8), dim=1)
                perm_score = torch.mean(perm_entropy).item()
            
            importance = perm_score - baseline_score  # Higher entropy = more important
            importances.append(importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df