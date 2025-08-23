"""
Training monitoring utilities for real-time progress tracking
"""

import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class TrainingMonitor:
    """Monitor training progress and provide real-time updates"""
    
    def __init__(self, status_file: str = "data/models/training_status.json"):
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def update_status(self, status: str, progress: float = 0, 
                     message: str = "", details: Optional[Dict[str, Any]] = None):
        """Update training status"""
        with self._lock:
            data = {
                'status': status,  # 'starting', 'data_loading', 'training', 'completed', 'failed'
                'progress': progress,  # 0-100
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'details': details or {}
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        if not self.status_file.exists():
            return {
                'status': 'idle',
                'progress': 0,
                'message': 'No training in progress',
                'timestamp': datetime.now().isoformat(),
                'details': {}
            }
        
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {
                'status': 'unknown',
                'progress': 0,
                'message': 'Unable to read status',
                'timestamp': datetime.now().isoformat(),
                'details': {}
            }
    
    def is_training_active(self) -> bool:
        """Check if training is currently active"""
        status = self.get_status()
        return status['status'] in ['starting', 'data_loading', 'training']
    
    def clear_status(self):
        """Clear training status"""
        if self.status_file.exists():
            self.status_file.unlink()


# Global monitor instance
monitor = TrainingMonitor()


def update_training_status(status: str, progress: float = 0, 
                          message: str = "", details: Optional[Dict[str, Any]] = None):
    """Convenience function to update training status"""
    monitor.update_status(status, progress, message, details)


def get_training_status() -> Dict[str, Any]:
    """Convenience function to get training status"""
    return monitor.get_status()