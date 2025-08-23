"""
Process management utilities to prevent orphaned workers
"""

import os
import signal
import psutil
import atexit
import logging
from contextlib import contextmanager
from typing import List, Set
from pathlib import Path
import json
import threading
import time

logger = logging.getLogger(__name__)

class ProcessManager:
    """Manages training processes and prevents orphaned workers"""
    
    def __init__(self):
        self.tracked_processes: Set[int] = set()
        self.is_training = False
        self.cleanup_thread = None
        self._lock = threading.Lock()
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all_processes)
        
        # Handle signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully"""
        logger.info(f"Received signal {signum}, cleaning up processes...")
        self.cleanup_all_processes()
        exit(0)
    
    def start_training_session(self):
        """Mark start of training session"""
        with self._lock:
            self.is_training = True
            self.tracked_processes.clear()
            
        # Start monitoring thread
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self._monitor_processes, daemon=True)
            self.cleanup_thread.start()
            
        logger.info("Training session started with process monitoring")
    
    def end_training_session(self):
        """Mark end of training session and cleanup"""
        with self._lock:
            self.is_training = False
            
        self.cleanup_joblib_workers()
        logger.info("Training session ended, processes cleaned up")
    
    def register_process(self, pid: int):
        """Register a process for tracking"""
        with self._lock:
            self.tracked_processes.add(pid)
    
    def _monitor_processes(self):
        """Background thread to monitor and cleanup orphaned processes"""
        while True:
            try:
                if not self.is_training:
                    time.sleep(10)
                    continue
                    
                # Check for orphaned joblib processes
                orphaned = self.find_orphaned_joblib_processes()
                if orphaned:
                    logger.warning(f"Found {len(orphaned)} orphaned joblib processes")
                    self._kill_processes(orphaned)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                time.sleep(60)
    
    def find_orphaned_joblib_processes(self) -> List[int]:
        """Find orphaned joblib/loky processes"""
        orphaned_pids = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    # Check for joblib/loky processes
                    if any(pattern in cmdline for pattern in [
                        'joblib.externals.loky',
                        'multiprocessing.spawn',
                        'multiprocessing.resource_tracker'
                    ]):
                        # Check if process is older than 5 minutes and not tracked
                        age = time.time() - proc.info['create_time']
                        if age > 300:  # 5 minutes
                            orphaned_pids.append(proc.info['pid'])
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Error finding orphaned processes: {e}")
            
        return orphaned_pids
    
    def cleanup_joblib_workers(self):
        """Clean up all joblib worker processes"""
        killed_count = 0
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    if any(pattern in cmdline for pattern in [
                        'joblib.externals.loky.backend.popen_loky_posix',
                        'joblib.externals.loky.backend.resource_tracker',
                        'multiprocessing.resource_tracker'
                    ]):
                        proc.terminate()
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            if killed_count > 0:
                logger.info(f"Cleaned up {killed_count} joblib worker processes")
                
        except Exception as e:
            logger.error(f"Error cleaning up joblib workers: {e}")
    
    def _kill_processes(self, pids: List[int]):
        """Kill processes by PID"""
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                logger.info(f"Terminated orphaned process {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def cleanup_all_processes(self):
        """Emergency cleanup of all tracked processes"""
        try:
            with self._lock:
                self.is_training = False
                
            self.cleanup_joblib_workers()
            
            # Kill any tracked processes
            for pid in list(self.tracked_processes):
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            self.tracked_processes.clear()
            logger.info("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in emergency cleanup: {e}")

# Global instance
process_manager = ProcessManager()

@contextmanager
def training_session():
    """Context manager for safe training sessions"""
    try:
        process_manager.start_training_session()
        yield process_manager
    finally:
        process_manager.end_training_session()

def cleanup_orphaned_processes():
    """Utility function to clean up orphaned processes"""
    process_manager.cleanup_joblib_workers()

def emergency_cleanup():
    """Emergency cleanup function"""
    process_manager.cleanup_all_processes()