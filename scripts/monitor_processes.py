#!/usr/bin/env python3
"""
Process monitoring utility to prevent and clean up orphaned workers
"""

import argparse
import time
import psutil
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.process_manager import emergency_cleanup, cleanup_orphaned_processes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_training_processes(duration_minutes: int = 60, check_interval: int = 30):
    """Monitor training processes and clean up orphaned workers"""
    
    logger.info(f"🔍 Starting process monitoring for {duration_minutes} minutes")
    logger.info(f"   Checking every {check_interval} seconds for orphaned processes")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    while time.time() < end_time:
        try:
            # Check for Python processes
            python_processes = []
            training_processes = []
            joblib_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'create_time']):
                try:
                    if 'python' not in proc.info['name'].lower():
                        continue
                        
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    # Categorize processes
                    if any(pattern in cmdline for pattern in ['train_models', 'training']):
                        training_processes.append(proc)
                    elif any(pattern in cmdline for pattern in ['joblib.externals.loky', 'multiprocessing']):
                        joblib_processes.append(proc)
                    elif any(pattern in cmdline for pattern in ['streamlit', 'uvicorn']):
                        continue  # Skip service processes
                    else:
                        python_processes.append(proc)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Report status
            current_time = time.strftime("%H:%M:%S")
            print(f"\n[{current_time}] Process Status:")
            print(f"  🐍 Python processes: {len(python_processes)}")
            print(f"  🚀 Training processes: {len(training_processes)}")
            print(f"  ⚙️  Joblib workers: {len(joblib_processes)}")
            
            # Check for orphaned joblib processes (older than 5 minutes with high CPU)
            orphaned_count = 0
            for proc in joblib_processes:
                try:
                    age_minutes = (time.time() - proc.info['create_time']) / 60
                    cpu_percent = proc.info['cpu_percent'] or 0
                    
                    if age_minutes > 5 and cpu_percent > 50:
                        orphaned_count += 1
                        logger.warning(f"Orphaned process detected: PID {proc.info['pid']}, Age: {age_minutes:.1f}min, CPU: {cpu_percent:.1f}%")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if orphaned_count > 0:
                logger.warning(f"🧹 Found {orphaned_count} potentially orphaned processes, cleaning up...")
                cleanup_orphaned_processes()
                print(f"  ✅ Cleaned up orphaned processes")
            else:
                print(f"  ✅ No orphaned processes detected")
            
            # Sleep until next check
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(check_interval)
    
    logger.info("🏁 Process monitoring completed")

def cleanup_all_orphaned():
    """Clean up all orphaned processes immediately"""
    
    logger.info("🧹 Starting emergency cleanup of all orphaned processes...")
    
    # Get initial count
    initial_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' not in proc.info['name'].lower():
                continue
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(pattern in cmdline for pattern in ['joblib.externals.loky', 'train_models']):
                initial_count += 1
        except:
            continue
    
    print(f"Found {initial_count} processes to clean up")
    
    # Perform cleanup
    emergency_cleanup()
    
    # Get final count
    time.sleep(2)  # Wait for cleanup to complete
    final_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' not in proc.info['name'].lower():
                continue
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(pattern in cmdline for pattern in ['joblib.externals.loky', 'train_models']):
                final_count += 1
        except:
            continue
    
    cleaned_count = initial_count - final_count
    print(f"✅ Cleanup completed: {cleaned_count} processes terminated")
    
    if final_count > 0:
        print(f"⚠️  {final_count} processes still running (may be legitimate)")

def main():
    parser = argparse.ArgumentParser(
        description='Monitor and clean up training processes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor for 30 minutes, check every 30 seconds
  python monitor_processes.py --monitor --duration 30
  
  # Clean up all orphaned processes immediately
  python monitor_processes.py --cleanup
  
  # Monitor continuously 
  python monitor_processes.py --monitor --duration 0
        """
    )
    
    parser.add_argument('--monitor', action='store_true',
                       help='Start monitoring for orphaned processes')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up orphaned processes immediately')
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in minutes (0 = infinite)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds')
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_all_orphaned()
    elif args.monitor:
        duration = float('inf') if args.duration == 0 else args.duration
        monitor_training_processes(duration, args.interval)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()