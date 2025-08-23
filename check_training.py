#!/usr/bin/env python3
"""
Quick training status checker
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def check_training_status():
    print("🔍 UK Road Risk - Quick Training Check")
    print("====================================")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        training_procs = [line for line in result.stdout.split('\n') 
                         if 'train_models_enhanced.py' in line and 'grep' not in line]
        
        print("🚀 Training Processes:")
        if training_procs:
            for proc in training_procs:
                parts = proc.split()
                if len(parts) >= 11:
                    pid, cpu, mem, time = parts[1], parts[2], parts[3], parts[9]
                    print(f"  PID: {pid} | CPU: {cpu}% | Memory: {mem}% | Time: {time}")
        else:
            print("  ❌ No training processes found")
    except Exception as e:
        print(f"  ❌ Error checking processes: {e}")
    
    print()
    
    # Check status file
    status_file = Path("data/models/training_status.json")
    print("📊 Training Status:")
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)
            print(f"  Status: {status.get('status', 'unknown')}")
            print(f"  Progress: {status.get('progress', 0)}%")
            print(f"  Message: {status.get('message', 'N/A')}")
            print(f"  Last Update: {status.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Error reading status: {e}")
    else:
        print("  ❌ No status file found")
    
    print()
    
    # Check latest models
    models_dir = Path("data/models")
    print("📁 Latest Models:")
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('model_v_')]
        if model_dirs:
            latest = sorted(model_dirs, key=lambda x: x.stat().st_mtime)[-1]
            print(f"  Latest: {latest.name}")
            print(f"  Created: {datetime.fromtimestamp(latest.stat().st_mtime)}")
        else:
            print("  ❌ No model directories found")
    else:
        print("  ❌ Models directory not found")

if __name__ == "__main__":
    check_training_status()