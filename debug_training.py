#!/usr/bin/env python3
"""
Debug script to test training from subprocess
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent
cmd = [
    "python", "scripts/train_models_enhanced.py", 
    "--quick", "--limit-rows", "500"
]

print(f"Testing command: {' '.join(cmd)}")
print(f"Working directory: {project_root}")

try:
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=60
    )
    
    print(f"\nReturn code: {result.returncode}")
    print("\n--- STDOUT ---")
    print(result.stdout)
    print("\n--- STDERR ---")
    print(result.stderr)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed with return code: {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("❌ Training timed out!")
except Exception as e:
    print(f"❌ Error running training: {e}")