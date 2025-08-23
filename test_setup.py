#!/usr/bin/env python3
"""
Quick test script to verify the restructured project works correctly
"""

import sys
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all core module imports"""
    print("🧪 Testing module imports...")
    
    modules = [
        'config.settings',
        'src.data_loader',
        'src.feature_engineering', 
        'src.model_training',
        'src.risk_predictor',
        'src.visualization',
        'api.main',
        'api.models',
        'api.endpoints',
    ]
    
    failed_imports = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_streamlit_pages():
    """Test Streamlit pages can be imported"""
    print("\n🎨 Testing Streamlit pages...")
    
    pages = [
        'app.streamlit_app',
        'app.utils',
        'app.pages.1_📊_Data_Overview',
        'app.pages.2_📈_Risk_Analysis', 
        'app.pages.3_🗺️_Geographic_Analysis',
        'app.pages.4_🤖_Risk_Prediction',
        'app.pages.5_📉_Model_Performance'
    ]
    
    failed_pages = []
    for page in pages:
        try:
            spec = importlib.util.spec_from_file_location(
                page, 
                str(project_root / page.replace('.', '/').replace('📊', '📊').replace('📈', '📈').replace('🗺️', '🗺️').replace('🤖', '🤖').replace('📉', '📉') + '.py')
            )
            if spec and spec.loader:
                print(f"  ✅ {page}")
            else:
                print(f"  ❌ {page}: File not found")
                failed_pages.append(page)
        except Exception as e:
            print(f"  ❌ {page}: {e}")
            failed_pages.append(page)
    
    return len(failed_pages) == 0

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config.settings import (
            PROJECT_ROOT, DATA_DIR, MODEL_CONFIG, 
            STREAMLIT_CONFIG, API_CONFIG, VIZ_CONFIG
        )
        
        print(f"  ✅ PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"  ✅ DATA_DIR: {DATA_DIR}")
        print(f"  ✅ MODEL_CONFIG loaded")
        print(f"  ✅ STREAMLIT_CONFIG loaded")
        print(f"  ✅ API_CONFIG loaded")
        print(f"  ✅ VIZ_CONFIG loaded")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_data_directories():
    """Test data directory structure"""
    print("\n📁 Testing directory structure...")
    
    directories = [
        'data',
        'data/raw',
        'data/processed', 
        'data/models',
        'src',
        'api',
        'app',
        'app/pages',
        'tests',
        'scripts',
        'config'
    ]
    
    missing_dirs = []
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (missing)")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0

def test_scripts():
    """Test script functionality"""
    print("\n📜 Testing scripts...")
    
    scripts = [
        'scripts/download_data.py',
        'scripts/train_models.py'
    ]
    
    failed_scripts = []
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} (missing)")
            failed_scripts.append(script)
    
    return len(failed_scripts) == 0

def main():
    """Run all tests"""
    print("🚀 Testing UK Road Risk Classification System Setup")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Streamlit Pages", test_streamlit_pages),
        ("Configuration", test_configuration), 
        ("Directory Structure", test_data_directories),
        ("Scripts", test_scripts)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Download data: python scripts/download_data.py --limit-rows 1000") 
        print("  3. Train models: python scripts/train_models.py --limit-rows 1000")
        print("  4. Run Streamlit: streamlit run app/streamlit_app.py")
        print("  5. Run API: uvicorn api.main:app --reload")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)