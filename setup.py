#!/usr/bin/env python3
"""
Setup script for ROI Detection Pipeline
This script helps set up the environment and validate the installation.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = [
        'torch', 'torchvision', 'cv2', 'PIL', 'pyvips', 
        'numpy', 'pandas', 'h5py', 'scipy', 'sklearn',
        'matplotlib', 'yaml', 'tqdm', 'alphashape'
    ]
    
    print("Checking package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available - will use CPU")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        'saved_models',
        'results',
        'logs',
        'data/patches',
        'data/annotations',
        'data/wsi'
    ]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

def validate_config():
    """Validate configuration file."""
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print(f"⚠ Configuration file not found: {config_file}")
        print("Please copy and modify config.yaml with your data paths")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for placeholder paths
        placeholder_paths = []
        def check_paths(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    check_paths(value, new_path)
            elif isinstance(obj, str) and obj.startswith('PATH_TO_'):
                placeholder_paths.append(f"{path}: {obj}")
        
        check_paths(config)
        
        if placeholder_paths:
            print("⚠ Configuration contains placeholder paths:")
            for path in placeholder_paths:
                print(f"  {path}")
            print("Please update these paths in config.yaml")
            return False
        
        print("✓ Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"✗ Error validating config: {e}")
        return False

def main():
    """Main setup function."""
    print("ROI Detection Pipeline Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Check imports
    if not check_imports():
        success = False
    
    # Check CUDA
    if not check_cuda():
        success = False
    
    # Create directories
    create_directories()
    
    # Validate config
    if not validate_config():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update config.yaml with your data paths")
        print("2. Run: python automated_pipeline.py --config config.yaml")
    else:
        print("✗ Setup completed with errors")
        print("Please fix the issues above before running the pipeline")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
