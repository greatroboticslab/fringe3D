#!/usr/bin/env python3
"""
Setup Script for Phase Map to Point Cloud Converter
===================================================

This script helps set up the environment and verify everything is working.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Try running manually: pip install -r requirements.txt")
        return False

def verify_imports():
    """Verify that all required modules can be imported"""
    print("\n🔍 Verifying imports...")
    required_modules = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_good = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not found")
            all_good = False
    
    return all_good

def test_converter():
    """Test the converter with sample data"""
    print("\n🧪 Testing converter...")
    try:
        from phase_to_pointcloud import PhaseToPointCloudConverter
        import numpy as np
        
        # Create simple test data
        test_phase = np.random.random((50, 50)) * np.pi
        np.save('test_phase.npy', test_phase)
        
        # Test conversion
        config = {
            'pixel_size_mm': 0.1,
            'wavelength_nm': 632.8,
            'scaling_factor': 1.0
        }
        
        converter = PhaseToPointCloudConverter(config)
        success = converter.convert(
            input_file='test_phase.npy',
            output_file='test_output.ply',
            file_format='ply',
            visualize=False
        )
        
        if success and os.path.exists('test_output.ply'):
            print("✅ Converter test passed")
            # Clean up test files
            os.remove('test_phase.npy')
            os.remove('test_output.ply')
            return True
        else:
            print("❌ Converter test failed")
            return False
            
    except Exception as e:
        print(f"❌ Converter test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Phase Map to Point Cloud Converter - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Verify imports
    if not verify_imports():
        return False
    
    # Test converter
    if not test_converter():
        return False
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("Your Phase Map to Point Cloud Converter is ready to use!")
    print("\nNext steps:")
    print("1. Export your phase map from PhaseWare as a .mat file")
    print("2. Edit 'convert_my_phase_map.py' with your file paths")
    print("3. Run: python convert_my_phase_map.py")
    print("4. Import the generated .ply file into MeshLab")
    print("\nFor examples and documentation, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
