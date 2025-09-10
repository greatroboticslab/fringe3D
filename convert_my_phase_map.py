#!/usr/bin/env python3
"""
Quick Start Script for Your PhaseWare Data
==========================================

Simple script to convert your PhaseWare phase map to a point cloud.
Just update the file paths and configuration below.
"""

from phase_to_pointcloud import PhaseToPointCloudConverter

def convert_my_data():
    """Convert your specific PhaseWare data"""
    
    # =================================================================
    # CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
    # =================================================================
    
    # Input/Output files
    INPUT_FILE = 'your_phase_map.mat'    # Path to your PhaseWare .mat file
    OUTPUT_FILE = 'my_object.ply'        # Output point cloud file
    
    # Optical setup parameters
    config = {
        'pixel_size_mm': 0.05,           # ADJUST: Physical size of each pixel in mm
        'wavelength_nm': 632.8,          # ADJUST: Your laser wavelength (HeNe = 632.8)
        'scaling_factor': 1.0,           # ADJUST: Start with 1.0, increase if heights too small
        'origin_offset': (0, 0),         # Usually keep as (0, 0)
        'remove_background': True,       # Usually keep as True
        'z_outlier_threshold': 3.0       # Remove extreme outliers (3 std devs)
    }
    
    # =================================================================
    # CONVERSION PROCESS
    # =================================================================
    
    print("Phase Map to Point Cloud Converter")
    print("=" * 50)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Pixel size: {config['pixel_size_mm']} mm")
    print(f"Wavelength: {config['wavelength_nm']} nm")
    print("=" * 50)
    
    # Create converter
    converter = PhaseToPointCloudConverter(config)
    
    # Convert with visualization
    success = converter.convert(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        file_format='ply',
        visualize=True  # Set to False to skip plots
    )
    
    # Results
    if success:
        print("\n" + "=" * 50)
        print("✓ SUCCESS!")
        print("=" * 50)
        print(f"Point cloud saved to: {OUTPUT_FILE}")
        print("\nNext steps:")
        print("1. Open MeshLab")
        print("2. File → Import Mesh...")
        print(f"3. Select {OUTPUT_FILE}")
        print("4. View your 3D reconstruction!")
        print("\nIf the result doesn't look right:")
        print("- Adjust 'scaling_factor' (try 10, 100, or 0.1, 0.01)")
        print("- Check 'pixel_size_mm' matches your setup")
        print("- Verify your phase map has height variations")
    else:
        print("\n" + "=" * 50)
        print("✗ CONVERSION FAILED")
        print("=" * 50)
        print("Check the error messages above.")
        print("Common fixes:")
        print(f"- Make sure {INPUT_FILE} exists")
        print("- Check that the .mat file contains a 2D phase array")
        print("- Verify all dependencies are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
    convert_my_data()
