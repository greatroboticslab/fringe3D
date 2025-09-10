#!/usr/bin/env python3
"""
Example Usage Script for Phase Map to Point Cloud Converter
===========================================================

This script demonstrates how to use the PhaseToPointCloudConverter
with your PhaseWare output files.

Usage:
    python example_usage.py
"""

from phase_to_pointcloud import PhaseToPointCloudConverter
import numpy as np

def create_sample_phase_map():
    """Create a sample phase map for testing (simulates PhaseWare output)"""
    # Create a simple test phase map with a circular object
    size = 200
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a circular phase pattern (simulating an object)
    radius = 0.6
    mask = (X**2 + Y**2) <= radius**2
    
    # Generate phase values (simulate surface height variations)
    phase_map = np.zeros((size, size))
    phase_map[mask] = np.sin(5 * np.sqrt(X[mask]**2 + Y[mask]**2)) * np.pi/2
    
    # Add some noise to make it more realistic
    phase_map += np.random.normal(0, 0.1, phase_map.shape)
    
    # Save as .npy file for testing
    np.save('sample_phase_map.npy', phase_map)
    print("Created sample phase map: sample_phase_map.npy")
    
    return phase_map

def example_basic_conversion():
    """Basic conversion example"""
    print("\n" + "="*60)
    print("BASIC CONVERSION EXAMPLE")
    print("="*60)
    
    # Configuration for your specific setup
    config = {
        'pixel_size_mm': 0.05,          # Adjust based on your setup
        'wavelength_nm': 632.8,         # HeNe laser wavelength
        'scaling_factor': 1.0,          # Start with 1.0, adjust as needed
        'origin_offset': (0, 0),        # Center the coordinate system
        'remove_background': True,      # Remove background tilt
        'z_outlier_threshold': 3.0      # Remove extreme outliers
    }
    
    # Create converter
    converter = PhaseToPointCloudConverter(config)
    
    # Convert phase map to point cloud
    success = converter.convert(
        input_file='sample_phase_map.npy',  # Your PhaseWare .mat file
        output_file='basic_output.ply',
        file_format='ply',
        visualize=True  # Set to False to skip plots
    )
    
    return success

def example_multiple_formats():
    """Example showing multiple output formats"""
    print("\n" + "="*60)
    print("MULTIPLE FORMATS EXAMPLE")
    print("="*60)
    
    config = {
        'pixel_size_mm': 0.05,
        'wavelength_nm': 632.8,
        'scaling_factor': 1.0,
        'origin_offset': (0, 0),
        'remove_background': True,
    }
    
    converter = PhaseToPointCloudConverter(config)
    
    # Export to different formats
    formats = ['ply', 'xyz', 'obj']
    
    for fmt in formats:
        output_file = f'output_example.{fmt}'
        success = converter.convert(
            input_file='sample_phase_map.npy',
            output_file=output_file,
            file_format=fmt,
            visualize=False  # Skip visualization for batch processing
        )
        
        if success:
            print(f"✓ Successfully created {output_file}")
        else:
            print(f"✗ Failed to create {output_file}")

def example_custom_configuration():
    """Example with custom configuration for different scenarios"""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Configuration for high-resolution setup
    high_res_config = {
        'pixel_size_mm': 0.01,          # Very fine pixel size
        'wavelength_nm': 532.0,         # Green laser
        'scaling_factor': 2.0,          # Amplify height variations
        'origin_offset': (-50, -50),    # Offset coordinate system
        'remove_background': True,
        'z_outlier_threshold': 2.5      # More aggressive outlier removal
    }
    
    converter = PhaseToPointCloudConverter(high_res_config)
    
    success = converter.convert(
        input_file='sample_phase_map.npy',
        output_file='high_res_output.ply',
        file_format='ply',
        visualize=True
    )
    
    return success

def main():
    """Main function demonstrating various usage scenarios"""
    print("Phase Map to Point Cloud Converter - Example Usage")
    print("=" * 60)
    
    # Create a sample phase map for testing
    create_sample_phase_map()
    
    # Run examples
    try:
        # Basic conversion
        success1 = example_basic_conversion()
        
        # Multiple formats
        example_multiple_formats()
        
        # Custom configuration
        success2 = example_custom_configuration()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if success1 and success2:
            print("✓ All examples completed successfully!")
            print("\nGenerated files:")
            print("- basic_output.ply")
            print("- output_example.ply")
            print("- output_example.xyz") 
            print("- output_example.obj")
            print("- high_res_output.ply")
            print("\nImport any .ply file into MeshLab to view your 3D reconstruction!")
        else:
            print("⚠ Some examples failed. Check error messages above.")
            
    except Exception as e:
        print(f"Error running examples: {e}")

# Instructions for using with your actual PhaseWare data
def instructions_for_real_data():
    """Print instructions for using with real PhaseWare data"""
    print("\n" + "="*60)
    print("USING WITH YOUR PHASEWARE DATA")
    print("="*60)
    print("""
1. Export your phase map from PhaseWare:
   - In MATLAB: save('my_phase_map.mat', 'phase_map')
   - Or export as CSV from PhaseWare interface

2. Update the configuration in your script:
   config = {
       'pixel_size_mm': YOUR_PIXEL_SIZE,    # Measure this from your setup
       'wavelength_nm': YOUR_LASER_WAVELENGTH,  # Usually 632.8 for HeNe
       'scaling_factor': 1.0,               # Adjust based on results
       'remove_background': True,           # Usually helpful
   }

3. Run the conversion:
   converter = PhaseToPointCloudConverter(config)
   converter.convert('my_phase_map.mat', 'my_object.ply', visualize=True)

4. Import the .ply file into MeshLab for 3D visualization

5. Fine-tune parameters:
   - Adjust pixel_size_mm based on your optical setup
   - Modify scaling_factor if heights seem too small/large
   - Try different wavelength values if using different laser
    """)

if __name__ == "__main__":
    main()
    instructions_for_real_data()
