# Phase Map to Point Cloud Converter

Convert phase maps from PhaseWare (MATLAB) to 3D point clouds for MeshLab visualization and processing.

## Overview

This tool bridges the gap between PhaseWare's phase extraction capabilities and 3D visualization software like MeshLab. It converts 2D phase maps (representing optical path differences) into 3D point clouds that can be imported into MeshLab for visualization, analysis, and further processing.

### Key Features

- **Multiple Input Formats**: MATLAB `.mat` files, CSV, NumPy `.npy`
- **Multiple Output Formats**: PLY (with colors), XYZ, OBJ
- **No Camera Calibration Required**: Works directly with PhaseWare's processed phase data
- **Automatic Background Removal**: Removes tilt and background planes
- **Built-in Visualization**: Preview your results before export
- **Configurable Parameters**: Easy adjustment for different optical setups

## Installation

1. **Clone or download** this repository to your computer

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** by running the example:
   ```bash
   python example_usage.py
   ```

## Quick Start

### Basic Usage

```python
from phase_to_pointcloud import PhaseToPointCloudConverter

# Configure for your setup
config = {
    'pixel_size_mm': 0.05,      # Physical size of each pixel
    'wavelength_nm': 632.8,     # Your laser wavelength
    'scaling_factor': 1.0,      # Height scaling factor
    'remove_background': True   # Remove background tilt
}

# Create converter and run
converter = PhaseToPointCloudConverter(config)
success = converter.convert(
    input_file='your_phase_map.mat',
    output_file='your_object.ply',
    visualize=True
)
```

### From PhaseWare to MeshLab

1. **Export from PhaseWare**: Save your phase map as a MATLAB file
2. **Convert to Point Cloud**: Run the Python script
3. **Import to MeshLab**: Open the generated `.ply` file in MeshLab

## Configuration Parameters

### Essential Parameters

| Parameter | Description | Example | Notes |
|-----------|-------------|---------|-------|
| `pixel_size_mm` | Physical size of each pixel in mm | `0.05` | Measure from your optical setup |
| `wavelength_nm` | Laser wavelength in nanometers | `632.8` | HeNe laser = 632.8nm |
| `scaling_factor` | Additional height scaling | `1.0` | Start with 1.0, adjust as needed |

### Optional Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `origin_offset` | Coordinate system offset (x,y) | `(0, 0)` | Centers the coordinate system |
| `remove_background` | Remove background tilt | `True` | Usually recommended |
| `z_outlier_threshold` | Remove extreme height outliers | `3.0` | Standard deviations from median |

## Workflow Integration

### PhaseWare → Python → MeshLab

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  PhaseWare  │───▶│    Python    │───▶│   MeshLab   │
│ (Phase Map) │    │ (Point Cloud)│    │ (3D Visual) │
└─────────────┘    └──────────────┘    └─────────────┘
```

### Step-by-Step Process

1. **PhaseWare Processing**:
   - Import your fringe image
   - Perform phase extraction (Fourier domain, carrier selection)
   - Apply post-processing (denoising, unwrapping, etc.)
   - Export phase map

2. **Python Conversion**:
   - Load phase map data
   - Convert phase values to physical heights
   - Generate 3D coordinates
   - Export as point cloud file

3. **MeshLab Visualization**:
   - Import point cloud file
   - View 3D reconstruction
   - Apply filters, measurements, or export to other formats

## File Formats

### Input Formats

- **MATLAB `.mat`**: Direct export from PhaseWare/MATLAB
- **CSV**: Comma-separated values (2D array)
- **NumPy `.npy`**: Python NumPy array format

### Output Formats

- **PLY**: Recommended for MeshLab (includes color information)
- **XYZ**: Simple text format (X Y Z coordinates)
- **OBJ**: Wavefront OBJ format (vertices only)

## Examples

### Example 1: Basic Conversion

```python
config = {
    'pixel_size_mm': 0.1,
    'wavelength_nm': 632.8,
    'scaling_factor': 1.0
}

converter = PhaseToPointCloudConverter(config)
converter.convert('phase_map.mat', 'output.ply', visualize=True)
```

### Example 2: High-Resolution Setup

```python
config = {
    'pixel_size_mm': 0.01,      # Fine pixel size
    'wavelength_nm': 532.0,     # Green laser
    'scaling_factor': 2.0,      # Amplify height variations
    'z_outlier_threshold': 2.5  # Aggressive outlier removal
}
```

### Example 3: Multiple Output Formats

```python
formats = ['ply', 'xyz', 'obj']
for fmt in formats:
    converter.convert('input.mat', f'output.{fmt}', file_format=fmt)
```

## Troubleshooting

### Common Issues

**Problem**: "No suitable 2D array found in MATLAB file"
- **Solution**: Ensure your phase map variable is saved properly in the `.mat` file
- **Check**: Variable names like 'phase_map', 'phase', 'data', 'image', or 'map'

**Problem**: Point cloud appears flat or has no height variation
- **Solution**: Adjust `scaling_factor` parameter (try values like 10, 100, or 0.1, 0.01)
- **Check**: Verify your phase map contains actual phase variations

**Problem**: Too many outlier points or noisy results
- **Solution**: Adjust `z_outlier_threshold` to a lower value (e.g., 2.0)
- **Check**: Enable `remove_background` to remove tilt

**Problem**: Coordinate system seems wrong
- **Solution**: Adjust `pixel_size_mm` based on your actual optical setup
- **Check**: Measure a known object in your setup to calibrate pixel size

### Determining Pixel Size

To find your `pixel_size_mm`:

1. Place a ruler or object of known size in your PhaseWare setup
2. Measure how many pixels it spans in the phase map
3. Calculate: `pixel_size_mm = actual_size_mm / pixels_spanned`

Example: If a 10mm ruler spans 200 pixels:
`pixel_size_mm = 10 / 200 = 0.05`

## Advanced Usage

### Custom Phase Processing

```python
# Access individual components for custom processing
loader = PhaseMapLoader()
phase_processor = PhaseProcessor(wavelength_nm=632.8, scaling_factor=2.0)

phase_map = loader.load_matlab_file('input.mat')
heights = phase_processor.phase_to_height(phase_map)
heights = phase_processor.remove_background_tilt(heights)
```

### Batch Processing

```python
import glob

config = {...}  # Your configuration
converter = PhaseToPointCloudConverter(config)

# Process all .mat files in a directory
for mat_file in glob.glob('*.mat'):
    output_file = mat_file.replace('.mat', '.ply')
    converter.convert(mat_file, output_file, visualize=False)
```

## Technical Details

### Phase to Height Conversion

The conversion uses the standard formula for optical path difference:
```
height = phase × wavelength / (4π)
```

This assumes:
- Phase represents optical path difference
- Single-pass measurement (not interferometric round-trip)
- Linear relationship between phase and height

### Coordinate System

- **X-axis**: Corresponds to image columns (left to right)
- **Y-axis**: Corresponds to image rows (top to bottom)
- **Z-axis**: Height derived from phase values
- **Units**: All coordinates in millimeters

### Background Removal

The tool fits a plane to the valid height data and subtracts it:
```
z_corrected = z_original - (a×x + b×y + c)
```

Where `a`, `b`, `c` are the fitted plane coefficients.

## Dependencies

- **NumPy**: Array operations and mathematical functions
- **SciPy**: MATLAB file I/O and scientific computing
- **Matplotlib**: Visualization and plotting
- **Python 3.7+**: Required Python version

## License

This project is provided as-is for research and educational purposes. Please cite appropriately if used in academic work.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your PhaseWare phase map export is correct
3. Test with the provided example script
4. Ensure all dependencies are properly installed

## Related Tools

- **PhaseWare**: Phase extraction from fringe patterns
- **MeshLab**: 3D mesh processing and visualization
- **CloudCompare**: Point cloud processing and analysis
- **Open3D**: Python library for 3D data processing
