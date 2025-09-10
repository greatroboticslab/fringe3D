"""
Phase Map to Point Cloud Converter
==================================

Converts phase maps from PhaseWare (MATLAB) to 3D point clouds for MeshLab.
Supports multiple input/output formats and includes visualization tools.

Author: AI Assistant
Created for: PhaseWare 3D Reconstruction Project
"""

import numpy as np
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from typing import Tuple, Optional, Dict, Any, Union
import warnings

class PhaseMapLoader:
    """Handle loading phase maps from various sources"""
    
    def load_matlab_file(self, filepath: str) -> np.ndarray:
        """Load phase map from MATLAB .mat file"""
        try:
            mat_data = scipy.io.loadmat(filepath)
            
            # Try common variable names for phase maps
            possible_names = ['phase_map', 'phase', 'data', 'image', 'map']
            
            phase_map = None
            for name in possible_names:
                if name in mat_data:
                    phase_map = mat_data[name]
                    break
            
            # If no common names found, get the largest 2D array
            if phase_map is None:
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                        if phase_map is None or value.size > phase_map.size:
                            phase_map = value
            
            if phase_map is None:
                raise ValueError("No suitable 2D array found in MATLAB file")
            
            # Ensure it's a 2D array
            if phase_map.ndim != 2:
                raise ValueError(f"Phase map must be 2D, got {phase_map.ndim}D")
            
            print(f"Loaded phase map: {phase_map.shape} from {filepath}")
            return phase_map.astype(np.float64)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MATLAB file {filepath}: {e}")
    
    def load_csv_file(self, filepath: str) -> np.ndarray:
        """Load phase map from CSV file"""
        try:
            phase_map = np.loadtxt(filepath, delimiter=',')
            
            if phase_map.ndim != 2:
                raise ValueError(f"CSV data must be 2D, got {phase_map.ndim}D")
            
            print(f"Loaded phase map: {phase_map.shape} from {filepath}")
            return phase_map.astype(np.float64)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file {filepath}: {e}")
    
    def load_numpy_file(self, filepath: str) -> np.ndarray:
        """Load phase map from NumPy .npy file"""
        try:
            phase_map = np.load(filepath)
            
            if phase_map.ndim != 2:
                raise ValueError(f"NumPy data must be 2D, got {phase_map.ndim}D")
            
            print(f"Loaded phase map: {phase_map.shape} from {filepath}")
            return phase_map.astype(np.float64)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load NumPy file {filepath}: {e}")
    
    def load_auto(self, filepath: str) -> np.ndarray:
        """Automatically detect file type and load phase map"""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.mat':
            return self.load_matlab_file(str(filepath))
        elif filepath.suffix.lower() == '.csv':
            return self.load_csv_file(str(filepath))
        elif filepath.suffix.lower() == '.npy':
            return self.load_numpy_file(str(filepath))
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def validate_phase_map(self, phase_map: np.ndarray) -> bool:
        """Validate phase map data integrity"""
        if not isinstance(phase_map, np.ndarray):
            return False
        
        if phase_map.ndim != 2:
            return False
        
        if phase_map.size == 0:
            return False
        
        # Check for reasonable phase values (typically -π to π or 0 to 2π)
        if np.all(np.isnan(phase_map)) or np.all(np.isinf(phase_map)):
            return False
        
        return True

class CoordinateMapper:
    """Transform pixel coordinates to physical space"""
    
    def __init__(self, pixel_size_mm: float = 1.0, origin_offset: Tuple[float, float] = (0, 0)):
        """
        Initialize coordinate mapper
        
        Args:
            pixel_size_mm: Physical size of each pixel in millimeters
            origin_offset: (x, y) offset for coordinate system origin
        """
        self.pixel_size = pixel_size_mm
        self.origin_offset = origin_offset
    
    def pixel_to_physical(self, row: int, col: int) -> Tuple[float, float]:
        """Convert pixel coordinates to physical x,y coordinates"""
        x = col * self.pixel_size + self.origin_offset[0]
        y = row * self.pixel_size + self.origin_offset[1]
        return x, y
    
    def create_coordinate_grids(self, phase_map_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate X,Y coordinate grids for entire phase map"""
        rows, cols = phase_map_shape
        
        # Create pixel coordinate arrays
        col_indices = np.arange(cols)
        row_indices = np.arange(rows)
        
        # Convert to physical coordinates
        x_coords = col_indices * self.pixel_size + self.origin_offset[0]
        y_coords = row_indices * self.pixel_size + self.origin_offset[1]
        
        # Create meshgrids
        X, Y = np.meshgrid(x_coords, y_coords)
        
        return X, Y

class PhaseProcessor:
    """Convert phase values to height information"""
    
    def __init__(self, wavelength_nm: float = 632.8, scaling_factor: float = 1.0):
        """
        Initialize phase processor
        
        Args:
            wavelength_nm: Laser wavelength in nanometers
            scaling_factor: Additional scaling factor for height conversion
        """
        self.wavelength = wavelength_nm * 1e-6  # Convert to meters
        self.scaling_factor = scaling_factor
    
    def phase_to_height(self, phase_map: np.ndarray) -> np.ndarray:
        """Convert phase values to physical heights"""
        # Basic phase-to-height conversion: height = phase * wavelength / (4π)
        # This assumes the phase represents optical path difference
        heights = phase_map * self.wavelength / (4 * np.pi)
        
        # Apply additional scaling
        heights *= self.scaling_factor
        
        # Convert to millimeters for easier handling
        heights *= 1000  # meters to mm
        
        return heights
    
    def apply_height_scaling(self, heights: np.ndarray, scale_factor: float) -> np.ndarray:
        """Apply calibration scaling to height values"""
        return heights * scale_factor
    
    def remove_background_tilt(self, heights: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Remove background tilt/plane from height data"""
        if mask is None:
            # Use all valid (non-NaN, non-inf) points
            mask = np.isfinite(heights)
        
        if not np.any(mask):
            warnings.warn("No valid points for background removal")
            return heights
        
        # Get coordinates of valid points
        rows, cols = np.where(mask)
        valid_heights = heights[mask]
        
        if len(valid_heights) < 3:
            warnings.warn("Not enough points for plane fitting")
            return heights
        
        # Fit a plane: z = ax + by + c
        A = np.column_stack([cols, rows, np.ones(len(cols))])
        
        try:
            # Solve for plane coefficients
            coeffs, residuals, rank, s = np.linalg.lstsq(A, valid_heights, rcond=None)
            
            # Create plane for entire image
            row_grid, col_grid = np.mgrid[0:heights.shape[0], 0:heights.shape[1]]
            background_plane = coeffs[0] * col_grid + coeffs[1] * row_grid + coeffs[2]
            
            # Remove background
            corrected_heights = heights - background_plane
            
            print(f"Background plane removed. Coefficients: a={coeffs[0]:.6f}, b={coeffs[1]:.6f}, c={coeffs[2]:.6f}")
            return corrected_heights
            
        except np.linalg.LinAlgError:
            warnings.warn("Failed to fit background plane")
            return heights

class PointCloudBuilder:
    """Generate 3D point cloud from processed data"""
    
    def __init__(self):
        self.points = None
        self.colors = None
    
    def build_point_cloud(self, x_coords: np.ndarray, y_coords: np.ndarray, 
                         heights: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create point cloud array from coordinate and height data"""
        
        if mask is None:
            # Create mask for valid points (finite values)
            mask = np.isfinite(heights) & np.isfinite(x_coords) & np.isfinite(y_coords)
        
        # Get valid points
        valid_x = x_coords[mask]
        valid_y = y_coords[mask]
        valid_z = heights[mask]
        
        # Stack into point cloud array (N x 3)
        points = np.column_stack([valid_x, valid_y, valid_z])
        
        print(f"Generated point cloud with {len(points)} points")
        return points
    
    def add_color_mapping(self, heights: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
        """Generate color values based on height for visualization"""
        # Normalize heights to 0-1 range
        valid_heights = heights[np.isfinite(heights)]
        
        if len(valid_heights) == 0:
            return np.zeros((heights.size, 3))
        
        h_min, h_max = np.min(valid_heights), np.max(valid_heights)
        
        if h_max == h_min:
            # All heights are the same, use single color
            colors = np.full((heights.size, 3), 0.5)
        else:
            # Normalize heights
            normalized_heights = (heights - h_min) / (h_max - h_min)
            
            # Apply colormap
            cmap = plt.cm.get_cmap(colormap)
            colors = cmap(normalized_heights.flatten())[:, :3]  # RGB only, no alpha
        
        return colors
    
    def filter_invalid_points(self, points: np.ndarray, z_threshold: Optional[float] = None) -> np.ndarray:
        """Remove NaN, infinite, or outlier points"""
        # Remove NaN and infinite points
        valid_mask = np.all(np.isfinite(points), axis=1)
        
        if z_threshold is not None:
            # Remove points with extreme Z values
            z_values = points[:, 2]
            z_median = np.median(z_values[np.isfinite(z_values)])
            z_std = np.std(z_values[np.isfinite(z_values)])
            
            z_outlier_mask = np.abs(z_values - z_median) < z_threshold * z_std
            valid_mask = valid_mask & z_outlier_mask
        
        filtered_points = points[valid_mask]
        
        if len(filtered_points) < len(points):
            print(f"Filtered {len(points) - len(filtered_points)} invalid/outlier points")
        
        return filtered_points

class FileExporter:
    """Export point clouds to various file formats"""
    
    def export_ply(self, points: np.ndarray, filepath: str, colors: Optional[np.ndarray] = None):
        """Export to PLY format for MeshLab"""
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # Write points
            for i, point in enumerate(points):
                if colors is not None:
                    # Convert colors from 0-1 to 0-255
                    r, g, b = (colors[i] * 255).astype(int)
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"Exported PLY file: {filepath}")
    
    def export_xyz(self, points: np.ndarray, filepath: str):
        """Export to simple XYZ text format"""
        filepath = Path(filepath)
        
        np.savetxt(filepath, points, fmt='%.6f', delimiter=' ', 
                   header='X Y Z', comments='')
        
        print(f"Exported XYZ file: {filepath}")
    
    def export_obj(self, points: np.ndarray, filepath: str):
        """Export to OBJ format"""
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            f.write("# Point cloud exported from PhaseWare\n")
            f.write(f"# {len(points)} vertices\n")
            
            for point in points:
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"Exported OBJ file: {filepath}")

class Visualizer:
    """Visualization and debugging tools"""
    
    def plot_phase_map(self, phase_map: np.ndarray, title: str = "Phase Map"):
        """Display 2D phase map"""
        plt.figure(figsize=(10, 8))
        plt.imshow(phase_map, cmap='hsv', origin='upper')
        plt.colorbar(label='Phase (radians)')
        plt.title(title)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.show()
    
    def plot_height_map(self, heights: np.ndarray, title: str = "Height Map"):
        """Display 2D height map"""
        plt.figure(figsize=(10, 8))
        plt.imshow(heights, cmap='viridis', origin='upper')
        plt.colorbar(label='Height (mm)')
        plt.title(title)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.show()
    
    def plot_3d_preview(self, points: np.ndarray, sample_rate: int = 10, colors: Optional[np.ndarray] = None):
        """Quick 3D scatter plot preview"""
        # Sample points for faster visualization
        if len(points) > 10000:
            indices = np.arange(0, len(points), sample_rate)
            sampled_points = points[indices]
            sampled_colors = colors[indices] if colors is not None else None
        else:
            sampled_points = points
            sampled_colors = colors
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        if sampled_colors is not None:
            ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                      c=sampled_colors, s=1, alpha=0.6)
        else:
            ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                      c=sampled_points[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Point Cloud Preview ({len(sampled_points)} points)')
        
        plt.show()

class PhaseToPointCloudConverter:
    """Main converter class that orchestrates the entire process"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize converter with configuration
        
        Args:
            config: Dictionary containing conversion parameters
        """
        self.loader = PhaseMapLoader()
        self.coord_mapper = CoordinateMapper(
            pixel_size_mm=config.get('pixel_size_mm', 1.0),
            origin_offset=config.get('origin_offset', (0, 0))
        )
        self.phase_processor = PhaseProcessor(
            wavelength_nm=config.get('wavelength_nm', 632.8),
            scaling_factor=config.get('scaling_factor', 1.0)
        )
        self.point_cloud_builder = PointCloudBuilder()
        self.exporter = FileExporter()
        self.visualizer = Visualizer()
        
        self.config = config
    
    def convert(self, input_file: str, output_file: str, 
                file_format: str = 'ply', visualize: bool = False) -> bool:
        """
        Main conversion pipeline
        
        Args:
            input_file: Path to input phase map file
            output_file: Path for output point cloud file
            file_format: Output format ('ply', 'xyz', 'obj')
            visualize: Whether to show visualization plots
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            print(f"Starting conversion: {input_file} -> {output_file}")
            
            # 1. Load phase map
            print("Loading phase map...")
            phase_map = self.loader.load_auto(input_file)
            
            if not self.loader.validate_phase_map(phase_map):
                raise ValueError("Invalid phase map data")
            
            # 2. Generate coordinate grids
            print("Generating coordinate grids...")
            x_coords, y_coords = self.coord_mapper.create_coordinate_grids(phase_map.shape)
            
            # 3. Convert phase to heights
            print("Converting phase to heights...")
            heights = self.phase_processor.phase_to_height(phase_map)
            
            # Optional background removal
            if self.config.get('remove_background', True):
                print("Removing background tilt...")
                heights = self.phase_processor.remove_background_tilt(heights)
            
            # 4. Build point cloud
            print("Building point cloud...")
            points = self.point_cloud_builder.build_point_cloud(x_coords, y_coords, heights)
            
            # Filter invalid points
            z_threshold = self.config.get('z_outlier_threshold', None)
            points = self.point_cloud_builder.filter_invalid_points(points, z_threshold)
            
            if len(points) == 0:
                raise ValueError("No valid points generated")
            
            # 5. Export to file
            print(f"Exporting to {file_format.upper()} format...")
            
            if file_format.lower() == 'ply':
                # Generate colors for PLY format
                colors = self.point_cloud_builder.add_color_mapping(points[:, 2])
                self.exporter.export_ply(points, output_file, colors)
            elif file_format.lower() == 'xyz':
                self.exporter.export_xyz(points, output_file)
            elif file_format.lower() == 'obj':
                self.exporter.export_obj(points, output_file)
            else:
                raise ValueError(f"Unsupported output format: {file_format}")
            
            # 6. Optional visualization
            if visualize:
                print("Generating visualizations...")
                self.visualizer.plot_phase_map(phase_map, "Original Phase Map")
                self.visualizer.plot_height_map(heights, "Height Map")
                
                # Generate colors for 3D preview
                colors = self.point_cloud_builder.add_color_mapping(points[:, 2])
                self.visualizer.plot_3d_preview(points, colors=colors)
            
            print(f"Conversion completed successfully!")
            print(f"Point cloud saved to: {output_file}")
            print(f"Total points: {len(points)}")
            
            return True
            
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

# Usage example and configuration
def main():
    """Example usage of the converter"""
    
    # Configuration parameters
    config = {
        'pixel_size_mm': 0.1,           # Physical size of each pixel in mm
        'wavelength_nm': 632.8,         # Laser wavelength (HeNe laser)
        'scaling_factor': 1.0,          # Additional height scaling
        'origin_offset': (0, 0),        # Coordinate system offset
        'remove_background': True,      # Remove background tilt
        'z_outlier_threshold': 3.0      # Remove Z outliers beyond 3 std devs
    }
    
    # Create converter
    converter = PhaseToPointCloudConverter(config)
    
    # Example conversion (update paths as needed)
    input_file = 'phase_map.mat'  # Your PhaseWare output file
    output_file = 'object_pointcloud.ply'
    
    # Convert phase map to point cloud
    success = converter.convert(
        input_file=input_file,
        output_file=output_file,
        file_format='ply',
        visualize=True
    )
    
    if success:
        print("\n" + "="*50)
        print("SUCCESS!")
        print("="*50)
        print(f"Point cloud generated: {output_file}")
        print("Import this file into MeshLab for 3D visualization")
        print("\nMeshLab import steps:")
        print("1. Open MeshLab")
        print("2. File -> Import Mesh...")
        print(f"3. Select {output_file}")
        print("4. View your 3D reconstruction!")
    else:
        print("\n" + "="*50)
        print("CONVERSION FAILED")
        print("="*50)
        print("Check the error messages above for troubleshooting")

if __name__ == "__main__":
    main()
