"""
GIMP 3 GEGL Interface Module

This module provides an interface to GIMP 3's GEGL (Generic Graphics Library) operations
through Python scripting. It allows for advanced image processing operations that leverage
GIMP's powerful GEGL-based image processing engine.

Requirements:
- GIMP 3.x installed
- PyGObject (gi) for GIMP Python bindings
- GEGL operations available through GIMP's Python API

Usage:
    from extensions.gimp.gimp_gegl_client import GimpGeglClient
    
    client = GimpGeglClient()
    result_path = client.apply_gegl_operation(input_path, "gegl:brightness-contrast", 
                                            {"brightness": 0.2, "contrast": 0.1})
"""

from enum import Enum
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, Any, Optional, List, Union

from utils.config import config
from utils.logging_setup import get_logger
from utils.subprocess_flags import NO_CONSOLE_WINDOW
from extensions.gimp.gimp_gegl_validator import validate_gegl_operation

logger = get_logger("gimp_gegl_client")


class GimpBlendMode(Enum):
    """GIMP blending modes for GEGL operations."""
    REPLACE = "replace"
    NORMAL = "normal"
    DISSOLVE = "dissolve"
    BEHIND = "behind"
    COLOR_ERASE = "color-erase"
    ERASE = "erase"
    MERGE = "merge"
    SPLIT = "split"
    LIGHTEN_ONLY = "lighten-only"
    DARKEN_ONLY = "darken-only"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft-light"
    HARD_LIGHT = "hard-light"
    DODGE = "dodge"
    BURN = "burn"
    ADD = "add"
    SUBTRACT = "subtract"
    DIFFERENCE = "difference"
    GRAIN_EXTRACT = "grain-extract"
    GRAIN_MERGE = "grain-merge"
    DIVIDE = "divide"
    HUE = "hue"
    SATURATION = "saturation"
    COLOR = "color"
    VALUE = "value"
    LCH_HUE = "lch-hue"
    LCH_CHROMA = "lch-chroma"
    LCH_LIGHTNESS = "lch-lightness"
    LCH_COLOR = "lch-color"
    LUMINANCE = "luminance"
    LUMINANCE_SHIFT = "luminance-shift"
    LUMINANCE_SHIFT_UP = "luminance-shift-up"
    LUMINANCE_SHIFT_DOWN = "luminance-shift-down"
    LUMINANCE_SHIFT_LEFT = "luminance-shift-left"
    LUMINANCE_SHIFT_RIGHT = "luminance-shift-right"
    LUMINANCE_SHIFT_DIAGONAL = "luminance-shift-diagonal"
    LUMINANCE_SHIFT_ANTIDIAGONAL = "luminance-shift-antidiagonal"
    LUMINANCE_SHIFT_RADIAL_CW = "luminance-shift-radial-cw"
    LUMINANCE_SHIFT_RADIAL_CCW = "luminance-shift-radial-ccw"
    LUMINANCE_SHIFT_SPIRAL_CW = "luminance-shift-spiral-cw"
    LUMINANCE_SHIFT_SPIRAL_CCW = "luminance-shift-spiral-ccw"
    LUMINANCE_SHIFT_HORIZONTAL = "luminance-shift-horizontal"
    LUMINANCE_SHIFT_VERTICAL = "luminance-shift-vertical"
    LUMINANCE_SHIFT_SQUARE = "luminance-shift-square"
    LUMINANCE_SHIFT_DIAMOND = "luminance-shift-diamond"
    LUMINANCE_SHIFT_CROSS = "luminance-shift-cross"
    LUMINANCE_SHIFT_X = "luminance-shift-x"
    LUMINANCE_SHIFT_PLUS = "luminance-shift-plus"
    LUMINANCE_SHIFT_STAR = "luminance-shift-star"
    LUMINANCE_SHIFT_CIRCLE = "luminance-shift-circle"
    LUMINANCE_SHIFT_TRIANGLE = "luminance-shift-triangle"
    LUMINANCE_SHIFT_HEXAGON = "luminance-shift-hexagon"
    LUMINANCE_SHIFT_OCTAGON = "luminance-shift-octagon"
    LUMINANCE_SHIFT_PENTAGON = "luminance-shift-pentagon"
    LUMINANCE_SHIFT_HEPTAGON = "luminance-shift-heptagon"
    LUMINANCE_SHIFT_NONAGON = "luminance-shift-nonagon"
    LUMINANCE_SHIFT_DECAGON = "luminance-shift-decagon"
    LUMINANCE_SHIFT_UNDECAGON = "luminance-shift-undecagon"
    LUMINANCE_SHIFT_DODECAGON = "luminance-shift-dodecagon"


class GimpGeglClient:
    """
    Client for interfacing with GIMP 3's GEGL operations through Python scripting.
    
    This class provides methods to apply various GEGL operations to images using
    GIMP's Python API. It handles the communication with GIMP and manages
    temporary files and operations.
    """
    
    # Common GEGL operations and their parameter schemas
    GEGL_OPERATIONS = {
        "gegl:brightness-contrast": {
            "brightness": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0},
            "contrast": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0}
        },
        "gegl:color-balance": {
            "cyan-red": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0},
            "magenta-green": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0},
            "yellow-blue": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0}
        },
        "gegl:color-temperature": {
            "original-temperature": {"type": "float", "default": 6500.0},
            "intended-temperature": {"type": "float", "default": 6500.0}
        },
        "gegl:exposure": {
            "black": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "exposure": {"type": "float", "default": 0.0, "min": -10.0, "max": 10.0},
            "gamma": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0}
        },
        "gegl:levels": {
            "in-low": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "in-high": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "out-low": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            "out-high": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            "gamma": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0}
        },
        "gegl:curves": {
            "curve": {"type": "string", "default": "0 0 1 1"}  # Control points as string
        },
        "gegl:hue-saturation": {
            "hue": {"type": "float", "default": 0.0, "min": -180.0, "max": 180.0},
            "saturation": {"type": "float", "default": 0.0, "min": -100.0, "max": 100.0},
            "lightness": {"type": "float", "default": 0.0, "min": -100.0, "max": 100.0}
        },
        "gegl:gaussian-blur": {
            "std-dev-x": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0},
            "std-dev-y": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0}
        },
        "gegl:unsharp-mask": {
            "std-dev": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            "scale": {"type": "float", "default": 0.5, "min": 0.0, "max": 10.0}
        },
        "gegl:noise-reduce": {
            "iterations": {"type": "int", "default": 1, "min": 1, "max": 10},
            "spatial-radius": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            "temporal-radius": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0}
        },
        "gegl:wavelet-decompose": {
            "levels": {"type": "int", "default": 4, "min": 1, "max": 10}
        },
        "gegl:wavelet-reconstruct": {
            "levels": {"type": "int", "default": 4, "min": 1, "max": 10}
        }
    }
    
    def __init__(self, gimp_executable: Optional[str] = None, 
                 default_opacity: float = 1.0, 
                 default_blend_mode: Union[str, GimpBlendMode] = GimpBlendMode.NORMAL):
        """
        Initialize the GIMP GEGL client.
        
        Args:
            gimp_executable: Path to GIMP executable. If None, uses config.gimp_exe_loc
            default_opacity: Default opacity for operations (0.0 to 1.0)
            default_blend_mode: Default blend mode for operations
        """
        self.gimp_exe = gimp_executable or config.gimp_exe_loc
        self.default_opacity = max(0.0, min(1.0, default_opacity))
        self.default_blend_mode = self._normalize_blend_mode(default_blend_mode)
        self._validate_gimp_installation()
        self._temp_dir = None
        self._script_path = None
    
    def _normalize_blend_mode(self, blend_mode: Union[str, GimpBlendMode]) -> str:
        """Normalize blend mode to string value."""
        if isinstance(blend_mode, GimpBlendMode):
            return blend_mode.value
        elif isinstance(blend_mode, str):
            # Try to find matching enum value
            for mode in GimpBlendMode:
                if mode.value.lower() == blend_mode.lower():
                    return mode.value
            # If not found, return as-is (might be a valid GIMP blend mode)
            return blend_mode.lower()
        else:
            return GimpBlendMode.NORMAL.value
        
    def _validate_gimp_installation(self):
        """Validate that GIMP 3 is properly installed and accessible."""
        if not self.gimp_exe:
            raise RuntimeError("GIMP executable not configured. Set 'gimp_exe_loc' in config.json")
        
        try:
            # Check if GIMP is accessible
            result = subprocess.run([self.gimp_exe, "--version"], 
                                 capture_output=True, text=True, timeout=10, creationflags=NO_CONSOLE_WINDOW)
            if result.returncode != 0:
                raise RuntimeError(f"GIMP executable failed: {result.stderr}")
            
            # Check if it's GIMP 3.x
            version_output = result.stdout
            if "GNU Image Manipulation Program" not in version_output:
                raise RuntimeError("Invalid GIMP installation")
                
            # Extract version number
            version_line = [line for line in version_output.split('\n') if 'GNU Image Manipulation Program' in line]
            if version_line:
                version_text = version_line[0]
                if '3.' not in version_text:
                    logger.warning(f"GIMP version may not be 3.x: {version_text}")
                    
            logger.info(f"GIMP validation successful: {self.gimp_exe}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            raise RuntimeError(f"Failed to validate GIMP installation: {e}")
    
    def _create_gimp_script(self) -> str:
        """Create a temporary GIMP Python script for GEGL operations."""
        if not self._temp_dir:
            self._temp_dir = tempfile.mkdtemp(prefix="gimp_gegl_")
        
        script_content = '''
import sys
import os
import json
import gi
gi.require_version('Gimp', '3.0')
gi.require_version('Gegl', '0.4')
from gi.repository import Gimp, Gegl, Gio

def apply_gegl_operation(input_path, output_path, operation_name, parameters, opacity=1.0, blend_mode="normal"):
    """Apply a GEGL operation to an image with opacity and blend mode."""
    try:
        # Initialize GIMP
        Gimp.init()
        
        # Load the image
        image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.file_new_for_path(input_path), None)
        if not image:
            return {"error": "Failed to load image"}
        
        # Get the drawable (active layer)
        drawable = image.get_active_drawable()
        if not drawable:
            return {"error": "No active drawable found"}
        
        # Create GEGL node
        node = Gegl.Node()
        
        # Add input node
        input_node = node.new_child("gegl:load", "input")
        input_node.set_property("path", input_path)
        
        # Add operation node
        op_node = node.new_child(operation_name, "operation")
        
        # Set operation parameters
        for param_name, param_value in parameters.items():
            try:
                op_node.set_property(param_name, param_value)
            except Exception as e:
                print(f"Warning: Failed to set parameter {param_name}: {e}")
        
        # Add opacity and blend mode if not 1.0 and normal
        if opacity != 1.0 or blend_mode != "normal":
            # Create a composite node for opacity and blend mode
            composite_node = node.new_child("gegl:composite", "composite")
            composite_node.set_property("opacity", opacity)
            composite_node.set_property("composite-mode", blend_mode)
            
            # Add a passthrough node for the original
            passthrough_node = node.new_child("gegl:passthrough", "passthrough")
            
            # Connect: input -> operation -> composite <- passthrough <- input
            input_node.connect_to("output", op_node, "input")
            input_node.connect_to("output", passthrough_node, "input")
            passthrough_node.connect_to("output", composite_node, "aux")
            op_node.connect_to("output", composite_node, "input")
            
            # Add output node
            output_node = node.new_child("gegl:save", "output")
            output_node.set_property("path", output_path)
            composite_node.connect_to("output", output_node, "input")
        else:
            # Simple case: no opacity/blend mode needed
            output_node = node.new_child("gegl:save", "output")
            output_node.set_property("path", output_path)
            input_node.connect_to("output", op_node, "input")
            op_node.connect_to("output", output_node, "input")
        
        # Process the image
        node.process()
        
        # Clean up
        image.delete()
        
        return {"success": True, "output_path": output_path}
        
    except Exception as e:
        return {"error": str(e)}

def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error": "Invalid arguments"}))
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    operation_name = sys.argv[3]
    parameters_json = sys.argv[4]
    opacity = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    blend_mode = sys.argv[6] if len(sys.argv) > 6 else "normal"
    
    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON parameters"}))
        return
    
    result = apply_gegl_operation(input_path, output_path, operation_name, parameters, opacity, blend_mode)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''
        
        self._script_path = os.path.join(self._temp_dir, "gimp_gegl_script.py")
        with open(self._script_path, 'w') as f:
            f.write(script_content)
        
        return self._script_path
    
    def _validate_operation(self, operation_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate operation name and parameters against known GEGL operations.
        
        Args:
            operation_name: Name of the GEGL operation
            parameters: Dictionary of parameters for the operation
            
        Returns:
            Validated and normalized parameters
        """
        if operation_name not in self.GEGL_OPERATIONS:
            available_ops = list(self.GEGL_OPERATIONS.keys())
            raise ValueError(f"Unknown GEGL operation: {operation_name}. Available: {available_ops}")
        
        schema = self.GEGL_OPERATIONS[operation_name]
        validated_params = {}
        
        for param_name, param_schema in schema.items():
            if param_name in parameters:
                value = parameters[param_name]
                param_type = param_schema["type"]
                
                # Type validation and conversion
                try:
                    if param_type == "float":
                        value = float(value)
                    elif param_type == "int":
                        value = int(value)
                    elif param_type == "string":
                        value = str(value)
                    
                    # Range validation
                    if "min" in param_schema and value < param_schema["min"]:
                        value = param_schema["min"]
                    if "max" in param_schema and value > param_schema["max"]:
                        value = param_schema["max"]
                    
                    validated_params[param_name] = value
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid parameter {param_name}={value}: {e}. Using default.")
                    validated_params[param_name] = param_schema["default"]
            else:
                # Use default value
                validated_params[param_name] = param_schema["default"]
        
        return validated_params
    
    def apply_gegl_operation(self, input_path: str, operation_name: str, 
                           parameters: Dict[str, Any], output_path: Optional[str] = None,
                           opacity: Optional[float] = None, 
                           blend_mode: Optional[Union[str, GimpBlendMode]] = None) -> str:
        """
        Apply a GEGL operation to an image.
        
        Args:
            input_path: Path to the input image
            operation_name: Name of the GEGL operation (e.g., "gegl:brightness-contrast")
            parameters: Dictionary of parameters for the operation
            output_path: Optional output path. If None, generates one automatically
            opacity: Opacity for the operation (0.0 to 1.0). If None, uses default
            blend_mode: Blend mode for the operation. If None, uses default
            
        Returns:
            Path to the processed image
            
        Raises:
            ValueError: If operation or parameters are invalid
            RuntimeError: If GIMP processing fails
        """
        # Comprehensive validation
        is_valid, errors = validate_gegl_operation(input_path, operation_name, parameters, output_path)
        if not is_valid:
            error_msg = "GEGL operation validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        # Validate operation and parameters
        validated_params = self._validate_operation(operation_name, parameters)
        
        # Handle opacity and blend mode
        final_opacity = opacity if opacity is not None else self.default_opacity
        final_blend_mode = self._normalize_blend_mode(blend_mode) if blend_mode is not None else self.default_blend_mode
        
        # Validate opacity
        final_opacity = max(0.0, min(1.0, final_opacity))
        
        # Generate output path if not provided
        if not output_path:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_gegl{input_path_obj.suffix}")
        
        # Create GIMP script
        script_path = self._create_gimp_script()
        
        try:
            # Prepare command
            cmd = [
                self.gimp_exe,
                "--batch-interpreter", "python-fu-eval",
                "-b", f"exec(open('{script_path}').read())",
                "--batch", f"main()",
                input_path,
                output_path,
                operation_name,
                json.dumps(validated_params),
                str(final_opacity),
                final_blend_mode
            ]
            
            logger.debug(f"Running GIMP command: {' '.join(cmd)}")
            
            # Execute GIMP script
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, creationflags=NO_CONSOLE_WINDOW)
            
            if result.returncode != 0:
                raise RuntimeError(f"GIMP execution failed: {result.stderr}")
            
            # Parse result
            try:
                result_data = json.loads(result.stdout.strip())
                if "error" in result_data:
                    raise RuntimeError(f"GEGL operation failed: {result_data['error']}")
                
                if not os.path.exists(output_path):
                    raise RuntimeError("Output file was not created")
                
                logger.info(f"GEGL operation '{operation_name}' completed successfully: {output_path}")
                return output_path
                
            except json.JSONDecodeError:
                raise RuntimeError(f"Failed to parse GIMP output: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("GIMP operation timed out")
        except Exception as e:
            raise RuntimeError(f"GEGL operation failed: {e}")
    
    def get_available_operations(self) -> List[str]:
        """Get list of available GEGL operations."""
        return list(self.GEGL_OPERATIONS.keys())
    
    def get_operation_schema(self, operation_name: str) -> Dict[str, Any]:
        """
        Get the parameter schema for a specific GEGL operation.
        
        Args:
            operation_name: Name of the GEGL operation
            
        Returns:
            Dictionary containing parameter schema
        """
        if operation_name not in self.GEGL_OPERATIONS:
            raise ValueError(f"Unknown GEGL operation: {operation_name}")
        
        return self.GEGL_OPERATIONS[operation_name].copy()
    
    def get_available_blend_modes(self) -> List[str]:
        """Get list of available blend modes."""
        return [mode.value for mode in GimpBlendMode]
    
    def set_default_opacity(self, opacity: float):
        """Set the default opacity for operations."""
        self.default_opacity = max(0.0, min(1.0, opacity))
    
    def set_default_blend_mode(self, blend_mode: Union[str, GimpBlendMode]):
        """Set the default blend mode for operations."""
        self.default_blend_mode = self._normalize_blend_mode(blend_mode)
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")
        
        self._temp_dir = None
        self._script_path = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python gimp_gegl_client.py <input_image> [operation]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    operation = sys.argv[2] if len(sys.argv) > 2 else "gegl:brightness-contrast"
    
    try:
        with GimpGeglClient() as client:
            result = client.apply_gegl_operation(input_image, operation, {})
            print(f"Processed image saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
