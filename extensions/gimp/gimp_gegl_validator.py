"""
GIMP GEGL Validation Module

This module provides comprehensive validation and error handling for GIMP GEGL operations.
It includes system checks, parameter validation, and detailed error reporting.
"""

import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Any

from utils.config import config
from utils.logging_setup import get_logger

logger = get_logger("gimp_gegl_validator")


class GimpGeglValidator:
    """
    Comprehensive validator for GIMP GEGL integration.
    
    This class provides methods to validate system requirements, GIMP installation,
    operation parameters, and file formats for GEGL operations.
    """
    
    # Supported image formats for GEGL operations
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', 
        '.gif', '.ico', '.xpm', '.xbm', '.pbm', '.pgm', '.ppm'
    }
    
    # Minimum system requirements
    MIN_MEMORY_MB = 512
    MIN_DISK_SPACE_MB = 100
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_cache = {}
        self.last_validation_time = 0
        
    def validate_system_requirements(self) -> Tuple[bool, List[str]]:
        """
        Validate system requirements for GIMP GEGL operations.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check Python version
        if sys.version_info < (3, 7):
            errors.append(f"Python 3.7+ required, found {sys.version}")
        
        # Check available memory (approximate)
        try:
            import psutil  # type: ignore
            memory = psutil.virtual_memory()
            if memory.available < self.MIN_MEMORY_MB * 1024 * 1024:
                errors.append(f"Insufficient memory: {memory.available // (1024*1024)}MB available, {self.MIN_MEMORY_MB}MB required")
        except ImportError:
            logger.warning("psutil not available, skipping memory check")
        
        # Check disk space
        try:
            import shutil
            free_space = shutil.disk_usage('.').free
            if free_space < self.MIN_DISK_SPACE_MB * 1024 * 1024:
                errors.append(f"Insufficient disk space: {free_space // (1024*1024)}MB available, {self.MIN_DISK_SPACE_MB}MB required")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        return len(errors) == 0, errors
    
    def validate_gimp_installation(self) -> Tuple[bool, List[str]]:
        """
        Validate GIMP installation and accessibility.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not config.gimp_exe_loc:
            errors.append("GIMP executable not configured (gimp_exe_loc)")
            return False, errors
        
        # Check if GIMP executable exists
        gimp_path = config.gimp_exe_loc
        if not os.path.isfile(gimp_path):
            # Try to find it in PATH
            import shutil
            gimp_path = shutil.which(config.gimp_exe_loc)
            if not gimp_path:
                errors.append(f"GIMP executable not found: {config.gimp_exe_loc}")
                return False, errors
        
        # Test GIMP execution
        try:
            result = subprocess.run(
                [config.gimp_version_probe_executable(gimp_path), "--version"],
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                errors.append(f"GIMP execution failed: {result.stderr}")
                return False, errors
            
            # Check if it's GIMP 3.x
            version_output = result.stdout
            if "GNU Image Manipulation Program" not in version_output:
                errors.append("Invalid GIMP installation")
                return False, errors
            
            # Check version
            if "3." not in version_output and "2.99" not in version_output:
                errors.append("GIMP 3.x required for GEGL operations")
                return False, errors
                
        except subprocess.TimeoutExpired:
            errors.append("GIMP version check timed out")
            return False, errors
        except (FileNotFoundError, OSError) as e:
            errors.append(f"Failed to execute GIMP: {e}")
            return False, errors
        
        return True, errors
    
    def validate_pygobject_installation(self) -> Tuple[bool, List[str]]:
        """
        Validate PyGObject installation for GIMP Python bindings.
        
        Note: This validation is optional when using subprocess mode.
        PyGObject is only required when using direct Python bindings.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            import gi  # type: ignore
            gi.require_version('Gimp', '3.0')
            gi.require_version('Gegl', '0.4')
            from gi.repository import Gimp, Gegl  # type: ignore
        except ImportError as e:
            # PyGObject not available - this is OK for subprocess mode
            logger.warning(f"PyGObject not available in system Python: {e}")
            logger.info("Using subprocess mode - PyGObject not required in system Python")
            return True, []  # Don't treat this as an error
        except ValueError as e:
            errors.append(f"GIMP/GEGL bindings not available: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error importing GIMP bindings: {e}")
            return False, errors
        
        return True, errors
    
    def validate_image_file(self, image_path: str) -> Tuple[bool, List[str]]:
        """
        Validate image file for GEGL operations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not image_path:
            errors.append("Image path is required")
            return False, errors
        
        if not os.path.exists(image_path):
            errors.append(f"Image file not found: {image_path}")
            return False, errors
        
        if not os.path.isfile(image_path):
            errors.append(f"Path is not a file: {image_path}")
            return False, errors
        
        # Check file extension
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            errors.append(f"Unsupported image format: {file_ext}. Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}")
            return False, errors
        
        # Check file size
        try:
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                errors.append("Image file is empty")
                return False, errors
            
            # Warn for very large files
            if file_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"Large image file: {file_size // (1024*1024)}MB")
                
        except OSError as e:
            errors.append(f"Cannot access image file: {e}")
            return False, errors
        
        return True, errors
    
    def validate_operation_parameters(self, operation_name: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters for a GEGL operation.
        
        Args:
            operation_name: Name of the GEGL operation
            parameters: Dictionary of parameters
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not operation_name:
            errors.append("Operation name is required")
            return False, errors
        
        if not operation_name.startswith("gegl:"):
            errors.append(f"Invalid operation name format: {operation_name}. Must start with 'gegl:'")
            return False, errors
        
        # Import operation schemas
        try:
            from extensions.gimp.gimp_gegl_client import GimpGeglClient
            client = GimpGeglClient()
            schema = client.get_operation_schema(operation_name)
        except Exception as e:
            errors.append(f"Cannot get operation schema: {e}")
            return False, errors
        
        if not schema:
            errors.append(f"Unknown operation: {operation_name}")
            return False, errors
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            if param_name not in schema:
                errors.append(f"Unknown parameter: {param_name}")
                continue
            
            param_schema = schema[param_name]
            param_type = param_schema.get("type", "string")
            
            # Type validation
            try:
                if param_type == "float":
                    float_value = float(param_value)
                    # Range validation
                    if "min" in param_schema and float_value < param_schema["min"]:
                        errors.append(f"Parameter {param_name} below minimum: {float_value} < {param_schema['min']}")
                    if "max" in param_schema and float_value > param_schema["max"]:
                        errors.append(f"Parameter {param_name} above maximum: {float_value} > {param_schema['max']}")
                elif param_type == "int":
                    int_value = int(param_value)
                    # Range validation
                    if "min" in param_schema and int_value < param_schema["min"]:
                        errors.append(f"Parameter {param_name} below minimum: {int_value} < {param_schema['min']}")
                    if "max" in param_schema and int_value > param_schema["max"]:
                        errors.append(f"Parameter {param_name} above maximum: {int_value} > {param_schema['max']}")
                elif param_type == "string":
                    if not isinstance(param_value, str):
                        errors.append(f"Parameter {param_name} must be string, got {type(param_value)}")
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid parameter {param_name}: {e}")
        
        return len(errors) == 0, errors
    
    def validate_output_path(self, output_path: str, create_dirs: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate output path for GEGL operations.
        
        Args:
            output_path: Path for the output file
            create_dirs: Whether to create parent directories if they don't exist
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not output_path:
            errors.append("Output path is required")
            return False, errors
        
        output_path_obj = Path(output_path)
        
        # Check parent directory
        parent_dir = output_path_obj.parent
        if not parent_dir.exists():
            if create_dirs:
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create output directory: {e}")
                    return False, errors
            else:
                errors.append(f"Output directory does not exist: {parent_dir}")
                return False, errors
        
        # Check write permissions
        if not os.access(parent_dir, os.W_OK):
            errors.append(f"No write permission for output directory: {parent_dir}")
            return False, errors
        
        # Check file extension
        file_ext = output_path_obj.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            errors.append(f"Unsupported output format: {file_ext}. Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}")
            return False, errors
        
        return True, errors
    
    def validate_complete_setup(self) -> Tuple[bool, List[str]]:
        """
        Perform complete validation of GIMP GEGL setup.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        all_errors = []
        
        # System requirements
        sys_valid, sys_errors = self.validate_system_requirements()
        all_errors.extend(sys_errors)
        
        # GIMP installation
        gimp_valid, gimp_errors = self.validate_gimp_installation()
        all_errors.extend(gimp_errors)
        
        # PyGObject installation
        pygobject_valid, pygobject_errors = self.validate_pygobject_installation()
        all_errors.extend(pygobject_errors)
        
        # Configuration
        if not config.gimp_gegl_enabled:
            all_errors.append("GIMP GEGL operations are disabled in configuration")
        
        is_valid = sys_valid and gimp_valid and pygobject_valid and config.gimp_gegl_enabled
        
        return is_valid, all_errors
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get detailed system information for debugging.
        
        Returns:
            Dictionary containing system information
        """
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "gimp_configured": bool(config.gimp_exe_loc),
            "gimp_path": config.gimp_exe_loc,
            "gegl_enabled": config.gimp_gegl_enabled,
            "gegl_timeout": config.gimp_gegl_timeout,
        }
        
        # Try to get GIMP version
        if config.gimp_exe_loc:
            try:
                result = subprocess.run(
                    [config.gimp_version_probe_executable(config.gimp_exe_loc), "--version"],
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    info["gimp_version"] = result.stdout.strip()
            except Exception:
                info["gimp_version"] = "Unknown"
        
        # Check PyGObject
        try:
            import gi  # type: ignore
            info["pygobject_available"] = True
            try:
                gi.require_version('Gimp', '3.0')
                info["gimp_bindings"] = True
            except ValueError:
                info["gimp_bindings"] = False
        except ImportError:
            info["pygobject_available"] = False
            info["gimp_bindings"] = False
        
        return info


def validate_gegl_operation(image_path: str, operation_name: str, parameters: Dict[str, Any], 
                          output_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a complete GEGL operation.
    
    Args:
        image_path: Path to input image
        operation_name: Name of GEGL operation
        parameters: Operation parameters
        output_path: Optional output path
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = GimpGeglValidator()
    all_errors = []
    
    # Validate setup
    setup_valid, setup_errors = validator.validate_complete_setup()
    all_errors.extend(setup_errors)
    
    # Validate input image
    image_valid, image_errors = validator.validate_image_file(image_path)
    all_errors.extend(image_errors)
    
    # Validate operation parameters
    param_valid, param_errors = validator.validate_operation_parameters(operation_name, parameters)
    all_errors.extend(param_errors)
    
    # Validate output path if provided
    if output_path:
        output_valid, output_errors = validator.validate_output_path(output_path)
        all_errors.extend(output_errors)
    
    is_valid = setup_valid and image_valid and param_valid and (not output_path or output_valid)
    
    return is_valid, all_errors


def get_validation_report() -> str:
    """
    Generate a comprehensive validation report.
    
    Returns:
        Formatted validation report string
    """
    validator = GimpGeglValidator()
    
    report = ["GIMP GEGL Integration Validation Report", "=" * 50, ""]
    
    # System requirements
    sys_valid, sys_errors = validator.validate_system_requirements()
    report.append(f"System Requirements: {'✅ PASS' if sys_valid else '❌ FAIL'}")
    for error in sys_errors:
        report.append(f"  - {error}")
    
    # GIMP installation
    gimp_valid, gimp_errors = validator.validate_gimp_installation()
    report.append(f"\nGIMP Installation: {'✅ PASS' if gimp_valid else '❌ FAIL'}")
    for error in gimp_errors:
        report.append(f"  - {error}")
    
    # PyGObject installation (optional for subprocess mode)
    pygobject_valid, pygobject_errors = validator.validate_pygobject_installation()
    if pygobject_errors:
        report.append(f"\nPyGObject Installation: {'✅ PASS' if pygobject_valid else '❌ FAIL'}")
        for error in pygobject_errors:
            report.append(f"  - {error}")
    else:
        report.append(f"\nPyGObject Installation: {'✅ AVAILABLE' if pygobject_valid else '⚠️  NOT AVAILABLE (using subprocess mode)'}")
    
    # Configuration
    config_status = "✅ ENABLED" if config.gimp_gegl_enabled else "❌ DISABLED"
    report.append(f"\nConfiguration: {config_status}")
    report.append(f"  - GIMP Path: {config.gimp_exe_loc or 'Not configured'}")
    report.append(f"  - Timeout: {config.gimp_gegl_timeout}s")
    
    # System info
    info = validator.get_system_info()
    report.append(f"\nSystem Information:")
    for key, value in info.items():
        report.append(f"  - {key}: {value}")
    
    # Overall status
    overall_valid = sys_valid and gimp_valid and pygobject_valid and config.gimp_gegl_enabled
    report.append(f"\nOverall Status: {'✅ READY' if overall_valid else '❌ NOT READY'}")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Generate and print validation report
    print(get_validation_report())
