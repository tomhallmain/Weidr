import os
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from utils.config import config
from utils.logging_setup import get_logger

from image.classifier_prediction_cache import classifier_prediction_cache, model_signature
from image.classifier_utils import (
    derive_neutral_categories_from_positive_groups,
    ensure_probabilities,
    format_prediction_line,
    from_pretrained_checked,
    infer_processor_input_shape,
    logits_to_probabilities,
    map_scores_to_categories,
    normalize_label,
    pick_split_positive,
    resolve_torch_device,
    top_category,
)
from image.image_classifier_model_config import ImageClassifierModelConfig

logger = get_logger("image_classifier")

# TODO: expose as config or model_kwarg (typical range ~0.05–0.10). Split-positive
# assignment only applies when a group's combined mass exceeds aggregate neutral mass
# by more than this margin (probability points on the model's output scale).
_SPLIT_GROUP_OVER_NEUTRAL_MARGIN = 0.05


class BackendType(Enum):
    """Backend type for image classifiers"""
    PYTORCH = "pytorch"
    HDF5 = "hdf5"
    ONNX = "onnx"
    TFLITE = "tflite"
    DETECTION = "detection"
    OTHER = "other"

    @staticmethod
    def config_values() -> List[str]:
        """Values a model config's ``backend`` can be set to: "auto" plus every loadable backend."""
        return ["auto"] + [b.value for b in BackendType if b is not BackendType.OTHER]

    @staticmethod
    def from_model_location(model_location: str) -> Optional["BackendType"]:
        """Backend implied by a model file's extension ("auto"), or None when there is none."""
        location = str(model_location).lower()
        if location.endswith('.h5'):
            return BackendType.HDF5
        if location.endswith(('.pth', '.pt', '.safetensors', '.bin')):
            return BackendType.PYTORCH
        if location.endswith('.onnx'):
            return BackendType.ONNX
        if location.endswith('.tflite'):
            return BackendType.TFLITE
        return None

    @staticmethod
    def parse(backend: Union["BackendType", str, None]) -> Optional["BackendType"]:
        """Parse backend input into a BackendType enum value."""
        if isinstance(backend, BackendType):
            return backend

        backend_str = str(backend).lower().strip()
        if backend_str == "auto":
            return None  # Will be determined from file extension
        if backend_str in ("tensorflow", "hdf5", "h5"):
            return BackendType.HDF5
        if backend_str in ("pytorch", "torch"):
            return BackendType.PYTORCH
        if backend_str == "onnx":
            return BackendType.ONNX
        if backend_str in ("tflite", "litert", "tf_lite"):
            return BackendType.TFLITE
        if backend_str in ("detection", "object_detection", "object-detection"):
            return BackendType.DETECTION
        return BackendType.OTHER


def import_model_architecture(import_path: str):
    """
    Import a model architecture class from a string path.
    
    Args:
        import_path: String in format "module:ClassName" or "module.ClassName"
    
    Returns:
        The model class
    """
    # Handle both colon and dot notation
    if ':' in import_path:
        module_path, class_name = import_path.split(':', 1)
    else:
        # Try to split by last dot
        parts = import_path.rsplit('.', 1)
        if len(parts) == 2:
            module_path, class_name = parts
        else:
            raise ValueError(f"Invalid import path format: {import_path}. "
                           f"Use 'module:ClassName' or 'module.ClassName'")
    
    try:
        # Try to import directly
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        logger.info(f"Successfully imported {class_name} from {module_path}")
        return model_class
    except ImportError as e:
        # If direct import fails, try with importlib
        try:
            import importlib
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            logger.info(f"Successfully imported {class_name} from {module_path} using importlib")
            return model_class
        except (ImportError, AttributeError) as e2:
            raise ImportError(f"Failed to import {class_name} from {module_path}: {e2}")


class BaseImageClassifier(ABC):
    """Abstract base class for image classifiers"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_loaded = False
        self.model = None
        self.input_shape = None
        self.model_architecture = None

    @classmethod
    def _find_model_class(cls, module) -> type:
        """Find the model class in a module"""
        raise NotImplementedError(f"Subclass {cls.__name__} must implement class method find_model_class")

    @classmethod
    def _import_from_file(cls, py_file_path, class_name: Optional[str] = None) -> type:
        """Import a model class from a Python file
        
        Args:
            py_file_path: Path to the Python file
            class_name: Name of the class to import, if None, will find the model class in the module
        
        Returns:
            The model class
        """
        import hashlib
        import importlib.util
        import sys

        # Get the directory and module name
        module_dir = os.path.dirname(py_file_path)
        base_name = os.path.splitext(os.path.basename(py_file_path))[0]
        # Third-party architecture files commonly reuse generic filenames (e.g.
        # multiple HF repos each shipping their own "model_architecture.py").
        # Suffix with a hash of the absolute path so two distinct files never
        # share a module identity, regardless of load order.
        path_hash = hashlib.sha1(os.path.abspath(py_file_path).encode("utf-8")).hexdigest()[:12]
        module_name = f"{base_name}_{path_hash}"

        # Add the directory to sys.path if not already there
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, py_file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            if class_name:
                return getattr(module, class_name)
            else:
                return cls._find_model_class(module)
            
        except Exception as e:
            raise ImportError(f"Failed to import from file {py_file_path}: {e}")

    @classmethod
    def load_model_architecture(cls,
                                architecture_module_name: str,
                                architecture_class_path: Optional[str] = None,
                                architecture_location: Optional[str] = None,
                                model_dir: Optional[str] = None) -> type:
        """
        Load a model architecture from various specification types.
        
        Args:
            architecture_module_name: Name of the module containing the model architecture
            architecture_class_path: Path to the class in the module, e.g. "module.ClassName"
            architecture_location: Path to the model architecture file
            model_dir: Path to model directory
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement class method load_model_architecture")

    @abstractmethod
    def load_model(self) -> bool:
        """Load the model from file"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def predict(self, preprocessed_image: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run prediction on preprocessed image"""
        pass
    
    def predict_image(self, image_path: str) -> np.ndarray:
        """Complete prediction pipeline"""
        preprocessed_img = self.preprocess_image(image_path)
        return self.predict(preprocessed_img)
    
    def _get_input_shape(self) -> Tuple[int, int]:
        """Get input dimensions (width, height) for PIL resize"""
        if self.input_shape is not None:
            return self.input_shape
            
        # Default implementation can be overridden
        raise NotImplementedError("Subclasses must implement _get_input_shape or set self.input_shape")

    def _get_model_base(self):
        """Get an instance of the base model architecture class"""
        import inspect

        if self.model_architecture is None:
            raise ValueError("Model architecture not set")
        if isinstance(self.model_architecture, type):
            model_base = self.model_architecture()
        elif isinstance(self.model_architecture, str):
            # Shouldn't happen, it should already be imported by this point, but just in case...
            model_base = import_model_architecture(self.model_architecture)
        elif inspect.isfunction(self.model_architecture) or inspect.ismethod(self.model_architecture):
            # A zero-arg factory function (e.g. a third-party `get_model()`-style
            # helper shipped as-is, rather than an nn.Module subclass) — call it
            # to build the model.
            model_base = self.model_architecture()
        else:
            print(f"Assuming runnable base model architecture type: {type(self.model_architecture)}")
            model_base = self.model_architecture
        if not hasattr(model_base, 'load_state_dict'):
            try:
                class_name = model_base.__class__.__name__
            except:
                class_name = "<unknown class>"
            raise ValueError(f"Model architecture base {class_name} of type {type(model_base)} does not support load_state_dict")
        return model_base


class H5ImageClassifier(BaseImageClassifier):
    """TensorFlow/Keras H5 model classifier"""
    
    def __init__(self, model_path: str, custom_objects: Optional[Dict] = None):
        """Image classifier for H5 models with version-independent loading
        
        Args:
            model_path: Path to .h5 model file
            custom_objects: Dictionary of custom layer classes {name: class}
        """
        super().__init__(model_path)
        self.custom_objects = custom_objects or {}
        self.load_errors = []
        self.load_model()
    
    def _register_common_layers(self, custom_objects: Dict):
        """Auto-register common custom layers and handle compatibility issues"""
        try:
            import tensorflow_hub as hub
            custom_objects['KerasLayer'] = hub.KerasLayer
        except ImportError:
            pass

    def _load_model_tensorflow_keras(self, model_path: str, custom_objects: Dict):
        """Attempt loading with TensorFlow's built-in Keras"""
        try:
            from tensorflow.keras.models import load_model as tf_load_model
            from tensorflow.keras.utils import custom_object_scope
            with custom_object_scope(custom_objects):
                model = tf_load_model(model_path)
                return model
        except Exception as e:
            self.load_errors.append(f"TensorFlow Keras load failed: {str(e)[:300]}")
            return None

    def _load_model_tf_keras(self, model_path: str, custom_objects: Dict):
        """Fallback to tf_keras package"""
        try:
            from tf_keras.models import load_model
            model = load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            self.load_errors.append(f"tf_keras load failed: {str(e)[:300]}")
            logger.info(f"[tf_keras] Load failed: {str(e)[:200]}")
            return None

    def _load_model_keras(self, model_path: str, custom_objects: Dict):
        """Last-resort standalone Keras attempt"""
        try:
            from keras.models import load_model as keras_load_model
            from keras.utils.custom_object_scope import custom_object_scope
            with custom_object_scope(custom_objects):
                model = keras_load_model(model_path)
                return model
        except Exception as e:
            self.load_errors.append(f"Standalone Keras load failed: {str(e)[:300]}")
            logger.info(f"[keras] Load failed: {str(e)[:200]}")
            return None

    def load_model(self) -> bool:
        """Load model with fallback strategies"""
        self._register_common_layers(self.custom_objects)
        
        loaders = [
            self._load_model_tensorflow_keras,
            self._load_model_tf_keras,
            self._load_model_keras
        ]
        
        for loader in loaders:
            self.model = loader(self.model_path, self.custom_objects)
            if self.model is not None:
                self.is_loaded = True
                self.input_shape = self._get_input_shape()
                self._verify_model_compatibility()
                return True

        logger.error(f"Failed to load model at {self.model_path}")
        for error in self.load_errors:
            logger.error(error)
        return False

    def _get_input_shape(self) -> Tuple[int, int]:
        """Get input dimensions with channels-last/channels-first awareness"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
            
        # Handle different data formats
        if len(input_shape) == 4:  # Batch dimension included
            _unused1, height, width, _unused2 = input_shape
        else:
            height, width, _unused2 = input_shape
        return (width, height)  # PIL uses (width, height) for resize

    def _verify_model_compatibility(self):
        """Check for common compatibility issues"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Loaded model doesn't support prediction interface")
        if len(self.input_shape) != 2:
            raise ValueError("Model expects unexpected input dimensions")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image with safety checks"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize(self.input_shape)
                img_array = np.array(img, dtype=np.float32) / 255.0
                return np.expand_dims(img_array, axis=0)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def predict(self, preprocessed_image: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run prediction with validation"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        if preprocessed_image.shape[1:3] != self.input_shape[::-1]:
            raise ValueError("Input image dimensions don't match model requirements")
        return self.model.predict(preprocessed_image, batch_size=batch_size)


class PyTorchImageClassifier(BaseImageClassifier):
    """PyTorch model classifier"""
    
    def __init__(self, model_path: str, 
                 device: str = 'auto',
                 normalize_mean: List[float] = None,
                 normalize_std: List[float] = None,
                 input_shape: Optional[Tuple[int, int]] = None,
                 model_architecture: Optional[Any] = None,
                 architecture_module_name: str = None,
                 architecture_class_path: str = None,
                 architecture_location: str = None,
                 weights_only: bool = False,  # Changed to False by default for converted models
                 safe_globals: List = None,
                 load_full_model: bool = True,  # New parameter to load full model directly
                 use_transformers_auto_model: bool = False,
                 hf_pretrained_path: Optional[str] = None):
        """PyTorch model classifier
        
        Args:
            model_path: Path to .pth/.pt model file
            device: 'auto', 'cuda', or 'cpu'
            normalize_mean: Normalization mean values (default: ImageNet)
            normalize_std: Normalization std values (default: ImageNet)
            input_shape: Optional (width, height) if not inferrable from model
            architecture_module_name: Name of the module containing the model architecture
            architecture_class_path: Path to the class in the module, e.g. "module.ClassName"
            architecture_location: Path to the model architecture file
            weights_only: Use safe loading (True) or allow arbitrary code execution (False)
            safe_globals: List of additional safe globals for weights_only=True
            load_full_model: If True, load full model directly (not state_dict)
        """
        super().__init__(model_path)
        self.device = self._get_device(device)
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self._input_shape_override = input_shape
        self.weights_only = weights_only
        self.safe_globals = safe_globals or []
        self.load_full_model = load_full_model
        self.use_transformers_auto_model = bool(use_transformers_auto_model)
        self.hf_pretrained_path = hf_pretrained_path
        self.transform = None
        self.processor = None
        
        # Handle model_architecture parameter - if using ImageClassifierWrapper, should already be imported and return early
        if model_architecture is not None:
            self.model_architecture = model_architecture
        elif architecture_module_name is not None:
            model_dir = os.path.dirname(model_path)
            self.model_architecture = PyTorchImageClassifier.load_model_architecture(
                architecture_module_name, architecture_class_path=architecture_class_path,
                architecture_location=architecture_location, model_dir=model_dir)
        
        self.load_model()

    @classmethod
    def _find_model_class(cls, module) -> type:
        # Find the model class (look for classes that are subclasses of nn.Module)
        import torch.nn as nn
        model_classes = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                model_classes.append(attr)
        
        if not model_classes:
            raise ValueError(f"No PyTorch model classes found in {module.__file__}")
        elif len(model_classes) > 1:
            logger.warning(f"Multiple model classes found in {module.__file__}, using {model_classes[0]}")
        return model_classes[0]

    @classmethod
    def load_model_architecture(cls, architecture_module_name: str,
                                architecture_class_path: Optional[str] = None,
                                architecture_location: Optional[str] = None,
                                model_dir: Optional[str] = None) -> type:
        """
        Load a model architecture from various specification types.
        
        Args:
            architecture_module_name: Name of the module containing the model architecture
            architecture_class_path: Path to the class in the module, e.g. "module.ClassName"
            architecture_location: Path to model architecture file
            model_dir: Path to model directory
        
        Returns:
            The model class
        """
        if architecture_module_name is None:
            return None
        
        # First, check if it's a filesystem path (absolute or relative)
        possible_paths = []

        if architecture_location:
            if os.path.isfile(architecture_location):
                possible_paths.append(architecture_location)
            
            architecture_file_path = os.path.join(architecture_location, architecture_module_name + '.py')
            if os.path.isfile(architecture_file_path):
                possible_paths.append(architecture_file_path)

            architecture_init_file_path = os.path.join(architecture_location, '__init__.py')
            if os.path.isfile(architecture_init_file_path):
                possible_paths.append(architecture_init_file_path)
        
        # It may be a file in the model directory
        if model_dir and os.path.isdir(model_dir):
            possible_paths.append(os.path.join(model_dir, architecture_module_name))
        
        # Try each possible path
        for path in possible_paths:
            # Check with and without .py extension
            if not path.endswith('.py'):
                path = path + '.py'
            if os.path.isfile(path):
                try:
                    if config.debug2:
                        print(f"Importing model architecture from {path}")
                    return cls._import_from_file(path, architecture_class_path)
                except ImportError:
                    continue
        
        # If no file found, assume it's an import string
        try:
            return import_model_architecture(f"{architecture_module_name}.{architecture_class_path}")
        except Exception as e:
            message = f"No valid location found for model architecture: {e}\n"
            message += f"Architecture module name: {architecture_module_name}\n"
            message += f"Architecture class path: {architecture_class_path}\n"
            message += f"Architecture location: {architecture_location}\n"
            message += f"Model directory: {model_dir}\n"
            logger.error(message)
            raise ValueError(f"No valid location found for model architecture: {e}")

    def _get_device(self, device: str):
        """Determine torch device"""
        return resolve_torch_device(device)

    def _log_input_shape_override_if_any(self) -> None:
        """Log once when the user supplied ``input_shape`` (applies to all load paths)."""
        if self._input_shape_override is None:
            return
        if self.use_transformers_auto_model:
            logger.info(
                "Using user-specified input_shape %s (HF processor still controls preprocessing tensors).",
                self.input_shape,
            )
        else:
            logger.info(
                "Using user-specified input_shape %s for torchvision resize and preprocessing.",
                self.input_shape,
            )

    def load_model(self) -> bool:
        """Load PyTorch model"""
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms
        except ImportError:
            logger.error("PyTorch or torchvision not installed. Install with: pip install torch torchvision")
            return False

        if self.use_transformers_auto_model:
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
            except ImportError:
                logger.error("transformers not installed. Install with: pip install transformers")
                return False

            model_root = self.hf_pretrained_path
            if not model_root:
                model_root = os.path.dirname(self.model_path) if os.path.isfile(self.model_path) else self.model_path
            if not model_root or not os.path.exists(model_root):
                logger.error(f"Invalid HF pretrained path for transformers auto model: {model_root}")
                return False

            try:
                self.processor = AutoImageProcessor.from_pretrained(model_root)
                self.model = from_pretrained_checked(
                    AutoModelForImageClassification, model_root, model_root, log=logger).to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.input_shape = (
                    self._input_shape_override or self._infer_transformers_input_shape()
                )
                self._log_input_shape_override_if_any()
                logger.info(f"Transformers auto image classifier loaded from: {model_root}")
                return True
            except Exception as e:
                logger.error(f"Failed to load transformers auto model from {model_root}: {e}")
                return False

        # Safetensors handling
        if self.model_path.lower().endswith('.safetensors'):
            logger.info(f"Loading safetensors file: {self.model_path}")
            try:
                import safetensors.torch
                
                # Load the state_dict from safetensors
                state_dict = safetensors.torch.load_file(self.model_path, device='cpu')
                logger.info(f"Loaded {len(state_dict)} tensors from safetensors")
                
                # We need a model architecture to load the state_dict into
                if self.model_architecture is None:
                    logger.error("For .safetensors file, model_architecture must be provided.")
                    logger.error("Please provide model_architecture parameter when loading .safetensors files.")
                    return False
                
                # Instantiate the model architecture
                self.model = self._get_model_base()
                
                # Load state dict
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                
            except ImportError:
                logger.error("safetensors library not installed. Install with: pip install safetensors")
                return False
            except Exception as e:
                logger.error(f"Failed to load safetensors: {str(e)}")
                return False
            
            # Set model to evaluation mode
            self.model.eval()
            self.is_loaded = True
            
            self.input_shape = self._input_shape_override or self._infer_input_shape()
            self._log_input_shape_override_if_any()
            self._setup_transforms()
            
            logger.info(f"Safetensors model loaded successfully on device: {self.device}")
            return True
        
        # PyTorch model handling (.pth, .pt files)
        try:
            # Add common safe globals for converted models
            # TODO: Maybe this needs to be moved to external model architecture
            if self.weights_only and not self.safe_globals:
                # Add torch.nn.modules.container.Sequential to safe globals
                self.safe_globals = [torch.nn.modules.container.Sequential]
            
            # Try to load with safe weights_only first
            try:
                if self.weights_only and self.safe_globals:
                    # Add safe globals if provided
                    for safe_global in self.safe_globals:
                        torch.serialization.add_safe_globals([safe_global])
                
                # Attempt to load the model
                if self.weights_only:
                    logger.info("Attempting safe loading (weights_only=True)...")
                    loaded_data = torch.load(self.model_path, map_location=self.device, weights_only=True)
                else:
                    logger.warning("Using unsafe loading (weights_only=False). Only use with trusted models!")
                    loaded_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    
            except (RuntimeError, ImportError) as e:
                if "weights_only" in str(e) and self.weights_only:
                    logger.warning(f"Safe loading failed: {e}")
                    logger.warning("Falling back to unsafe loading for compatibility...")
                    loaded_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
                else:
                    raise
            
            # Handle different save formats
            if isinstance(loaded_data, dict):
                if 'state_dict' in loaded_data:
                    # Handle models saved with state_dict in a dict
                    state_dict = loaded_data['state_dict']
                else:
                    # Assume it's a state_dict directly
                    state_dict = loaded_data
                
                # If we have a state_dict but no architecture, we need it
                if self.model_architecture is None:
                    logger.warning("Model file contains state_dict but no model_architecture was provided.")
                    logger.warning("Trying to infer if this is a full model...")
                    # Check if it might be a full model saved in an unexpected way
                    if hasattr(loaded_data, 'eval') and hasattr(loaded_data, 'parameters'):
                        self.model = loaded_data.to(self.device)
                    else:
                        logger.error("Cannot load state_dict without model_architecture.")
                        logger.error("Please provide model_architecture parameter when loading state_dict files.")
                        return False
                else:
                    # Load state dict into model architecture
                    self.model = self._get_model_base()
                    
                    # Strip 'module.' prefix if saved from DataParallel
                    from collections import OrderedDict
                    if all(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
                    
                    # Load state dict
                    self.model.load_state_dict(state_dict)
                    self.model = self.model.to(self.device)
                    
            elif hasattr(loaded_data, 'eval') and hasattr(loaded_data, 'parameters'):
                # Model is already a nn.Module (full model saved directly)
                self.model = loaded_data.to(self.device)
            else:
                logger.error(f"Unsupported PyTorch model format in {self.model_path}")
                return False
            
            # Set model to evaluation mode
            self.model.eval()
            self.is_loaded = True
            
            self.input_shape = self._input_shape_override or self._infer_input_shape()
            self._log_input_shape_override_if_any()
            self._setup_transforms()
                
            logger.info(f"PyTorch model loaded successfully on device: {self.device}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _setup_transforms(self):
        """Setup image transformations"""
        try:
            from torchvision import transforms
            
            if self.input_shape is None:
                # Default to 224x224 if not set
                self.input_shape = (224, 224)
                
            self.transform = transforms.Compose([
                transforms.Resize(self.input_shape[::-1]),  # (height, width) for torch
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        except ImportError:
            logger.error("torchvision not available for transforms")

    def _infer_transformers_input_shape(self) -> Tuple[int, int]:
        """Infer input shape from HF AutoImageProcessor metadata."""
        return infer_processor_input_shape(self.processor)
    
    def _infer_input_shape(self) -> Tuple[int, int]:
        """Try to infer input shape from model"""
        try:
            import torch
            
            # Check if model has expected_input_size attribute
            if hasattr(self.model, 'expected_input_size'):
                size = self.model.expected_input_size
                if isinstance(size, (tuple, list)) and len(size) >= 2:
                    return (size[1], size[0])  # Convert to (width, height)
            
            # Check if model has input_size attribute
            if hasattr(self.model, 'input_size'):
                size = self.model.input_size
                if isinstance(size, (tuple, list)) and len(size) >= 2:
                    return (size[1], size[0])  # Convert to (width, height)
                elif isinstance(size, int):
                    return (size, size)
            
            # For ResNet models, input is typically 224x224
            model_str = str(self.model).lower()
            if 'resnet' in model_str or 'fastai' in model_str:
                return (224, 224)
            
            # Try to get from first conv layer
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    # This is a heuristic - actual input might be different
                    logger.info("Inferring input size from Conv2d layer (may not be accurate)")
                    return (224, 224)  # Common default
            
        except Exception as e:
            logger.warning(f"Could not infer input shape: {e}")
        
        # Default fallback
        logger.warning("Could not infer input shape, using default (224, 224)")
        return (224, 224)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for PyTorch model"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                if self.use_transformers_auto_model:
                    if self.processor is None:
                        raise ValueError("Transformers processor is not initialized")
                    inputs = self.processor(images=img, return_tensors="pt")
                    import torch
                    return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                tensor = self.transform(img)
                # Add batch dimension
                tensor = tensor.unsqueeze(0).to(self.device)
                return tensor
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")
    
    def predict(self, preprocessed_image: Any, batch_size: int = 32) -> np.ndarray:
        """Run prediction with PyTorch model"""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            import torch
            
            with torch.no_grad():
                if self.use_transformers_auto_model:
                    if not isinstance(preprocessed_image, dict):
                        raise ValueError("Expected dict model inputs for transformers auto model")
                    output = self.model(**preprocessed_image)
                    logits = output.logits if hasattr(output, "logits") else output[0]
                    return logits_to_probabilities(logits.float().cpu().numpy())
                output = self.model(preprocessed_image)
                return ensure_probabilities(output.float().cpu().numpy())
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


class ONNXImageClassifier(BaseImageClassifier):
    """ONNX Runtime model classifier.

    Unlike the PyTorch backend, an ONNX graph is self-contained -- no
    architecture_module_name/architecture_class_path is needed to reconstruct
    it from raw weights, since the computation graph is embedded in the file.
    """

    def __init__(self, model_path: str,
                 device: str = 'auto',
                 normalize_mean: Optional[List[float]] = None,
                 normalize_std: Optional[List[float]] = None,
                 input_shape: Optional[Tuple[int, int]] = None,
                 rescale: bool = True,
                 channels_first: bool = True):
        """ONNX model classifier

        Args:
            model_path: Path to .onnx model file
            device: 'auto', 'cuda', or 'cpu' -- selects the ONNX Runtime execution provider
            normalize_mean: Normalization mean values (default: ImageNet)
            normalize_std: Normalization std values (default: ImageNet)
            input_shape: Optional (width, height) if not inferrable from the model's
                declared input tensor shape
            rescale: Divide pixel values by 255 before mean/std normalization (default
                True). Set False for models expecting raw 0-255 input (rare; some
                older Caffe-lineage exports also expect BGR channel order and
                per-channel mean subtraction only -- not handled here, see USAGE.md)
            channels_first: Whether the model expects NCHW input (default True,
                matching most PyTorch-exported ONNX models). Set False for NHWC-exported
                (e.g. TensorFlow-lineage) models.
        """
        super().__init__(model_path)
        self.device = device
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self._input_shape_override = input_shape
        self.rescale = rescale
        self.channels_first = channels_first
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()

    def _get_providers(self, ort) -> List[str]:
        """Select ONNX Runtime execution providers based on ``device``."""
        available = ort.get_available_providers()
        if self.device == 'cpu':
            return ["CPUExecutionProvider"]
        if self.device == 'cuda':
            if "CUDAExecutionProvider" not in available:
                logger.warning("CUDAExecutionProvider not available, falling back to CPU")
                return ["CPUExecutionProvider"]
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # auto
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _infer_input_shape_from_dims(shape: list, channels_first: bool) -> Optional[Tuple[int, int]]:
        """Infer (width, height) from a 4-D [N, C, H, W] or [N, H, W, C] shape.
        Dynamic/symbolic dims (non-positive-int, e.g. a string batch-size axis)
        are treated as unresolvable. Returns None when the shape can't be read."""
        numeric = [d if isinstance(d, int) and d > 0 else None for d in shape]
        if len(numeric) != 4:
            return None
        if channels_first:
            _batch, _channels, height, width = numeric
        else:
            _batch, height, width, _channels = numeric
        if height is None or width is None:
            return None
        return (width, height)

    def load_model(self) -> bool:
        """Load ONNX model via onnxruntime.InferenceSession"""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error(
                "onnxruntime not installed. Install with: pip install onnxruntime "
                "(or onnxruntime-gpu for CUDA)"
            )
            return False

        try:
            providers = self._get_providers(ort)
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.output_name = self.session.get_outputs()[0].name

            inferred = self._infer_input_shape_from_dims(list(input_meta.shape), self.channels_first)
            self.input_shape = self._input_shape_override or inferred
            if self.input_shape is None:
                logger.warning(
                    f"Could not infer input shape from ONNX model metadata {input_meta.shape}, "
                    "using default (224, 224)"
                )
                self.input_shape = (224, 224)

            self.is_loaded = True
            logger.info(f"ONNX model loaded successfully with providers: {self.session.get_providers()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            return False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for ONNX Runtime input"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize(self.input_shape)
                img_array = np.array(img, dtype=np.float32)
                if self.rescale:
                    img_array = img_array / 255.0
                    mean = np.array(self.normalize_mean, dtype=np.float32)
                    std = np.array(self.normalize_std, dtype=np.float32)
                    img_array = (img_array - mean) / std
                if self.channels_first:
                    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
                return np.expand_dims(img_array, axis=0)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def predict(self, preprocessed_image: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run prediction with ONNX Runtime"""
        if self.session is None:
            raise ValueError("Model not loaded")

        try:
            outputs = self.session.run(
                [self.output_name], {self.input_name: preprocessed_image.astype(np.float32)}
            )
            return ensure_probabilities(outputs[0])
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


class TFLiteImageClassifier(BaseImageClassifier):
    """TensorFlow Lite (LiteRT) model classifier.

    Like ONNX, a .tflite file embeds its own graph, so no architecture module is
    needed. Input is NHWC; preprocessing follows the input tensor's dtype (uint8:
    raw pixels, int8: quantized from a float range, float: ONNX-style rescale and
    mean/std). A single-value output is a binary positive score (sigmoid applied
    when it is a logit), which ImageClassifierWrapper expands to ``[1 - p, p]``.
    """

    # (module to import, attribute path on it), tried in order; the first that
    # resolves wins. tensorflow is the fallback since it is a core requirement.
    # tf.lite.Interpreter has to be read as an attribute of the tensorflow module:
    # importing "tensorflow.lite" yields the empty source package, while the public
    # tf.lite namespace is generated under tensorflow._api.
    _INTERPRETER_SOURCES = (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow", "lite.Interpreter"),
    )

    def __init__(self, model_path: str,
                 input_shape: Optional[Tuple[int, int]] = None,
                 normalize_mean: Optional[List[float]] = None,
                 normalize_std: Optional[List[float]] = None,
                 rescale: Optional[bool] = None,
                 num_threads: Optional[int] = None):
        """TFLite model classifier

        Args:
            model_path: Path to .tflite model file
            input_shape: Optional (width, height). Applied when the model's spatial
                dims are dynamic (the input tensor is resized to it); ignored with a
                warning when it conflicts with a fixed declared shape, since the
                interpreter rejects mismatched input.
            normalize_mean: float-input models only (default: ImageNet)
            normalize_std: float-input models only (default: ImageNet)
            rescale: float input: divide by 255 before mean/std (None means True).
                int8 input: True quantizes from [0, 1], False from raw 0-255, None
                infers the float range from the input quantization scale.
                uint8 input: ignored, raw pixels are fed.
            num_threads: Interpreter CPU thread count (None: runtime default)
        """
        super().__init__(model_path)
        self._input_shape_override = input_shape
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.rescale = rescale
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        # A single Interpreter instance does not support concurrent invoke() calls.
        self._lock = threading.Lock()
        self.load_model()

    @classmethod
    def _import_interpreter_class(cls):
        """Return (Interpreter class, source name) from the first available runtime, or
        (None, None) after logging why each source was unusable."""
        import importlib
        failures = []
        for module_name, attr_path in cls._INTERPRETER_SOURCES:
            source = f"{module_name}.{attr_path}"
            try:
                target = importlib.import_module(module_name)
                for attr in attr_path.split("."):
                    target = getattr(target, attr)
                return target, source
            except Exception as e:
                failures.append(f"{source}: {type(e).__name__}: {e}")
        for failure in failures:
            logger.error(f"TFLite runtime unavailable - {failure}")
        return None, None

    @staticmethod
    def _dims_with_dynamic_marked(details: dict) -> list:
        """The input shape with dynamic dims (-1 in ``shape_signature``) replaced by None."""
        shape = [int(d) for d in details["shape"]]
        signature = details.get("shape_signature")
        if signature is None or len(signature) != len(shape):
            return shape
        return [None if int(sig) < 0 else dim for dim, sig in zip(shape, signature)]

    @staticmethod
    def _describe_tensor(details: dict) -> str:
        """Shape, dtype and (scale, zero_point) of a tensor, for logging."""
        shape = [int(d) for d in details["shape"]]
        text = f"{shape} {np.dtype(details['dtype']).name}"
        scale, zero_point = details.get("quantization", (0.0, 0))
        if scale:
            text += f" (scale={float(scale):.6g}, zero_point={int(zero_point)})"
        return text

    @staticmethod
    def _int8_input_range(scale: float) -> Tuple[float, float]:
        """Infer the float range an int8 input was quantized from: 256 steps of
        ``scale`` span about 1 for [0, 1], 2 for [-1, 1] and 255 for raw pixels."""
        span = 255.0 * scale
        if span > 10.0:
            return (0.0, 255.0)
        if span > 1.5:
            return (-1.0, 1.0)
        return (0.0, 1.0)

    def load_model(self) -> bool:
        """Load the .tflite model into an Interpreter and read its tensor details"""
        interpreter_cls, source = self._import_interpreter_class()
        if interpreter_cls is None:
            logger.error(
                "No usable TFLite runtime found (reasons logged above). Supported: "
                "ai-edge-litert, tflite-runtime, or tensorflow with tf.lite.Interpreter"
            )
            return False

        try:
            kwargs = {"model_path": self.model_path}
            if self.num_threads is not None:
                kwargs["num_threads"] = int(self.num_threads)
            self.interpreter = interpreter_cls(**kwargs)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]

            dims = self._dims_with_dynamic_marked(self.input_details)
            inferred = ONNXImageClassifier._infer_input_shape_from_dims(dims, channels_first=False)
            override = tuple(self._input_shape_override) if self._input_shape_override else None
            if override and inferred and override != inferred:
                logger.warning(
                    f"Ignoring input_shape {override}: TFLite model declares a fixed input of {inferred}"
                )
                override = None
            self.input_shape = override or inferred
            if self.input_shape is None:
                logger.warning(
                    f"Could not infer input shape from TFLite model metadata {self.input_details['shape']}, "
                    "using default (224, 224)"
                )
                self.input_shape = (224, 224)

            if len(dims) == 4 and (dims[1] is None or dims[2] is None):
                width, height = self.input_shape
                channels = dims[3] or 3
                self.interpreter.resize_tensor_input(
                    self.input_details["index"], [1, height, width, channels])
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()[0]

            self.output_details = self.interpreter.get_output_details()[0]
            self.is_loaded = True
            logger.info(
                f"TFLite model loaded via {source}: input {self._describe_tensor(self.input_details)}, "
                f"output {self._describe_tensor(self.output_details)}"
            )
            return True
        except Exception as e:
            message = str(e)
            lowered = message.lower()
            if "ethos" in lowered or "custom op" in lowered:
                logger.error(
                    f"Failed to load TFLite model {self.model_path}: it contains a custom operator "
                    f"that the desktop runtime cannot execute ({message}). Models compiled for an NPU "
                    "(e.g. with Arm Vela for Ethos-U) only run on that hardware."
                )
            else:
                logger.error(f"Failed to load TFLite model: {message}")
            return False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for the TFLite input tensor's dtype"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize(self.input_shape)
                pixels = np.array(img, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

        dtype = np.dtype(self.input_details["dtype"])
        if dtype == np.uint8:
            img_array = pixels.astype(np.uint8)
        elif dtype == np.int8:
            scale, zero_point = self.input_details.get("quantization", (0.0, 0))
            if not scale:
                img_array = np.clip(pixels - 128.0, -128, 127).astype(np.int8)
            else:
                if self.rescale is None:
                    low, high = self._int8_input_range(scale)
                else:
                    low, high = (0.0, 1.0) if self.rescale else (0.0, 255.0)
                real = low + pixels / 255.0 * (high - low)
                img_array = np.clip(np.round(real / scale + zero_point), -128, 127).astype(np.int8)
        elif np.issubdtype(dtype, np.floating):
            img_array = pixels
            if self.rescale is None or self.rescale:
                img_array = img_array / 255.0
                mean = np.array(self.normalize_mean, dtype=np.float32)
                std = np.array(self.normalize_std, dtype=np.float32)
                img_array = (img_array - mean) / std
            img_array = img_array.astype(dtype)
        else:
            raise ValueError(f"Unsupported TFLite input dtype: {dtype.name}")
        return np.expand_dims(img_array, axis=0)

    def predict(self, preprocessed_image: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run prediction with the TFLite interpreter (one image per invoke; batch_size is ignored)"""
        if self.interpreter is None:
            raise ValueError("Model not loaded")

        try:
            with self._lock:
                self.interpreter.set_tensor(self.input_details["index"], preprocessed_image)
                self.interpreter.invoke()
                raw = self.interpreter.get_tensor(self.output_details["index"])

            output = raw.astype(np.float32)
            scale, zero_point = self.output_details.get("quantization", (0.0, 0))
            if np.issubdtype(raw.dtype, np.integer) and scale:
                output = (output - zero_point) * scale
            return ensure_probabilities(output.reshape(1, -1))
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


def _as_list(values) -> list:
    """Plain list from a tensor, array or sequence."""
    return values.tolist() if hasattr(values, "tolist") else list(values)


class DetectionImageClassifier(BaseImageClassifier):
    """Object detector (transformers AutoModelForObjectDetection) scored as a classifier.

    Each non-background category scores the highest confidence among detections
    whose label maps to it, after the score and box-area filters. The background
    category scores 1 minus the best of those, so classify_image picks a target
    category exactly when its best detection exceeds 0.5. Categories match the
    model's id2label via normalize_label; label_aliases maps one category to
    several model labels.
    """

    DEFAULT_SCORE_THRESHOLD = 0.3
    DEFAULT_MIN_BOX_AREA_RATIO = 0.0

    def __init__(self, model_path: str,
                 model_categories: List[str],
                 background_category: Optional[str] = None,
                 score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 min_box_area_ratio: float = DEFAULT_MIN_BOX_AREA_RATIO,
                 label_aliases: Optional[Dict[str, Any]] = None,
                 hf_pretrained_path: Optional[str] = None,
                 device: str = 'auto',
                 input_shape: Optional[Tuple[int, int]] = None):
        """Object detection classifier

        Args:
            model_path: Weights file in the model directory, or the directory itself
            model_categories: Categories to score, in output order
            background_category: The "none of the above" category, which must be one
                of model_categories. Without one, an image with no qualifying
                detection classifies as the first category.
            score_threshold: Detections below this confidence are discarded
            min_box_area_ratio: Detections whose box covers less than this fraction
                of the image are discarded (e.g. 0.02 ignores small background figures)
            label_aliases: {category: [model label, ...]} for categories that cover
                several model labels or are named differently from them
            hf_pretrained_path: Model directory (default: model_path's directory)
            device: 'auto', 'cuda', or 'cpu'
            input_shape: Informational only; the model's processor sets the input size
        """
        super().__init__(model_path)
        self.model_categories = list(model_categories)
        self.background_category = background_category or None
        self.score_threshold = float(score_threshold)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.label_aliases = {
            str(category): [str(label) for label in (labels if isinstance(labels, (list, tuple)) else [labels])]
            for category, labels in (label_aliases or {}).items()
        }
        self.hf_pretrained_path = hf_pretrained_path
        self._device_setting = device
        self._input_shape_override = input_shape
        self.device = None
        self.processor = None
        self._id2label: Dict[int, str] = {}
        self._category_label_ids: Dict[str, set] = {}
        self._last_detections: Optional[Tuple[str, List[Dict[str, Any]]]] = None
        self.load_model()

    @staticmethod
    def read_id2label(model_root: str) -> Dict[int, str]:
        """The id2label map from *model_root*'s config.json ({} when unreadable)."""
        import json
        try:
            with open(os.path.join(model_root, "config.json"), "r", encoding="utf-8") as f:
                id2label = json.load(f).get("id2label") or {}
            return {int(k): str(v) for k, v in id2label.items()}
        except Exception:
            return {}

    @staticmethod
    def split_background_category(categories: List[str], labels) -> Tuple[Optional[str], List[str]]:
        """(background category, categories matching no label). The background is the
        single category that is not a model label; when several are not, none is
        picked and all of them are returned as unmatched."""
        known = {normalize_label(label) for label in labels}
        unmatched = [c for c in categories if normalize_label(c) not in known]
        if len(unmatched) == 1:
            return unmatched[0], []
        return None, unmatched

    def _model_root(self) -> str:
        if self.hf_pretrained_path:
            return self.hf_pretrained_path
        return os.path.dirname(self.model_path) if os.path.isfile(self.model_path) else self.model_path

    def _load_components(self, model_root: str):
        """(image processor, model on self.device in eval mode) from *model_root*."""
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        processor = AutoImageProcessor.from_pretrained(model_root)
        model = from_pretrained_checked(AutoModelForObjectDetection, model_root, model_root, log=logger)
        return processor, model.to(self.device).eval()

    def _resolve_category_labels(self) -> Dict[str, set]:
        """{non-background category: model label ids}. Raises ValueError for a label the
        model does not produce or a background category outside model_categories."""
        if self.background_category and self.background_category not in self.model_categories:
            raise ValueError(
                f"background_category {self.background_category!r} is not one of the model "
                f"categories {self.model_categories}"
            )
        if not self.background_category:
            logger.warning(
                f"No background_category set for object detection model {self.model_path}: "
                "an image without a qualifying detection classifies as the first category"
            )
        ids_by_label: Dict[str, set] = {}
        for label_id, label in self._id2label.items():
            ids_by_label.setdefault(normalize_label(label), set()).add(label_id)

        resolved: Dict[str, set] = {}
        unknown: List[str] = []
        for category in self.model_categories:
            if category == self.background_category:
                continue
            ids: set = set()
            for label in self.label_aliases.get(category, [category]):
                found = ids_by_label.get(normalize_label(label))
                if found is None:
                    unknown.append(label)
                else:
                    ids |= found
            resolved[category] = ids
        if unknown:
            raise ValueError(
                f"Labels not produced by this model: {unknown}. "
                f"Model labels: {sorted(set(self._id2label.values()))}"
            )
        return resolved

    def load_model(self) -> bool:
        """Load the processor and model, and map categories onto model labels"""
        model_root = self._model_root()
        if not model_root or not os.path.isdir(model_root):
            logger.error(f"Invalid model directory for object detection: {model_root}")
            return False
        try:
            self.device = resolve_torch_device(self._device_setting)
            self.processor, self.model = self._load_components(model_root)
            id2label = getattr(getattr(self.model, "config", None), "id2label", None) or {}
            self._id2label = {int(k): str(v) for k, v in id2label.items()}
            self._category_label_ids = self._resolve_category_labels()
            self.input_shape = self._input_shape_override or infer_processor_input_shape(self.processor)
            self.is_loaded = True
            logger.info(
                f"Object detection model loaded from {model_root} on {self.device}: "
                f"categories {self.model_categories}, background {self.background_category!r}, "
                f"score_threshold {self.score_threshold}, min_box_area_ratio {self.min_box_area_ratio}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load object detection model from {model_root}: {e}")
            return False

    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Processor inputs on self.device plus the original (height, width)"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                width, height = img.size
                inputs = self.processor(images=img, return_tensors="pt")
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return {"inputs": inputs, "size": (height, width)}

    def _forward(self, inputs: Dict[str, Any]):
        import torch
        with torch.no_grad():
            return self.model(**inputs)

    def _detections(self, preprocessed: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs = self._forward(preprocessed["inputs"])
        height, width = preprocessed["size"]
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(height, width)]
        )[0]
        image_area = float(width * height) or 1.0
        detections = []
        for score, label_id, box in zip(
            _as_list(result["scores"]), _as_list(result["labels"]), _as_list(result["boxes"])
        ):
            x0, y0, x1, y1 = (float(v) for v in box)
            area_ratio = max(0.0, x1 - x0) * max(0.0, y1 - y0) / image_area
            if area_ratio < self.min_box_area_ratio:
                continue
            label_id = int(label_id)
            detections.append({
                "label": self._id2label.get(label_id, str(label_id)),
                "label_id": label_id,
                "score": float(score),
                "box": (x0, y0, x1, y1),
                "area_ratio": area_ratio,
            })
        return detections

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """Detections that pass the score and box-area filters, as dicts with label,
        label_id, score, box (x0, y0, x1, y1 in pixels) and area_ratio."""
        return self._detections(self.preprocess_image(image_path))

    def predict_image(self, image_path: str) -> np.ndarray:
        """Per-category scores for *image_path*; its detections are kept for last_detections."""
        detections = self.detect(image_path)
        self._last_detections = (image_path, detections)
        return self._scores_from_detections(detections)

    def last_detections(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """Detections behind the most recent predict_image call, if it was for
        *image_path* (another image may have been scored since, e.g. on another thread)."""
        last = self._last_detections
        if last is not None and last[0] == image_path:
            return last[1]
        return None

    def categories_for_label_id(self, label_id: int) -> List[str]:
        """Categories a detection with *label_id* counts toward (empty if none)."""
        return [c for c, label_ids in self._category_label_ids.items() if label_id in label_ids]

    def _scores_from_detections(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        best = {category: 0.0 for category in self._category_label_ids}
        for detection in detections:
            for category, label_ids in self._category_label_ids.items():
                if detection["label_id"] in label_ids and detection["score"] > best[category]:
                    best[category] = detection["score"]
        top = max(best.values(), default=0.0)
        row = [1.0 - top if c == self.background_category else best[c] for c in self.model_categories]
        return np.array([row], dtype=np.float32)

    def predict(self, preprocessed_image: Dict[str, Any], batch_size: int = 32) -> np.ndarray:
        """Per-category scores for one preprocessed image (batch_size is ignored)"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        try:
            return self._scores_from_detections(self._detections(preprocessed_image))
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


_BACKEND_CLASSES = {
    BackendType.PYTORCH: PyTorchImageClassifier,
    BackendType.HDF5: H5ImageClassifier,
    BackendType.ONNX: ONNXImageClassifier,
    BackendType.TFLITE: TFLiteImageClassifier,
    BackendType.DETECTION: DetectionImageClassifier,
}
# Constructor arguments ImageClassifierWrapper supplies itself, never from model_kwargs.
_WRAPPER_SUPPLIED_ARGS = frozenset({"self", "model_path", "model_categories", "custom_objects"})


def accepted_model_kwargs(backend: Optional[BackendType]) -> Optional[frozenset]:
    """model_kwargs keys *backend*'s classifier accepts, or None when it has no classifier class."""
    import inspect
    backend_cls = _BACKEND_CLASSES.get(backend)
    if backend_cls is None:
        return None
    return frozenset(inspect.signature(backend_cls.__init__).parameters) - _WRAPPER_SUPPLIED_ARGS


def model_kwargs_for_backend_change(
    model_kwargs: Dict[str, Any],
    old_backend: str, old_location: str,
    new_backend: str, new_location: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """(kept model_kwargs, dropped keys) when a model config moves from one backend to
    another ("auto" resolved from each location). Keys the new backend's classifier
    does not accept are dropped, since passing them fails the load. Nothing is dropped
    when the backend is unchanged or the new one is unknown."""
    old = BackendType.parse(old_backend) or BackendType.from_model_location(old_location)
    new = BackendType.parse(new_backend) or BackendType.from_model_location(new_location)
    accepted = accepted_model_kwargs(new)
    if old == new or accepted is None:
        return dict(model_kwargs), []
    kept = {k: v for k, v in model_kwargs.items() if k in accepted}
    return kept, sorted(k for k in model_kwargs if k not in accepted)


class ImageClassifierWrapper:
    def __init__(self, model_config: ImageClassifierModelConfig):
        """Load and run an image classifier from a single :class:`ImageClassifierModelConfig`.

        Split-positive options (``positive_groups``, ``neutral_categories``, ``severity_order``)
        are read from ``model_config``. If ``positive_groups`` is non-empty and
        ``neutral_categories`` is omitted or empty, neutrals are derived as the complement of
        the union of all positive groups within ``model_categories``.
        """
        self.model_name = model_config.model_name
        self.model_categories = list(model_config.model_categories)
        self.model_location = model_config.model_location
        self.use_hub_keras_layers = model_config.use_hub_keras_layers
        self.backend = BackendType.parse(model_config.backend)
        self.model_kwargs = dict(model_config.model_kwargs)
        self.input_shape = model_config.input_shape
        self.positive_groups = [list(g) for g in model_config.positive_groups]
        self.neutral_categories = list(model_config.neutral_categories)
        self.severity_order = list(model_config.severity_order)

        if self.positive_groups and not self.neutral_categories:
            self.neutral_categories = derive_neutral_categories_from_positive_groups(
                self.model_categories, self.positive_groups
            )
            if config.debug2 and self.neutral_categories:
                logger.debug(
                    f"Derived neutral_categories for {self.model_name}: {self.neutral_categories}"
                )

        self.can_run = True
        self.classifier = None
        self.predictions_cache = {}
        self._prediction_signature: Optional[str] = None
        
        if self.can_run:
            try:
                self.model_name = str(self.model_name).strip()
                if self.model_name is None or self.model_name == "":
                    raise Exception("Invalid model name: " + self.model_name)
                if not type(self.model_categories) == list or len(self.model_categories) == 0 \
                        or any([type(c) != str for c in self.model_categories]):
                    raise Exception(f"Invalid model categories: {self.model_categories}")
                if not type(self.model_location) == str or not (os.path.isfile(self.model_location) or os.path.isdir(self.model_location)):
                    raise Exception(f"Invalid model location: {self.model_location}")
                if not type(self.use_hub_keras_layers) == bool:
                    raise Exception(f"Invalid use hub keras layers flag, must be boolean: {self.use_hub_keras_layers}")
                allowed = set(self.model_categories)
                if self.positive_groups:
                    if not isinstance(self.positive_groups, list):
                        raise Exception(f"positive_groups must be a list, got {type(self.positive_groups)}")
                    for grp in self.positive_groups:
                        if not isinstance(grp, list):
                            raise Exception(f"positive_groups entries must be lists, got {type(grp)}")
                        for c in grp:
                            if c not in allowed:
                                raise Exception(
                                    f"positive_groups references unknown category {c!r} "
                                    f"(not in model_categories)"
                                )
                for c in self.neutral_categories:
                    if c not in allowed:
                        raise Exception(
                            f"neutral_categories references unknown category {c!r} "
                            f"(not in model_categories)"
                        )
            except Exception:
                self.can_run = False
                logger.exception(
                    "Image classifier %r: config validation failed before load (location=%r, "
                    "positive_groups=%s, neutral_categories=%s)",
                    self.model_name,
                    self.model_location,
                    self.positive_groups,
                    self.neutral_categories,
                )
            if self.can_run:
                self.load_classifier()

    def load_classifier(self):
        """Load appropriate classifier based on backend and file extension"""
        assert self.can_run is True
        
        # Determine backend if auto
        if self.backend is None:
            self.backend = BackendType.from_model_location(self.model_location)
            if self.backend is None:
                self.can_run = False
                logger.error(f"Cannot determine backend for file: {self.model_location}")
                return

        # Special handling for safetensors: require model_architecture (a detection
        # model's architecture comes from its config.json)
        if self.model_location.lower().endswith('.safetensors') and self.backend != BackendType.DETECTION:
            if not self.model_kwargs.get("use_transformers_auto_model", False) and 'architecture_module_name' not in self.model_kwargs:
                message = "For safetensors files, architecture_module_name must be provided.\n"
                message += f"Found model_kwargs: {self.model_kwargs}\n"
                message += "Must include architecture_module_name in model_kwargs.\n"
                message += "Example: model_kwargs={'architecture_module_name': 'model_architecture', 'architecture_class_path': 'ClassName'}"
                logger.error(message)
                self.can_run = False
                return

        # Initialize appropriate classifier
        try:
            if self.backend == BackendType.HDF5:
                custom_objects = {}
                if self.use_hub_keras_layers:
                    try:
                        import tensorflow_hub as hub
                        custom_objects['KerasLayer'] = hub.KerasLayer
                    except ImportError:
                        logger.error("Failed to import tensorflow hub to support h5 model, please install it using pip")
                        self.can_run = False
                        return
                
                # NOTE: Separate model architecture loading is not supported for h5 models as a standard
                
                self.classifier = H5ImageClassifier(
                    self.model_location,
                    custom_objects=custom_objects,
                    **self.model_kwargs
                )
                
            elif self.backend == BackendType.PYTORCH:
                # Default kwargs for PyTorch
                pytorch_kwargs = self.model_kwargs.copy()
                if self.input_shape is not None:
                    pytorch_kwargs["input_shape"] = self.input_shape
                
                # Set defaults for converted models if not specified
                if 'weights_only' not in pytorch_kwargs:
                    pytorch_kwargs['weights_only'] = False  # Safer for converted models

                # Handle model architecture import if provided
                if 'architecture_module_name' in pytorch_kwargs and pytorch_kwargs['architecture_module_name'] is not None:
                    architecture_module_name = pytorch_kwargs['architecture_module_name']
                    architecture_class_path = pytorch_kwargs.get('architecture_class_path', None)
                    architecture_location = pytorch_kwargs.get('architecture_location', None)
                    try:
                        model_dir = os.path.dirname(self.model_location)
                        model_class = PyTorchImageClassifier.load_model_architecture(
                            architecture_module_name, architecture_class_path=architecture_class_path,
                            architecture_location=architecture_location, model_dir=model_dir)
                        pytorch_kwargs['model_architecture'] = model_class
                    except Exception as e:
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.error(f"Failed to import model architecture: {e}")
                        self.can_run = False
                        return

                self.classifier = PyTorchImageClassifier(
                    self.model_location,
                    **pytorch_kwargs
                )

            elif self.backend == BackendType.ONNX:
                onnx_kwargs = self.model_kwargs.copy()
                if self.input_shape is not None:
                    onnx_kwargs["input_shape"] = self.input_shape

                self.classifier = ONNXImageClassifier(
                    self.model_location,
                    **onnx_kwargs
                )

            elif self.backend == BackendType.TFLITE:
                tflite_kwargs = self.model_kwargs.copy()
                if self.input_shape is not None:
                    tflite_kwargs["input_shape"] = self.input_shape

                self.classifier = TFLiteImageClassifier(
                    self.model_location,
                    **tflite_kwargs
                )

            elif self.backend == BackendType.DETECTION:
                detection_kwargs = self.model_kwargs.copy()
                if self.input_shape is not None:
                    detection_kwargs["input_shape"] = self.input_shape

                self.classifier = DetectionImageClassifier(
                    self.model_location,
                    self.model_categories,
                    **detection_kwargs
                )
            else:
                logger.error(f"Unsupported backend: {self.backend}")
                self.can_run = False
                return
                
            self.can_run = bool(self.classifier.is_loaded)
            
        except Exception as e:
            self.can_run = False
            logger.error(e)
            logger.warning(f"Failed to initialize {self.backend} model for image classifier: {self.model_name}")

    def _persisted_prediction_key(self) -> tuple:
        """(model key, settings signature) for classifier_prediction_cache."""
        if self._prediction_signature is None:
            self._prediction_signature = model_signature({
                "location": self.model_location,
                "categories": self.model_categories,
                "backend": self.backend.value if self.backend else None,
                "model_kwargs": self.model_kwargs,
                "input_shape": self.input_shape,
            })
        return f"image:{self.model_name}", self._prediction_signature

    @staticmethod
    def input_image_path(media_path: str) -> str:
        """The raster image a classifier reads for *media_path*: the rendered first
        frame/page (FrameCache) for video, GIF, PDF, ePub, SVG and HTML, else the
        path itself. A FrameCache render passes through unchanged."""
        from image.frame_cache import FrameCache
        return FrameCache.get_image_path(media_path)

    def discard_cached_prediction(self, image_path) -> None:
        """Forget the session and persisted scores for *image_path*."""
        image_path = self.input_image_path(image_path)
        self.predictions_cache.pop(image_path, None)
        model_key, _sig = self._persisted_prediction_key()
        classifier_prediction_cache.discard(model_key, image_path)

    def predict_image(self, image_path):
        # Callers may pass any media path; scores are kept under the rendered
        # image's path, which the persisted cache maps back to the source file.
        image_path = self.input_image_path(image_path)
        if image_path in self.predictions_cache:
            return self.predictions_cache[image_path]

        model_key, signature = self._persisted_prediction_key()
        persisted = classifier_prediction_cache.get(
            model_key, signature, self.model_location, image_path)
        if persisted is not None:
            self.predictions_cache[image_path] = persisted
            return dict(persisted)

        if self.classifier is None:
            raise ValueError("Classifier not initialized")
        
        predictions = self.classifier.predict_image(image_path)
        classed_predictions = map_scores_to_categories(
            predictions[0], self.model_categories, self.model_name, log=logger)
        
        self.predictions_cache[image_path] = dict(classed_predictions)
        classifier_prediction_cache.put(
            model_key, signature, self.model_location, image_path, classed_predictions)
        return classed_predictions

    def predict_image_ranked(self, image_path) -> list[tuple[str, float]]:
        """Return (category, score) pairs sorted by score descending (rank 1 = highest)."""
        return sorted(self.predict_image(image_path).items(), key=lambda kv: kv[1], reverse=True)

    def classify_image(self, image_path):
        if not self.can_run:
            raise Exception(f"Invalid state: Image classifier details failed to initialize, unable to classify image")

        classed_predictions = self.predict_image(image_path)

        if self.positive_groups:
            split = pick_split_positive(
                classed_predictions, self.positive_groups, self.neutral_categories,
                self.severity_order, _SPLIT_GROUP_OVER_NEUTRAL_MARGIN,
            )
            if split is not None:
                best_category, best_group, best_combined_prob = split
                if config.debug2:
                    logger.debug(
                        f"Image classifier prediction map ({self.model_name}): "
                        f"{format_prediction_line(classed_predictions)}"
                    )
                    logger.debug(
                        f"Split positive in group '{'+'.join(best_group)}': "
                        f"combined={best_combined_prob:.6f}, assigned={best_category}"
                    )
                return best_category

        classed_category = top_category(classed_predictions, self.model_categories)
        if config.debug2:
            logger.debug(
                f"Image classifier prediction map ({self.model_name}): "
                f"{format_prediction_line(classed_predictions)}"
            )
        return classed_category

    def test_image_for_categories(self, image_path, categories):
        if not self.can_run:
            raise Exception(f"Invalid state: Image classifier details failed to initialize, unable to classify image")
        category = self.classify_image(image_path)
        return category in categories

    def test_image_for_category(self, image_path, category, threshold):
        if self.can_run:
            return self.predict_image(image_path).get(category, 0.0) > threshold
        raise Exception(f"Invalid state: Image classifier details failed to initialize, unable to classify image")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', categories={self.model_categories}, backend={self.backend})"

    def __hash__(self) -> int:
        return hash(self.model_name)
    
    def __eq__(self, other):
        if not isinstance(other, ImageClassifierWrapper):
            raise TypeError(f"Invalid type for comparison: {type(other)}")
        return self.model_name == other.model_name


# Factory function for convenience
def create_image_classifier(model_name: str = "",
                           model_categories: Optional[List[str]] = None,
                           model_location: str = "",
                           use_hub_keras_layers: bool = False,
                           backend: Union[str, BackendType] = "auto",
                           **kwargs) -> ImageClassifierWrapper:
    """Convenience wrapper that builds an :class:`ImageClassifierModelConfig` and classifier.

    For split-positive settings or full control, construct :class:`ImageClassifierModelConfig`
    (or use :meth:`ImageClassifierModelConfig.from_dict`) and pass it to
    :class:`ImageClassifierWrapper` directly.

    **kwargs: Passed as ``model_kwargs`` on the config.
    """
    if model_categories is None:
        model_categories = ["drawing", "photograph"]
    parsed_backend = BackendType.parse(backend)
    backend_str = "auto" if parsed_backend is None else parsed_backend.value
    if not backend_str:
        backend_str = "auto"

    mc = ImageClassifierModelConfig(
        model_name=str(model_name).strip(),
        model_location=model_location,
        model_categories=list(model_categories),
        use_hub_keras_layers=use_hub_keras_layers,
        backend=backend_str,
        model_kwargs=dict(kwargs),
    )
    return ImageClassifierWrapper(mc)


