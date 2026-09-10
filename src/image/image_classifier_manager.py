import json
from typing import Dict, Any, Optional, List

from image.image_classifier import ImageClassifierWrapper
from image.image_classifier_model_config import ImageClassifierModelConfig
from utils.config import config
from utils.logging_setup import get_logger

logger = get_logger("image_classifier_manager")


def _format_model_entry_for_log(model_details: Any, max_chars: int = 4000) -> str:
    """Compact JSON for logs when a config entry is rejected (truncate very long values)."""
    if not isinstance(model_details, dict):
        return repr(model_details)
    try:
        text = json.dumps(model_details, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = repr(model_details)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... ({len(text) - max_chars} more chars)"
    return text


class ImageClassifierManager:
    classifier_metadata: Dict[str, ImageClassifierModelConfig]
    classifiers: Dict[str, ImageClassifierWrapper]

    def resolve_registered_model_name(self, model_name: Optional[str]) -> Optional[str]:
        """Map a user-facing id to the canonical ``model_name`` key in ``classifier_metadata``.

        Prevalidations and classifier actions may reference either ``model_name`` or, for HF
        models, ``hf_repo_id`` when that is what was stored in settings.
        """
        if model_name is None:
            return None
        key = str(model_name).strip()
        if not key:
            return None
        if key in self.classifier_metadata:
            return key
        for cfg in self.classifier_metadata.values():
            repo = (cfg.hf_repo_id or "").strip()
            if repo and repo == key:
                return cfg.model_name
        return None

    def __init__(self) -> None:
        self.classifier_metadata = {}
        self.classifiers = {}
        models = getattr(config, 'image_classifier_models', [])
        if isinstance(models, list):
            for model_details in models:
                try:
                    model_config = ImageClassifierModelConfig.from_dict(model_details, logger=logger)
                    self.classifier_metadata[model_config.model_name] = model_config
                except Exception:
                    mn = model_details.get("model_name") if isinstance(model_details, dict) else None
                    logger.exception(
                        "Skipping invalid image_classifier_models entry (model_name=%r); "
                        "this model will be absent from the registry and prevalidations that reference it will be dropped at init.",
                        mn,
                    )
                    logger.error(
                        "Rejected image classifier config body follows (search logs for this model_name):\n%s",
                        _format_model_entry_for_log(model_details),
                    )

    def set_classifier_metadata(self, model_details_list: List[Dict[str, Any]]) -> None:
        """Replace all configured classifier metadata and trim stale runtime classifiers."""
        self.classifier_metadata.clear()
        if isinstance(model_details_list, list):
            for model_details in model_details_list:
                try:
                    model_config = ImageClassifierModelConfig.from_dict(model_details, logger=logger)
                    self.classifier_metadata[model_config.model_name] = model_config
                except Exception:
                    mn = model_details.get("model_name") if isinstance(model_details, dict) else None
                    logger.exception(
                        "Skipping invalid image_classifier_models entry (model_name=%r); "
                        "this model will be absent from the registry and prevalidations that reference it will be dropped at init.",
                        mn,
                    )
                    logger.error(
                        "Rejected image classifier config body follows (search logs for this model_name):\n%s",
                        _format_model_entry_for_log(model_details),
                    )
        stale_names = [name for name in self.classifiers if name not in self.classifier_metadata]
        for stale_name in stale_names:
            self.classifiers.pop(stale_name, None)

    def can_classify(self) -> bool:
        return len(self.get_model_names()) > 0

    def classify_image(self, model_name: str, image_path: str) -> Any:
        try:
            return self.get_classifier(model_name).classify_image(image_path)
        except KeyError as e:
            resolved = self.resolve_registered_model_name(model_name)
            if resolved is None:
                classifier_model_names = list(self.classifier_metadata.keys())
                raise Exception(f"Image classifier model name not found: {model_name}\n"
                                f"Valid classifier model names: {classifier_model_names}")
            raise e

    def add_classifier(self, image_classifier: ImageClassifierWrapper) -> None:
        if not isinstance(image_classifier, ImageClassifierWrapper):
            raise Exception(f"Invalid image classifier argument: {image_classifier}")
        if image_classifier.can_run:
            self.classifiers[image_classifier.model_name] = image_classifier
            logger.info(f"Added image classifier: {image_classifier}")
        else:
            logger.warning(f"Image classifier not runnable: {image_classifier}")

    def get_classifier(self, model_name: Optional[str]) -> Optional[ImageClassifierWrapper]:
        if model_name is None or model_name.strip() == "":
            return None
        key = self.resolve_registered_model_name(model_name)
        if key is None:
            keys = list(self.classifier_metadata.keys())
            hf_hint = [
                f"{cfg.model_name!r} (hf_repo_id={cfg.hf_repo_id!r})"
                for cfg in self.classifier_metadata.values()
                if cfg.hf_repo_id
            ]
            suffix = f"; Hugging Face ids: {', '.join(hf_hint)}" if hf_hint else ""
            raise Exception(
                f"Failed to find image classifier with model name: \"{model_name}\". "
                f"Registered model_name keys: {keys}{suffix}"
            )
        if key in self.classifiers:
            return self.classifiers[key]
        model_config = self.classifier_metadata[key]
        classifier = ImageClassifierWrapper(model_config)
        self.classifiers[key] = classifier
        if not classifier.can_run:
            logger.error(
                "Registered metadata for image classifier %r exists but the runtime wrapper failed to initialize "
                "(can_run=False). Prevalidations may reference this name; classification will not run until the "
                "model_location, architecture, or split fields are fixed. See image_classifier log lines above for this model.",
                key,
            )
        return classifier

    def get_model_names(self) -> List[str]:
        return list(self.classifier_metadata.keys())

    def is_loaded(self, model_name: Optional[str]) -> bool:
        """Whether the classifier is already instantiated and cached (vs. just registered metadata)."""
        key = self.resolve_registered_model_name(model_name)
        return key is not None and key in self.classifiers

    def add_classifier_metadata(self, model_details: Dict[str, Any]) -> None:
        model_config = ImageClassifierModelConfig.from_dict(model_details, logger=logger)
        self.classifier_metadata[model_config.model_name] = model_config

    def remove_classifier_metadata(self, model_name: str) -> None:
        self.classifier_metadata.pop(model_name, None)
        self.classifiers.pop(model_name, None)

    def get_model_configs(self) -> List[ImageClassifierModelConfig]:
        return list(self.classifier_metadata.values())


image_classifier_manager = ImageClassifierManager()





