from typing import List


from utils.logging_setup import get_logger
from utils.translations import _
logger = get_logger("lookahead")


class Lookahead:
    """Represents a lookahead check for a prevalidation or classifier action."""

    lookaheads: List['Lookahead'] = []
    
    def __init__(self, name="", name_or_text="", threshold=0.23, is_prevalidation_name=False):
        self.name = name  # Unique name to identify this lookahead
        self.name_or_text = name_or_text
        self.threshold = threshold
        self.is_prevalidation_name = is_prevalidation_name
        self.run_result = None  # Cached result for the current prevalidate call (None = not run yet, True = triggered, False = not triggered)

    @staticmethod
    def eval_cache_key(lookahead_name: str, image_path: str) -> str:
        """
        Build a stable cache key for per-frame lookahead evaluation reuse.
        """
        return f"{lookahead_name}|{image_path}"

    def __eq__(self, other):
        """Check equality based on name, name_or_text, threshold, and is_prevalidation_name."""
        if not isinstance(other, Lookahead):
            return False
        return (self.name == other.name and 
                self.name_or_text == other.name_or_text and
                self.threshold == other.threshold and
                self.is_prevalidation_name == other.is_prevalidation_name)
    
    def __hash__(self):
        """Hash based on name, name_or_text, threshold, and is_prevalidation_name."""
        return hash((self.name, self.name_or_text, self.threshold, self.is_prevalidation_name))

    def validate(self):
        if self.name is None or self.name.strip() == "":
            return False
        if self.name_or_text is None or self.name_or_text.strip() == "":
            return False
        if self.is_prevalidation_name:
            from compare.classifier_actions_manager import ClassifierActionsManager
            prevalidation = ClassifierActionsManager.get_prevalidation_by_name(self.name_or_text)
            return prevalidation is not None
        return True
    
    def to_dict(self):
        """Serialize to dictionary for caching."""
        return {
            "name": self.name,
            "name_or_text": self.name_or_text,
            "threshold": self.threshold,
            "is_prevalidation_name": self.is_prevalidation_name,
        }
    
    @staticmethod
    def from_dict(d):
        """Deserialize from dictionary."""
        name = d.get("name", "")
        name_or_text = d.get("name_or_text", "")
        threshold = d.get("threshold", 0.23)
        is_prevalidation_name = d.get("is_prevalidation_name", False)
        
        return Lookahead(
            name=name,
            name_or_text=name_or_text,
            threshold=threshold,
            is_prevalidation_name=is_prevalidation_name
        )

    @staticmethod
    def get_lookahead_by_name(name: str) -> 'Lookahead':
        """Get a lookahead by name. Returns None if not found."""
        for lookahead in Lookahead.lookaheads:
            if name == lookahead.name:
                return lookahead
        return None

