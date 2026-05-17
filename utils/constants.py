from enum import Enum

from utils.translations import I18N
_ = I18N._


class AppInfo:
    SERVICE_NAME = "MyPersonalApplicationsService"
    APP_IDENTIFIER = "simple_image_compare"  # TODO update to new name


class Mode(Enum):
    BROWSE = _("Browsing Mode")
    SEARCH = _("Searching Mode")
    GROUP = _("Group Comparison Mode")
    DUPLICATES = _("Duplicate Detection Mode")

    # NOTE need this method because this class is initialized before the config
    # locale overwrites the default I18N settings.
    def get_text(self):
        if self == Mode.BROWSE:
            return _("Browsing Mode")
        elif self == Mode.SEARCH:
            return _("Searching Mode")
        elif self == Mode.GROUP:
            return _("Group Comparison Mode")
        elif self == Mode.DUPLICATES:
            return _("Duplicate Detection Mode")
        raise Exception("Unhandled Mode text: " + str(self))

    def __str__(self):
        return self.value


class CompareMode(Enum):
    COLOR_MATCHING = _("Color Matching")
    CLIP_EMBEDDING = _("CLIP Embedding")
    SIGLIP_EMBEDDING = _("SigLIP Embedding")
    FLAVA_EMBEDDING = _("FLAVA Embedding")
    ALIGN_EMBEDDING = _("ALIGN Embedding")
    XVLM_EMBEDDING = _("X-VLM Embedding")
    LAION_EMBEDDING = _("LAION Embedding")
    PROMPTS = _("Prompts")
    PROMPTS_EXACT = _("Prompts (Exact Match)")
    SIZE = _("Size")
    MODELS = _("Models")

    def get_text(self):
        if self == CompareMode.COLOR_MATCHING:
            return _("Color Matching")
        elif self == CompareMode.CLIP_EMBEDDING:
            return _("CLIP Embedding")
        elif self == CompareMode.SIGLIP_EMBEDDING:
            return _("SigLIP Embedding")
        elif self == CompareMode.FLAVA_EMBEDDING:
            return _("FLAVA Embedding")
        elif self == CompareMode.ALIGN_EMBEDDING:
            return _("ALIGN Embedding")
        elif self == CompareMode.XVLM_EMBEDDING:
            return _("X-VLM Embedding")
        elif self == CompareMode.LAION_EMBEDDING:
            return _("LAION Embedding")
        elif self == CompareMode.PROMPTS:
            return _("Prompts")
        elif self == CompareMode.PROMPTS_EXACT:
            return _("Prompts (Exact Match)")
        elif self == CompareMode.SIZE:
            return _("Size")
        elif self == CompareMode.MODELS:
            return _("Models")
        raise Exception("Unhandled Compare Mode text: " + str(self))

    def __str__(self):
        return self.value

    @staticmethod
    def get(name):
        # If name is already a CompareMode enum, return it directly
        if isinstance(name, CompareMode):
            return name
        # Otherwise, try to match by text, name, or value
        for key, value in CompareMode.__members__.items():
            if value.get_text() == name or value.name == name or value.value == name:
                return value
        raise Exception(f"Not a valid compare mode: {name}")

    @staticmethod
    def get_translated_names():
        return [value.get_text() for key, value in CompareMode.__members__.items()]

    def threshold_str(self):
        if self == CompareMode.COLOR_MATCHING:
            return _("Color diff threshold")
        if self.is_embedding():
            return _("Embedding similarity threshold")
        elif self == CompareMode.SIZE:
            return _("Size tolerance")
        elif self == CompareMode.MODELS:
            return _("Model similarity threshold")
        raise Exception("Unhandled Compare Mode text: " + str(self))

    def threshold_vals(self):
        if self == CompareMode.COLOR_MATCHING:
            return [str(i) for i in list(range(31))]
        if self.is_embedding():
            return [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.98, 0.99]
        elif self == CompareMode.PROMPTS_EXACT:
            # PROMPTS_EXACT uses similarity thresholds similar to embedding modes
            return [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.98, 0.99]
        elif self == CompareMode.SIZE:
            return [str(i) for i in list(range(0, 201, 10))]  # 0-200 pixel tolerance
        elif self == CompareMode.MODELS:
            return [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        # Default fallback for any unhandled modes
        return [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.98, 0.99]

    def is_embedding(self):
        return (self != CompareMode.COLOR_MATCHING and 
                self != CompareMode.SIZE and 
                self != CompareMode.MODELS and
                self != CompareMode.PROMPTS_EXACT)

    @staticmethod
    def embedding_modes():
        return [mode for mode in CompareMode if mode.is_embedding()]

    @staticmethod
    def text_search_modes():
        return [mode for mode in CompareMode if mode != CompareMode.SIZE and mode != CompareMode.COLOR_MATCHING]


class SortBy(Enum):
    NAME = _("Name")
    FULL_PATH = _("Full Path")
    CREATION_TIME = _("Creation Time")
    MODIFY_TIME = _("Modify Time")
    TYPE = _("Type")
    SIZE = _("Size")
    NAME_LENGTH = _("Name Length")
    RELATED_IMAGE = _("Related Image")
    RANDOMIZE = _("Random")
    IMAGE_PIXELS = _("Image Pixels")
    IMAGE_HEIGHT = _("Image Height")
    IMAGE_WIDTH = _("Image Width")

    def get_text(self):
        if self == SortBy.NAME:
            return _("Name")
        elif self == SortBy.FULL_PATH:
            return _("Full Path")
        elif self == SortBy.CREATION_TIME:
            return _("Creation Time")
        elif self == SortBy.MODIFY_TIME:
            return _("Modify Time")
        elif self == SortBy.TYPE:
            return _("Type")
        elif self == SortBy.SIZE:
            return _("Size")
        elif self == SortBy.NAME_LENGTH:
            return _("Name Length")
        elif self == SortBy.RELATED_IMAGE:
            return _("Related Image")
        elif self == SortBy.RANDOMIZE:
            return _("Random")
        elif self == SortBy.IMAGE_PIXELS:
            return _("Image Pixels")
        elif self == SortBy.IMAGE_HEIGHT:
            return _("Image Height")
        elif self == SortBy.IMAGE_WIDTH:
            return _("Image Width")
        raise Exception("Unhandled Sort By text: " + str(self))

    def __str__(self):
        return self.value

    @staticmethod
    def get(name):
        for key, value in SortBy.__members__.items():
            if str(value.value) == name or value.get_text() == name:
                return value
        raise Exception(f"Not a valid sort by: {name}")

    @staticmethod
    def members():
        return [value.get_text() for key, value in SortBy.__members__.items()]


class Sort(Enum):
    ASC = _("ascending")
    DESC = _("descending")
    RANDOM = _("random")

    def get_text(self):
        return _(self.value)

    def __str__(self):
        return self.value


class HfHubVisualMediaTask(Enum):
    ALL_VISUAL_MEDIA = ""
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    IMAGE_CLASSIFICATION = "image-classification"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    DEPTH_ESTIMATION = "depth-estimation"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"
    VIDEO_CLASSIFICATION = "video-classification"
    TEXT_TO_VIDEO = "text-to-video"
    IMAGE_FEATURE_EXTRACTION = "image-feature-extraction"

    def __str__(self):
        return self.value

    def display(self):
        return {
            HfHubVisualMediaTask.ALL_VISUAL_MEDIA: _("All visual media"),
            HfHubVisualMediaTask.TEXT_TO_IMAGE: _("Text to Image"),
            HfHubVisualMediaTask.IMAGE_TO_IMAGE: _("Image to Image"),
            HfHubVisualMediaTask.IMAGE_TO_TEXT: _("Image to Text"),
            HfHubVisualMediaTask.IMAGE_CLASSIFICATION: _("Image Classification"),
            HfHubVisualMediaTask.ZERO_SHOT_IMAGE_CLASSIFICATION: _("Zero-shot Image Classification"),
            HfHubVisualMediaTask.OBJECT_DETECTION: _("Object Detection"),
            HfHubVisualMediaTask.IMAGE_SEGMENTATION: _("Image Segmentation"),
            HfHubVisualMediaTask.DEPTH_ESTIMATION: _("Depth Estimation"),
            HfHubVisualMediaTask.VISUAL_QUESTION_ANSWERING: _("Visual Question Answering"),
            HfHubVisualMediaTask.VIDEO_CLASSIFICATION: _("Video Classification"),
            HfHubVisualMediaTask.TEXT_TO_VIDEO: _("Text to Video"),
            HfHubVisualMediaTask.IMAGE_FEATURE_EXTRACTION: _("Image Feature Extraction"),
        }[self]

    @classmethod
    def display_values(cls):
        return [item.display() for item in cls]

    @staticmethod
    def get(name):
        for key, value in HfHubVisualMediaTask.__members__.items():
            if key == name or value.value == name or value.display() == name:
                return value
        raise Exception(f"Not a valid HF Hub visual media task: {name}")

    @staticmethod
    def api_values():
        return [item.value for item in HfHubVisualMediaTask if item != HfHubVisualMediaTask.ALL_VISUAL_MEDIA]


class HfHubSortOption(Enum):
    DOWNLOADS = "downloads"
    LIKES = "likes"
    LAST_MODIFIED = "lastModified"
    CREATED_AT = "createdAt"
    TRENDING_SCORE = "trendingScore"

    def __str__(self):
        return self.value

    def display(self):
        return {
            HfHubSortOption.DOWNLOADS: _("Downloads"),
            HfHubSortOption.LIKES: _("Likes"),
            HfHubSortOption.LAST_MODIFIED: _("Last Modified"),
            HfHubSortOption.CREATED_AT: _("Created At"),
            HfHubSortOption.TRENDING_SCORE: _("Trending Score"),
        }[self]

    @classmethod
    def display_values(cls):
        return [item.display() for item in cls]

    @staticmethod
    def get(name):
        for key, value in HfHubSortOption.__members__.items():
            if key == name or value.value == name or value.display() == name:
                return value
        raise Exception(f"Not a valid HF Hub sort option: {name}")


class HfHubSortDirection(Enum):
    DESCENDING = -1
    ASCENDING = 1

    def __str__(self):
        return str(self.value)

    def display(self):
        return {
            HfHubSortDirection.DESCENDING: _("Descending"),
            HfHubSortDirection.ASCENDING: _("Ascending"),
        }[self]

    @classmethod
    def display_values(cls):
        return [item.display() for item in cls]

    @staticmethod
    def get(name):
        for key, value in HfHubSortDirection.__members__.items():
            if key == name or str(value.value) == str(name) or value.display() == name:
                return value
        raise Exception(f"Not a valid HF Hub sort direction: {name}")


class ImageGenerationType(Enum):
    REDO_PROMPT = "redo_prompt"
    TAKE_PROMPT = "take_prompt"
    CONTROL_NET = "control_net"
    IP_ADAPTER = "ip_adapter"
    RENOISER = "renoiser"
    IMG2IMG = "img2img"
    LAST_SETTINGS = "last_settings"
    CANCEL = "cancel"
    REVERT_TO_SIMPLE_GEN = "revert_to_simple_gen"

    def __str__(self):
        return self.value

    def get_text(self):
        if self == ImageGenerationType.REDO_PROMPT:
            return _("Redo Prompt")
        elif self == ImageGenerationType.TAKE_PROMPT:
            return _("Take Prompt")
        elif self == ImageGenerationType.CONTROL_NET:
            return _("Control Net")
        elif self == ImageGenerationType.IP_ADAPTER:
            return _("IP Adapter")
        elif self == ImageGenerationType.RENOISER:
            return _("Renoiser")
        elif self == ImageGenerationType.IMG2IMG:
            return _("Image to Image")
        elif self == ImageGenerationType.LAST_SETTINGS:
            return _("Last Settings")
        elif self == ImageGenerationType.CANCEL:
            return _("Cancel")
        elif self == ImageGenerationType.REVERT_TO_SIMPLE_GEN:
            return _("Revert to Simple Generation")
        raise Exception("Unhandled image generation type text: " + str(self))

    @staticmethod
    def get(name):
        if isinstance(name, ImageGenerationType):
            return name

        for key, value in ImageGenerationType.__members__.items():
            if value.name == name or value.value == name or value.get_text() == name:
                return value
        raise Exception(f"Not a valid prompt mode: {name}")

    @staticmethod
    def members():
        return [value.name for key, value in ImageGenerationType.__members__.items()]

    @staticmethod
    def translated_members():
        return [value.get_text() for key, value in ImageGenerationType.__members__.items()]


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "back"

    def get_correction(self, backward_value=0):
        return backward_value if self == Direction.BACKWARD else -1


class ViewMode(Enum):
    FULL = "full"        # single MediaFrame (default)
    MASONRY = "masonry"  # thumbnail grid browser


class CompareMediaType(Enum):
    """Enumeration of supported media types for comparison."""
    IMAGE = "image"
    GIF = "gif"
    VIDEO = "video"
    PDF = "pdf"
    SVG = "svg"
    HTML = "html"
    #: Path missing/invalid, or extension matches a disabled media category (see :func:`~utils.media_utils.get_media_type_for_path`).
    UNCONFIGURED = "unconfigured"

    def get_translation(self):
        """Get the translated string for this media type."""
        if self == CompareMediaType.IMAGE:
            return _("Image")
        elif self == CompareMediaType.GIF:
            return _("GIF")
        elif self == CompareMediaType.VIDEO:
            return _("Video")
        elif self == CompareMediaType.PDF:
            return _("PDF")
        elif self == CompareMediaType.SVG:
            return _("SVG")
        elif self == CompareMediaType.HTML:
            return _("HTML")
        elif self == CompareMediaType.UNCONFIGURED:
            return _("Unconfigured")
        raise Exception("Unhandled media type translation: " + str(self))

    def is_video(self) -> bool:
        return self == CompareMediaType.VIDEO

    def is_non_video(self) -> bool:
        """True for any type other than :data:`VIDEO` (includes :data:`UNCONFIGURED`)."""
        return self != CompareMediaType.VIDEO

    def is_gif(self) -> bool:
        return self == CompareMediaType.GIF

    def is_unconfigured(self) -> bool:
        """True when the path is invalid or the matching media type is disabled in config."""
        return self == CompareMediaType.UNCONFIGURED

    def supports_raster_image_details(self) -> bool:
        """True when the details dialog can use FrameCache + PIL / prompt extraction (not video or unconfigured)."""
        return self not in (
            CompareMediaType.VIDEO,
            CompareMediaType.UNCONFIGURED,
        )


# UI-friendly alias (same members and methods as :class:`CompareMediaType`).
MediaType = CompareMediaType


class ActionType(Enum):
    """Enumeration of supported action types for notifications."""
    MOVE_FILE = "move_file"
    COPY_FILE = "copy_file"
    REMOVE_FILE = "remove_file"
    MARK_FILE = "mark_file"
    UNMARK_FILE = "unmark_file"
    SEARCH_FILE = "search_file"
    COMPARE_FILES = "compare_files"
    GENERATE_IMAGE = "generate_image"
    CHANGE_DIRECTORY = "change_directory"
    SYSTEM = "system"  # For general system notifications

    def get_translation(self):
        """Get the translated string for this action type."""
        if self == ActionType.MOVE_FILE:
            return _("Moved files")
        elif self == ActionType.COPY_FILE:
            return _("Copied files")
        elif self == ActionType.REMOVE_FILE:
            return _("Removed files")
        elif self == ActionType.MARK_FILE:
            return _("Marked files")
        elif self == ActionType.UNMARK_FILE:
            return _("Unmarked files")
        elif self == ActionType.SEARCH_FILE:
            return _("Searched files")
        elif self == ActionType.COMPARE_FILES:
            return _("Compared files")
        elif self == ActionType.GENERATE_IMAGE:
            return _("Generated images")
        elif self == ActionType.CHANGE_DIRECTORY:
            return _("Changed directory")
        elif self == ActionType.SYSTEM:
            return _("System")
        raise Exception("Unhandled action type translation: " + str(self))


class ClassifierActionClass(Enum):
    """Classifier item type: classifier action or prevalidation."""
    CLASSIFIER_ACTION = "classifier_action"
    PREVALIDATION = "prevalidation"

    def get_display_value(self):
        """Return the translated display string for this type."""
        if self == ClassifierActionClass.CLASSIFIER_ACTION:
            return _("Classifier Action")
        elif self == ClassifierActionClass.PREVALIDATION:
            return _("Prevalidation")
        raise Exception("Unhandled ClassifierActionClass: " + str(self))

    @staticmethod
    def from_key(key):
        """Get enum from string key ('classifier_action' or 'prevalidation')."""
        if isinstance(key, ClassifierActionClass):
            return key
        key = (key or "").strip().lower()
        if key in ("classifier_action", "classifier action"):
            return ClassifierActionClass.CLASSIFIER_ACTION
        if key == "prevalidation":
            return ClassifierActionClass.PREVALIDATION
        raise ValueError(f"Unknown ClassifierActionClass key: {key!r}")

    @staticmethod
    def from_display_value(display_value):
        """Get enum from translated display string."""
        if isinstance(display_value, ClassifierActionClass):
            return display_value
        display_value = (display_value or "").strip()
        for member in ClassifierActionClass:
            if member.get_display_value() == display_value:
                return member
        raise ValueError(f"Unknown ClassifierActionClass display value: {display_value!r}")



class ClassifierActionType(Enum):
    SKIP = "SKIP"
    HIDE = "HIDE"
    NOTIFY = "NOTIFY"
    MOVE = "MOVE"
    COPY = "COPY"
    DELETE = "DELETE"
    ADD_MARK = "ADD_MARK"

    def is_cache_type(self):
        # If the action is not one of these types, it should have been moved out of the directory.
        return (self == ClassifierActionType.HIDE or self == ClassifierActionType.NOTIFY 
            or self == ClassifierActionType.SKIP or self == ClassifierActionType.ADD_MARK)

    def get_translation(self):
        if self == ClassifierActionType.HIDE:
            return _("Hide")
        elif self == ClassifierActionType.NOTIFY:
            return _("Notify")
        elif self == ClassifierActionType.SKIP:
            return _("Skip")
        elif self == ClassifierActionType.MOVE:
            return _("Move")
        elif self == ClassifierActionType.COPY:
            return _("Copy")
        elif self == ClassifierActionType.DELETE:
            return _("Delete")
        elif self == ClassifierActionType.ADD_MARK:
            return _("Add Mark")
        raise Exception("Classifier action translation not found: " + str(self))

    @staticmethod
    def get_action(action_name):
        for name, value in ClassifierActionType.__members__.items():
            if action_name.upper() == name.upper() or action_name == value.get_translation():
                return value
        raise Exception("Invalid classifier action: " + action_name)


class ProtectedActions(Enum):
    """Enumeration of actions that can be password protected."""
    OPEN_APPLICATION = "start_application"
    RUN_COMPARES = "run_compares"
    RUN_SEARCH = "run_search"
    RUN_SEARCH_PRESET = "run_search_preset"
    VIEW_MEDIA_DETAILS = "view_media_details"
    VIEW_RECENT_DIRECTORIES = "view_recent_directories"
    VIEW_FILE_ACTIONS = "view_file_actions"
    RUN_FILE_ACTIONS = "run_file_actions"
    EDIT_PREVALIDATIONS = "edit_prevalidations"
    RUN_PREVALIDATIONS = "run_prevalidations"
    RUN_IMAGE_GENERATION = "run_image_generation"
    RUN_REFACDIR = "run_refacdir"
    DELETE_MEDIA = "delete_media"
    CONFIGURE_MEDIA_TYPES = "configure_media_types"
    SET_HOTKEY_ACTIONS = "set_hotkey_actions"
    ACCESS_ADMIN = "access_admin"
    
    @staticmethod
    def get_action(action_name: str):
        """Get the ProtectedActions enum value for a given action name."""
        try:
            return ProtectedActions(action_name.lower().replace(" ", "_"))
        except ValueError:
            return None

    def get_description(self):
        """Get the user-friendly description for this action."""
        descriptions = {
            ProtectedActions.OPEN_APPLICATION: _("Start Application"),
            ProtectedActions.RUN_COMPARES: _("Run Compares"),
            ProtectedActions.RUN_SEARCH: _("Run Search"),
            ProtectedActions.RUN_SEARCH_PRESET: _("Run Search Preset"),
            ProtectedActions.VIEW_MEDIA_DETAILS: _("View Media Details"),
            ProtectedActions.VIEW_RECENT_DIRECTORIES: _("View Recent Directories"),
            ProtectedActions.VIEW_FILE_ACTIONS: _("View File Actions"),
            ProtectedActions.RUN_FILE_ACTIONS: _("Run File Actions"),
            ProtectedActions.EDIT_PREVALIDATIONS: _("Edit Prevalidations"),
            ProtectedActions.RUN_PREVALIDATIONS: _("Run Prevalidations"),
            ProtectedActions.RUN_IMAGE_GENERATION: _("Run Image Generation"),
            ProtectedActions.RUN_REFACDIR: _("Run RefacDir"),
            ProtectedActions.DELETE_MEDIA: _("Delete Media"),
            ProtectedActions.CONFIGURE_MEDIA_TYPES: _("Configure Media Types"),
            ProtectedActions.SET_HOTKEY_ACTIONS: _("Set Hotkey Actions"),
            ProtectedActions.ACCESS_ADMIN: _("Access Password Administration")
        }
        return descriptions.get(self, self.value)
