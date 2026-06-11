from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from utils.config import config
from utils.constants import CompareMode, Mode

if TYPE_CHECKING:
    from compare.compare_filters import CompareFilter


class CompareArgs:
    def __init__(self, base_dir=".", listener=None, mode=Mode.GROUP, compare_mode=CompareMode.CLIP_EMBEDDING,
                 recursive=True, searching_image=False, search_media_path=None, search_text=None, search_text_negative=None,
                 find_duplicates=False, counter_limit=config.file_counter_limit, compare_threshold=config.embedding_similarity_threshold,
                 file_filter=None, overwrite=False, store_checkpoints=config.store_checkpoints,
                 use_matrix_comparison=True, app_actions=None):
        self.base_dir = base_dir
        self.listener = listener
        self.mode = mode
        self.compare_mode = compare_mode
        self.recursive = recursive
        self.searching_image = searching_image
        self.search_media_path = search_media_path
        self.negative_search_media_path = None
        self.search_text = search_text
        self.search_text_negative = search_text_negative
        self.find_duplicates = find_duplicates
        self.counter_limit = counter_limit
        self.threshold = compare_threshold
        self.file_filter = file_filter
        self.overwrite = overwrite
        self.store_checkpoints = store_checkpoints
        self.include_videos = config.enable_videos
        self.include_gifs = config.enable_gifs
        self.include_pdfs = config.enable_pdfs
        self.match_dims = False
        self.verbose = True
        self.use_matrix_comparison = use_matrix_comparison
        self.app_actions = app_actions
        self.data_filter: Optional['CompareFilter'] = None

    def not_searching(self):
        def _empty(v):
            if v is None:
                return True
            if isinstance(v, str):
                return v.strip() == ""
            return True  # e.g. dict or other type treated as "no search"
        return (_empty(self.search_media_path) and _empty(self.search_text)
                and _empty(self.search_text_negative) and _empty(self.negative_search_media_path))

    def _is_new_data_request_required(self, other):
        return (self.threshold != other.threshold
                or self.counter_limit != other.counter_limit
                or self.file_filter != other.file_filter
                or self.recursive != other.recursive
                or (not self.overwrite and other.overwrite))

    def clone(self):
        clone = CompareArgs()
        for k, v in self.__dict__.items():
            if k not in ("listener", "app_actions"):
                clone.__dict__[k] = deepcopy(v)
        return clone
