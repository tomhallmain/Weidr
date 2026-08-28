from datetime import datetime, timedelta
import os
from typing import Optional

from utils.config import config
from utils.constants import FileActionKind
from utils.utils import Utils
from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("file_action")


def _delete_file_sentinel(path: str, *args, **kwargs) -> str:
    return path


# Names of the manually-invoked image ops FileAction can record
# (crop/rotate/enhance/scramble/flip/etc.). Classification goes through these
# names rather than the registry below so that asking "is this an image op?"
# -- done for every entry by action_kind(), get_action_statistics(), and
# get_history_action() -- never has to import anything. Must stay in sync with
# _image_op_registry()'s keys; it asserts that.
_IMAGE_OP_NAMES = frozenset({
    "rotate_image",
    "enhance_image",
    "flip_image",
    "change_aspect_ratio",
    "convert_to_jpg",
    "scramble_image",
    "semi_scramble_image",
    "random_crop_and_upscale",
    "randomly_modify_image",
    "crop_image_to_rect",
    "crop_image_to_polygon",
    "draw_box_at_rect",
    "draw_box_at_polygon",
    "draw_background_box_at_rect",
    "draw_background_box_at_polygon",
    "smart_crop_multi_detect",
    "copy_without_exif",
})

_image_op_registry_cache: Optional[dict] = None


def _image_op_registry() -> dict:
    """Lazily built name -> function map, for resolving a persisted action name
    back to the function that produced it (from_dict, at startup).

    Built lazily and only on that path: resolving it pulls in image/smart_crop
    and therefore scipy and sklearn, which callers that only deal in
    move/copy/delete should never pay for.
    """
    global _image_op_registry_cache
    if _image_op_registry_cache is None:
        from image.image_data_extractor import image_data_extractor
        from image.image_ops import ImageOps
        from image.smart_crop import Cropper
        _image_op_registry_cache = {
            "rotate_image": ImageOps.rotate_image,
            "enhance_image": ImageOps.enhance_image,
            "flip_image": ImageOps.flip_image,
            "change_aspect_ratio": ImageOps.change_aspect_ratio,
            "convert_to_jpg": ImageOps.convert_to_jpg,
            "scramble_image": ImageOps.scramble_image,
            "semi_scramble_image": ImageOps.semi_scramble_image,
            "random_crop_and_upscale": ImageOps.random_crop_and_upscale,
            "randomly_modify_image": ImageOps.randomly_modify_image,
            "crop_image_to_rect": ImageOps.crop_image_to_rect,
            "crop_image_to_polygon": ImageOps.crop_image_to_polygon,
            "draw_box_at_rect": ImageOps.draw_box_at_rect,
            "draw_box_at_polygon": ImageOps.draw_box_at_polygon,
            "draw_background_box_at_rect": ImageOps.draw_background_box_at_rect,
            "draw_background_box_at_polygon": ImageOps.draw_background_box_at_polygon,
            "smart_crop_multi_detect": Cropper.smart_crop_multi_detect,
            "copy_without_exif": image_data_extractor.copy_without_exif,
        }
        assert set(_image_op_registry_cache) == _IMAGE_OP_NAMES, (
            "image op registry and _IMAGE_OP_NAMES disagree: "
            f"{set(_image_op_registry_cache) ^ _IMAGE_OP_NAMES}"
        )
    return _image_op_registry_cache


class FileAction():
    MAX_ACTIONS = config.file_actions_history_max
    MAX_ACTION_ROWS = config.file_actions_window_rows_max

    action_history: list['FileAction'] = []

    permanent_action: Optional['FileAction'] = None
    hotkey_actions: dict[int, 'FileAction'] = {}


    @staticmethod
    def setup_permanent_action():
        permanent_mark_target = app_info_cache.get_meta("permanent_mark_target")
        permanent_action = app_info_cache.get_meta("permanent_action")
        if permanent_action and permanent_mark_target:
            FileAction.permanent_action = FileAction(
                FileAction.convert_action_from_text(permanent_action), permanent_mark_target
            )

    @staticmethod
    def setup_hotkey_actions():
        hotkey_actions_dict = app_info_cache.get_meta("hotkey_actions", default_val={})
        assert type(hotkey_actions_dict) == dict
        hotkey_actions = {}
        for number, action in hotkey_actions_dict.items():
            hotkey_actions[int(number)] = FileAction(
                FileAction.convert_action_from_text(action["action"]), action["target"]
            )
        FileAction.hotkey_actions = hotkey_actions

    @staticmethod
    def store_actions():
        action_dicts = []
        for action in FileAction.action_history:
            action_dicts.append(action.to_dict())
        app_info_cache.set_meta("file_actions", action_dicts)

        # Persist permanent action
        if FileAction.permanent_action is not None:
            app_info_cache.set_meta("permanent_action", FileAction.permanent_action.action.__name__)
            app_info_cache.set_meta("permanent_mark_target", FileAction.permanent_action.target)

        # Persist hotkey actions
        hotkey_actions_dict = {}
        for number, action in FileAction.hotkey_actions.items():
            hotkey_actions_dict[number] = {"action": action.action.__name__, "target": action.target}
        app_info_cache.set_meta("hotkey_actions", hotkey_actions_dict)
    
    @staticmethod
    def load_actions():
        action_history_dicts = app_info_cache.get_meta("file_actions", default_val=[])
        for action_dict in action_history_dicts[:FileAction.MAX_ACTIONS]:
            FileAction.action_history.append(FileAction.from_dict(action_dict))
        FileAction.setup_permanent_action()
        FileAction.setup_hotkey_actions()

    @staticmethod
    def get_history_action(start_index=0, auto=False, include_deletes=False):
        # Get a previous action that is not equivalent to the permanent action if possible.
        # Returns None when nothing qualifies -- callers feed the result straight
        # into move_marks_to_dir_static as (target_dir, move_func), so handing
        # back a non-returnable entry would run the wrong operation.
        action = None
        seen_actions = []
        for i in range(len(FileAction.action_history)):
            candidate = FileAction.action_history[i]
            is_returnable_action = (
                candidate != FileAction.permanent_action
                and (auto is None or candidate.auto == auto)
                and (include_deletes or not candidate.is_delete_action())
                # Image-op actions have no move/copy destination directory
                # (target is just the source's own directory) and don't accept
                # a move_func signature, so they're never returnable here.
                and not candidate.is_image_op_action()
            )
            if not is_returnable_action or candidate in seen_actions:
                start_index += 1
            seen_actions.append(candidate)
#            logger.debug(f"i={i}, start_index={start_index}, candidate={candidate}")
            if i < start_index:
                continue
            if is_returnable_action:
                action = candidate
                break
        return action

    @staticmethod
    def get_last_auto_file() -> Optional[str]:
        """Return the output path of the most recent system-initiated file action, or None."""
        action = FileAction.get_history_action(auto=True)
        return action.new_files[0] if action and action.new_files else None

    @staticmethod
    def was_recently_actioned(path: str, auto: Optional[bool] = None, max_actions: int = 100) -> bool:
        """True when *path*'s filename matches the FIRST new file of a recent
        move/copy action.

        Used to exempt directly-requested media from advisory prevalidation:
        a file that was recently moved or copied out of a context was
        displayable there (manual moves) or already skipped once (auto moves).

        - Delete actions are not under consideration. Neither are image ops:
          an in-place edit output was never moved or copied into a context,
          so it carries none of the history this exemption relies on.
        - Multi-file actions only check their first file (bounds the scan to
          one comparison per action regardless of transfer size).
        - Only the filename is compared, not the full path.
        - *auto*: None matches any initiator; True/False filters by initiator.
        """
        filename = os.path.normcase(os.path.basename(path))
        if not filename:
            return False
        for action in FileAction.action_history[:max_actions]:
            if auto is not None and bool(action.auto) != auto:
                continue
            if action.is_delete_action() or action.is_image_op_action():
                continue
            if not action.new_files:
                continue
            if os.path.normcase(os.path.basename(action.new_files[0])) == filename:
                return True
        return False


    @staticmethod
    def set_permanent_action(target_dir, move_func, toast_callback):
        FileAction.permanent_action = FileAction(move_func, target_dir, timestamp=datetime.now())
        app_info_cache.set_meta("permanent_action", move_func.__name__)
        app_info_cache.set_meta("permanent_mark_target", target_dir)
        toast_callback(f"Set permanent action:\n{move_func.__name__} to {target_dir}")


    @staticmethod
    def set_hotkey_action(number, target_dir, move_func, toast_callback):
        FileAction.hotkey_actions[number] = FileAction(move_func, target_dir, timestamp=datetime.now())
        hotkey_actions = app_info_cache.get_meta("hotkey_actions", default_val={})
        assert type(hotkey_actions) == dict
        hotkey_actions[number] = {"action": move_func.__name__, "target": target_dir}
        app_info_cache.set_meta("hotkey_actions", hotkey_actions)


    @staticmethod
    def update_history(latest_action):
        FileAction.action_history.insert(0, latest_action)
        if len(FileAction.action_history) > FileAction.MAX_ACTIONS:
            del FileAction.action_history[-1]

    @staticmethod
    def add_delete_action(
        deleted_path: str,
        *,
        rest_path: str | None = None,
        auto: bool = False,
    ) -> None:
        source_dir = os.path.normpath(os.path.dirname(deleted_path))
        new_files = [rest_path] if rest_path else []
        entry = FileAction(
            _delete_file_sentinel,
            source_dir,
            [deleted_path],
            new_files,
            auto,
        )
        FileAction.update_history(entry)

    @staticmethod
    def add_file_action(action, source, target, auto=True, overwrite_existing=False):
        # Use lock to ensure thread-safe file operations
        with Utils.file_operation_lock:
            new_filepath = str(action(source, target, overwrite_existing=overwrite_existing))
        logger.info("Moved file to " + new_filepath)
        new_action = FileAction(action, target, [source], [new_filepath], auto)
        FileAction.update_history(new_action)

    @staticmethod
    def add_image_op_action(op, source: str, new_files) -> None:
        """Record a manual image-ops invocation in the file action history.

        *op* must be named in _IMAGE_OP_NAMES so it round-trips through
        to_dict/from_dict. *new_files* is one output path or a list (smart crop
        can produce several).

        Only for user-initiated edits: classifier-pipeline image ops are
        automated triage, not user file operations, and must not be recorded.

        An output equal to *source* means the op did nothing
        (ImageOps.convert_to_jpg returns its input for a JPG with no EXIF) and
        is dropped -- undo deletes new_files, so recording it would arm a
        delete of the original.
        """
        if isinstance(new_files, str):
            new_files = [new_files]
        source_key = os.path.normcase(os.path.abspath(source))
        new_files = [
            f for f in new_files
            if f and os.path.exists(f)
            and os.path.normcase(os.path.abspath(f)) != source_key
        ]
        if not new_files:
            return
        target = os.path.dirname(os.path.abspath(source))
        new_action = FileAction(op, target, [source], new_files, auto=False)
        FileAction.update_history(new_action)

    @staticmethod
    def get_action_statistics(today_only=False, kind: 'FileActionKind | None' = None, auto: 'bool | None' = False):
        """
        Calculate statistics from the action history.
        Args:
            today_only: If True, only include actions performed today
            kind: If set, restrict to actions of that FileActionKind
            auto: None = all initiators, False = user only, True = auto only
        Returns a dictionary mapping target directories to their move/copy/delete counts.
        """
        stats = {}
        for action in FileAction.action_history:
            if today_only and not action.is_today():
                continue
            if auto is not None and action.auto != auto:
                continue
            if kind is not None and action.action_kind() != kind:
                continue

            target_dir = action.target
            if target_dir not in stats:
                stats[target_dir] = {"moved": 0, "copied": 0, "deleted": 0, "image_ops": 0}

            if action.is_delete_action():
                stats[target_dir]["deleted"] += len(action.original_marks)
            elif action.is_image_op_action():
                stats[target_dir]["image_ops"] += len(action.new_files)
            elif action.is_move_action():
                stats[target_dir]["moved"] += len(action.new_files)
            else:
                stats[target_dir]["copied"] += len(action.new_files)

        # Add total count for each directory
        for target_dir in stats:
            stats[target_dir]["total"] = (
                stats[target_dir]["moved"]
                + stats[target_dir]["copied"]
                + stats[target_dir]["deleted"]
                + stats[target_dir]["image_ops"]
            )
        
        return stats

    def __init__(self, action, target, original_marks=[], new_files=[], auto=False, timestamp=None):
        self.action = action
        self.target = target
        self.original_marks = original_marks[:]
        self.new_files = new_files[:]
        self.auto = auto
        self.timestamp = timestamp or datetime.now()

    def add_file(self, file):
        self.new_files.append(file)

    def get_original_directory(self):
        if len(self.original_marks) == 0:
            raise Exception("No original marks")
        return os.path.dirname(os.path.abspath(self.original_marks[-1]))

    def is_move_action(self):
        return self.action is not None and self.action.__name__.startswith("move")

    def is_delete_action(self) -> bool:
        return self.action is _delete_file_sentinel

    def is_image_op_action(self) -> bool:
        return getattr(self.action, "__name__", None) in _IMAGE_OP_NAMES

    def action_kind(self) -> FileActionKind:
        if self.is_delete_action():
            return FileActionKind.DELETE
        if self.is_image_op_action():
            return FileActionKind.IMAGE_OP
        if self.is_move_action():
            return FileActionKind.MOVE
        return FileActionKind.COPY

    @property
    def relevant_files(self) -> list[str]:
        """Paths that identify what was affected: original_marks for deletes, new_files otherwise."""
        return self.original_marks if self.is_delete_action() else self.new_files

    _DAY_START_HOUR = 5  # "today" begins at 5 AM; actions before this belong to the previous session

    def is_today(self):
        """Check if this action falls within the current session day (since 5 AM today, or since 5 AM yesterday if it's still before 5 AM)."""
        if not self.timestamp:
            return False

        now = datetime.now()
        if now.hour < self._DAY_START_HOUR:
            day_start = (now - timedelta(days=1)).replace(
                hour=self._DAY_START_HOUR, minute=0, second=0, microsecond=0
            )
        else:
            day_start = now.replace(hour=self._DAY_START_HOUR, minute=0, second=0, microsecond=0)
        return self.timestamp >= day_start

    def any_new_files_exist(self):
        for file in self.new_files:
            if os.path.exists(file):
                return True
        return False

    def remove_new_files(self):
        for f in self.new_files[:]:
            try:
                os.remove(f)
            except Exception as e:
                logger.error(e)

    def get_action(self, do_flip=False):
        action = self.action
        if do_flip:
            if action == Utils.move_file:
                action = Utils.copy_file
            elif action == Utils.copy_file:
                action = Utils.move_file
        return self.action
    
    def to_dict(self):
        return {
            "action": FileAction.convert_action_to_text(self.action),
            "target": self.target,
            "original_marks": self.original_marks[:],
            "new_files": self.new_files[:],
            "auto": self.auto,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
            }

    @staticmethod
    def from_dict(dct):
        timestamp = None
        if "timestamp" in dct and dct["timestamp"]:
            try:
                timestamp = datetime.fromisoformat(dct["timestamp"])
            except ValueError:
                # Fallback for old format or invalid timestamps
                timestamp = None
        
        return FileAction(FileAction.convert_action_from_text(dct["action"]),
                      dct["target"], dct["original_marks"][:], dct["new_files"][:],
                      dct["auto"] if "auto" in dct else False, timestamp)

    def __eq__(self, other):
        if not isinstance(other, FileAction):
            return False
        return self.action == other.action and self.target == other.target
    
    def __hash__(self):
        return hash((self.action, self.target))

    def __str__(self):
        if self.action is None:
            return "unknown action to " + self.target
        if self.is_delete_action():
            return "delete_file from " + self.target
        return self.action.__name__ + " to " + self.target

    @staticmethod
    def convert_action_from_text(action_text):
        if action_text == "move_file":
            return Utils.move_file
        elif action_text == "copy_file":
            return Utils.copy_file
        elif action_text == "delete_file":
            return _delete_file_sentinel
        elif action_text in _IMAGE_OP_NAMES:
            return _image_op_registry().get(action_text)
        else:
            return None

    @staticmethod
    def convert_action_to_text(action_func):
        if action_func == Utils.move_file:
            return "move_file"
        elif action_func == Utils.copy_file:
            return "copy_file"
        elif action_func is _delete_file_sentinel:
            return "delete_file"
        name = getattr(action_func, "__name__", None)
        return name if name in _IMAGE_OP_NAMES else None

    @staticmethod
    def _is_matching_action_in_list(action_list, action):
        for _action in action_list:
            if action == _action:
                if len(action.new_files) != len(_action.new_files):
                    continue
                if tuple(action.new_files) == tuple(_action.new_files):
                    return True
        return False



