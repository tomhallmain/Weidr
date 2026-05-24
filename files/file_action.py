from datetime import datetime
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
        for action_dict in action_history_dicts:
            FileAction.action_history.append(FileAction.from_dict(action_dict))
        FileAction.setup_permanent_action()
        FileAction.setup_hotkey_actions()

    @staticmethod
    def get_history_action(start_index=0, auto=False, include_deletes=False):
        # Get a previous action that is not equivalent to the permanent action if possible.
        action = None
        seen_actions = []
        for i in range(len(FileAction.action_history)):
            action = FileAction.action_history[i]
            is_returnable_action = (
                action != FileAction.permanent_action
                and (auto is None or action.auto == auto)
                and (include_deletes or not action.is_delete_action())
            )
            if not is_returnable_action or action in seen_actions:
                start_index += 1
            seen_actions.append(action)
#            logger.debug(f"i={i}, start_index={start_index}, action={action}")
            if i < start_index:
                continue
            if is_returnable_action:
                break
        return action

    @staticmethod
    def get_last_auto_file() -> Optional[str]:
        """Return the output path of the most recent system-initiated file action, or None."""
        action = FileAction.get_history_action(auto=True)
        return action.new_files[0] if action and action.new_files else None


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
    def get_action_statistics(today_only=False, kind: 'FileActionKind | None' = None):
        """
        Calculate statistics from the action history.
        Args:
            today_only: If True, only include actions performed today
            kind: If set, restrict to actions of that FileActionKind
        Returns a dictionary mapping target directories to their move/copy/delete counts.
        """
        stats = {}
        for action in FileAction.action_history:
            # Skip if filtering for today and action is not from today
            if action.auto or (today_only and not action.is_today()):
                continue
            if kind is not None and action.action_kind() != kind:
                continue

            target_dir = action.target
            if target_dir not in stats:
                stats[target_dir] = {"moved": 0, "copied": 0, "deleted": 0}

            if action.is_delete_action():
                stats[target_dir]["deleted"] += len(action.original_marks)
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

    def action_kind(self) -> FileActionKind:
        if self.is_delete_action():
            return FileActionKind.DELETE
        if self.is_move_action():
            return FileActionKind.MOVE
        return FileActionKind.COPY

    @property
    def relevant_files(self) -> list[str]:
        """Paths that identify what was affected: original_marks for deletes, new_files otherwise."""
        return self.original_marks if self.is_delete_action() else self.new_files

    def is_today(self):
        """Check if this action was performed today or within 24 hours if it's early morning."""
        if not self.timestamp:
            return False
        
        now = datetime.now()
        today = now.date()
        action_date = self.timestamp.date()
        
        # If it's the same date, it's definitely today
        if action_date == today:
            return True
        
        # If it's early morning (before 5 AM), include actions from the past 24 hours
        if now.hour < 5:
            # Check if the action was within the last 24 hours
            time_diff = now - self.timestamp
            return time_diff.total_seconds() <= 24 * 3600  # 24 hours in seconds
        
        return False

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
        else:
            return None

    @staticmethod
    def _is_matching_action_in_list(action_list, action):
        for _action in action_list:
            if action == _action:
                if len(action.new_files) != len(_action.new_files):
                    continue
                if tuple(action.new_files) == tuple(_action.new_files):
                    return True
        return False



