from utils.app_info_cache import app_info_cache
from utils.utils import Utils


class ActionStep:
    def __init__(self, action: str, target: str):
        self.action = action  # "move_file" or "copy_file"
        self.target = target

    def is_move(self) -> bool:
        return self.action == "move_file"

    def action_label(self) -> str:
        return "Move" if self.is_move() else "Copy"

    def action_func(self):
        return Utils.move_file if self.is_move() else Utils.copy_file

    def matches(self, other: "ActionStep") -> bool:
        return self.action == other.action and self.target == other.target

    def to_dict(self) -> dict:
        return {"action": self.action, "target": self.target}

    @staticmethod
    def from_dict(d: dict) -> "ActionStep":
        return ActionStep(d.get("action", "move_file"), d.get("target", ""))


class ActionSet:
    def __init__(self, name: str, steps: list):
        self.name = name
        self.steps = steps  # list[ActionStep]

    def summary(self) -> str:
        if not self.steps:
            return "(empty)"
        return "; ".join(f"{s.action_label()} → {s.target}" for s in self.steps)

    def to_dict(self) -> dict:
        return {"name": self.name, "steps": [s.to_dict() for s in self.steps]}

    @staticmethod
    def from_dict(d: dict) -> "ActionSet":
        steps = [ActionStep.from_dict(s) for s in d.get("steps", [])]
        return ActionSet(d.get("name", ""), steps)


class FileActionSets:
    all_actions: list = []       # list[ActionStep] — the full pool shown as cards
    selected_indices: list = []  # list[int] — indices into all_actions that are selected
    action_sets: list = []       # list[ActionSet] — saved presets

    @staticmethod
    def load() -> None:
        pool_data = app_info_cache.get_meta("file_action_set_pool", default_val=[])
        FileActionSets.all_actions = [ActionStep.from_dict(d) for d in pool_data]
        raw = app_info_cache.get_meta("file_action_set_selection", default_val=[])
        FileActionSets.selected_indices = [i for i in raw if i < len(FileActionSets.all_actions)]
        sets_data = app_info_cache.get_meta("file_action_sets", default_val=[])
        FileActionSets.action_sets = [ActionSet.from_dict(d) for d in sets_data]

    @staticmethod
    def store() -> None:
        app_info_cache.set_meta("file_action_set_pool", [a.to_dict() for a in FileActionSets.all_actions])
        app_info_cache.set_meta("file_action_set_selection", FileActionSets.selected_indices)
        app_info_cache.set_meta("file_action_sets", [s.to_dict() for s in FileActionSets.action_sets])

    @staticmethod
    def get_selected_actions() -> list:
        return [FileActionSets.all_actions[i] for i in FileActionSets.selected_indices]

    @staticmethod
    def selected_move_index() -> int:
        """Return the pool index of the currently selected move action, or -1."""
        for i in FileActionSets.selected_indices:
            if FileActionSets.all_actions[i].is_move():
                return i
        return -1

    @staticmethod
    def add_to_pool(action: str, target: str) -> int:
        """Add an action to the pool if not already present. Returns its index."""
        for i, existing in enumerate(FileActionSets.all_actions):
            if existing.action == action and existing.target == target:
                return i
        FileActionSets.all_actions.append(ActionStep(action, target))
        return len(FileActionSets.all_actions) - 1
