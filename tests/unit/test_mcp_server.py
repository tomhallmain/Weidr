"""Tests for extensions/mcp_server.py.

Written against a bare fake session object rather than a real AppWindow or
headless entry point -- extensions/mcp_server.py is deliberately mode-agnostic
(see its module docstring), so nothing here needs Qt or a live process.
"""

from __future__ import annotations

import sys

import pytest

from extensions.mcp_server import (
    PASSWORD_GATES,
    MCPServerExtension,
    MCPToolError,
    resource_descriptors,
    tool_descriptors,
)


# ---------------------------------------------------------------------------
# Fake session
# ---------------------------------------------------------------------------

class _FakeSession:
    """Implements the session interface extensions/mcp_server.py calls
    through, recording what it was asked to do so tests can assert on it."""

    def __init__(self, current_file="/base/a.png", base_dir="/base"):
        self.current_file = current_file
        self.base_dir = base_dir
        self.marks = ["/base/a.png", "/base/b.png"]
        self.compare_running = False
        self.compare_mode = "GROUP"
        self.prevalidations_running = True
        self.protected = set()
        self.pipeline_runs = []
        self.has_compare_result = False
        self.run_mode = "BROWSE"
        self.file_groups = {}
        self.files_matched = []
        self.calls = []

    def get_current_file(self):
        return self.current_file

    def next_file(self):
        self.calls.append("next_file")
        self.current_file = "/base/b.png"
        return self.current_file

    def previous_file(self):
        self.calls.append("previous_file")
        self.current_file = "/base/z.png"
        return self.current_file

    def get_index(self):
        return 2, 5

    def go_to_file(self, path):
        self.calls.append(("go_to_file", path))
        if path == "missing.png":
            return None
        self.current_file = path
        return path

    def go_to_index(self, index):
        self.calls.append(("go_to_index", index))
        if index == 999:
            raise ValueError("go_to_index only works while browsing, not in compare results")
        if index < 1:
            return None
        self.current_file = f"/base/index_{index}.png"
        return self.current_file

    def extract_frames(
        self, strategy=None, media_path=None, k=None, fps=None, target_dir=None,
        action_name=None, kind=None, start_slot=0, sample_ratio=None,
    ):
        path = media_path or self.current_file
        if strategy == "nonsense":
            raise ValueError(f"Unknown frame extraction strategy: {strategy}")
        self.calls.append(("extract_frames", strategy, path, action_name))
        # None means every enabled strategy ran; name the frames for whichever
        # was asked for so a test can tell the two apart.
        return {
            "media_path": path,
            "frames_written": [f"{path}_{strategy or 'all'}.png"],
            "duplicates_skipped": 0,
            "errors": [],
        }

    def extract_frames_batch(
        self, strategy=None, k=None, fps=None, target_dir=None,
        action_name=None, kind=None, start_slot=0, sample_ratio=None,
    ):
        self.calls.append(("extract_frames_batch", strategy, action_name))
        return {
            "extracted": 2, "frames_written": 3, "failed": 0,
            "skipped": 1, "duplicates_skipped": 1,
        }

    def extract_peek_frames(self, media_path=None, k=None, fps=None, target_dir=None):
        return self.extract_frames(
            "peek", media_path=media_path, k=k, fps=fps, target_dir=target_dir,
        )

    def extract_peek_frames_batch(self, k=None, fps=None, target_dir=None):
        return self.extract_frames_batch("peek", k=k, fps=fps, target_dir=target_dir)

    def get_base_dir(self):
        return self.base_dir

    def set_base_dir(self, path):
        self.calls.append(("set_base_dir", path))
        self.base_dir = path

    def list_marks(self):
        return self.marks

    def toggle_mark(self, path):
        filepath = path if path is not None else self.current_file
        if filepath == "/locked/file.png":
            raise ValueError("marks are locked while a transfer is in progress")
        if filepath in self.marks:
            self.marks.remove(filepath)
            return False, filepath
        self.marks.append(filepath)
        return True, filepath

    def go_to_mark(self, backward):
        if not self.marks:
            raise ValueError("no marks set")
        self.mark_index = getattr(self, "mark_index", -1) + (-1 if backward else 1)
        self.mark_index %= len(self.marks)
        self.current_file = self.marks[self.mark_index]
        self.calls.append(("go_to_mark", backward))
        return self.current_file

    def clear_marks(self):
        if self.current_file == "/locked/file.png":
            raise ValueError("marks are locked while a transfer is in progress")
        cleared = len(self.marks)
        self.marks = []
        return cleared

    def add_marks_series(self):
        if not self.marks:
            raise ValueError("no existing mark to start the series from")
        self.calls.append("add_marks_series")
        added = ["/base/b.png", "/base/c.png"]
        self.marks.extend(added)
        return {"added": len(added), "marks": self.marks}

    def move_marks(self, target_dir, copy):
        if target_dir == "/no/such/dir":
            raise ValueError("a marks transfer is already in progress")
        self.calls.append(("move_marks", target_dir, copy))
        self.marks = []
        return {"already_present": False, "had_errors": False, "marks_remaining": self.marks}

    def convert_to_jpg(self, overwrite_existing):
        self.calls.append(("convert_to_jpg", overwrite_existing))
        return {"converted": 3, "failed": 0, "skipped_existing": 1}

    def convert_svg_to_png(self, overwrite_existing):
        self.calls.append(("convert_svg_to_png", overwrite_existing))
        return {"converted": 2, "failed": 0, "skipped_existing": 0}

    def scale_images(self, target_side):
        self.calls.append(("scale_images", target_side))
        return {"scaled": 4, "skipped": 1, "failed": 0}

    def strip_video_metadata(self):
        self.calls.append("strip_video_metadata")
        return {"written": 2, "failed": 0}

    def delete_file(self, path):
        self.calls.append(("delete_file", path))

    def hide_current_file(self, path):
        self.calls.append(("hide_current_file", path))

    def run_compare(self, mode, find_duplicates, run_mode="GROUP"):
        if mode == "NOT_A_REAL_MODE":
            raise ValueError(f"unknown compare mode: {mode}")
        if run_mode not in ("GROUP", "GROUP_COMPLEMENT"):
            raise ValueError(f"run_mode must be GROUP or GROUP_COMPLEMENT, not {run_mode}")
        self.calls.append(("run_compare", mode, find_duplicates, run_mode))
        self.compare_running = True

    def run_search(
        self, mode,
        search_text=None, search_text_negative=None,
        search_media_path=None, negative_search_media_path=None,
    ):
        if mode == "NOT_A_REAL_MODE":
            raise ValueError(f"unknown compare mode: {mode}")
        if not any([search_text, search_text_negative, search_media_path, negative_search_media_path]):
            raise ValueError("run_search needs at least one search input")
        self.calls.append((
            "run_search", mode,
            search_text, search_text_negative, search_media_path, negative_search_media_path,
        ))
        self.compare_running = True

    def is_compare_running(self):
        return self.compare_running

    def get_compare_mode(self):
        return self.compare_mode

    def get_prevalidations_running(self):
        return self.prevalidations_running

    def compare_results(self):
        return {
            "has_compare": self.has_compare_result,
            "run_mode": self.run_mode,
            "file_groups": self.file_groups,
            "files_matched": self.files_matched,
        }

    def run_image_generation(self, edit_suffix, target_dir):
        self.calls.append(("run_image_generation", edit_suffix, target_dir))

    def find_trigger(self, action_name, kind="classifier_action", media_path=None,
                     start_slot=0, sample_ratio=None):
        if action_name == "Missing":
            raise ValueError("no classifier_action named 'Missing'")
        self.calls.append(("find_trigger", action_name, kind, media_path, start_slot, sample_ratio))
        return {"media_kind": "still", "matched": True, "detail": None}

    def password_blocked(self, action_names):
        return next((name for name in action_names if name in self.protected), None)

    def set_prevalidations_running(self, enabled):
        self.calls.append(("set_prevalidations_running", enabled))
        self.prevalidations_running = enabled

    def run_pipeline(self, pipeline_name, profile_name=None, continue_without_sd_runner=False):
        if pipeline_name == "Missing":
            raise ValueError("no pipeline named 'Missing'")
        self.pipeline_runs.append((pipeline_name, profile_name, continue_without_sd_runner))
        return {"status": "started", "pipeline": pipeline_name, "profile": profile_name or "Photos",
                "directories": ["/p"], "warnings": []}

    def pipeline_status(self):
        return {"running": bool(self.pipeline_runs), "pipeline": None}

    def list_pipelines(self):
        return {"pipelines": [{"name": "Sorter", "is_active": True, "kind": "action",
                               "last_profile": None}], "selected_profile": "Photos"}

    def list_directory_profiles(self):
        return {"profiles": [{"name": "Photos", "directories": ["/p"]}]}

    def list_trigger_actions(self):
        return {"classifier_actions": [{"name": "Cats", "applies_to_media_types": None}], "prevalidations": []}


def _extension(session=None, **kwargs):
    kwargs.setdefault("host", "localhost")
    kwargs.setdefault("port", 6100)
    kwargs.setdefault("token", "")
    resolved = session if session is not None else _FakeSession()
    return MCPServerExtension(session_resolver=lambda: resolved, **kwargs), resolved


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

class TestDescriptors:
    def test_tool_descriptors_have_name_and_description(self):
        for tool in tool_descriptors():
            assert isinstance(tool["name"], str) and tool["name"]
            assert isinstance(tool["description"], str) and tool["description"]

    def test_resource_descriptors_have_name_uri_and_description(self):
        for resource in resource_descriptors():
            assert isinstance(resource["name"], str) and resource["name"]
            assert resource["uri"].startswith("weidr://")
            assert isinstance(resource["description"], str) and resource["description"]

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in tool_descriptors()]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# refuses_to_start
# ---------------------------------------------------------------------------

class TestRefusesToStart:
    def test_no_port_is_refused(self):
        ext, _ = _extension(port=0)
        assert ext.refuses_to_start() is not None

    def test_non_loopback_host_is_refused(self):
        ext, _ = _extension(host="0.0.0.0")
        assert ext.refuses_to_start() is not None

    def test_token_is_refused(self):
        ext, _ = _extension(token="secret")
        assert ext.refuses_to_start() is not None

    def test_loopback_with_port_and_no_token_is_allowed(self):
        ext, _ = _extension(host="127.0.0.1", port=6100, token="")
        assert ext.refuses_to_start() is None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_no_session_raises(self):
        ext = MCPServerExtension(session_resolver=lambda: None, port=6100)
        with pytest.raises(MCPToolError):
            ext.dispatch("get_current_file")

    def test_get_current_file(self):
        ext, session = _extension()
        assert ext.dispatch("get_current_file") == {"path": session.current_file}

    def test_go_to_file_found(self):
        ext, _ = _extension()
        result = ext.dispatch("go_to_file", {"path": "/base/c.png"})
        assert result == {"found": True, "path": "/base/c.png"}

    def test_go_to_file_not_found(self):
        ext, _ = _extension()
        result = ext.dispatch("go_to_file", {"path": "missing.png"})
        assert result == {"found": False, "path": None}

    def test_go_to_file_requires_path(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("go_to_file", {})

    def test_go_to_index(self):
        ext, _ = _extension()
        result = ext.dispatch("go_to_index", {"index": 3})
        assert result == {"found": True, "path": "/base/index_3.png"}

    def test_go_to_index_wraps_value_error_as_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("go_to_index", {"index": 999})

    def test_go_to_index_requires_index(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("go_to_index", {})

    def test_next_file(self):
        ext, session = _extension()
        result = ext.dispatch("next_file")
        assert result == {"advanced": True, "path": "/base/b.png"}
        assert "next_file" in session.calls

    def test_previous_file(self):
        ext, session = _extension()
        result = ext.dispatch("previous_file")
        assert result == {"moved": True, "path": "/base/z.png"}
        assert "previous_file" in session.calls

    def test_get_index(self):
        ext, _ = _extension()
        assert ext.dispatch("get_index") == {"index": 2, "count": 5}

    def test_set_base_dir(self):
        ext, session = _extension()
        result = ext.dispatch("set_base_dir", {"path": "/other"})
        assert result == {"base_dir": "/other"}
        assert ("set_base_dir", "/other") in session.calls

    def test_set_base_dir_requires_path(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("set_base_dir", {})

    def test_hide_current_file(self):
        ext, session = _extension()
        assert ext.dispatch("hide_current_file", {"path": "/base/a.png"}) == {}
        assert ("hide_current_file", "/base/a.png") in session.calls

    def test_delete_file(self):
        ext, session = _extension()
        assert ext.dispatch("delete_file", {"path": "/base/a.png"}) == {}
        assert ("delete_file", "/base/a.png") in session.calls

    def test_delete_file_requires_path(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("delete_file", {})

    def test_list_marks(self):
        ext, session = _extension()
        assert ext.dispatch("list_marks") == {"marks": session.marks}

    def test_toggle_mark_adds_when_absent(self):
        ext, session = _extension()
        result = ext.dispatch("toggle_mark", {"path": "/base/c.png"})
        assert result == {"marked": True, "path": "/base/c.png", "marks": session.marks}
        assert "/base/c.png" in session.marks

    def test_toggle_mark_removes_when_present(self):
        ext, session = _extension()
        assert "/base/a.png" in session.marks
        result = ext.dispatch("toggle_mark", {"path": "/base/a.png"})
        assert result == {"marked": False, "path": "/base/a.png", "marks": session.marks}
        assert "/base/a.png" not in session.marks

    def test_toggle_mark_defaults_to_current_file(self):
        ext, session = _extension()
        session.current_file = "/base/d.png"
        result = ext.dispatch("toggle_mark", {})
        assert result["path"] == "/base/d.png"
        assert result["marked"] is True

    def test_toggle_mark_wraps_value_error_as_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("toggle_mark", {"path": "/locked/file.png"})

    def test_go_to_mark_advances(self):
        ext, session = _extension()
        result = ext.dispatch("go_to_mark", {})
        assert result == {"path": session.marks[0]}
        assert ("go_to_mark", False) in session.calls

    def test_go_to_mark_backward(self):
        ext, session = _extension()
        result = ext.dispatch("go_to_mark", {"backward": True})
        assert ("go_to_mark", True) in session.calls
        assert result["path"] in session.marks

    def test_go_to_mark_defaults_backward_to_false(self):
        ext, session = _extension()
        ext.dispatch("go_to_mark")
        assert ("go_to_mark", False) in session.calls

    def test_go_to_mark_wraps_value_error_as_tool_error(self):
        ext, session = _extension()
        session.marks = []
        with pytest.raises(MCPToolError):
            ext.dispatch("go_to_mark", {})

    def test_clear_marks(self):
        ext, session = _extension()
        assert ext.dispatch("clear_marks") == {"cleared": 2, "marks": []}
        assert session.marks == []

    def test_clear_marks_wraps_value_error_as_tool_error(self):
        ext, _ = _extension(_FakeSession(current_file="/locked/file.png"))
        with pytest.raises(MCPToolError):
            ext.dispatch("clear_marks")

    def test_add_marks_series(self):
        ext, session = _extension()
        result = ext.dispatch("add_marks_series")
        assert result == {"added": 2, "marks": session.marks}
        assert "add_marks_series" in session.calls

    def test_add_marks_series_wraps_value_error_as_tool_error(self):
        ext, session = _extension()
        session.marks = []
        with pytest.raises(MCPToolError):
            ext.dispatch("add_marks_series")

    def test_move_marks(self):
        ext, session = _extension()
        result = ext.dispatch("move_marks", {"target_dir": "/out", "copy": True})
        assert result == {"already_present": False, "had_errors": False, "marks_remaining": []}
        assert ("move_marks", "/out", True) in session.calls

    def test_move_marks_defaults_copy_to_false(self):
        ext, session = _extension()
        ext.dispatch("move_marks", {"target_dir": "/out"})
        assert ("move_marks", "/out", False) in session.calls

    def test_move_marks_requires_target_dir(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("move_marks", {})

    def test_move_marks_wraps_value_error_as_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("move_marks", {"target_dir": "/no/such/dir"})

    def test_convert_to_jpg(self):
        ext, session = _extension()
        result = ext.dispatch("convert_to_jpg", {"overwrite_existing": True})
        assert result == {"converted": 3, "failed": 0, "skipped_existing": 1}
        assert ("convert_to_jpg", True) in session.calls

    def test_convert_to_jpg_defaults_overwrite_to_false(self):
        ext, session = _extension()
        ext.dispatch("convert_to_jpg")
        assert ("convert_to_jpg", False) in session.calls

    def test_convert_svg_to_png(self):
        ext, session = _extension()
        result = ext.dispatch("convert_svg_to_png", {"overwrite_existing": True})
        assert result == {"converted": 2, "failed": 0, "skipped_existing": 0}
        assert ("convert_svg_to_png", True) in session.calls

    def test_scale_images(self):
        ext, session = _extension()
        result = ext.dispatch("scale_images", {"target_side": 512})
        assert result == {"scaled": 4, "skipped": 1, "failed": 0}
        assert ("scale_images", 512) in session.calls

    def test_scale_images_defaults_target_side_to_320(self):
        ext, session = _extension()
        ext.dispatch("scale_images")
        assert ("scale_images", 320) in session.calls

    def test_strip_video_metadata(self):
        ext, session = _extension()
        result = ext.dispatch("strip_video_metadata")
        assert result == {"written": 2, "failed": 0}
        assert "strip_video_metadata" in session.calls

    def test_run_compare(self):
        ext, session = _extension()
        result = ext.dispatch("run_compare", {"mode": "GROUP", "find_duplicates": True})
        assert result == {"status": "started"}
        assert ("run_compare", "GROUP", True, "GROUP") in session.calls

    def test_run_compare_passes_run_mode_through(self):
        ext, session = _extension()
        ext.dispatch("run_compare", {"mode": "CLIP_EMBEDDING", "run_mode": "GROUP_COMPLEMENT"})
        assert ("run_compare", "CLIP_EMBEDDING", False, "GROUP_COMPLEMENT") in session.calls

    def test_run_compare_rejected_run_mode_is_a_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_compare", {"mode": "CLIP_EMBEDDING", "run_mode": "SEARCH"})

    def test_run_compare_requires_mode(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_compare", {})

    def test_run_compare_wraps_value_error_as_tool_error(self):
        """A session rejecting an unknown mode raises ValueError; dispatch
        must translate that into MCPToolError, not let it escape raw -- an
        MCP client only expects MCPToolError out of dispatch()."""
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_compare", {"mode": "NOT_A_REAL_MODE"})

    def test_run_search(self):
        ext, session = _extension()
        result = ext.dispatch("run_search", {"mode": "CLIP_EMBEDDING", "search_text": "cat"})
        assert result == {"status": "started"}
        assert ("run_search", "CLIP_EMBEDDING", "cat", None, None, None) in session.calls

    def test_run_search_requires_mode(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_search", {"search_text": "cat"})

    def test_run_search_wraps_value_error_as_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_search", {"mode": "CLIP_EMBEDDING"})

    def test_run_image_generation(self):
        ext, session = _extension()
        result = ext.dispatch(
            "run_image_generation", {"edit_suffix": "upscale", "target_dir": "/out"},
        )
        assert result == {"status": "started"}
        assert ("run_image_generation", "upscale", "/out") in session.calls

    def test_find_trigger_passes_defaults(self):
        ext, session = _extension()
        result = ext.dispatch("find_trigger", {"action_name": "Cats"})
        assert result == {"media_kind": "still", "matched": True, "detail": None}
        assert ("find_trigger", "Cats", "classifier_action", None, 0, None) in session.calls

    def test_find_trigger_passes_every_argument(self):
        ext, session = _extension()
        ext.dispatch("find_trigger", {
            "action_name": "NoCats", "kind": "prevalidation", "media_path": "/base/v.mp4",
            "start_slot": 4, "sample_ratio": 0.5,
        })
        assert ("find_trigger", "NoCats", "prevalidation", "/base/v.mp4", 4, 0.5) in session.calls

    def test_find_trigger_requires_action_name(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("find_trigger", {})

    def test_find_trigger_wraps_value_error_as_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("find_trigger", {"action_name": "Missing"})

    def test_list_trigger_actions(self):
        ext, _ = _extension()
        assert ext.dispatch("list_trigger_actions") == {
            "classifier_actions": [{"name": "Cats", "applies_to_media_types": None}],
            "prevalidations": [],
        }

    def test_set_prevalidations_running(self):
        ext, session = _extension()
        assert ext.dispatch("set_prevalidations_running", {"enabled": False}) == {
            "prevalidations_running": False,
        }
        assert ("set_prevalidations_running", False) in session.calls

    def test_set_prevalidations_running_requires_enabled(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("set_prevalidations_running", {})

    def test_run_pipeline_passes_defaults(self):
        ext, session = _extension()
        result = ext.dispatch("run_pipeline", {"pipeline_name": "Sorter"})
        assert result["status"] == "started"
        assert session.pipeline_runs == [("Sorter", None, False)]

    def test_run_pipeline_passes_every_argument(self):
        ext, session = _extension()
        ext.dispatch("run_pipeline", {
            "pipeline_name": "Sorter", "profile_name": "Videos", "continue_without_sd_runner": True,
        })
        assert session.pipeline_runs == [("Sorter", "Videos", True)]

    def test_run_pipeline_requires_pipeline_name(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_pipeline", {})

    def test_run_pipeline_wraps_value_error_as_tool_error(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("run_pipeline", {"pipeline_name": "Missing"})

    @pytest.mark.parametrize("tool_name, arguments", [
        ("delete_file", {"path": "/base/a.png"}),
        ("move_marks", {"target_dir": "/out"}),
        ("run_compare", {"mode": "GROUP"}),
        ("run_search", {"mode": "CLIP_EMBEDDING", "search_text": "cat"}),
        ("run_image_generation", {}),
        ("set_prevalidations_running", {"enabled": False}),
        ("run_pipeline", {"pipeline_name": "Sorter"}),
    ])
    def test_gated_tool_is_refused_while_its_password_is_required(self, tool_name, arguments):
        ext, session = _extension()
        session.protected = set(PASSWORD_GATES[tool_name])
        with pytest.raises(MCPToolError):
            ext.dispatch(tool_name, arguments)
        assert session.calls == []
        assert session.pipeline_runs == []

    def test_ungated_tool_ignores_protected_actions(self):
        ext, session = _extension()
        session.protected = {gate for gates in PASSWORD_GATES.values() for gate in gates}
        assert ext.dispatch("next_file")["advanced"] is True

    def test_password_gates_name_real_tools(self):
        names = {t["name"] for t in tool_descriptors()}
        assert set(PASSWORD_GATES) <= names

    def test_health_check(self):
        ext, session = _extension()
        session.compare_running = True
        assert ext.dispatch("health_check") == {
            "session_available": True, "compare_running": True,
        }

    def test_unknown_tool_raises(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("not_a_real_tool")

    def test_session_resolved_fresh_each_call(self):
        """A resolver returning a different session on the second call must
        have dispatch act on that one, not a value cached from construction
        or from an earlier call -- see the module docstring's note on why
        the session is never cached."""
        first = _FakeSession(current_file="/first/a.png")
        second = _FakeSession(current_file="/second/z.png")
        sessions = [first, second]
        ext = MCPServerExtension(session_resolver=lambda: sessions.pop(0), port=6100)

        assert ext.dispatch("get_current_file") == {"path": "/first/a.png"}
        assert ext.dispatch("get_current_file") == {"path": "/second/z.png"}


# ---------------------------------------------------------------------------
# read_resource
# ---------------------------------------------------------------------------

class TestReadResource:
    def test_no_session_raises(self):
        ext = MCPServerExtension(session_resolver=lambda: None, port=6100)
        with pytest.raises(MCPToolError):
            ext.read_resource("current_file")

    def test_current_file(self):
        ext, session = _extension()
        assert ext.read_resource("current_file") == {"path": session.current_file}

    def test_marks(self):
        ext, session = _extension()
        assert ext.read_resource("marks") == {"marks": session.marks}

    def test_compare_status(self):
        ext, session = _extension()
        session.compare_running = True
        session.compare_mode = "SEARCH"
        session.prevalidations_running = False
        assert ext.read_resource("compare_status") == {
            "running": True, "mode": "SEARCH", "prevalidations_running": False,
        }

    def test_compare_results_empty_before_any_run(self):
        ext, _ = _extension()
        assert ext.read_resource("compare_results") == {
            "has_compare": False, "run_mode": "BROWSE", "file_groups": {}, "files_matched": [],
        }

    def test_compare_results_after_a_group_run(self):
        ext, session = _extension()
        session.has_compare_result = True
        session.run_mode = "GROUP"
        session.file_groups = {0: {"/base/a.png": 0.0, "/base/b.png": 0.12}}
        result = ext.read_resource("compare_results")
        assert result == {
            "has_compare": True,
            "run_mode": "GROUP",
            "file_groups": {0: {"/base/a.png": 0.0, "/base/b.png": 0.12}},
            "files_matched": [],
        }

    def test_pipelines(self):
        ext, _ = _extension()
        assert ext.read_resource("pipelines")["selected_profile"] == "Photos"

    def test_directory_profiles(self):
        ext, _ = _extension()
        assert ext.read_resource("directory_profiles") == {
            "profiles": [{"name": "Photos", "directories": ["/p"]}],
        }

    def test_pipeline_status(self):
        ext, _ = _extension()
        assert ext.read_resource("pipeline_status") == {"running": False, "pipeline": None}

    def test_unknown_resource_raises(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.read_resource("not_a_real_resource")


# ---------------------------------------------------------------------------
# Frame extraction tools
# ---------------------------------------------------------------------------

class TestFrameExtractionTools:
    def test_strategy_and_current_file_reach_the_session(self):
        ext, session = _extension()

        result = ext.dispatch("extract_frames", {"strategy": "first"})

        assert ("extract_frames", "first", session.current_file, None) in session.calls
        assert result["frames_written"] == [f"{session.current_file}_first.png"]
        assert result["duplicates_skipped"] == 0

    def test_no_strategy_means_every_enabled_one(self):
        """The normal call: one pass, nothing to pick per file."""
        ext, session = _extension()

        result = ext.dispatch("extract_frames", {})

        assert session.calls[-1][1] is None
        assert result["frames_written"] == [f"{session.current_file}_all.png"]

    def test_trigger_arguments_are_passed_through(self):
        ext, session = _extension()

        ext.dispatch("extract_frames", {
            "strategy": "trigger", "action_name": "Rotate check",
            "kind": "prevalidation", "start_slot": 3, "sample_ratio": 0.5,
        })

        assert ("extract_frames", "trigger", session.current_file, "Rotate check") in session.calls

    def test_an_explicit_media_path_wins(self):
        ext, session = _extension()

        ext.dispatch("extract_frames", {"strategy": "last", "media_path": "/base/other.mp4"})

        assert ("extract_frames", "last", "/base/other.mp4", None) in session.calls

    def test_a_bad_strategy_is_reported_as_a_tool_error(self):
        ext, _ = _extension()

        with pytest.raises(MCPToolError):
            ext.dispatch("extract_frames", {"strategy": "nonsense"})

    def test_batch_returns_the_run_counts(self):
        ext, session = _extension()

        result = ext.dispatch("extract_frames_batch", {"strategy": "first"})

        assert ("extract_frames_batch", "first", None) in session.calls
        assert result == {
            "extracted": 2, "frames_written": 3, "failed": 0,
            "skipped": 1, "duplicates_skipped": 1,
        }

    def test_the_peek_tools_still_work(self):
        """The names the surface started with, now strategy='peek'."""
        ext, session = _extension()

        single = ext.dispatch("extract_peek_frames", {})
        batch = ext.dispatch("extract_peek_frames_batch", {})

        assert single["frames_written"] == [f"{session.current_file}_peek.png"]
        assert batch["extracted"] == 2

    def test_both_new_tools_are_declared(self):
        names = [t["name"] for t in tool_descriptors()]
        assert "extract_frames" in names
        assert "extract_frames_batch" in names

    def test_frame_extraction_needs_no_password(self):
        """Its GUI route is ungated, so the tools are too."""
        from extensions.mcp_server import PASSWORD_GATES
        assert "extract_frames" not in PASSWORD_GATES
        assert "extract_frames_batch" not in PASSWORD_GATES


# ---------------------------------------------------------------------------
# start() without the optional SDK installed
# ---------------------------------------------------------------------------

class TestStartWithoutSdk:
    def test_start_returns_false_when_mcp_not_installed(self, monkeypatch):
        # Force the optional import in MCPServerExtension.start() to fail
        # regardless of whether 'mcp' happens to be installed in the
        # environment running this test.
        monkeypatch.setitem(sys.modules, "mcp", None)
        ext, _ = _extension(host="127.0.0.1", port=6100, token="")
        assert ext.start() is False
        assert ext.is_running() is False

    def test_start_returns_false_when_refused(self):
        ext, _ = _extension(port=0)
        assert ext.start() is False
        assert ext.is_running() is False
