"""Tests for extensions/mcp_server.py.

Written against a bare fake session object rather than a real AppWindow or
headless entry point -- extensions/mcp_server.py is deliberately mode-agnostic
(see its module docstring), so nothing here needs Qt or a live process.
"""

from __future__ import annotations

import sys

import pytest

from extensions.mcp_server import (
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

    def compare_results(self):
        return {
            "has_compare": self.has_compare_result,
            "run_mode": self.run_mode,
            "file_groups": self.file_groups,
            "files_matched": self.files_matched,
        }

    def run_image_generation(self, edit_suffix, target_dir):
        self.calls.append(("run_image_generation", edit_suffix, target_dir))


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
        assert ext.read_resource("compare_status") == {"running": True, "mode": "SEARCH"}

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

    def test_unknown_resource_raises(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.read_resource("not_a_real_resource")


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
