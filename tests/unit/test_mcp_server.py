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
        self.calls = []

    def get_current_file(self):
        return self.current_file

    def next_file(self):
        self.calls.append("next_file")
        self.current_file = "/base/b.png"
        return self.current_file

    def go_to_file(self, path):
        self.calls.append(("go_to_file", path))
        if path == "missing.png":
            return None
        self.current_file = path
        return path

    def go_to_index(self, index):
        self.calls.append(("go_to_index", index))
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

    def delete_file(self, path):
        self.calls.append(("delete_file", path))

    def hide_current_file(self, path):
        self.calls.append(("hide_current_file", path))

    def run_compare(self, mode, find_duplicates):
        if mode == "NOT_A_REAL_MODE":
            raise ValueError(f"unknown compare mode: {mode}")
        self.calls.append(("run_compare", mode, find_duplicates))
        self.compare_running = True

    def is_compare_running(self):
        return self.compare_running

    def get_compare_mode(self):
        return self.compare_mode

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

    def test_go_to_index_requires_index(self):
        ext, _ = _extension()
        with pytest.raises(MCPToolError):
            ext.dispatch("go_to_index", {})

    def test_next_file(self):
        ext, session = _extension()
        result = ext.dispatch("next_file")
        assert result == {"advanced": True, "path": "/base/b.png"}
        assert "next_file" in session.calls

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

    def test_run_compare(self):
        ext, session = _extension()
        result = ext.dispatch("run_compare", {"mode": "GROUP", "find_duplicates": True})
        assert result == {"status": "started"}
        assert ("run_compare", "GROUP", True) in session.calls

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
