"""Compare settings and runs over MCP against a live AppWindow, end to end:
MCPServerExtension.dispatch/read_resource -> QtWindowMCPSession -> the
window's app actions -> SearchController -> CompareManager.

Only model-free modes run (COLOR_MATCHING, SIZE). The compare_colors_dir
families differ by LAB distances well under 15 within a family and well over
it across families.
"""

import pytest
from PIL import Image

from extensions.mcp_server import MCPServerExtension, MCPToolError
from tests.ui.app_window_fixtures import _teardown_app_window
from ui.app_window.app_window import AppWindow
from ui.app_window.mcp_session_qt import QtWindowMCPSession
from ui.compare.compare_settings_window_qt import CompareSettingsWindow
from utils.config import config


def _open_window(qtbot, monkeypatch, immediate_compare_debounce, base_dir):
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.set_base_dir(base_dir)
    qtbot.waitUntil(lambda: win.base_dir == base_dir, timeout=3000)
    immediate_compare_debounce(win.search_ctrl)
    # Engine alerts reach the GUI through a blocking cross-thread call; a real
    # dialog would hold the worker forever.
    monkeypatch.setitem(win.app_actions._actions, "_alert", lambda *a, **k: None)
    return win


@pytest.fixture
def colors_window(qtbot, monkeypatch, immediate_compare_debounce, compare_colors_dir):
    win = _open_window(qtbot, monkeypatch, immediate_compare_debounce, compare_colors_dir["dir"])
    yield win, compare_colors_dir
    _teardown_app_window(win)


def _server(win):
    session = QtWindowMCPSession(win)
    ext = MCPServerExtension(session_resolver=lambda: session, host="localhost", port=6100, token="")
    return ext, session


def _wait_for_compare(qtbot, session):
    qtbot.waitUntil(lambda: not session.is_compare_running(), timeout=30000)


def _groups(ext) -> list:
    return [set(g) for g in ext.read_resource("compare_results")["file_groups"].values()]


class TestThresholdBelongsToItsMode:
    def test_threshold_for_another_mode_does_not_reach_the_run(self, colors_window, qtbot):
        win, colors = colors_window
        ext, session = _server(win)
        ext.dispatch("set_compare_settings", {"settings": {
            "threshold": 0.9, "threshold_mode": "CLIP_EMBEDDING", "store_checkpoints": False,
        }})

        ext.dispatch("run_compare", {"mode": "COLOR_MATCHING"})
        _wait_for_compare(qtbot, session)

        assert ext.read_resource("compare_status")["mode"] == "COLOR_MATCHING"
        assert ext.read_resource("compare_settings")["effective_threshold"] == config.color_diff_threshold
        groups = _groups(ext)
        assert len(groups) >= 3, f"expected the three colour families to group, got {groups}"
        family_of = {p: name for name in ("red", "blue", "green") for p in colors[name]}
        for group in groups:
            assert len({family_of[p] for p in group if p in family_of}) <= 1

    def test_open_settings_window_shows_the_mcp_change(self, colors_window):
        win, _colors = colors_window
        ext, _session = _server(win)
        CompareSettingsWindow.open(parent=win, compare_manager=win.compare_manager)
        settings_win = CompareSettingsWindow._open_windows[win.compare_manager]
        try:
            ext.dispatch("set_compare_settings", {"settings": {
                "instances": [
                    {"compare_mode": "COLOR_MATCHING", "weight": 2.0},
                    {"compare_mode": "SIZE"},
                ],
                "combination_logic": "WEIGHTED",
                "counter_limit": 77,
            }})
            assert settings_win._counter_limit_edit.text() == "77"
            assert [e.text() for e in settings_win._weight_vars.values()] == ["2.0", "1.0"]

            # Apply writes the window's values back; they must be the MCP ones.
            settings_win._on_apply()
            settings = ext.read_resource("compare_settings")
            assert settings["counter_limit"] == 77
            assert [i["weight"] for i in settings["instances"]] == [2.0, 1.0]
        finally:
            CompareSettingsWindow._open_windows.pop(win.compare_manager, None)
            settings_win.close()


class TestCompositeSetup:
    @pytest.fixture
    def composite_window(self, qtbot, monkeypatch, immediate_compare_debounce, tmp_path_factory):
        d = tmp_path_factory.mktemp("composite_media")

        def png(name, color, size=(48, 48)):
            path = str(d / name)
            Image.new("RGB", size, color).save(path, format="PNG")
            return path

        images = {
            "red": [png("red_a.png", (220, 0, 0)), png("red_b.png", (230, 0, 0))],
            # Red like the others, but another size: grouped by COLOR_MATCHING,
            # not by SIZE, so AND drops it.
            "red_large": png("red_large.png", (225, 0, 0), size=(96, 96)),
            "blue": [png("blue_a.png", (0, 0, 220)), png("blue_b.png", (0, 0, 230))],
        }
        win = _open_window(qtbot, monkeypatch, immediate_compare_debounce, str(d))
        yield win, images
        _teardown_app_window(win)

    def _configure(self, ext):
        ext.dispatch("set_compare_settings", {"settings": {
            "instances": [
                {"compare_mode": "COLOR_MATCHING"},
                {"compare_mode": "SIZE", "threshold": 0},
            ],
            "combination_logic": "AND",
            "store_checkpoints": False,
        }})

    def test_group_run_without_mode_in_the_window(self, composite_window, qtbot):
        win, images = composite_window
        ext, session = _server(win)
        self._configure(ext)

        ext.dispatch("run_compare", {})
        _wait_for_compare(qtbot, session)

        groups = _groups(ext)
        assert set(images["red"]) in groups, f"red pair not grouped: {groups}"
        assert not any(images["red_large"] in g for g in groups)
        assert win.compare_manager.is_composite_mode()

    def test_search_runs_without_mode_in_the_window(self, composite_window, qtbot):
        win, images = composite_window
        ext, session = _server(win)
        self._configure(ext)

        ext.dispatch("run_search", {"search_media_path": images["red"][0]})
        _wait_for_compare(qtbot, session)

        matched = ext.read_resource("compare_results")["files_matched"]
        assert images["red"][1] in matched, f"same-size red image missing: {matched}"
        assert images["red_large"] not in matched
        assert not set(images["blue"]) & set(matched)
        assert win.compare_manager.is_composite_mode()

    def test_run_with_a_mode_is_refused(self, composite_window):
        win, _images = composite_window
        ext, session = _server(win)
        self._configure(ext)

        with pytest.raises(MCPToolError):
            ext.dispatch("run_compare", {"mode": "COLOR_MATCHING"})

        assert not session.is_compare_running()
        assert win.compare_manager.is_composite_mode()


def test_settings_change_refused_while_a_compare_runs(colors_window, monkeypatch):
    win, _colors = colors_window
    ext, session = _server(win)
    monkeypatch.setattr(session, "is_compare_running", lambda: True)
    before = ext.read_resource("compare_settings")
    with pytest.raises(MCPToolError):
        ext.dispatch("set_compare_settings", {"settings": {"counter_limit": 5}})
    assert ext.read_resource("compare_settings") == before
