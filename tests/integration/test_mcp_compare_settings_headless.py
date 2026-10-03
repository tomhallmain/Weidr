"""Compare settings over MCP, end to end with no Qt: MCPServerExtension's
dispatch/read_resource against a real HeadlessMCPSession, running real
compares on generated images.

Only model-free modes run (COLOR_MATCHING, SIZE); CLIP_EMBEDDING appears only
as the mode a threshold is set for, never as one that runs.
"""

import os
import time

import pytest
from PIL import Image

import app_headless
from extensions.mcp_server import MCPServerExtension, MCPToolError
from tests.fixtures.compare_image_fixtures import compare_colors_dir  # noqa: F401
from utils.config import config


def _png(path, color, size=(48, 48)) -> str:
    Image.new("RGB", size, color).save(str(path), format="PNG")
    return str(path)


def _server(base_dir):
    session = app_headless.HeadlessMCPSession(str(base_dir))
    ext = MCPServerExtension(session_resolver=lambda: session, host="localhost", port=6100, token="")
    return ext, session


def _wait_for_compare(ext, timeout=60.0) -> None:
    # The run happens on the session's background runner.
    deadline = time.monotonic() + timeout
    while ext.read_resource("compare_status")["running"]:
        if time.monotonic() > deadline:
            pytest.fail("compare did not finish in time")
        time.sleep(0.05)


def _groups(ext) -> list:
    return [set(group) for group in ext.read_resource("compare_results")["file_groups"].values()]


class TestThresholdBelongsToItsMode:
    def test_a_threshold_set_for_another_mode_does_not_reach_the_run(self, compare_colors_dir):
        """0.9 set for CLIP_EMBEDDING must not become COLOR_MATCHING's LAB
        distance when run_compare switches modes; COLOR_MATCHING then runs
        with its own default and the colour families group."""
        ext, _session = _server(compare_colors_dir["dir"])
        ext.dispatch("set_compare_settings", {"settings": {
            "threshold": 0.9, "threshold_mode": "CLIP_EMBEDDING", "store_checkpoints": False,
        }})

        ext.dispatch("run_compare", {"mode": "COLOR_MATCHING"})
        _wait_for_compare(ext)

        settings = ext.read_resource("compare_settings")
        assert settings["compare_mode"] == "COLOR_MATCHING"
        assert (settings["threshold"], settings["threshold_mode"]) == (0.9, "CLIP_EMBEDDING")
        assert settings["effective_threshold"] == config.color_diff_threshold
        groups = _groups(ext)
        assert len(groups) >= 3, f"expected the three colour families to group, got {groups}"
        family_of = {p: name for name in ("red", "blue", "green") for p in compare_colors_dir[name]}
        for group in groups:
            assert len({family_of[p] for p in group if p in family_of}) <= 1

    def test_a_threshold_prepared_for_the_next_mode_is_used(self, compare_colors_dir):
        """threshold_mode lets a threshold be set before run_compare switches
        to that mode. A LAB distance of 0 groups none of the families, whose
        members all differ slightly."""
        ext, _session = _server(compare_colors_dir["dir"])
        ext.dispatch("set_compare_settings", {"settings": {
            "threshold": 0, "threshold_mode": "COLOR_MATCHING", "store_checkpoints": False,
        }})

        ext.dispatch("run_compare", {"mode": "COLOR_MATCHING"})
        _wait_for_compare(ext)

        assert ext.read_resource("compare_settings")["effective_threshold"] == 0
        family_files = set(compare_colors_dir["red"])
        assert not any(len(group & family_files) > 1 for group in _groups(ext))


class TestCompositeSetup:
    @pytest.fixture
    def images(self, tmp_path):
        d = tmp_path / "media"
        d.mkdir()
        return {
            "dir": str(d),
            "red": [_png(d / "red_a.png", (220, 0, 0)), _png(d / "red_b.png", (230, 0, 0))],
            # Red like the others, but another size: COLOR_MATCHING groups
            # it with them, SIZE does not.
            "red_large": _png(d / "red_large.png", (225, 0, 0), size=(96, 96)),
            "blue": [_png(d / "blue_a.png", (0, 0, 220)), _png(d / "blue_b.png", (0, 0, 230))],
        }

    def _configure(self, ext):
        return ext.dispatch("set_compare_settings", {"settings": {
            "instances": [
                {"compare_mode": "COLOR_MATCHING"},
                {"compare_mode": "SIZE", "threshold": 0},
            ],
            "combination_logic": "AND",
            "store_checkpoints": False,
        }})

    def test_configured_over_mcp_and_reported(self, images):
        ext, _session = _server(images["dir"])
        result = self._configure(ext)
        assert result["composite"] is True
        assert [i["compare_mode"] for i in result["instances"]] == ["COLOR_MATCHING", "SIZE"]
        assert result["combination_logic"] == "AND"
        assert ext.read_resource("compare_settings") == result

    def test_group_run_without_mode_combines_with_and(self, images):
        """COLOR_MATCHING groups all three red images; SIZE (tolerance 0)
        groups the same-size ones; AND leaves the larger red image out."""
        ext, session = _server(images["dir"])
        self._configure(ext)

        ext.dispatch("run_compare", {})
        _wait_for_compare(ext)

        groups = _groups(ext)
        assert set(images["red"]) in groups, f"red pair not grouped: {groups}"
        assert not any(images["red_large"] in group for group in groups)
        assert session.get_compare_settings()["composite"] is True

    def test_search_runs_without_mode_and_combines_with_and(self, images):
        """COLOR_MATCHING finds every red image; SIZE (tolerance 0) only those
        of the search image's size; AND keeps the red image of that size."""
        ext, _session = _server(images["dir"])
        self._configure(ext)

        ext.dispatch("run_search", {"search_media_path": images["red"][0]})
        _wait_for_compare(ext)

        matched = ext.read_resource("compare_results")["files_matched"]
        assert images["red"][1] in matched, f"same-size red image missing: {matched}"
        assert images["red_large"] not in matched
        assert not set(images["blue"]) & set(matched)

    def test_group_run_of_two_group_modes(self, images):
        """A composite GROUP run end to end; whatever AND keeps, no group
        mixes red and blue."""
        ext, _session = _server(images["dir"])
        ext.dispatch("set_compare_settings", {"settings": {
            "instances": [{"compare_mode": "COLOR_MATCHING"}, {"compare_mode": "COLOR_HISTOGRAM"}],
            "combination_logic": "AND",
            "store_checkpoints": False,
        }})

        ext.dispatch("run_compare", {})
        _wait_for_compare(ext)

        blue = set(images["blue"])
        for group in _groups(ext):
            assert not (group & blue and group - blue), f"group mixes colours: {group}"

    def test_run_with_a_mode_is_refused_and_keeps_the_setup(self, images):
        ext, _session = _server(images["dir"])
        self._configure(ext)

        with pytest.raises(MCPToolError):
            ext.dispatch("run_compare", {"mode": "COLOR_MATCHING"})

        assert ext.read_resource("compare_status")["running"] is False
        assert ext.read_resource("compare_settings")["composite"] is True

    def test_invalid_settings_change_nothing(self, images):
        ext, _session = _server(images["dir"])
        before = ext.read_resource("compare_settings")
        with pytest.raises(MCPToolError):
            ext.dispatch("set_compare_settings", {"settings": {
                "instances": [{"compare_mode": "COLOR_MATCHING"}, {"compare_mode": "SIZE"}],
                "combination_logic": "XOR",
            }})
        assert ext.read_resource("compare_settings") == before


def test_compare_runs_leave_no_checkpoint_when_disabled(compare_colors_dir):
    """store_checkpoints false over MCP reaches the run: no checkpoint file
    is written into the media directory."""
    ext, _session = _server(compare_colors_dir["dir"])
    ext.dispatch("set_compare_settings", {"settings": {"store_checkpoints": False}})
    ext.dispatch("run_compare", {"mode": "COLOR_MATCHING"})
    _wait_for_compare(ext)
    assert not [n for n in os.listdir(compare_colors_dir["dir"]) if n.startswith("weidr_result_")]
