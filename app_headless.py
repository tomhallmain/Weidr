#!/usr/bin/env python3
"""Weidr -- persistent headless entry point.

No QApplication, no window: this process holds one browsing/comparison
session open for its lifetime and serves it over MCP
(extensions/mcp_server.py), for an agent to drive with no GUI involved.

This is the persistent counterpart to scripts/agent_headless_demo.py, which
builds the same kind of session but throws it away after one script run.
Both rest on the same Qt-free layer: files/file_browser.py,
compare/compare_manager.py, files/marked_files.py, and
utils/headless_app_actions.py.

Run it:
    python app_headless.py --base-dir /path/to/media --port 6200
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from compare.compare_args import CompareArgs
from compare.compare_manager import CompareManager
from extensions.mcp_server import MCPServerExtension
from files.file_browser import FileBrowser
from files.marked_files import MarkedFiles
from files.skip_aware_navigation import advance_past_skipped
from utils.background_runner import ThreadedTaskRunner
from utils.config import config
from utils.constants import CompareMode, ImageGenerationType, Mode
from utils.headless_app_actions import build_headless_app_actions
from utils.logging_setup import get_logger
from utils.ui_responsiveness import NullResponsiveness

logger = get_logger("app_headless")

# Mirrors ui/image/media_details.py's MediaDetails.get_image_specific_generation_mode()
# fallback for a type that isn't one of the four "sticky" ones a Qt session
# can select interactively. There's no interactive selection here, so this is
# the only type headless generation requests use.
_DEFAULT_IMAGE_GENERATION_TYPE = ImageGenerationType.CONTROL_NET


class HeadlessMCPSession:
    """Session backed by one Qt-free FileBrowser/CompareManager pair.

    Implements the same interface extensions/mcp_server.py calls through for
    a live AppWindow (ui/app_window/mcp_session_qt.py), but directly against
    the Qt-free layer -- there is no display to update, so none of these
    methods do anything beyond the underlying state change.
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._runner = ThreadedTaskRunner()
        self._file_browser = FileBrowser(base_dir)
        self._file_browser.set_directory(base_dir)
        self._actions = build_headless_app_actions({
            "get_base_dir": lambda: self._base_dir,
            "is_compare_running": lambda: self._runner.is_running(),
        })
        self._compare_manager = CompareManager(
            master=None, app_actions=self._actions,
            get_base_dir=lambda: self._base_dir,
            responsiveness=NullResponsiveness(),
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def get_current_file(self) -> Optional[str]:
        return self._file_browser.current_file()

    def next_file(self) -> Optional[str]:
        start = self._file_browser.current_file()
        candidate = self._file_browser.next_file()
        return advance_past_skipped(
            self._file_browser, self._compare_manager.skip_media,
            backward=False, start=start, current=candidate,
        )

    def go_to_file(self, path: str) -> Optional[str]:
        return self._file_browser.find(search_text=path, exact_match=True)

    def go_to_index(self, index: int) -> Optional[str]:
        try:
            return self._file_browser.go_to_index(index)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    def get_base_dir(self) -> Optional[str]:
        return self._base_dir

    def set_base_dir(self, path: str) -> None:
        self._base_dir = path
        self._file_browser.set_directory(path)

    # ------------------------------------------------------------------
    # Marks / file operations
    # ------------------------------------------------------------------
    def list_marks(self) -> list:
        return list(MarkedFiles.file_marks)

    def toggle_mark(self, path: Optional[str]) -> "tuple[bool, str]":
        filepath = path if path is not None else self.get_current_file()
        if not filepath:
            raise ValueError("no file to mark")
        try:
            marked = MarkedFiles.toggle_mark(filepath, self._actions)
        except Exception as e:
            raise ValueError(str(e))
        return marked, filepath

    def go_to_mark(self, backward: bool) -> Optional[str]:
        try:
            marked_file, _wrapped = MarkedFiles.advance_mark_cursor(backward=backward)
        except Exception as e:
            raise ValueError(str(e))
        return self.go_to_file(marked_file)

    def delete_file(self, path: str) -> None:
        MarkedFiles.delete_file_static(path, self._actions, toast=False, manual_delete=True)

    def hide_current_file(self, path: Optional[str]) -> None:
        filepath = path if path is not None else self.get_current_file()
        if filepath is not None and filepath not in self._compare_manager.hidden_media:
            self._compare_manager.hidden_media.append(filepath)
        self.next_file()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    def run_compare(self, mode: str, find_duplicates: bool) -> None:
        try:
            compare_mode = CompareMode[mode]
        except KeyError:
            raise ValueError(f"unknown compare mode: {mode}")

        # COLOR_MATCHING reads compare_threshold as a LAB colour distance
        # rather than an embedding similarity -- see
        # scripts/agent_headless_demo.py's API notes on this same gotcha.
        threshold = (
            config.color_diff_threshold
            if compare_mode == CompareMode.COLOR_MATCHING
            else config.embedding_similarity_threshold
        )

        compare_args = CompareArgs(
            base_dir=self._base_dir,
            mode=Mode.GROUP,
            compare_mode=compare_mode,
            recursive=False,
            store_checkpoints=False,
            app_actions=self._actions,
            compare_threshold=threshold,
        )
        compare_args.find_duplicates = find_duplicates
        self._runner.start(
            self._compare_manager.run, [compare_args],
            on_error=lambda msg: logger.error("Compare run failed: %s", msg),
        )

    def is_compare_running(self) -> bool:
        return self._runner.is_running()

    def get_compare_mode(self) -> str:
        return self._compare_manager.compare_mode.name

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def run_image_generation(self, edit_suffix: Optional[str], target_dir: Optional[str]) -> None:
        from extensions.sd_runner_client import SDRunnerClient

        media_path = self.get_current_file()
        if media_path is None:
            raise ValueError("no current file to generate from")
        SDRunnerClient().run(
            _DEFAULT_IMAGE_GENERATION_TYPE, media_path,
            edit_suffix=edit_suffix, target_dir=target_dir,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a Weidr browsing/comparison session over MCP, with no GUI.",
    )
    parser.add_argument("--base-dir", required=True, help="directory to browse")
    parser.add_argument("--host", default=None, help="default: config.mcp_server_host")
    parser.add_argument("--port", type=int, default=None, help="default: config.mcp_server_port")
    parser.add_argument("--token", default=None, help="default: config.mcp_server_token")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.base_dir):
        parser.error(f"not a directory: {args.base_dir}")

    session = HeadlessMCPSession(args.base_dir)
    server = MCPServerExtension(
        session_resolver=lambda: session,
        host=args.host, port=args.port, token=args.token,
    )
    refusal = server.refuses_to_start()
    if refusal:
        logger.error("MCP server not started: %s", refusal)
        return 1

    logger.info("Serving MCP for %s", args.base_dir)
    started = server.start()
    return 0 if started else 1


if __name__ == "__main__":
    sys.exit(main())
