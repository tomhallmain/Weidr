"""Model Context Protocol front end.

A front end for driving a running Weidr session -- browsing, marking,
deleting/hiding files, triggering a compare or an sd-runner image-generation
request -- from an external agent.

This module is deliberately mode-agnostic: it knows nothing about Qt or
about the headless entry point. It calls through a *session* object handed
back by ``session_resolver``, a callable supplied by whichever entry point
constructs this class. ``ui/app_window/mcp_session_qt.py`` adapts a live
``AppWindow`` to that session shape for ``app_qt.py``; ``app_headless.py``
builds one directly from a Qt-free ``FileBrowser``/``CompareManager`` pair.
Either way the session already answers every call safely -- in the Qt case
because ``AppWindow.app_actions`` entries are already wrapped onto the GUI
thread, in the headless case because there is no GUI thread to marshal onto.
This module contains no thread-marshaling of its own.

The session is resolved fresh on every ``dispatch``/``read_resource`` call,
never cached. Weidr can have zero, one, or several windows open at once, and
which one is "current" changes over time -- capturing one at construction
time would go stale exactly the way ``MarkedFileMover._current_media`` does
when the dialog it belongs to is left open across other UI activity.

Everything except :meth:`MCPServerExtension._serve` is ordinary code and is
tested as such. ``_serve`` is the only part that touches the MCP SDK, kept
small deliberately -- see its docstring.

HTTP rather than stdio, which is MCP's usual default: under stdio the
*client* launches the server as a subprocess, and a subprocess has none of
the state every callback here depends on -- a Qt window or a headless
session that belongs to an already-running process the client did not start.

**Loopback only**, and ``refuses_to_start`` explains why. When remote access
is actually wanted there are three ways out, smallest first: a reverse proxy
that terminates authentication, which needs nothing here and is the usual
answer for an HTTP service that speaks one protocol; implementing
``TokenVerifier`` plus the minimum ``AuthSettings`` the SDK insists on; or
leaving remote access out of scope. The proxy is the recommendation. This
becomes pressing the moment a headless process listens on anything but
localhost, since a machine with no display implies no trusted desktop
around it.
"""

from typing import Any, Callable, Optional

from utils.logging_setup import get_logger

logger = get_logger("mcp_server")


class MCPToolError(Exception):
    """A request that cannot be honoured, reported to the client as-is."""


def tool_descriptors() -> list:
    """The tool surface, as plain data.

    Deliberately not the SDK's decorators: expressed this way the surface can
    be asserted without the SDK installed, and ``_serve`` becomes a loop over
    it rather than a second place the tools are defined.

    Names and descriptions only. A tool's *parameters* come from its
    handler's annotations in ``_register_tools``, which is what the SDK
    reads, so restating them here would be a second answer that could
    disagree.

    The names and descriptions are protocol payload, not UI text, and are
    deliberately not translated. A tool surface that changed shape with the
    user's interface language would describe different tools to a client
    depending on a setting the client cannot see, and the names have to stay
    fixed regardless -- they are what a client calls.
    """
    return [
        {
            "name": "get_current_file",
            "description": "The path of the file currently on screen (or held by a headless session), or null if none.",
        },
        {
            "name": "go_to_file",
            "description": "Navigate to a file by name or path. Searches the current session first, opens it directly if it's a valid path on disk.",
        },
        {
            "name": "go_to_index",
            "description": "Navigate to a file by its 1-based position in the current browse listing.",
        },
        {
            "name": "next_file",
            "description": "Advance to the next file in the current browse listing.",
        },
        {
            "name": "previous_file",
            "description": "Go back to the previous file in the current browse listing.",
        },
        {
            "name": "get_index",
            "description": (
                "The current file's 1-based position in the browse listing (null if "
                "it isn't in the listing) and the listing's length. go_to_index with "
                "1 or with that length jumps to the first or last file."
            ),
        },
        {
            "name": "set_base_dir",
            "description": "Change the directory the current session is browsing.",
        },
        {
            "name": "hide_current_file",
            "description": "Hide a file from the current listing without deleting it. Defaults to the current file.",
        },
        {
            "name": "delete_file",
            "description": "Delete a file from disk (honouring the configured trash folder, if any).",
        },
        {
            "name": "list_marks",
            "description": "The current mark list. Marks are shared process-wide, not per-session.",
        },
        {
            "name": "toggle_mark",
            "description": (
                "Mark a file if it isn't marked, unmark it if it is. Defaults to the "
                "current file. Marks are shared process-wide, not per-session."
            ),
        },
        {
            "name": "go_to_mark",
            "description": (
                "Navigate to the next marked file (or the previous one, if backward "
                "is set), wrapping around at the end of the mark list."
            ),
        },
        {
            "name": "clear_marks",
            "description": "Empty the mark list. Marks are shared process-wide, not per-session.",
        },
        {
            "name": "add_marks_series",
            "description": (
                "Mark every file between the last existing mark and the current file, "
                "in file-listing order. Needs at least one existing mark to start from."
            ),
        },
        {
            "name": "move_marks",
            "description": (
                "Move (or copy) every marked file into target_dir. Successfully "
                "transferred files are removed from the mark list; files that failed "
                "or already existed at the target stay marked."
            ),
        },
        {
            "name": "convert_to_jpg",
            "description": (
                "Convert every convertible image in the current directory scope to JPG. "
                "overwrite_existing=True also re-encodes files already JPG (stripping "
                "EXIF) and overwrites existing .jpg targets; False converts only "
                "non-JPG sources and leaves existing targets alone."
            ),
        },
        {
            "name": "convert_svg_to_png",
            "description": (
                "Rasterise every SVG in the current directory scope to a PNG beside it. "
                "overwrite_existing controls whether an existing .png target is replaced."
            ),
        },
        {
            "name": "scale_images",
            "description": (
                "Scale every image in the current directory scope so its pixel count "
                "matches target_side squared (aspect ratio preserved). Modifies files "
                "in place."
            ),
        },
        {
            "name": "strip_video_metadata",
            "description": (
                "Write a metadata-stripped sibling copy of every video in the current "
                "directory scope (container tags and chapters removed, stream copy)."
            ),
        },
        {
            "name": "run_compare",
            "description": (
                "Start a GROUP-mode compare/grouping run (results land in "
                "compare_results' file_groups) with the given compare mode; in an app "
                "window this switches the window's compare mode, and is refused if it "
                "runs a composite multi-mode setup. run_mode=GROUP_COMPLEMENT (not with "
                "find_duplicates) then switches to the scanned files no group contains, "
                "in browse-listing order, in compare_results' files_matched; if every "
                "file was grouped, compare_results' run_mode stays GROUP. Returns once "
                "the run is started, not once it has finished -- poll the "
                "compare_status resource or call health_check to find out when it's done."
            ),
        },
        {
            "name": "run_search",
            "description": (
                "Start a SEARCH-mode compare run for the given search_text and/or "
                "search_media_path (positive) and search_text_negative and/or "
                "negative_search_media_path (negative) -- at least one is required. "
                "The compare mode is applied as for run_compare. "
                "Results land in compare_results' files_matched. Returns once the run "
                "is started, not once it has finished -- poll compare_status or call "
                "health_check to find out when it's done."
            ),
        },
        {
            "name": "run_image_generation",
            "description": (
                "Ask sd-runner to generate from the current file. Returns once the "
                "request is queued, not once it has finished."
            ),
        },
        {
            "name": "extract_peek_frames",
            "description": (
                "Run PEEK frame detection on the current (or given) video/GIF file and "
                "write its most essential frames as PNGs (each tagged related_image back "
                "at the source file). Requires the optional peek dependency; requires "
                "media_path (or the current file) to be a video or GIF under the "
                "configured duration limit."
            ),
        },
        {
            "name": "extract_peek_frames_batch",
            "description": (
                "Run PEEK frame detection over every video/GIF in the current directory "
                "scope, writing each file's most essential frames as PNGs (each tagged "
                "related_image back at its source). Requires the optional peek dependency."
            ),
        },
        {
            "name": "health_check",
            "description": "Whether a session is available to drive, and whether a compare is currently running.",
        },
    ]


def resource_descriptors() -> list:
    """The read-only surface, as plain data.

    Resources answer rather than act, which is the whole distinction from
    tools: a client reads these to find out what it is driving before it
    asks for anything. Expressed the same way as ``tool_descriptors`` so
    both can be asserted without the SDK installed.

    The URIs are the client's address for each one and are protocol payload,
    so they are fixed and untranslated for the same reason the tool names
    are.
    """
    return [
        {
            "name": "current_file",
            "uri": "weidr://media/current",
            "description": "The path of the file currently on screen, or null if none.",
        },
        {
            "name": "marks",
            "uri": "weidr://marks",
            "description": "The current mark list.",
        },
        {
            "name": "compare_status",
            "uri": "weidr://compare/status",
            "description": "Whether a compare is running, and the active compare mode.",
        },
        {
            "name": "compare_results",
            "uri": "weidr://compare/results",
            "description": (
                "What the last (or currently running) compare found: file_groups "
                "for a GROUP-mode run, files_matched for a SEARCH-mode run (the "
                "matches) or a GROUP_COMPLEMENT run (the ungrouped files). run_mode "
                "names which of those the session is in. Empty if no compare has run yet."
            ),
        },
    ]


class MCPServerExtension:
    """Serves the MCP tool surface over HTTP, on its own thread.

    Takes a ``session_resolver`` rather than a fixed session so it can be
    constructed once and still see whichever session is "current" at each
    call -- see the module docstring on why that matters here.
    """

    def __init__(
        self,
        session_resolver: Callable[[], Optional[Any]],
        host: str = None,
        port: int = None,
        token: str = None,
    ):
        # Resolved at call time rather than from the module-level binding:
        # tests swap in a fresh Config per test, and only a fresh attribute
        # lookup sees the swap.
        from utils.config import config as _config

        self._session_resolver = session_resolver
        self._host = host if host is not None else getattr(_config, "mcp_server_host", "localhost")
        self._port = port if port is not None else getattr(_config, "mcp_server_port", 0)
        self._token = token if token is not None else getattr(_config, "mcp_server_token", "")
        self._running = False
        self._server = None

    # ------------------------------------------------------------------
    # Authorisation
    # ------------------------------------------------------------------
    @staticmethod
    def _is_loopback(host: str) -> bool:
        return str(host or "").strip().lower() in ("", "localhost", "127.0.0.1", "::1")

    def refuses_to_start(self) -> "str | None":
        """Why this must not listen, or None when it may.

        Loopback only, for now. The SDK's own answer for authenticating a
        remote client is an OAuth resource server -- a ``token_verifier`` is
        rejected unless full ``AuthSettings`` with an issuer URL come with it
        -- which is more than a shared secret.

        Until that is built, a configured token is refused rather than
        ignored. Serving while a token sits unenforced would leave someone
        believing they are protected, which is worse than not serving.
        """
        if not self._port:
            return "no mcp_server_port configured"
        if not self._is_loopback(self._host):
            return (
                f"refusing to serve MCP on {self._host}: only a loopback bind is "
                "supported until authentication is implemented"
            )
        if self._token:
            return (
                "mcp_server_token is set but cannot be enforced yet; clear it to "
                "serve on loopback, where the port is reachable only as this user"
            )
        return None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _resolve_session(self) -> Any:
        session = self._session_resolver()
        if session is None:
            raise MCPToolError("no session available")
        return session

    def dispatch(self, tool_name: str, arguments: dict = None) -> dict:
        """Run one tool call and return what the client should see."""
        arguments = arguments or {}
        session = self._resolve_session()

        if tool_name == "get_current_file":
            return {"path": session.get_current_file()}
        if tool_name == "go_to_file":
            path = arguments.get("path")
            if not path:
                raise MCPToolError("go_to_file needs a path")
            found_path = session.go_to_file(str(path))
            return {"found": found_path is not None, "path": found_path}
        if tool_name == "go_to_index":
            index = arguments.get("index")
            if index is None:
                raise MCPToolError("go_to_index needs an index")
            try:
                found_path = session.go_to_index(int(index))
            except ValueError as e:
                raise MCPToolError(str(e))
            return {"found": found_path is not None, "path": found_path}
        if tool_name == "next_file":
            new_path = session.next_file()
            return {"advanced": new_path is not None, "path": new_path}
        if tool_name == "previous_file":
            new_path = session.previous_file()
            return {"moved": new_path is not None, "path": new_path}
        if tool_name == "get_index":
            index, count = session.get_index()
            return {"index": index, "count": count}
        if tool_name == "set_base_dir":
            path = arguments.get("path")
            if not path:
                raise MCPToolError("set_base_dir needs a path")
            session.set_base_dir(str(path))
            return {"base_dir": session.get_base_dir()}
        if tool_name == "hide_current_file":
            session.hide_current_file(arguments.get("path"))
            return {}
        if tool_name == "delete_file":
            path = arguments.get("path")
            if not path:
                raise MCPToolError("delete_file needs a path")
            session.delete_file(str(path))
            return {}
        if tool_name == "list_marks":
            return {"marks": list(session.list_marks())}
        if tool_name == "toggle_mark":
            try:
                marked, path = session.toggle_mark(arguments.get("path"))
            except ValueError as e:
                raise MCPToolError(str(e))
            return {"marked": marked, "path": path, "marks": list(session.list_marks())}
        if tool_name == "go_to_mark":
            try:
                path = session.go_to_mark(bool(arguments.get("backward", False)))
            except ValueError as e:
                raise MCPToolError(str(e))
            return {"path": path}
        if tool_name == "clear_marks":
            try:
                cleared = session.clear_marks()
            except ValueError as e:
                raise MCPToolError(str(e))
            return {"cleared": cleared, "marks": list(session.list_marks())}
        if tool_name == "add_marks_series":
            try:
                return session.add_marks_series()
            except ValueError as e:
                raise MCPToolError(str(e))
        if tool_name == "move_marks":
            target_dir = arguments.get("target_dir")
            if not target_dir:
                raise MCPToolError("move_marks needs a target_dir")
            try:
                return session.move_marks(str(target_dir), bool(arguments.get("copy", False)))
            except ValueError as e:
                raise MCPToolError(str(e))
        if tool_name == "convert_to_jpg":
            return session.convert_to_jpg(bool(arguments.get("overwrite_existing", False)))
        if tool_name == "convert_svg_to_png":
            return session.convert_svg_to_png(bool(arguments.get("overwrite_existing", False)))
        if tool_name == "scale_images":
            return session.scale_images(int(arguments.get("target_side", 320)))
        if tool_name == "strip_video_metadata":
            return session.strip_video_metadata()
        if tool_name == "run_compare":
            mode = arguments.get("mode")
            if not mode:
                raise MCPToolError("run_compare needs a mode")
            try:
                session.run_compare(
                    str(mode), bool(arguments.get("find_duplicates", False)),
                    run_mode=str(arguments.get("run_mode") or "GROUP"),
                )
            except ValueError as e:
                raise MCPToolError(str(e))
            return {"status": "started"}
        if tool_name == "run_search":
            mode = arguments.get("mode")
            if not mode:
                raise MCPToolError("run_search needs a mode")
            try:
                session.run_search(
                    str(mode),
                    search_text=arguments.get("search_text"),
                    search_text_negative=arguments.get("search_text_negative"),
                    search_media_path=arguments.get("search_media_path"),
                    negative_search_media_path=arguments.get("negative_search_media_path"),
                )
            except ValueError as e:
                raise MCPToolError(str(e))
            return {"status": "started"}
        if tool_name == "run_image_generation":
            session.run_image_generation(
                arguments.get("edit_suffix"), arguments.get("target_dir"),
            )
            return {"status": "started"}
        if tool_name == "extract_peek_frames":
            try:
                return session.extract_peek_frames(
                    media_path=arguments.get("media_path"),
                    k=arguments.get("k"),
                    fps=arguments.get("fps"),
                    target_dir=arguments.get("target_dir"),
                )
            except ValueError as e:
                raise MCPToolError(str(e))
        if tool_name == "extract_peek_frames_batch":
            return session.extract_peek_frames_batch(
                k=arguments.get("k"),
                fps=arguments.get("fps"),
                target_dir=arguments.get("target_dir"),
            )
        if tool_name == "health_check":
            return {"session_available": True, "compare_running": session.is_compare_running()}
        raise MCPToolError(f"unknown tool: {tool_name}")

    def read_resource(self, name: str) -> dict:
        """Read one resource by name. The counterpart to ``dispatch``.

        Kept separate from the tool dispatch rather than folded into it: a
        resource takes no arguments and changes nothing, and a client that
        can only read should not have to go through the surface that acts.
        """
        session = self._resolve_session()
        if name == "current_file":
            return {"path": session.get_current_file()}
        if name == "marks":
            return {"marks": list(session.list_marks())}
        if name == "compare_status":
            return {"running": session.is_compare_running(), "mode": session.get_compare_mode()}
        if name == "compare_results":
            return session.compare_results()
        raise MCPToolError(f"unknown resource: {name}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Serve until stopped. Returns False without serving when it must not.

        Called on its own thread from ``app_qt.py``; called directly (and
        left to block) from ``app_headless.py``. A missing SDK is not an
        error: this is an optional dependency and a user who never wanted
        MCP should not see a failure for it.
        """
        refusal = self.refuses_to_start()
        if refusal:
            logger.warning(f"MCP server not started: {refusal}")
            return False

        try:
            from mcp.server import MCPServer  # noqa: F401
        except ImportError:
            logger.info(
                "MCP server not started: the 'mcp' package is not installed "
                "(see requirements-optional.txt)"
            )
            return False

        self._running = True
        try:
            self._serve()
            return True
        except Exception as e:
            logger.error(f"MCP server stopped: {e}")
            return False
        finally:
            self._running = False

    def stop(self) -> None:
        """Mark it stopped. The listener itself ends with the process.

        ``MCPServer.run`` blocks and exposes no shutdown, so there is
        nothing to call. The thread it runs on is a daemon, so process exit
        ends it -- which is enough for an app-lifetime server, and would not
        be for one that needed restarting in place.
        """
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def _serve(self) -> None:
        """Build the SDK server and serve it. The only SDK-dependent code.

        ``run`` is synchronous and blocks, which is what the calling thread
        is for -- see :meth:`start`.
        """
        from mcp.server import MCPServer

        server = MCPServer("Weidr")
        self._server = server
        self._register_tools(server)
        self._register_resources(server)
        server.run(transport="streamable-http", host=self._host, port=self._port)

    def _register_tools(self, server) -> None:
        """Bind each tool to a handler whose signature *is* its schema.

        The SDK derives a tool's parameters from the handler's annotations,
        so these are written out rather than generated from a ``**kwargs``
        shim -- a shim would advertise a tool that takes no arguments, and a
        client would have no way to call it properly. Descriptions come from
        ``tool_descriptors`` so the catalogue stays the one place they are
        said.
        """
        described = {d["name"]: d["description"] for d in tool_descriptors()}

        @server.tool(name="get_current_file", description=described["get_current_file"])
        def get_current_file() -> dict:
            return self.dispatch("get_current_file")

        @server.tool(name="go_to_file", description=described["go_to_file"])
        def go_to_file(path: str) -> dict:
            return self.dispatch("go_to_file", {"path": path})

        @server.tool(name="go_to_index", description=described["go_to_index"])
        def go_to_index(index: int) -> dict:
            return self.dispatch("go_to_index", {"index": index})

        @server.tool(name="next_file", description=described["next_file"])
        def next_file() -> dict:
            return self.dispatch("next_file")

        @server.tool(name="previous_file", description=described["previous_file"])
        def previous_file() -> dict:
            return self.dispatch("previous_file")

        @server.tool(name="get_index", description=described["get_index"])
        def get_index() -> dict:
            return self.dispatch("get_index")

        @server.tool(name="set_base_dir", description=described["set_base_dir"])
        def set_base_dir(path: str) -> dict:
            return self.dispatch("set_base_dir", {"path": path})

        @server.tool(name="hide_current_file", description=described["hide_current_file"])
        def hide_current_file(path: str | None = None) -> dict:
            return self.dispatch("hide_current_file", {"path": path})

        @server.tool(name="delete_file", description=described["delete_file"])
        def delete_file(path: str) -> dict:
            return self.dispatch("delete_file", {"path": path})

        @server.tool(name="list_marks", description=described["list_marks"])
        def list_marks() -> dict:
            return self.dispatch("list_marks")

        @server.tool(name="toggle_mark", description=described["toggle_mark"])
        def toggle_mark(path: str | None = None) -> dict:
            return self.dispatch("toggle_mark", {"path": path})

        @server.tool(name="go_to_mark", description=described["go_to_mark"])
        def go_to_mark(backward: bool = False) -> dict:
            return self.dispatch("go_to_mark", {"backward": backward})

        @server.tool(name="clear_marks", description=described["clear_marks"])
        def clear_marks() -> dict:
            return self.dispatch("clear_marks")

        @server.tool(name="add_marks_series", description=described["add_marks_series"])
        def add_marks_series() -> dict:
            return self.dispatch("add_marks_series")

        @server.tool(name="move_marks", description=described["move_marks"])
        def move_marks(target_dir: str, copy: bool = False) -> dict:
            return self.dispatch("move_marks", {"target_dir": target_dir, "copy": copy})

        @server.tool(name="convert_to_jpg", description=described["convert_to_jpg"])
        def convert_to_jpg(overwrite_existing: bool = False) -> dict:
            return self.dispatch("convert_to_jpg", {"overwrite_existing": overwrite_existing})

        @server.tool(name="convert_svg_to_png", description=described["convert_svg_to_png"])
        def convert_svg_to_png(overwrite_existing: bool = False) -> dict:
            return self.dispatch("convert_svg_to_png", {"overwrite_existing": overwrite_existing})

        @server.tool(name="scale_images", description=described["scale_images"])
        def scale_images(target_side: int = 320) -> dict:
            return self.dispatch("scale_images", {"target_side": target_side})

        @server.tool(name="strip_video_metadata", description=described["strip_video_metadata"])
        def strip_video_metadata() -> dict:
            return self.dispatch("strip_video_metadata")

        @server.tool(name="run_compare", description=described["run_compare"])
        def run_compare(mode: str, find_duplicates: bool = False, run_mode: str = "GROUP") -> dict:
            return self.dispatch("run_compare", {
                "mode": mode, "find_duplicates": find_duplicates, "run_mode": run_mode,
            })

        @server.tool(name="run_search", description=described["run_search"])
        def run_search(
            mode: str,
            search_text: str | None = None,
            search_text_negative: str | None = None,
            search_media_path: str | None = None,
            negative_search_media_path: str | None = None,
        ) -> dict:
            return self.dispatch("run_search", {
                "mode": mode,
                "search_text": search_text, "search_text_negative": search_text_negative,
                "search_media_path": search_media_path,
                "negative_search_media_path": negative_search_media_path,
            })

        @server.tool(name="run_image_generation", description=described["run_image_generation"])
        def run_image_generation(edit_suffix: str | None = None, target_dir: str | None = None) -> dict:
            return self.dispatch(
                "run_image_generation", {"edit_suffix": edit_suffix, "target_dir": target_dir},
            )

        @server.tool(name="extract_peek_frames", description=described["extract_peek_frames"])
        def extract_peek_frames(
            media_path: str | None = None, k: int | None = None,
            fps: float | None = None, target_dir: str | None = None,
        ) -> dict:
            return self.dispatch("extract_peek_frames", {
                "media_path": media_path, "k": k, "fps": fps, "target_dir": target_dir,
            })

        @server.tool(name="extract_peek_frames_batch", description=described["extract_peek_frames_batch"])
        def extract_peek_frames_batch(
            k: int | None = None, fps: float | None = None, target_dir: str | None = None,
        ) -> dict:
            return self.dispatch("extract_peek_frames_batch", {
                "k": k, "fps": fps, "target_dir": target_dir,
            })

        @server.tool(name="health_check", description=described["health_check"])
        def health_check() -> dict:
            return self.dispatch("health_check")

    def _register_resources(self, server) -> None:
        """Bind each resource to a reader, by URI.

        A loop rather than one decorated function per resource: unlike a
        tool, a resource has no parameters, so there is no signature that
        would differ between them.

        The name is captured by a factory rather than by a default argument.
        The SDK reads a handler's signature as its schema -- the same thing
        that gives tools their parameters -- so a captured default would
        advertise the internal name as something a client passes in, on a
        surface that is supposed to take nothing.
        """
        for descriptor in resource_descriptors():
            server.resource(
                descriptor["uri"],
                name=descriptor["name"],
                description=descriptor["description"],
            )(self._resource_reader(descriptor["name"]))

    def _resource_reader(self, name: str):
        """A zero-argument reader bound to one resource name."""
        def read() -> dict:
            return self.read_resource(name)
        return read
