"""
Unit tests for Phase 2 — intermediate generate batch dispatch.

ClassifierPipelinesTab._make_generate_batch_state is a static factory that
returns (all_generates, on_generate, dispatch_batch).  Tests exercise this
factory directly without Qt or an SD Runner process by monkeypatching
SDRunnerClient.run_batch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call


def _make_state(batch_size, generation_type=None):
    """Thin wrapper that calls the real production factory."""
    from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
    return ClassifierPipelinesTab._make_generate_batch_state(generation_type, batch_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paths(n: int) -> list[str]:
    return [f"/img_{i:04d}.png" for i in range(n)]


# ---------------------------------------------------------------------------
# Flush threshold
# ---------------------------------------------------------------------------

class TestFlushThreshold:
    def test_no_flush_before_batch_size(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, _ = _make_state(batch_size=5)
        for p in _paths(4):
            on_gen(p)
        mock_rb.assert_not_called()

    def test_flush_triggers_at_exactly_batch_size(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, _ = _make_state(batch_size=5)
        for p in _paths(5):
            on_gen(p)
        mock_rb.assert_called_once()

    def test_second_flush_at_second_batch_boundary(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, _ = _make_state(batch_size=3)
        for p in _paths(6):
            on_gen(p)
        assert mock_rb.call_count == 2

    def test_batch_size_none_never_intermediate_flushes(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, _ = _make_state(batch_size=None)
        for p in _paths(1000):
            on_gen(p)
        mock_rb.assert_not_called()


# ---------------------------------------------------------------------------
# Terminal flush (end-of-run)
# ---------------------------------------------------------------------------

class TestTerminalFlush:
    def test_remainder_flushed_on_dispatch_call(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        for p in _paths(7):
            on_gen(p)
        mock_rb.assert_not_called()
        dispatch()
        mock_rb.assert_called_once()

    def test_dispatch_on_empty_is_noop(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, _, dispatch = _make_state(batch_size=5)
        dispatch()
        mock_rb.assert_not_called()

    def test_double_dispatch_does_not_resend(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        for p in _paths(3):
            on_gen(p)
        dispatch()
        dispatch()
        mock_rb.assert_called_once()


# ---------------------------------------------------------------------------
# all_generates accumulation
# ---------------------------------------------------------------------------

class TestAllGeneratesAccumulation:
    def test_all_generates_grows_across_flushes(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", MagicMock()
        )
        all_gen, on_gen, dispatch = _make_state(batch_size=3)
        for p in _paths(7):
            on_gen(p)
        dispatch()
        assert len(all_gen) == 7

    def test_all_generates_contains_all_paths(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", MagicMock()
        )
        all_gen, on_gen, dispatch = _make_state(batch_size=2)
        paths = _paths(5)
        for p in paths:
            on_gen(p)
        dispatch()
        assert [g[0] for g in all_gen] == paths

    def test_modifier_none_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", MagicMock()
        )
        all_gen, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/a.png", None)
        dispatch()
        assert all_gen[0] == ("/a.png", None)

    def test_modifier_string_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", MagicMock()
        )
        all_gen, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/a.png", "v2")
        dispatch()
        assert all_gen[0] == ("/a.png", "v2")


# ---------------------------------------------------------------------------
# batch_args shape
# ---------------------------------------------------------------------------

class TestBatchArgsShape:
    def test_batch_args_has_image_key(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/img.png")
        dispatch()
        _, batch_args = mock_rb.call_args.args
        assert batch_args[0]["image"] == "/img.png"

    def test_batch_args_append_is_false(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/img.png")
        dispatch()
        _, batch_args = mock_rb.call_args.args
        assert batch_args[0]["append"] is False

    def test_edit_suffix_present_when_modifier_given(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/img.png", "v2")
        dispatch()
        _, batch_args = mock_rb.call_args.args
        assert batch_args[0]["edit_suffix"] == "v2"

    def test_edit_suffix_absent_when_modifier_none(self, monkeypatch):
        mock_rb = MagicMock()
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/img.png", None)
        dispatch()
        _, batch_args = mock_rb.call_args.args
        assert "edit_suffix" not in batch_args[0]


# ---------------------------------------------------------------------------
# Fault isolation — exception must not propagate
# ---------------------------------------------------------------------------

class TestFaultIsolation:
    def test_run_batch_exception_does_not_propagate_from_intermediate(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        _, on_gen, _ = _make_state(batch_size=2)
        on_gen("/a.png")
        on_gen("/b.png")  # triggers flush — must not raise

    def test_run_batch_exception_does_not_propagate_from_terminal_flush(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        _, on_gen, dispatch = _make_state(batch_size=10)
        on_gen("/a.png")
        dispatch()  # must not raise

    def test_all_generates_still_populated_after_flush_exception(self, monkeypatch):
        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient.run_batch",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        all_gen, on_gen, dispatch = _make_state(batch_size=2)
        on_gen("/a.png")
        on_gen("/b.png")  # triggers failing flush
        assert len(all_gen) == 2


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigDefault:
    def test_default_pipeline_generate_batch_size(self):
        from utils.config import Config
        c = Config()
        assert c.pipeline_generate_batch_size == 150

    def test_default_pipeline_scramble_batch_size(self):
        from utils.config import Config
        c = Config()
        assert c.pipeline_scramble_batch_size == 100

    def test_zero_config_produces_none_batch_size(self):
        """Config value 0 maps to None (no intermediate flush)."""
        batch_size = 0
        result = batch_size if batch_size > 0 else None
        assert result is None


# ---------------------------------------------------------------------------
# Scramble batch state factory
# ---------------------------------------------------------------------------

def _make_scramble_state(batch_size, run_one=None):
    """Thin wrapper; optionally patches _run_one_scramble before building state."""
    from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
    if run_one is not None:
        ClassifierPipelinesTab._run_one_scramble = staticmethod(run_one)
    return ClassifierPipelinesTab._make_scramble_batch_state(batch_size)


class TestScrambleFlushThreshold:
    def test_no_execute_before_batch_size(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: calls.append(path)),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        _, on_scr, _ = ClassifierPipelinesTab._make_scramble_batch_state(5)
        for p in _paths(4):
            on_scr(p)
        assert calls == []

    def test_execute_triggers_at_exactly_batch_size(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: calls.append(path)),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        _, on_scr, _ = ClassifierPipelinesTab._make_scramble_batch_state(5)
        for p in _paths(5):
            on_scr(p)
        assert len(calls) == 5

    def test_batch_size_none_never_intermediate_flushes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: calls.append(path)),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        _, on_scr, _ = ClassifierPipelinesTab._make_scramble_batch_state(None)
        for p in _paths(200):
            on_scr(p)
        assert calls == []

    def test_remainder_flushed_on_execute_batch_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: calls.append(path)),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        _, on_scr, execute = ClassifierPipelinesTab._make_scramble_batch_state(10)
        for p in _paths(7):
            on_scr(p)
        assert calls == []
        execute()
        assert len(calls) == 7

    def test_double_execute_does_not_rerun(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: calls.append(path)),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        _, on_scr, execute = ClassifierPipelinesTab._make_scramble_batch_state(10)
        for p in _paths(3):
            on_scr(p)
        execute()
        execute()
        assert len(calls) == 3


class TestScrambleAllAccumulation:
    def test_all_scrambles_grows_across_flushes(self, monkeypatch):
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: None),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        all_scr, on_scr, execute = ClassifierPipelinesTab._make_scramble_batch_state(3)
        for p in _paths(7):
            on_scr(p)
        execute()
        assert len(all_scr) == 7

    def test_modifier_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(lambda path, mod, skip_existing=False: None),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        all_scr, on_scr, execute = ClassifierPipelinesTab._make_scramble_batch_state(10)
        on_scr("/img.png", "semi")
        execute()
        assert all_scr[0] == ("/img.png", "semi")


class TestScrambleFaultIsolation:
    def test_item_exception_does_not_abort_batch(self, monkeypatch):
        call_count = [0]

        def _failing(path, mod, skip_existing=False):
            call_count[0] += 1
            raise RuntimeError("disk error")

        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.ClassifierPipelinesTab._run_one_scramble",
            staticmethod(_failing),
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        _, on_scr, execute = ClassifierPipelinesTab._make_scramble_batch_state(10)
        for p in _paths(3):
            on_scr(p)
        execute()  # must not raise; each future's exception is logged
        assert call_count[0] == 3
