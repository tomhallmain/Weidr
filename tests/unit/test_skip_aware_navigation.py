"""Tests for files/skip_aware_navigation.py.

Uses a small fake file browser rather than a real FileBrowser -- the
function under test only calls next_file()/previous_file() on it, so a
stand-in list-walker is enough and needs no real filesystem or directory
scan.
"""

from __future__ import annotations

from files.skip_aware_navigation import advance_past_skipped


class _FakeFileBrowser:
    """Cycles through a fixed file list, wrapping at either end."""

    def __init__(self, files):
        self._files = files
        self._index = 0

    def current_file(self):
        return self._files[self._index]

    def next_file(self):
        self._index = (self._index + 1) % len(self._files)
        return self._files[self._index]

    def previous_file(self):
        self._index = (self._index - 1) % len(self._files)
        return self._files[self._index]


def _no_skip(_path):
    return False


def test_returns_current_when_not_skipped(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.touch()
    b.touch()
    fb = _FakeFileBrowser([str(a), str(b)])
    start = fb.current_file()
    current = fb.next_file()
    result = advance_past_skipped(fb, _no_skip, backward=False, start=start, current=current)
    assert result == str(b)


def test_skips_a_vetoed_file(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    for f in (a, b, c):
        f.touch()
    fb = _FakeFileBrowser([str(a), str(b), str(c)])
    start = fb.current_file()
    current = fb.next_file()  # lands on b
    result = advance_past_skipped(
        fb, skip=lambda p: p == str(b), backward=False, start=start, current=current,
    )
    assert result == str(c)


def test_skips_a_missing_file(tmp_path):
    a, c = tmp_path / "a.png", tmp_path / "c.png"
    missing = tmp_path / "b.png"  # never created
    a.touch()
    c.touch()
    fb = _FakeFileBrowser([str(a), str(missing), str(c)])
    start = fb.current_file()
    current = fb.next_file()  # lands on the missing file
    result = advance_past_skipped(fb, _no_skip, backward=False, start=start, current=current)
    assert result == str(c)


def test_stops_at_wraparound_when_everything_is_skipped(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.touch()
    b.touch()
    fb = _FakeFileBrowser([str(a), str(b)])
    start = fb.current_file()
    current = fb.next_file()
    result = advance_past_skipped(
        fb, skip=lambda _p: True, backward=False, start=start, current=current,
    )
    assert result == start


def test_backward_walks_previous_file(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    for f in (a, b, c):
        f.touch()
    fb = _FakeFileBrowser([str(a), str(b), str(c)])
    fb._index = 2  # start on c
    start = fb.current_file()
    current = fb.previous_file()  # lands on b
    result = advance_past_skipped(
        fb, skip=lambda p: p == str(b), backward=True, start=start, current=current,
    )
    assert result == str(a)
