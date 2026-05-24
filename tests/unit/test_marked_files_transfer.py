"""Unit tests for marks-transfer session locking."""

from files.marked_files import MarkedFiles


def test_guard_mark_mutation_allows_when_idle():
    MarkedFiles.is_performing_action = False
    assert MarkedFiles.guard_mark_mutation() is True


def test_guard_mark_mutation_blocks_during_transfer():
    MarkedFiles.is_performing_action = True
    try:
        assert MarkedFiles.guard_mark_mutation() is False
        assert MarkedFiles.add_mark_if_not_present("/tmp/extra.jpg") is False
    finally:
        MarkedFiles.is_performing_action = False


def test_apply_file_marks_clears_successful_and_keeps_failed():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/c.jpg"]
    MarkedFiles._apply_file_marks_after_transfer(
        files_to_move=["/dir/a.jpg", "/dir/c.jpg"],
        exceptions={"/dir/c.jpg": ("error", "/target/c.jpg")},
        invalid_files=[],
        single_image=False,
    )
    assert MarkedFiles.file_marks == ["/dir/c.jpg"]


def test_is_transfer_running_aliases_is_performing_action():
    MarkedFiles.is_performing_action = True
    try:
        assert MarkedFiles.is_transfer_running() is True
    finally:
        MarkedFiles.is_performing_action = False
