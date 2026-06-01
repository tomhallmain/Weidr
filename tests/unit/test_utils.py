"""
Unit tests for Utils static methods.
All methods under test are pure functions with no I/O or external dependencies.
"""

import os
import tempfile
import pytest

from utils.utils import Utils


class TestScaleDims:
    def test_image_fits_exactly(self):
        assert Utils.scale_dims((100, 200), (100, 200)) == (100, 200)

    def test_image_smaller_than_max(self):
        assert Utils.scale_dims((50, 80), (100, 200)) == (50, 80)

    def test_landscape_constrained_by_width(self):
        w, h = Utils.scale_dims((400, 200), (100, 200))
        assert w == 100
        assert h == 50

    def test_portrait_constrained_by_height(self):
        w, h = Utils.scale_dims((200, 400), (200, 100))
        assert w == 50
        assert h == 100

    def test_both_dims_over_max_width_is_binding(self):
        # 400x200, max 100x200 → scale by 0.25 (width is binding)
        w, h = Utils.scale_dims((400, 200), (100, 200))
        assert w == 100
        assert h == 50

    def test_both_dims_over_max_height_is_binding(self):
        # 200x400, max 200x100 → scale by 0.25 (height is binding)
        w, h = Utils.scale_dims((200, 400), (200, 100))
        assert w == 50
        assert h == 100

    def test_maximize_width_limited(self):
        # Image is 50x100, max is 200x200 — maximize by height gives (100, 200)
        w, h = Utils.scale_dims((50, 100), (200, 200), maximize=True)
        assert h == 200
        assert w == 100

    def test_maximize_height_limited(self):
        # Image is 100x50, max is 200x200 — maximize by width gives (200, 100)
        w, h = Utils.scale_dims((100, 50), (200, 200), maximize=True)
        assert w == 200
        assert h == 100

    def test_maximize_already_fills(self):
        # Already at max — no change
        assert Utils.scale_dims((200, 200), (200, 200), maximize=True) == (200, 200)


class TestAlphanumericSort:
    def test_pure_alpha(self):
        assert Utils.alphanumeric_sort(["b", "a", "c"]) == ["a", "b", "c"]

    def test_numeric_order(self):
        result = Utils.alphanumeric_sort(["img10.jpg", "img2.jpg", "img1.jpg"])
        assert result == ["img1.jpg", "img2.jpg", "img10.jpg"]

    def test_reverse(self):
        result = Utils.alphanumeric_sort(["img1.jpg", "img10.jpg", "img2.jpg"], reverse=True)
        assert result == ["img10.jpg", "img2.jpg", "img1.jpg"]

    def test_text_lambda(self):
        items = [("b", 2), ("a", 1), ("c", 3)]
        result = Utils.alphanumeric_sort(items, text_lambda=lambda x: x[0])
        assert result == [("a", 1), ("b", 2), ("c", 3)]

    def test_empty_list(self):
        assert Utils.alphanumeric_sort([]) == []


class TestSplit:
    def test_simple(self):
        assert Utils.split("a,b,c") == ["a", "b", "c"]

    def test_single_element(self):
        assert Utils.split("hello") == ["hello"]

    def test_escaped_delimiter(self):
        assert Utils.split(r"a\,b,c") == ["a,b", "c"]

    def test_custom_delimiter(self):
        assert Utils.split("a|b|c", delimiter="|") == ["a", "b", "c"]

    def test_empty_string(self):
        assert Utils.split("") == []

    def test_trailing_delimiter(self):
        parts = Utils.split("a,b,")
        assert parts[0] == "a"
        assert parts[1] == "b"


class TestWrapTextToFitLength:
    def test_short_string_unchanged(self):
        assert Utils._wrap_text_to_fit_length("hello", 20) == "hello"

    def test_exact_fit(self):
        assert Utils._wrap_text_to_fit_length("hello", 5) == "hello"

    def test_wraps_at_space(self):
        result = Utils._wrap_text_to_fit_length("hello world", 7)
        assert "\n" in result
        lines = result.split("\n")
        assert all(len(line) <= 7 for line in lines)

    def test_wraps_long_word_without_spaces(self):
        result = Utils._wrap_text_to_fit_length("abcdefghij", 4)
        lines = result.split("\n")
        assert all(len(line) <= 4 for line in lines)

    def test_multiline_all_lines_fit(self):
        long_text = "the quick brown fox jumps over the lazy dog"
        result = Utils._wrap_text_to_fit_length(long_text, 10)
        for line in result.split("\n"):
            assert len(line) <= 10


class TestGetCentrallyTruncatedString:
    def test_short_string_unchanged(self):
        assert Utils.get_centrally_truncated_string("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert Utils.get_centrally_truncated_string("hello", 5) == "hello"

    def test_truncated_contains_ellipsis(self):
        result = Utils.get_centrally_truncated_string("abcdefghijklmnop", 8)
        assert "..." in result

    def test_truncated_length_bounded(self):
        result = Utils.get_centrally_truncated_string("abcdefghijklmnop", 8)
        # Result should be shorter than the original
        assert len(result) < len("abcdefghijklmnop")


class TestGetRelativeDirpathSplit:
    def test_unix_style(self):
        dirpath, basename = Utils.get_relative_dirpath_split("/base", "/base/sub/file.jpg")
        assert basename == "file.jpg"
        assert dirpath == "sub"

    def test_file_in_base(self):
        dirpath, basename = Utils.get_relative_dirpath_split("/base", "/base/file.jpg")
        assert basename == "file.jpg"
        assert dirpath == ""


class TestGetRelativeDirpath:
    def test_single_level(self):
        result = Utils.get_relative_dirpath("/a/b/c", levels=1)
        assert result == "c"

    def test_two_levels(self):
        result = Utils.get_relative_dirpath("/a/b/c", levels=2)
        assert result == "b/c"

    def test_no_separator(self):
        result = Utils.get_relative_dirpath("standalone")
        assert result == "standalone"

    def test_levels_exceeds_depth(self):
        result = Utils.get_relative_dirpath("/a/b", levels=10)
        assert result == "/a/b"


class TestIsInvalidFile:
    def test_none_path_is_invalid(self):
        assert Utils.is_invalid_file(None, 1, False, None) is True

    def test_valid_path_no_pattern(self):
        assert Utils.is_invalid_file("/some/file.jpg", 1, False, None) is False

    def test_run_search_counter_zero_is_valid(self):
        assert Utils.is_invalid_file("/some/file.jpg", 0, True, None) is False

    def test_file_filter_matches(self):
        assert Utils.is_invalid_file("/some/cats/file.jpg", 1, False, "cats") is False

    def test_file_filter_no_match(self):
        assert Utils.is_invalid_file("/some/dogs/file.jpg", 1, False, "cats") is True


class TestGetValidFile:
    def test_none_returns_none(self):
        assert Utils.get_valid_file("/base", None) is None

    def test_empty_string_returns_none(self):
        assert Utils.get_valid_file("/base", "  ") is None

    def test_existing_absolute_path(self, tmp_path):
        f = tmp_path / "image.jpg"
        f.write_bytes(b"")
        assert Utils.get_valid_file(str(tmp_path), str(f)) == str(f)

    def test_relative_to_base(self, tmp_path):
        f = tmp_path / "image.jpg"
        f.write_bytes(b"")
        result = Utils.get_valid_file(str(tmp_path), "image.jpg")
        assert result is not None
        assert "image.jpg" in result

    def test_nonexistent_returns_none(self, tmp_path):
        assert Utils.get_valid_file(str(tmp_path), "ghost.jpg") is None

    def test_quoted_path_stripped(self, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(b"")
        assert Utils.get_valid_file(str(tmp_path), f'"{f}"') == str(f)


class TestParseJsonStringsInDict:
    def test_converts_json_string_value(self):
        d = {"key": '{"a": 1}'}
        result = Utils.parse_json_strings_in_dict(d)
        assert result["key"] == {"a": 1}

    def test_leaves_non_json_string_intact(self):
        d = {"key": "plain text"}
        result = Utils.parse_json_strings_in_dict(d)
        assert result["key"] == "plain text"

    def test_keys_to_check_filter(self):
        d = {"a": '{"x": 1}', "b": '{"y": 2}'}
        result = Utils.parse_json_strings_in_dict(d, keys_to_check=["a"])
        assert result["a"] == {"x": 1}
        assert result["b"] == '{"y": 2}'

    def test_does_not_mutate_original(self):
        d = {"key": '{"a": 1}'}
        Utils.parse_json_strings_in_dict(d)
        assert d["key"] == '{"a": 1}'

    def test_non_string_values_untouched(self):
        d = {"num": 42, "lst": [1, 2]}
        result = Utils.parse_json_strings_in_dict(d)
        assert result["num"] == 42
        assert result["lst"] == [1, 2]


class TestCalculateHash:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        h1 = Utils.calculate_hash(str(f))
        h2 = Utils.calculate_hash(str(f))
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert Utils.calculate_hash(str(f1)) != Utils.calculate_hash(str(f2))

    def test_returns_hex_string(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"data")
        h = Utils.calculate_hash(str(f))
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


class TestRoundUp:
    def test_already_multiple(self):
        assert Utils.round_up(10, 5) == 10

    def test_rounds_up(self):
        assert Utils.round_up(11, 5) == 15

    def test_zero(self):
        assert Utils.round_up(0, 5) == 0
