"""
Unit tests for compare.classifier_pipeline_runner._stem_matches_any_suffix.

Covers the flexible matching rules introduced to support:
- Extra or double underscores/spaces as separators between base stem and suffix
- Right-side truncation of the suffix word
- Trailing variant markers (_2, ' 2', 2) after the suffix
"""

import pytest

from compare.classifier_pipeline_runner import _stem_matches_any_suffix


FRUIT_SUFFIXES = ["_apple", "_banana", "_cherry"]


# ---------------------------------------------------------------------------
# Exact matches
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_single_underscore_exact(self):
        assert _stem_matches_any_suffix("TEST_123_cherry", FRUIT_SUFFIXES)

    def test_exact_apple(self):
        assert _stem_matches_any_suffix("TEST_123_apple", FRUIT_SUFFIXES)

    def test_exact_banana(self):
        assert _stem_matches_any_suffix("TEST_123_banana", FRUIT_SUFFIXES)

    def test_case_insensitive(self):
        assert _stem_matches_any_suffix("TEST_123_Cherry", FRUIT_SUFFIXES)
        assert _stem_matches_any_suffix("TEST_123_CHERRY", FRUIT_SUFFIXES)

    def test_suffix_spec_case_insensitive(self):
        assert _stem_matches_any_suffix("TEST_123_cherry", ["_Cherry", "_Apple"])


# ---------------------------------------------------------------------------
# Separator flexibility
# ---------------------------------------------------------------------------

class TestSeparatorFlexibility:
    def test_double_underscore(self):
        assert _stem_matches_any_suffix("TEST_17820535380315060__cherry", FRUIT_SUFFIXES)

    def test_triple_underscore(self):
        assert _stem_matches_any_suffix("TEST_17820535380315060___cherry", FRUIT_SUFFIXES)

    def test_space_separator(self):
        assert _stem_matches_any_suffix("TEST_123 cherry", FRUIT_SUFFIXES)

    def test_underscore_space_separator(self):
        assert _stem_matches_any_suffix("TEST_123_ cherry", FRUIT_SUFFIXES)

    def test_arbitrary_intermediate_content(self):
        # Intermediate tokens (e.g. an edit label or date) between the base stem
        # and the suffix should not prevent recognition.
        assert _stem_matches_any_suffix(
            "TEST_17820535380315060_upscaled_2x_cherry", FRUIT_SUFFIXES
        )


# ---------------------------------------------------------------------------
# Right-side truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_cher_matches_cherry(self):
        assert _stem_matches_any_suffix("TEST_17820535380315060__cher", FRUIT_SUFFIXES)

    def test_che_matches_cherry(self):
        assert _stem_matches_any_suffix("TEST_123_che", FRUIT_SUFFIXES)

    def test_ban_matches_banana(self):
        assert _stem_matches_any_suffix("TEST_123__ban", FRUIT_SUFFIXES)

    def test_app_matches_apple(self):
        assert _stem_matches_any_suffix("TEST_123_app", FRUIT_SUFFIXES)

    def test_single_char_truncation(self):
        assert _stem_matches_any_suffix("TEST_123_c", FRUIT_SUFFIXES)


# ---------------------------------------------------------------------------
# Trailing variant markers
# ---------------------------------------------------------------------------

class TestVariantMarkers:
    def test_underscore_digit(self):
        assert _stem_matches_any_suffix("TEST_123_cherry_2", FRUIT_SUFFIXES)

    def test_space_digit(self):
        assert _stem_matches_any_suffix("TEST_123_cherry 2", FRUIT_SUFFIXES)

    def test_digit_no_separator(self):
        assert _stem_matches_any_suffix("TEST_123_cherry2", FRUIT_SUFFIXES)

    def test_truncated_plus_variant(self):
        # __cher_2 should match _cherry
        assert _stem_matches_any_suffix("TEST_17820535380315060__cher_2", FRUIT_SUFFIXES)

    def test_truncated_space_variant(self):
        assert _stem_matches_any_suffix("TEST_17820535380315060__cher 2", FRUIT_SUFFIXES)

    def test_multi_digit_variant(self):
        assert _stem_matches_any_suffix("TEST_123_cherry_10", FRUIT_SUFFIXES)

    def test_large_number_in_stem_not_stripped_as_variant(self):
        # A pure base stem with a long numeric ID should NOT match any suffix.
        assert not _stem_matches_any_suffix("TEST_17820535380315060", FRUIT_SUFFIXES)


# ---------------------------------------------------------------------------
# Negative cases — should NOT match
# ---------------------------------------------------------------------------

class TestNoMatch:
    def test_pure_base_stem_no_suffix(self):
        assert not _stem_matches_any_suffix("TEST_17820535380315060", FRUIT_SUFFIXES)

    def test_unknown_suffix(self):
        assert not _stem_matches_any_suffix("TEST_123_unknown", FRUIT_SUFFIXES)

    def test_no_separator_before_suffix_word(self):
        # 'notacherry' — 'a' before 'cherry' is alphanumeric, not a separator
        assert not _stem_matches_any_suffix("TEST_123notacherry", FRUIT_SUFFIXES)

    def test_empty_suffixes_list(self):
        assert not _stem_matches_any_suffix("TEST_123_cherry", [])

    def test_empty_stem(self):
        assert not _stem_matches_any_suffix("", FRUIT_SUFFIXES)

    def test_suffix_spec_with_only_separators_ignored(self):
        # A suffix spec that is nothing but underscores has no core — skip it.
        assert not _stem_matches_any_suffix("TEST_123_cherry", ["___"])

    def test_unrelated_word_ending(self):
        assert not _stem_matches_any_suffix("TEST_123_mango", FRUIT_SUFFIXES)


# ---------------------------------------------------------------------------
# Leading-separator normalisation in suffix spec
# ---------------------------------------------------------------------------

class TestSuffixSpecNormalisation:
    def test_spec_with_leading_underscore(self):
        assert _stem_matches_any_suffix("TEST_123_cherry", ["_cherry"])

    def test_spec_without_leading_underscore(self):
        assert _stem_matches_any_suffix("TEST_123_cherry", ["cherry"])

    def test_spec_with_double_leading_underscore(self):
        assert _stem_matches_any_suffix("TEST_123_cherry", ["__cherry"])
