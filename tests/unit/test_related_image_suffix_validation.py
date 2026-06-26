"""Unit tests for files.related_image.validate_related_image_suffix."""

from files.related_image import validate_related_image_suffix


class TestValidateRelatedImageSuffix:

    def test_valid_alpha_word(self):
        assert validate_related_image_suffix("edit") is None

    def test_valid_mixed_alphanumeric(self):
        assert validate_related_image_suffix("v2") is None

    def test_valid_exactly_eight_chars(self):
        assert validate_related_image_suffix("a1b2c3d4") is None

    def test_invalid_purely_numeric(self):
        assert validate_related_image_suffix("123") is not None

    def test_invalid_empty(self):
        assert validate_related_image_suffix("") is not None

    def test_invalid_too_long(self):
        assert validate_related_image_suffix("toolongname") is not None

    def test_invalid_underscore_returns_nonempty_string(self):
        result = validate_related_image_suffix("my_edit")
        assert isinstance(result, str) and result
