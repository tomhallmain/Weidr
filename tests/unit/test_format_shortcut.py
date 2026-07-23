"""Unit tests for format_shortcut modifier localization."""

import pytest

from utils.translations import I18N, format_shortcut


@pytest.fixture
def restore_locale():
    previous = I18N.locale
    yield
    I18N.install_locale(previous, verbose=False)


class TestFormatShortcut:
    def test_english_unchanged(self, restore_locale):
        I18N.install_locale("en", verbose=False)
        assert format_shortcut("Ctrl+Shift+Y") == "Ctrl+Shift+Y"
        assert format_shortcut("Ctrl+T") == "Ctrl+T"
        assert format_shortcut("Alt+Y") == "Alt+Y"

    def test_german_ctrl_to_strg(self, restore_locale):
        I18N.install_locale("de", verbose=False)
        assert format_shortcut("Ctrl+T") == "Strg+T"
        assert format_shortcut("Ctrl+Shift+Y") == "Strg+Umschalt+Y"
        assert format_shortcut("Ctrl+Alt+Return") == "Strg+Alt+Return"
        assert format_shortcut("Ctrl/Alt/Shift") == "Strg/Alt/Umschalt"

    def test_leaves_non_modifiers(self, restore_locale):
        I18N.install_locale("de", verbose=False)
        assert format_shortcut("Left/Right Arrow") == "Left/Right Arrow"
        assert format_shortcut("Command") == "Command"
        assert format_shortcut("0-9") == "0-9"

    def test_french_shift_to_maj(self, restore_locale):
        I18N.install_locale("fr", verbose=False)
        assert format_shortcut("Ctrl+Shift+G") == "Ctrl+Maj+G"
