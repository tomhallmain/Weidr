"""Tests for ui.auth.password_core.first_password_protected, the check MCP
sessions use to refuse password-protected actions (Qt-free)."""

from __future__ import annotations

import pytest

from ui.auth import password_core
from utils.constants import ProtectedActions


class _Config:
    def __init__(self, protected):
        self._protected = set(protected)

    def is_action_protected(self, name):
        return name in self._protected


@pytest.fixture
def security(monkeypatch):
    keychain_reads = []

    def install(protected, password_set):
        monkeypatch.setattr(password_core, "get_security_config", lambda: _Config(protected))

        def is_security_configured():
            keychain_reads.append(True)
            return password_set

        monkeypatch.setattr(password_core.PasswordManager, "is_security_configured",
                            staticmethod(is_security_configured))
        return keychain_reads
    return install


def test_flagged_action_with_a_password_set_is_blocked(security):
    security({"delete_media"}, password_set=True)
    blocked = password_core.first_password_protected(
        [ProtectedActions.RUN_COMPARES, ProtectedActions.DELETE_MEDIA]
    )
    assert blocked == ProtectedActions.DELETE_MEDIA


def test_flagged_action_without_a_password_is_allowed(security):
    security({"delete_media"}, password_set=False)
    assert password_core.first_password_protected([ProtectedActions.DELETE_MEDIA]) is None


def test_unflagged_actions_never_read_the_keychain(security):
    keychain_reads = security(set(), password_set=True)
    assert password_core.first_password_protected([ProtectedActions.RUN_COMPARES]) is None
    assert keychain_reads == []
