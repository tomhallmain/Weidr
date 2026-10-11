"""Password protection fails closed when the stored password cannot be
decrypted (quantum keys, no OQS)."""

import pytest

from ui.auth import password_core, password_utils
from utils.constants import ProtectedActions


class _Config:
    def __init__(self, protected):
        self._protected = set(protected)

    def is_action_protected(self, name):
        return name in self._protected


@pytest.fixture
def unreadable(monkeypatch):
    monkeypatch.setattr(password_core, "stored_keys_need_unavailable_oqs", lambda service, app: True)
    monkeypatch.setattr(password_core.PasswordManager, "_security_configured_cache", None)
    alerts = []
    monkeypatch.setattr(password_utils, "show_password_unavailable", lambda master: alerts.append(master))

    def prompt(*args, **kwargs):
        raise AssertionError("must not prompt for a password that cannot be verified")

    monkeypatch.setattr(password_utils.PasswordDialog, "prompt_password", staticmethod(prompt))
    return alerts


def test_unreadable_password_counts_as_configured(unreadable):
    assert password_core.PasswordManager.is_security_configured() is True


def test_protected_action_is_refused(unreadable, monkeypatch):
    monkeypatch.setattr(password_utils, "get_security_config", lambda: _Config({"delete_media"}))
    results = []
    assert password_utils.check_password_required(
        [ProtectedActions.DELETE_MEDIA], master=None, callback=results.append) is False
    assert results == [False]
    assert unreadable == [None]


def test_action_requiring_authentication_is_refused(unreadable, monkeypatch):
    monkeypatch.setattr(password_utils, "get_security_config", lambda: _Config(set()))
    assert password_utils.check_password_required(
        [ProtectedActions.DELETE_MEDIA], master=None, allow_unauthenticated=False) is False


def test_unprotected_action_is_allowed(unreadable, monkeypatch):
    monkeypatch.setattr(password_utils, "get_security_config", lambda: _Config(set()))
    results = []
    assert password_utils.check_password_required(
        [ProtectedActions.DELETE_MEDIA], master=None, callback=results.append) is True
    assert results == [True]
    assert unreadable == []


def test_require_password_keeps_the_method_name():
    # A Nuitka build ran the wrong slot when every decorated method was named "wrapper".
    class Owner:
        @password_utils.require_password(ProtectedActions.DELETE_MEDIA)
        def delete_current(self):
            pass

        @password_utils.require_password(ProtectedActions.DELETE_MEDIA)
        def delete_marked(self):
            pass

    assert Owner.delete_current.__name__ == "delete_current"
    assert Owner.delete_marked.__name__ == "delete_marked"
