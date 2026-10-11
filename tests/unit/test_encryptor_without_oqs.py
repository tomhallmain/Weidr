"""Detecting quantum (OQS) keys in a process without OQS, as a build made
without --with-oqs is."""

import pytest

import utils.encryptor as enc

SERVICE = "TestService"
APP = "test_app"


@pytest.fixture
def stored_type(monkeypatch):
    def install(key_type, oqs_available):
        monkeypatch.setattr(enc, "_stored_encryptor_type", lambda service, app: key_type)
        monkeypatch.setattr(enc, "KeyEncapsulation", object() if oqs_available else None)
    return install


@pytest.mark.parametrize("key_type, oqs_available, expected", [
    ("quantum", False, True),
    ("quantum", True, False),
    ("standard", False, False),
    (None, False, False),
])
def test_stored_keys_need_unavailable_oqs(stored_type, key_type, oqs_available, expected):
    stored_type(key_type, oqs_available)
    assert enc.stored_keys_need_unavailable_oqs(SERVICE, APP) is expected


def test_quantum_keys_without_oqs_raise_a_distinct_error(stored_type):
    stored_type("quantum", False)
    with pytest.raises(enc.OQSUnavailableError):
        enc._determine_encryptor(SERVICE, APP)


def test_distinct_error_is_still_a_runtime_error():
    assert issubclass(enc.OQSUnavailableError, RuntimeError)
