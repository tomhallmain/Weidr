"""Windows passphrase lockdown follows keyring's credential targets."""

import sys
import types

from utils.encryptor import read_windows_credential, windows_credential_targets


def test_targets_match_winvault_keyring():
    service = "MyPersonalApplicationsService"
    username = "weidr__passphrase"
    assert windows_credential_targets(service, username) == (
        service,
        f"{username}@{service}",
    )


def _install_fake_cred(monkeypatch, store):
    class CredError(Exception):
        def __init__(self, winerror, funcname):
            self.winerror = winerror
            self.funcname = funcname

    def cred_read(target, _typ, _flags):
        if target not in store:
            raise CredError(1168, "CredRead")
        return store[target]

    monkeypatch.setitem(sys.modules, "win32cred", types.SimpleNamespace(
        CredRead=cred_read, CRED_TYPE_GENERIC=1,
    ))
    monkeypatch.setitem(sys.modules, "pywintypes", types.SimpleNamespace(error=CredError))


def test_primary_slot_is_used_when_username_matches(monkeypatch):
    service = "MyPersonalApplicationsService"
    username = "weidr__passphrase"
    primary = {"UserName": username, "TargetName": service}
    _install_fake_cred(monkeypatch, {service: primary})
    assert read_windows_credential(service, username) is primary


def test_compound_slot_is_used_when_another_username_holds_the_service(monkeypatch):
    service = "MyPersonalApplicationsService"
    username = "weidr__passphrase"
    compound_name = f"{username}@{service}"
    compound = {"UserName": username, "TargetName": compound_name}
    _install_fake_cred(monkeypatch, {
        service: {"UserName": "protokoll__passphrase", "TargetName": service},
        compound_name: compound,
    })
    assert read_windows_credential(service, username) is compound


def test_missing_targets_return_none(monkeypatch):
    _install_fake_cred(monkeypatch, {})
    assert read_windows_credential("MyPersonalApplicationsService", "weidr__passphrase") is None
