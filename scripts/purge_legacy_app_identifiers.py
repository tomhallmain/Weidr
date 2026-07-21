#!/usr/bin/env python3
"""
Purge legacy app-identifier credentials from the OS keyring, once you've
confirmed the encryption identifier migration (AppInfo.LEGACY_APP_IDENTIFIERS
in utils/constants.py) succeeded.

Background:
  utils/app_info_cache.py falls back to legacy app identifiers when it can't
  decrypt app_info_cache.enc under the current identifier, then re-encrypts
  the cache under the current one. It deliberately leaves the legacy
  identifier's keyring entries in place afterward -- they're the only copy
  of the old decryption key until a user confirms the migration worked.
  This script performs that manual cleanup step.

Safety:
  - Refuses to run unless app_info_cache.enc decrypts successfully under the
    CURRENT app identifier only (no legacy fallback) -- proof the migration
    actually completed.
  - Only ever deletes keyring entries for identifiers listed in
    AppInfo.LEGACY_APP_IDENTIFIERS -- never the current AppInfo.APP_IDENTIFIER.
  - Dry run by default; pass --execute to actually delete anything, and
    confirm interactively unless --yes is also given.
  - Does NOT touch BaseEncryptor.purge_keys's purge_files cleanup (the
    quantum_pub.key / standard_pub.key files) -- those filenames aren't
    namespaced by app identifier, so deleting them could remove a file the
    current identifier still relies on. Only keyring entries are purged.

Usage:
  python scripts/purge_legacy_app_identifiers.py                 # dry run
  python scripts/purge_legacy_app_identifiers.py --execute        # prompts, then purges
  python scripts/purge_legacy_app_identifiers.py --execute --yes  # no prompt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.constants import AppInfo  # noqa: E402
from utils.encryptor import BaseEncryptor, decrypt_data_from_file, namespaced_key  # noqa: E402
import keyring  # noqa: E402


def _cache_path() -> str:
    override = os.environ.get("WEIDR_CACHE_DIR")
    base = override if override else str(REPO_ROOT)
    return os.path.join(base, "app_info_cache.enc")


def verify_migration_succeeded(cache_path: str) -> None:
    """Abort unless *cache_path* decrypts under the CURRENT app identifier,
    with no legacy fallback -- the same proof AppInfoCache itself relies on."""
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"No cache file found at {cache_path} -- nothing to verify against, "
            f"aborting without purging anything."
        )
    try:
        decrypt_data_from_file(cache_path, AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER)
    except Exception as e:
        raise SystemExit(
            f"Refusing to purge: {cache_path} does not decrypt under the current "
            f"app identifier '{AppInfo.APP_IDENTIFIER}' ({e}).\n"
            f"Run the app first so it can complete the migration, then re-run this script."
        )
    print(f"Verified: {cache_path} decrypts under current identifier '{AppInfo.APP_IDENTIFIER}'.")


def _has_any_keyring_entries(service_name: str, app_identifier: str) -> bool:
    probe_keys = [
        namespaced_key(app_identifier, "salt"),
        namespaced_key(app_identifier, "nonce"),
        namespaced_key(app_identifier, "tag"),
        namespaced_key(app_identifier, "encryptor_type"),
        namespaced_key(app_identifier, "passphrase"),
    ]
    return any(keyring.get_password(service_name, k) for k in probe_keys)


def purge_identifier(service_name: str, app_identifier: str, execute: bool) -> None:
    if app_identifier == AppInfo.APP_IDENTIFIER:
        # Should be unreachable given the caller filters this out, but this is
        # exactly the mistake that would silently wipe live credentials.
        raise SystemExit(
            f"Refusing to purge '{app_identifier}' -- it is the CURRENT app "
            f"identifier, not a legacy one."
        )
    if not _has_any_keyring_entries(service_name, app_identifier):
        print(f"  {app_identifier}: no keyring entries found, nothing to do.")
        return
    if not execute:
        print(f"  {app_identifier}: would purge keyring entries (dry run).")
        return
    BaseEncryptor.purge_keys(service_name, app_identifier, purge_files=False)
    print(f"  {app_identifier}: purged.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--identifier", action="append", dest="identifiers",
        help="Legacy identifier to purge (repeatable). Defaults to all of "
             "AppInfo.LEGACY_APP_IDENTIFIERS.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually delete keyring entries (default: dry run).")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the cache-decrypts-under-current-identifier check. Dangerous -- "
             "only use if you've verified the migration succeeded some other way.",
    )
    args = parser.parse_args()

    identifiers = args.identifiers or list(AppInfo.LEGACY_APP_IDENTIFIERS)
    identifiers = [i for i in identifiers if i != AppInfo.APP_IDENTIFIER]
    if not identifiers:
        print("No legacy identifiers to purge.")
        return

    print(f"Service: {AppInfo.SERVICE_NAME}")
    print(f"Current identifier (kept): {AppInfo.APP_IDENTIFIER}")
    print(f"Legacy identifiers targeted: {identifiers}")

    if args.skip_verify:
        print("WARNING: skipping migration verification (--skip-verify).")
    else:
        verify_migration_succeeded(_cache_path())

    if args.execute and not args.yes:
        answer = input(
            f"\nThis will permanently delete keyring credentials for {identifiers} "
            f"under service '{AppInfo.SERVICE_NAME}'. Continue? (y/N): "
        )
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    mode = "EXECUTING" if args.execute else "DRY RUN"
    print(f"\n{mode}:")
    for identifier in identifiers:
        purge_identifier(AppInfo.SERVICE_NAME, identifier, args.execute)

    if not args.execute:
        print("\nDry run complete. Re-run with --execute to actually delete these entries.")


if __name__ == "__main__":
    main()
