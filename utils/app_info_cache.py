from abc import ABC
import json
import os
import shutil
import sys
import threading
from typing import Any, Dict, List

from lib.equivalence import are_equivalent

from utils.constants import AppInfo
from utils.encryptor import encrypt_data_to_file, decrypt_data_from_file
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class InflationMonitor(ABC):
    """
    Abstract base class for cache systems that need to monitor list inflation.
    Concrete classes must implement get_cache_dict() and optionally get_meta() to provide the cache dictionary.
    """
    
    # Configuration
    INFLATION_ENABLED_KEY = "__monitor_list_inflation"
    INFLATION_MIN_LIST_SIZE = 5
    INFLATION_GROWTH_THRESHOLD = 2.0
    INFLATION_MAX_LIST_SIZE = 1000
    
    def __init__(self):
        self._monitor_inflation = False
        self._initial_list_sizes = {}
        self._suspected_keys = set()
        self._check_monitoring_enabled()
    
    def get_cache_dict(self, scope_key: str = "info") -> Dict:
        """Get the cache dictionary to monitor (required by InflationMonitor)."""
        raise NotImplementedError("get_cache_dict is not implemented in the base class")
    
    def get_meta(self, key: str, default_val: Any = None) -> Any:
        """
        Optional method to get meta value. If not implemented, monitoring will only check environment variable.
        Override this if your cache supports meta storage.
        """
        return default_val
    
    def _check_monitoring_enabled(self, scope_key: str = "info"):
        """Check if monitoring should be enabled via environment variable or meta flag."""
        # Check environment variable first
        if os.environ.get("MONITOR_CACHE_INFLATION", "0") == "1":
            self.enable_inflation_monitoring(True, scope_key)
            return
        
        # Check meta flag if get_meta is implemented
        try:
            monitor_flag = self.get_meta(self.INFLATION_ENABLED_KEY, False)
            if monitor_flag:
                self.enable_inflation_monitoring(True, scope_key)
        except (AttributeError, NotImplementedError):
            # get_meta not implemented or not available yet, skip
            pass
    
    def enable_inflation_monitoring(self, enable: bool = True, scope_key: str = "info"):
        """Enable or disable inflation monitoring."""
        if enable:
            self._monitor_inflation = True
            self._record_initial_list_sizes(scope_key)
            logger.info(f"Inflation monitoring enabled. Tracking {len(self._initial_list_sizes)} lists.")
        else:
            self._monitor_inflation = False
            self._initial_list_sizes = {}
            self._suspected_keys = set()
            logger.info("Inflation monitoring disabled.")
    
    def _record_initial_list_sizes(self, scope_key: str):
        """Record initial list sizes from cache with debug logging."""
        cache_dict = self.get_cache_dict(scope_key)
        self._initial_list_sizes = {}
        
        def _traverse_and_record(data: Any, path: str):
            """Recursively find and record list sizes in dictionaries."""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, list):
                        self._initial_list_sizes[current_path] = len(value)
                        logger.debug(f"Tracking list: {current_path} = {len(value)} items")
                        
                        # Check for immediate issues with large lists
                        if len(value) > self.INFLATION_MAX_LIST_SIZE:
                            logger.warning(f"Large initial list at {current_path}: {len(value)} items")
                    elif isinstance(value, dict):
                        _traverse_and_record(value, current_path)
            elif isinstance(data, list):
                self._initial_list_sizes[path] = len(data)
                logger.debug(f"Tracking list: {path} = {len(data)} items")
                if len(data) > self.INFLATION_MAX_LIST_SIZE:
                    logger.warning(f"Large initial list at {path}: {len(data)} items")
        
        if cache_dict:
            _traverse_and_record(cache_dict, path=scope_key)
    
    def _collect_list_sizes(self, data: Any, path: str = "") -> Dict[str, int]:
        """Collect sizes of all lists in a data structure."""
        sizes = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    sizes.update(self._collect_list_sizes(value, current_path))
        elif isinstance(data, list):
            sizes[path] = len(data)
        
        return sizes
    
    def check_for_inflation(self, scope_key: str = "info") -> List[str]:
        """
        Check for list inflation and return list of suspected keys with detailed logging.
        
        Returns:
            List of key paths suspected of inflation
        """
        if not self._monitor_inflation:
            return []
        
        cache_dict = self.get_cache_dict(scope_key)
        current_sizes = self._collect_list_sizes(cache_dict, scope_key)
        
        suspected = []
        inflation_detected = False
        
        for list_path, initial_size in self._initial_list_sizes.items():
            if list_path in current_sizes:
                current_size = current_sizes[list_path]
                
                # Skip very small lists
                if initial_size < self.INFLATION_MIN_LIST_SIZE:
                    continue
                
                # Check for doubling or significant growth
                if current_size >= initial_size * self.INFLATION_GROWTH_THRESHOLD:
                    inflation_detected = True
                    suspected.append(list_path)
                    
                    if list_path not in self._suspected_keys:
                        logger.warning(
                            f"⚠️ LIST INFLATION DETECTED: {list_path}\n"
                            f"   Initial size: {initial_size}\n"
                            f"   Current size: {current_size}\n"
                            f"   Growth factor: {current_size/initial_size:.1f}x\n"
                            f"   This may indicate duplicate entries or append-instead-of-update."
                        )
                        
                        # Try to identify if it's likely a duplication issue
                        if current_size % initial_size == 0:
                            multiples = current_size // initial_size
                            logger.info(f"   Note: Current size is exactly {multiples} times initial size.")
        
        if not inflation_detected and self._initial_list_sizes:
            logger.debug(f"No list inflation detected in '{scope_key}' dictionary.")
        
        self._suspected_keys = set(suspected)
        return suspected
    
    def get_suspected_inflation_keys(self, scope_key: str = "info") -> List[str]:
        """
        Get list of keys suspected of inflation with formatted details.
        
        Returns:
            List of formatted strings describing suspected inflation
        """
        if not self._monitor_inflation:
            return []
        
        suspected = []
        cache_dict = self.get_cache_dict(scope_key)
        current_sizes = self._collect_list_sizes(cache_dict, scope_key)
        
        for list_path, initial_size in self._initial_list_sizes.items():
            if list_path in current_sizes:
                current_size = current_sizes[list_path]
                if (initial_size >= self.INFLATION_MIN_LIST_SIZE and 
                    current_size >= initial_size * self.INFLATION_GROWTH_THRESHOLD):
                    suspected.append(f"{list_path} ({initial_size} → {current_size}, {current_size/initial_size:.1f}x)")
        
        return suspected
    
    def get_inflation_report(self) -> Dict[str, Any]:
        """Get inflation monitoring report."""
        return {
            "enabled": self._monitor_inflation,
            "tracked_lists": len(self._initial_list_sizes),
            "suspected_keys": list(self._suspected_keys)
        }


class AppInfoCache(InflationMonitor):
    CACHE_LOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_info_cache.enc")
    JSON_LOC = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "app_info_cache.json")
    META_INFO_KEY = "info"
    DIRECTORIES_KEY = "directories"
    NUM_BACKUPS = 4  # Number of backup files to maintain

    def __init__(self):
        super().__init__()
        self._lock = threading.RLock()
        self._has_changes = False
        self._cache = {self.META_INFO_KEY: {}, self.DIRECTORIES_KEY: {}}
        _override = os.environ.get("WEIDR_CACHE_DIR")
        self._cache_loc = os.path.join(_override, "app_info_cache.enc") if _override else self.CACHE_LOC
        self._json_loc  = os.path.join(_override, "app_info_cache.json") if _override else self.JSON_LOC
        self.load()
        self.validate()

    @staticmethod
    def _is_qt_runtime_active() -> bool:
        try:
            from PySide6.QtWidgets import QApplication
            return QApplication.instance() is not None
        except Exception:
            return False

    @staticmethod
    def _get_position_data_class():
        # Prefer Qt in active Qt runtime, fallback to Tk implementation.
        if AppInfoCache._is_qt_runtime_active():
            try:
                from lib.position_data_qt import PositionData as QtPositionData
                return QtPositionData
            except Exception:
                pass
        raise Exception("Invalid runtime environment: no Qt runtime active")

    def get_cache_dict(self, scope_key: str = "info") -> Dict:
        """Get the cache dictionary to monitor (required by InflationMonitor)."""
        with self._lock:
            if scope_key == "info":
                return self._cache.get(self.META_INFO_KEY, {})
            elif scope_key == "directories":
                return self._cache.get(self.DIRECTORIES_KEY, {})
            return self._cache.get(scope_key, {})

    @property
    def has_changes(self) -> bool:
        """True if there are pending writes that have not yet been stored."""
        with self._lock:
            return self._has_changes

    def store(self):
        """Persist cache to encrypted file. Returns True on success, False if encrypted store failed but JSON fallback succeeded. Raises on encoding or JSON fallback failure."""
        with self._lock:
            if self._monitor_inflation:
                self.check_for_inflation()

            try:
                cache_data = json.dumps(self._cache).encode('utf-8')
            except Exception as e:
                raise Exception(f"Error compiling application cache: {e}")

            try:
                encrypt_data_to_file(
                    cache_data,
                    AppInfo.SERVICE_NAME,
                    AppInfo.APP_IDENTIFIER,
                    self._cache_loc
                )
                self._has_changes = False
                return True  # Encryption successful
            except Exception as e:
                logger.error(f"Error encrypting cache: {e}")

            try:
                with open(self._json_loc, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f)
                self._has_changes = False
                return False  # Encryption failed, but JSON fallback succeeded
            except Exception as e:
                raise Exception(f"Error storing application cache: {e}")

    def _try_load_cache_from_file(self, path, app_identifier=None):
        """Attempt to load and decrypt the cache from the given file path. Raises on failure."""
        encrypted_data = decrypt_data_from_file(
            path,
            AppInfo.SERVICE_NAME,
            app_identifier or AppInfo.APP_IDENTIFIER
        )
        return json.loads(encrypted_data.decode('utf-8'))

    def _load_cache_from_file_with_fallback(self, path):
        """
        Attempt to load and decrypt from path using the current app identifier, falling
        back to legacy identifiers (e.g. after an app rename) if that fails. Returns the
        legacy identifier that succeeded, or None if the current identifier worked directly.
        Raises the original exception if no identifier can decrypt the file.
        """
        try:
            self._cache = self._try_load_cache_from_file(path)
            return None
        except Exception as current_identifier_error:
            for legacy_identifier in AppInfo.LEGACY_APP_IDENTIFIERS:
                try:
                    self._cache = self._try_load_cache_from_file(path, legacy_identifier)
                    return legacy_identifier
                except Exception:
                    continue
            raise current_identifier_error

    def load(self):
        with self._lock:
            try:
                if os.path.exists(self._json_loc):
                    logger.info(f"Detected JSON-format application cache, will attempt migration to encrypted store")
                    with open(self._json_loc, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                    if self.store():
                        logger.info(f"Migrated application cache from {self._json_loc} to encrypted store")
                        os.remove(self._json_loc)
                    else:
                        logger.warning("Encrypted store of application cache failed; keeping JSON cache file")
                    return

                # Try encrypted cache and backups in order
                cache_paths = [self._cache_loc] + self._get_backup_paths()
                any_exist = any(os.path.exists(path) for path in cache_paths)
                if not any_exist:
                    logger.info(f"No cache file found at {self._cache_loc}, creating new cache")
                    return

                for path in cache_paths:
                    if os.path.exists(path):
                        try:
                            migrated_from = self._load_cache_from_file_with_fallback(path)
                            if migrated_from:
                                logger.warning(
                                    f"Decrypted {path} using legacy app identifier "
                                    f"'{migrated_from}' instead of '{AppInfo.APP_IDENTIFIER}'. "
                                    f"Credentials for '{migrated_from}' were left in place; "
                                    f"only remove them manually once you've confirmed this "
                                    f"migration succeeded."
                                )
                            # Only rotate backups if we loaded from the main file
                            if path == self._cache_loc:
                                message = f"Loaded cache from {self._cache_loc}"
                                rotated_count = self._rotate_backups()
                                if rotated_count > 0:
                                    message += f", rotated {rotated_count} backups"
                                logger.info(message)
                            else:
                                logger.warning(f"Loaded cache from backup: {path}")
                            if migrated_from and path == self._cache_loc:
                                self.store()
                                logger.info(
                                    f"Re-encrypted cache under '{AppInfo.APP_IDENTIFIER}'"
                                )
                            return
                        except Exception as e:
                            logger.error(f"Failed to load cache from {path}: {e}")
                            continue
                # If we get here, all attempts failed (but at least one file existed)
                raise Exception(f"Failed to load cache from all locations: {cache_paths}")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                raise e

    def validate(self):
        with self._lock:
            directory_info = self._get_directory_info()
            directories = list(directory_info.keys())
            for d in directories:
                if not os.path.isdir(d):
                    # The external drive this reference is pointing to may not be mounted, might still be valid
                    if sys.platform == "win32" and not d.upper().startswith("C:\\"):
                        base_dir = os.path.split("\\")[0] + "\\"
                        if not os.path.isdir(base_dir):
                            continue
                    del directory_info[d]
                    logger.info(f"Removed stale directory reference: {d}")

    def _get_directory_info(self):
        with self._lock:
            if AppInfoCache.DIRECTORIES_KEY not in self._cache:
                self._cache[AppInfoCache.DIRECTORIES_KEY] = {}
            return self._cache[AppInfoCache.DIRECTORIES_KEY]

    def set_meta(self, key, value):
        with self._lock:
            if AppInfoCache.META_INFO_KEY not in self._cache:
                self._cache[AppInfoCache.META_INFO_KEY] = {}
            if self._has_changes or not are_equivalent(self._cache[AppInfoCache.META_INFO_KEY].get(key), value):
                self._cache[AppInfoCache.META_INFO_KEY][key] = value
                self._has_changes = True

    def get_meta(self, key, default_val=None):
        with self._lock:
            if AppInfoCache.META_INFO_KEY not in self._cache or key not in self._cache[AppInfoCache.META_INFO_KEY]:
                return default_val
            return self._cache[AppInfoCache.META_INFO_KEY][key]

    def set_display_position(self, master):
        """Store the main window's display position and size."""
        pd_cls = AppInfoCache._get_position_data_class()
        self.set_meta("display_position", pd_cls.from_master(master).to_dict())

    def set_virtual_screen_info(self, master):
        """Store the virtual screen information."""
        try:
            pd_cls = AppInfoCache._get_position_data_class()
            pd = pd_cls.from_master_virtual_screen(master)
            if pd is not None:
                self.set_meta("virtual_screen_info", pd.to_dict())
        except Exception as e:
            logger.warning(f"Failed to store virtual screen info: {e}")

    def get_virtual_screen_info(self):
        """Get the cached virtual screen info, returns None if not set or invalid."""
        virtual_screen_data = self.get_meta("virtual_screen_info")
        if not virtual_screen_data:
            return None
        pd_cls = AppInfoCache._get_position_data_class()
        try:
            return pd_cls.from_dict(virtual_screen_data)
        except Exception:
            return TkPositionData.from_dict(virtual_screen_data)

    def get_display_position(self):
        """Get the cached display position, returns None if not set or invalid."""
        position_data = self.get_meta("display_position")
        if not position_data:
            return None
        pd_cls = AppInfoCache._get_position_data_class()
        try:
            return pd_cls.from_dict(position_data)
        except Exception:
            return TkPositionData.from_dict(position_data)

    def get_directory_color(self, directory: str):
        """Return the stored custom title-bar color for *directory*, or None."""
        return self.get(directory, "title_bar_color")

    def set_directory_color(self, directory: str, color_hex: str | None):
        """Persist/clear custom title-bar color for *directory*."""
        self.set(directory, "title_bar_color", color_hex)

    def set(self, directory, key, value):
        with self._lock:
            if not directory or (isinstance(directory, str) and not directory.strip()):
                raise Exception(f"Invalid directory provided to app_info_cache.set(). key={key} value={value}")
            directory = AppInfoCache.normalize_directory_key(directory)
            if directory is None or directory.strip() == "":
                raise Exception(f"Invalid directory provided to app_info_cache.set(). key={key} value={value}")
            directory_info = self._get_directory_info()
            if directory not in directory_info:
                directory_info[directory] = {}
            if self._has_changes or not are_equivalent(directory_info[directory].get(key), value):
                directory_info[directory][key] = value
                self._has_changes = True

    def get(self, directory, key, default_val=None):
        with self._lock:
            directory = AppInfoCache.normalize_directory_key(directory)
            directory_info = self._get_directory_info()
            if directory not in directory_info or key not in directory_info[directory]:
                return default_val
            return directory_info[directory][key]

    @staticmethod
    def normalize_directory_key(directory):
        return os.path.normpath(os.path.abspath(directory))

    def export_as_json(self, json_path=None):
        """Export the current cache as a JSON file (not encrypted)."""
        if json_path is None:
            json_path = self._json_loc
        with self._lock:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        return json_path

    def clear_directory_cache(self, base_dir: str) -> None:
        """
        Clear all cache entries related to a specific base directory.
        This includes secondary_base_dirs, per-directory settings, and meta base_dir.
        """
        with self._lock:
            normalized_base_dir = self.normalize_directory_key(base_dir)

            # Remove from secondary_base_dirs
            try:
                secondary_base_dirs = self.get_meta("secondary_base_dirs", default_val=[])
                if base_dir in secondary_base_dirs:
                    secondary_base_dirs = [d for d in secondary_base_dirs if d != base_dir]
                    self.set_meta("secondary_base_dirs", secondary_base_dirs)
            except Exception as e:
                logger.error(f"Error updating secondary base dirs during delete: {e}")

            # Clear per-directory cached settings (favorites, cursors, etc.)
            try:
                directory_info = self._get_directory_info()
                if normalized_base_dir in directory_info:
                    del directory_info[normalized_base_dir]
                    self._has_changes = True
            except Exception as e:
                logger.error(f"Error clearing directory cache during delete: {e}")

            # If this was the stored main base_dir, clear it
            try:
                if self.get_meta("base_dir") == base_dir:
                    self.set_meta("base_dir", "")
            except Exception as e:
                logger.error(f"Error clearing meta base_dir during delete: {e}")

    def _get_backup_paths(self):
        """Get list of backup file paths in order of preference"""
        backup_paths = []
        for i in range(1, self.NUM_BACKUPS + 1):
            index = "" if i == 1 else f"{i}"
            path = f"{self._cache_loc}.bak{index}"
            backup_paths.append(path)
        return backup_paths

    def _rotate_backups(self):
        """Rotate backup files: move each backup to the next position, oldest gets overwritten"""
        backup_paths = self._get_backup_paths()
        rotated_count = 0
        
        # Remove the oldest backup if it exists
        if os.path.exists(backup_paths[-1]):
            os.remove(backup_paths[-1])
        
        # Shift backups: move each backup to the next position
        for i in range(len(backup_paths) - 1, 0, -1):
            if os.path.exists(backup_paths[i - 1]):
                shutil.copy2(backup_paths[i - 1], backup_paths[i])
                rotated_count += 1
        
        # Copy main cache to first backup position
        shutil.copy2(self._cache_loc, backup_paths[0])
        
        return rotated_count


app_info_cache = AppInfoCache()
# app_info_cache.enable_inflation_monitoring(True)
