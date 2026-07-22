from enum import Enum
import hashlib
import re
import os
import psutil
import shutil
import subprocess
import sys
import threading

from utils.running_tasks_registry import start_thread
from utils.logging_setup import get_logger

class Utils:
    _logger = get_logger("utils")
    
    # Global lock for thread-safe file operations
    file_operation_lock = threading.Lock()
    
    @staticmethod
    def safe_write(textfile, data):
        try:
            textfile.write(data)
        except UnicodeEncodeError as e:
            Utils._logger.error(e)

    @staticmethod
    def calculate_hash(filepath):
        with open(filepath, 'rb') as f:
            sha256 = hashlib.sha256()
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

    @staticmethod
    def trace(frame, event, arg):
        if event != 'call':
            return
        co = frame.f_code
        func_name = co.co_name
        if func_name == 'write':  # Ignore write() calls from print statements
            return
        func_filename = co.co_filename
        func_line_no = frame.f_lineno
        caller = frame.f_back
        app_name = "weidr"
        site_packages = "site-packages"
        lib = "\\Lib\\"
        if caller is None:
            if app_name in func_filename:
                func_simple_name = os.path.splitext(os.path.basename(func_filename))[0]
                print(f'{func_simple_name}:{func_line_no}:{func_name}()')
            return
        caller_filename = caller.f_code.co_filename
        if app_name in func_filename or app_name in caller_filename:
            if app_name in func_filename:
                func_simple_name = os.path.splitext(os.path.basename(func_filename))[0]
            elif site_packages in func_filename:
                func_simple_name = func_filename[func_filename.find(site_packages)+len(site_packages)+1:]
            elif lib in func_filename:
                func_simple_name = func_filename[func_filename.find(lib)+len(lib):]
            else:
                func_simple_name = func_filename
            if app_name in caller_filename:
                caller_simple_name = os.path.splitext(os.path.basename(func_filename))[0]
            elif site_packages in caller_filename:
                caller_simple_name = caller_filename[caller_filename.find(site_packages)+len(site_packages)+1:]
            elif lib in func_filename:
                func_simple_name = func_filename[func_filename.find(lib)+len(lib):]
            else:
                caller_simple_name = caller_filename
            caller_line_no = caller.f_lineno
            print(f'{caller_simple_name}:{caller_line_no} called {func_simple_name}:{func_line_no}:{func_name}()')

    @staticmethod
    def get_user_dir():
        return os.path.expanduser("~")

    @staticmethod
    def get_no_directory_compare_cache_dir() -> str:
        """
        Designated on-disk location for a compare's embedding/checkpoint cache
        when there is no real base_dir to key it off of -- e.g.
        FileActionsWindow's "Search in New Window", whose corpus is a file
        list spanning many directories rather than one.

        This is purely an internal detail of where the compare engine (whose
        core, self.base_dir, is used pervasively as a real path -- os.path.join,
        log strings, Utils.get_valid_file, etc. -- and would raise on None)
        persists its own cache files in that case. It must never be surfaced
        as AppWindow.base_dir or passed to prevalidation's directory-profile
        matching (ClassifierActionsManager.prevalidate_media), which should
        keep seeing no directory at all so no profile is ever scoped to it.
        """
        cache_dir = os.path.join(Utils.get_user_dir(), ".weidr", "compare_cache_no_dir")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def get_pictures_dir():
        """Get the Pictures directory for the current OS."""
        import platform
        system = platform.system()
        
        if system == "Windows":
            # Windows: Use Pictures folder
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        elif system == "Darwin":  # macOS
            # macOS: Use Pictures folder
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        else:  # Linux and others
            # Linux: Use Pictures folder, fallback to user dir
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
        
        # If Pictures directory doesn't exist, fallback to user directory
        if not os.path.exists(pictures_dir):
            pictures_dir = Utils.get_user_dir()
        
        return pictures_dir

    @staticmethod
    def calculate_available_ram():
        return psutil.virtual_memory().available / (1024 ** 3)

    @staticmethod
    def alphanumeric_sort(l, text_lambda=lambda i: i, reverse=False):
        """ Sort the given iterable in the way that humans expect."""
        def convert(text): return int(text) if text.isdigit() else text
        def alphanum_key(item): return [convert(c)
                                        for c in re.split('([0-9]+)', text_lambda(item))]
        return sorted(l, key=alphanum_key, reverse=reverse)

    @staticmethod
    def scale_dims(dims, max_dims, maximize=False):
        x = dims[0]
        y = dims[1]
        max_x = max_dims[0]
        max_y = max_dims[1]
        if x <= max_x and y <= max_y:
            if maximize:
                x_scale = max_x / x
                y_scale = max_y / y
                scale = min(x_scale, y_scale)
                return (int(x * scale), int(y * scale))
            return (x, y)
        elif x <= max_x:
            return (int(x * max_y/y), max_y)
        elif y <= max_y:
            return (max_x, int(y * max_x/x))
        else:
            x_scale = max_x / x
            y_scale = max_y / y
            if x_scale < y_scale:
                return (int(x * x_scale), int(y * x_scale))
            else:
                return (int(x * y_scale), int(y * y_scale))

    @staticmethod
    def _wrap_text_to_fit_length(text: str, fit_length: int):
        if len(text) <= fit_length:
            return text

        if " " in text and text.index(" ") < len(text) - 2:
            test_new_text = text[:fit_length]
            if " " in test_new_text:
                last_space_block = re.findall(" +", test_new_text)[-1]
                last_space_block_index = test_new_text.rfind(last_space_block)
                new_text = text[:last_space_block_index]
                text = text[(last_space_block_index+len(last_space_block)):]
            else:
                new_text = test_new_text
                text = text[fit_length:]
            while len(text) > 0:
                new_text += "\n"
                test_new_text = text[:fit_length]
                if len(test_new_text) <= fit_length:
                    new_text += test_new_text
                    text = text[fit_length:]
                elif " " in test_new_text and test_new_text.index(" ") < len(test_new_text) - 2:
                    last_space_block = re.findall(" +", test_new_text)[-1]
                    last_space_block_index = test_new_text.rfind(
                        last_space_block)
                    new_text += text[:last_space_block_index]
                    text = text[(last_space_block_index
                                 + len(last_space_block)):]
                else:
                    new_text += test_new_text
                    text = text[fit_length:]
        else:
            new_text = text[:fit_length]
            text = text[fit_length:]
            while len(text) > 0:
                new_text += "\n"
                new_text += text[:fit_length]
                text = text[fit_length:]

        return new_text

    @staticmethod
    def get_relative_dirpath_split(base_dir, filepath):
        # split the filepath from base directory
        relative_filepath = filepath.split(base_dir)[-1]

        # remove leading slash if exists
        if relative_filepath[0] == '/' or relative_filepath[0] == "\\":
            relative_filepath = relative_filepath[1:]

        basename = os.path.basename(relative_filepath)
        relative_dirpath = relative_filepath.replace(basename, "")

        if len(relative_dirpath) > 0 and (relative_dirpath[-1] == '/' or relative_dirpath[-1] == "\\"):
            relative_dirpath = relative_dirpath[:-1]

        return relative_dirpath, basename

    @staticmethod
    def get_relative_dirpath(base_dir, levels=1):
        # get relative dirpath from base directory
        if "/" not in base_dir and "\\" not in base_dir:
            return base_dir
        if "/" in base_dir:
            # temp = base_dir
            # if "/" == base_dir[0]:
            #     temp = base_dir[1:]
            dir_parts = base_dir.split("/")
        else:
            dir_parts = base_dir.split("\\")
        if len(dir_parts) <= levels:
            return base_dir
        relative_dirpath = ""
        for i in range(len(dir_parts) - 1, len(dir_parts) - levels - 1, -1):
            if relative_dirpath == "":
                relative_dirpath = dir_parts[i]
            else:
                relative_dirpath = dir_parts[i] + "/" + relative_dirpath
        return relative_dirpath

    @staticmethod
    def get_centrally_truncated_string(s, maxlen):
        # get centrally truncated string
        if len(s) <= maxlen:
            return s
        max_left_index = int((maxlen)/2-2)
        min_right_index = int(-(maxlen)/2-1)
        return s[:max_left_index] + "..." + s[min_right_index:]

    @staticmethod
    def unique_sibling_path(path: str, append_part: str) -> str:
        """Build a collision-safe sibling path: ``dir/stem<append_part>.ext``.

        If that path already exists, ``_1``, ``_2``, ... is appended until a free
        path is found, so repeated operations on the same source (e.g. cropping
        the same image/video twice) never overwrite a previous output.
        """
        dirname = os.path.dirname(os.path.abspath(path)) or "."
        stem, ext = os.path.splitext(os.path.basename(path))
        candidate = os.path.join(dirname, f"{stem}{append_part}{ext}")
        n = 1
        while os.path.exists(candidate):
            candidate = os.path.join(dirname, f"{stem}{append_part}_{n}{ext}")
            n += 1
        return candidate

    @staticmethod
    # NOTE: Maybe want to raise Exception if either existing filepath or target dir are not valid
    def move_file(existing_filepath, target_dir, overwrite_existing=False):
        new_filepath = os.path.join(
            target_dir, os.path.basename(existing_filepath))
        if not overwrite_existing and os.path.exists(new_filepath):
            raise Exception("File already exists: " + new_filepath)
        return shutil.move(existing_filepath, new_filepath)

    @staticmethod
    def copy_file(existing_filepath, target_dir, overwrite_existing=False):
        new_filepath = os.path.join(
            target_dir, os.path.basename(existing_filepath))
        if not overwrite_existing and os.path.exists(new_filepath):
            raise Exception("File already exists: " + new_filepath)
        return shutil.copy2(existing_filepath, new_filepath)

    @staticmethod
    def remove_path(
        path: str,
        delete_instantly: bool = False,
        trash_folder: str | None = None,
        is_directory: bool = False,
    ) -> None:
        """Remove a file or directory.

        - If delete_instantly is True, permanently removes (file or directory).
        - Else if trash_folder is set, moves into the trash folder (basename only).
        - Else attempts to send to OS trash; on failure, permanently removes.
        """
        if path is None or str(path).strip() == "":
            return
        normalized_path = os.path.normpath(path)

        with Utils.file_operation_lock:
            if delete_instantly:
                if is_directory:
                    shutil.rmtree(normalized_path)
                else:
                    os.remove(normalized_path)
                return

            if trash_folder is not None:
                normalized_trash = os.path.normpath(trash_folder)
                sep = "\\" if "\\" in normalized_path else "/"
                basename = normalized_path[normalized_path.rfind(sep)+1:]
                target_path = os.path.join(normalized_trash, basename)
                os.rename(normalized_path, target_path)
                return

            from send2trash import send2trash  # type: ignore
            send2trash(normalized_path)
    
    @staticmethod
    def isdir_with_retry(path, max_retries=3, retry_delay=1.0, wake_drive=True):
        """
        Check if a path is a directory, with retry logic for sleeping external drives.
        
        On Windows, external drives may be in a sleep/standby state and report paths
        as invalid before they have time to spin up. This function retries the check
        with delays to allow the drive to wake.
        
        Args:
            path: The path to check
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Seconds to wait between retries (default: 1.0)
            wake_drive: If True, attempt to wake the drive by accessing its root first
            
        Returns:
            bool: True if the path is a valid directory, False otherwise
        """
        import time
        external_drive_root = Utils._get_external_drive_root(path)
        drive_root = external_drive_root if wake_drive else None
        retries = max_retries if external_drive_root else 0

        for attempt in range(retries + 1):
            # On first attempt, probe external drive root to help wake sleeping drives.
            if wake_drive and drive_root and attempt == 0:
                try:
                    os.path.exists(drive_root)
                except OSError:
                    pass  # Drive may not be accessible yet
            
            if os.path.isdir(path):
                return True
            
            if attempt < retries:
                logger.debug(f"Directory check failed for '{path}', retrying in {retry_delay}s (attempt {attempt + 1}/{retries})")
                time.sleep(retry_delay)
        
        return False

    @staticmethod
    def isfile_with_retry(path, max_retries=3, retry_delay=1.0, wake_drive=True):
        """
        Check if a path is a file, with retry logic for sleeping external drives.
        
        On Windows, external drives may be in a sleep/standby state and report paths
        as invalid before they have time to spin up. This function retries the check
        with delays to allow the drive to wake.
        
        Args:
            path: The path to check
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Seconds to wait between retries (default: 1.0)
            wake_drive: If True, attempt to wake the drive by accessing its root first
            
        Returns:
            bool: True if the path is a valid file, False otherwise
        """
        import time
        external_drive_root = Utils._get_external_drive_root(path)
        drive_root = external_drive_root if wake_drive else None
        retries = max_retries if external_drive_root else 0

        for attempt in range(retries + 1):
            if wake_drive and drive_root and attempt == 0:
                try:
                    os.path.exists(drive_root)
                except OSError:
                    pass
            
            if os.path.isfile(path):
                return True
            
            if attempt < retries:
                logger.debug(f"File check failed for '{path}', retrying in {retry_delay}s (attempt {attempt + 1}/{retries})")
                time.sleep(retry_delay)
        
        return False

    @staticmethod
    def _get_external_drive_root(path):
        """
        Return an external/removable drive root for path, or None if not external.

        Windows:
            Treat drive letters E: and above as external/removable.
        Non-Windows:
            Best-effort check for common removable-media mount roots.
        """
        if not path:
            return None

        normalized = os.path.normpath(os.path.abspath(path))

        if sys.platform == "win32":
            drive = os.path.splitdrive(normalized)[0]  # e.g. "E:"
            if len(drive) == 2 and drive[1] == ":" and drive[0].isalpha():
                if drive[0].upper() >= "E":
                    return drive + os.sep
            return None

        # Common removable mount roots on macOS/Linux
        removable_roots = (
            "/Volumes",
            "/media",
            "/run/media",
            "/mnt",
        )
        for root in removable_roots:
            root_norm = os.path.normpath(root)
            if normalized == root_norm or normalized.startswith(root_norm + os.sep):
                return root_norm + os.sep

        return None

    @staticmethod
    def open_file(filepath):
        if sys.platform == 'win32':
            os.startfile(filepath)
        elif sys.platform == 'darwin':
            os.system('open "%s"' % filepath)
        else:
            os.system('xdg-open "%s"' % filepath)

    @staticmethod
    def open_file_location(filepath):
        if sys.platform == 'win32':
            os.startfile(filepath)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', filepath])
        else:
            try:
                subprocess.Popen(['xdg-open', filepath])
            except OSError:
                # er, think of something else to try
                # xdg-open *should* be supported by recent Gnome, KDE, Xfce
                raise Exception(
                    "Unsupported distribution for opening file location.")

    @staticmethod
    def open_media_file(filepath: str, is_video: bool = False) -> None:
        """
        Open a media file with the appropriate application.
        
        Args:
            filepath: Path to the media file to open
            is_video: Whether the file is a video (determines opening method)
        """
        if is_video:
            Utils._open_video_file(filepath)
        else:
            Utils.open_file_location(filepath)

    @staticmethod
    def _open_video_file(filepath: str) -> None:
        """Open a video file with VLC or default video player."""
        if sys.platform == 'win32':
            try:
                subprocess.Popen(['vlc', filepath])
                return
            except FileNotFoundError:
                pass
        elif sys.platform == 'darwin':
            try:
                subprocess.Popen(['vlc', filepath])
                return
            except FileNotFoundError:
                pass
        else:
            try:
                subprocess.Popen(['vlc', filepath])
                return
            except FileNotFoundError:
                pass
        Utils.open_file_location(filepath)

    @staticmethod
    def open_file_in_gimp(filepath, gimp_exe_loc="gimp-2.10"):
        def gimp_process():
            command = ["set", "LANG=en", "&&", gimp_exe_loc, filepath]
            process = subprocess.call(command, shell=True)
            if process != 0:
                raise Exception("Could not open file in GIMP")
        start_thread(gimp_process)

    @staticmethod
    def is_external_drive(filepath):
        if sys.platform == 'win32':
            return os.path.splitdrive(filepath)[0] != 'C:'
        else:
            # TODO figure out how to detect external drives on other platforms
            return False

    @staticmethod
    def get_default_user_language():
        if sys.platform == 'win32':
            import ctypes
            import locale
            windll = ctypes.windll.kernel32
            windll.GetUserDefaultUILanguage()
            _locale = locale.windows_locale[windll.GetUserDefaultUILanguage()]
            if _locale is not None and "_" in _locale:
                return _locale[:_locale.index("_")]
        # TODO support finding default languages on other platforms
        return 'en'

    @staticmethod
    def modifier_key_pressed(event, keys_to_check=[]):
        if not event:
            return (False)
        is_pressed = []
        for key in keys_to_check:
            if not isinstance(key, ModifierKey):
                raise Exception("Invalid modifier key: " + str(key))
            is_pressed.append(event.state & key.value != 0)
        if len(keys_to_check) == 1:
            return is_pressed[0]
        return tuple(is_pressed)

    @staticmethod
    def round_up(number, to):
        if number % to == 0:
            return number
        else:
            return number - (number % to) + to

    @staticmethod
    def is_invalid_file(file_path, counter, run_search, file_filter):
        if file_path is None:
            return True
        elif run_search and counter == 0:
            return False
        elif file_filter is not None:
            terms = [t.strip() for t in file_filter.split(";") if t.strip()]
            inclusions = [t for t in terms if not t.startswith("!")]
            exclusions = [t[1:] for t in terms if t.startswith("!") and len(t) > 1]
            if exclusions and any(term in file_path for term in exclusions):
                return True
            if inclusions and not any(term in file_path for term in inclusions):
                return True
            return False
        else:
            return False

    @staticmethod
    def get_valid_file(base_dir, input_filepath):
        if (not isinstance(input_filepath, str) or input_filepath is None
                or input_filepath.strip() == ""):
            return None
        if input_filepath.startswith('"') and input_filepath.endswith('"'):
            input_filepath = input_filepath[1:-1]
        elif input_filepath.startswith("'") and input_filepath.endswith("'"):
            input_filepath = input_filepath[1:-1]
        if os.path.exists(input_filepath):
            return input_filepath
        elif base_dir is not None and os.path.exists(os.path.join(base_dir, input_filepath)):
            return base_dir + "/" + input_filepath
        else:
            return None

    @staticmethod
    def parse_json_strings_in_dict(data_dict, keys_to_check=None):
        """
        Convert JSON string values in a dictionary to JSON objects.
        
        Args:
            data_dict: Dictionary that may contain JSON strings as values
            keys_to_check: Optional list of keys to check. If None, checks all string values.
            
        Returns:
            A copy of the dictionary with JSON strings parsed into objects.
        """
        import json
        
        # Create a copy to avoid modifying the original
        processed_dict = data_dict.copy()
        
        # Determine which keys to check
        if keys_to_check is None:
            # Check all string values in the dict
            keys_to_check = [k for k, v in processed_dict.items() if isinstance(v, str)]
        
        # Convert string values to JSON objects if they're valid JSON
        for key in keys_to_check:
            if key in processed_dict and isinstance(processed_dict[key], str):
                try:
                    processed_dict[key] = json.loads(processed_dict[key])
                except json.JSONDecodeError:
                    # If parsing fails, keep the original string
                    pass
        
        return processed_dict

    @staticmethod
    def split(string, delimiter=","):
        # Split the string by the delimiter and clean any delimiter escapes present in the string
        parts = []
        i = 0
        while i < len(string):
            if string[i] == delimiter:
                if i == 0 or string[i-1] != "\\":
                    parts.append(string[:i])
                    string = string[i+1:]
                    i = -1
                elif i != 0 and string[i-1] == "\\":
                    string = string[:i-1] + delimiter + string[i+1:]
            elif i == len(string) - 1:
                parts.append(string[:i+1])
            i += 1
        if len(parts) == 0 and len(string) != 0:
            parts.append(string)
        return parts

    @staticmethod
    def check_single_instance(app_name="Weidr", mutex_name=None, lock_filename=None):
        """
        Check if another instance of the application is already running.
        
        Args:
            app_name: Display name for error messages
            mutex_name: Name for Windows mutex (defaults to app_name + "_SingleInstance")
            lock_filename: Name for lock file (defaults to app_name.lower() + ".lock")
            
        Returns:
            tuple: (lock_file_path, cleanup_function) where lock_file_path is the path to the lock file
                   (None if using Windows mutex) and cleanup_function is a function to call for cleanup
                   
        Raises:
            SystemExit: If another instance is already running
        """
        if mutex_name is None:
            mutex_name = f"{app_name}_SingleInstance"
        if lock_filename is None:
            lock_filename = f"{app_name.lower()}.lock"
        
        # Try Windows mutex first (most reliable on Windows)
        if sys.platform == 'win32':
            try:
                import win32event
                import win32api
                import winerror
                
                # Try to create a named mutex
                mutex = win32event.CreateMutex(None, False, mutex_name)
                if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
                    # Another instance is already running
                    print(f"Another instance of {app_name} is already running.")
                    print("Please close the existing instance or use that one.")
                    input("Press Enter to exit...")
                    os._exit(1)
                
                # Successfully created mutex, return None for lock file and dummy cleanup
                return None, lambda: None
                
            except ImportError:
                pass  # Fall through to file-based method
        
        # Try file locking (works on Unix and Windows)
        try:
            import tempfile
            
            lock_file = os.path.join(tempfile.gettempdir(), lock_filename)
            
            # Open the lock file
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)
            
            # Try to acquire an exclusive lock
            if sys.platform == 'win32':
                # Windows file locking
                try:
                    import msvcrt
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    lock_acquired = True
                except (ImportError, OSError):
                    lock_acquired = False
            else:
                # Unix file locking
                try:
                    import fcntl
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                except (ImportError, OSError):
                    lock_acquired = False
            
            if not lock_acquired:
                # Could not acquire lock, another instance is running
                os.close(lock_fd)
                print(f"Another instance of {app_name} is already running.")
                print("Please close the existing instance or use that one.")
                input("Press Enter to exit...")
                os._exit(1)
            
            # Successfully acquired lock, write PID to file
            os.ftruncate(lock_fd, 0)  # Clear file contents
            os.write(lock_fd, str(os.getpid()).encode())
            os.fsync(lock_fd)  # Ensure data is written to disk
            
            # Return cleanup function
            def cleanup_lock():
                try:
                    if sys.platform == 'win32':
                        try:
                            import msvcrt
                            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                        except (ImportError, OSError):
                            pass
                    else:
                        try:
                            import fcntl
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        except (ImportError, OSError):
                            pass
                    
                    os.close(lock_fd)
                    try:
                        if os.path.exists(lock_file):
                            os.remove(lock_file)
                    except (OSError, PermissionError):
                        pass  # Ignore cleanup errors
                except Exception:
                    pass  # Ignore any cleanup errors
            
            return lock_file, cleanup_lock
            
        except (OSError, PermissionError, ImportError):
            pass  # Fall through to socket-based method
        
        # Final fallback: try socket-based method
        try:
            import socket
            
            # Try to bind to a specific port
            port = 0  # Let OS choose a port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                sock.bind(('localhost', port))
                sock.listen(1)
                
                # Successfully bound, return cleanup function
                def cleanup_lock():
                    try:
                        sock.close()
                    except Exception:
                        pass
                
                return None, cleanup_lock
                
            except OSError:
                # Port is in use, another instance is running
                sock.close()
                print(f"Another instance of {app_name} is already running.")
                print("Please close the existing instance or use that one.")
                input("Press Enter to exit...")
                os._exit(1)
                
        except ImportError:
            # Socket module not available, this is very unlikely
            print(f"Warning: Could not implement single instance check for {app_name}.")
            print("Multiple instances may run simultaneously.")
            return None, lambda: None


class ModifierKey(Enum):
    SHIFT = 0x1
    CAPS_LOCK = 0x2
    CTRL = 0x4
    ALT = 0x20000
