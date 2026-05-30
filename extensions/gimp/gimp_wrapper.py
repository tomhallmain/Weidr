"""
GIMP Wrapper Module

Note: This entire module exists solely because GIMP's UI file export process is notoriously slow
when working with directories containing many files. If GIMP properly optimized their file I/O
operations, this wrapper would be unnecessary.

This module provides a wrapper around GIMP initialization to handle temporary file operations
when working with directories containing many files. It avoids slowness by copying files
to a temporary directory for editing and then handling the results appropriately.

The wrapper handles several scenarios:
1. No change occurred to the file copied over - delete the temp file
2. The original file was overwritten - move the file from temp to original location
3. New files present with no filename conflicts - move files to source directory
4. New files present with filename conflicts - ask user to confirm overwrite

Process Management:
The wrapper handles multiple initializations within the same GIMP session:
- Opening the same file multiple times is ignored with user notification
- Opening different files in an existing GIMP process preserves previous work and transitions cleanly
- Automatic cleanup of stale temporary directories on application startup
- Thread-safe process monitoring with proper wrapper instance management
"""

import os
import shutil
import subprocess
import tempfile
import time
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
import traceback

from utils.config import config
from utils.logging_setup import get_logger
from utils.running_tasks_registry import start_thread
from utils.translations import _
from utils.utils import Utils

logger = get_logger("gimp_wrapper")


class FileState(Enum):
    """Enumeration of possible file states."""
    NO_EXIST = "no_exist"
    NO_CONFLICT = "no_conflict"
    CONFLICT = "conflict"


class FileGroup(Enum):
    """Enumeration of file group contexts."""
    ORIGINAL_FILE = "original_file"
    NEW_FILES = "new_files"


# Global state for GIMP process management
_gimp_process_lock = threading.Lock()
_gimp_process = None
_current_filepath = None
_is_gimp_running = False
_current_wrapper = None  # Reference to the current active wrapper instance
_process_monitor_thread = None  # Reference to the process monitoring thread


def _cleanup_stale_temp_directories() -> None:
    """
    Clean up stale temporary directories from previous application runs.
    This should be called on application startup.
    """
    import tempfile
    import glob
    
    try:
        # Get the system temp directory
        temp_base = tempfile.gettempdir()
        
        # Look for sic_gimp_wrapper_* directories
        pattern = os.path.join(temp_base, "sic_gimp_wrapper_*")
        stale_dirs = glob.glob(pattern)
        
        for stale_dir in stale_dirs:
            try:
                if os.path.isdir(stale_dir):
                    # Check if directory is empty or contains only old files
                    files = os.listdir(stale_dir)
                    if not files:  # Empty directory
                        shutil.rmtree(stale_dir)
                        logger.info(f"Cleaned up empty stale temp directory: {stale_dir}")
                    else:
                        # Check if files are older than 1 hour (safety check)
                        current_time = time.time()
                        all_old = True
                        for file in files:
                            file_path = os.path.join(stale_dir, file)
                            if os.path.isfile(file_path):
                                file_age = current_time - os.path.getmtime(file_path)
                                if file_age < 3600:  # Less than 1 hour old
                                    all_old = False
                                    break
                        
                        if all_old:
                            shutil.rmtree(stale_dir)
                            logger.info(f"Cleaned up stale temp directory: {stale_dir}")
                        else:
                            logger.debug(f"Keeping temp directory with recent files: {stale_dir}")
                            
            except Exception as e:
                logger.warning(f"Error cleaning up stale temp directory {stale_dir}: {e}")
                
    except Exception as e:
        logger.error(f"Error during stale temp directory cleanup: {e}")


# Clean up stale temp directories on module import
_cleanup_stale_temp_directories()


class GimpWrapper:
    """
    Wrapper for GIMP operations that handles temporary file management
    for directories with many files to avoid performance issues.
    """
    
    def __init__(self, files_threshold_reached: Callable[[], bool], app_actions):
        """
        Initialize the GIMP wrapper.
        
        Args:
            files_threshold_reached: Callback function that returns True if directory has many files
            app_actions: App actions instance for UI interactions
        """
        self.files_threshold_reached = files_threshold_reached
        self.app_actions = app_actions
        
        # Validate app_actions parameter
        if app_actions is None:
            raise ValueError("app_actions cannot be None")
        
        # Check that app_actions has the required methods
        required_methods = ['toast', 'alert']
        for method_name in required_methods:
            if not hasattr(app_actions, method_name):
                raise ValueError(f"app_actions must have a '{method_name}' method")
            if not callable(getattr(app_actions, method_name)):
                raise ValueError(f"app_actions.{method_name} must be callable")
        
        self._temp_dir = None
        self._original_file_path = None
        self._temp_file_path = None
        self._original_file_hash = None
        
    def _wait_for_file_release(self, filepath: str, max_wait_seconds: int = 30) -> bool:
        """
        Wait for file to be released by GIMP process.
        
        Args:
            filepath: Path to the file to check
            max_wait_seconds: Maximum time to wait for file release
            
        Returns:
            True if file was released, False if timeout exceeded
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            try:
                # Try to open file in exclusive mode to test if it's released
                with open(filepath, 'r+b') as f:
                    pass  # If we can open it, it's released
                logger.debug(f"File {filepath} released after {time.time() - start_time:.1f}s")
                return True
            except (PermissionError, OSError, IOError):
                # File is still locked, wait a bit before retry
                time.sleep(0.5)
        
        logger.warning(f"File {filepath} still locked after {max_wait_seconds}s timeout")
        return False
    
    def _safe_calculate_hash(self, filepath: str) -> Optional[str]:
        """
        Safely calculate file hash with retry logic for locked files.
        
        Args:
            filepath: Path to the file to hash
            
        Returns:
            Hash string if successful, None if failed
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return Utils.calculate_hash(filepath)
            except (PermissionError, OSError, IOError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Hash calculation failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Hash calculation failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error calculating hash: {e}")
                return None
        return None
    
    def _monitor_gimp_process(self, process) -> None:
        """
        Monitor GIMP process and handle completion properly.
        Uses a custom loop to avoid stranded threads when wrapper changes.
        
        Args:
            process: The subprocess.Popen object for the GIMP process
        """
        try:
            # Use a custom loop instead of blocking wait
            start_time = time.time()
            while True:
                # Check if this wrapper is still active
                global _current_wrapper
                if _current_wrapper != self:
                    logger.info("This wrapper is no longer active, stopping monitoring")
                    return
                
                # Check if process has terminated (non-blocking)
                if process.poll() is not None:
                    elapsed_time = time.time() - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    if minutes > 0:
                        logger.info(f"GIMP process completed (took {minutes}m {seconds:.1f}s)")
                    else:
                        logger.info(f"GIMP process completed (took {seconds:.1f}s)")
                    break
                
                # Wait a bit before checking again
                time.sleep(1)
            
            # Double-check we're still the current wrapper after process termination
            if _current_wrapper != self:
                logger.info("This wrapper is no longer active, skipping completion handling")
                return
            
            # Additional check: ensure file handles are released
            if not self._wait_for_file_release(self._temp_file_path):
                logger.warning("GIMP process terminated but file still locked, proceeding anyway")
            
            # Handle the completion
            self._handle_gimp_completion()
            
        except Exception as e:
            logger.error(f"Error monitoring GIMP process: {e}")
            traceback.print_exc()
            self.app_actions.alert(
                _("Process Error"),
                _("Error monitoring GIMP process: {0}").format(str(e)),
                kind="error"
            )
            self._cleanup_temp_directory()
    
    def _handle_previous_wrapper_completion(self, previous_wrapper: "GimpWrapper") -> None:
        """
        Handle the completion of the previous wrapper to avoid data loss.
        This ensures the user's changes are processed before opening a new file.
        
        Args:
            previous_wrapper: The previous wrapper instance to handle (type: GimpWrapper)
        """
        try:
            logger.info("Processing previous wrapper completion to preserve user changes")
            
            # If the previous wrapper was using temp directory, we need to handle its completion
            if previous_wrapper._temp_dir and os.path.exists(previous_wrapper._temp_dir):
                logger.info("Previous wrapper had temp directory, processing its completion")
                
                # Wait for file handles to be released
                if previous_wrapper._temp_file_path:
                    if not previous_wrapper._wait_for_file_release(previous_wrapper._temp_file_path):
                        logger.warning("Previous wrapper's temp file still locked, proceeding anyway")
                
                # Process the completion as if GIMP had just closed
                previous_wrapper._handle_gimp_completion()
                
            else:
                logger.info("Previous wrapper was using direct mode or temp directory was deleted, no completion needed")
                
        except Exception as e:
            logger.error(f"Error handling previous wrapper completion: {e}")
            traceback.print_exc()
            # Don't show error to user as this is internal cleanup
            # Just clean up the temp directory to avoid resource leaks
            try:
                previous_wrapper._cleanup_temp_directory()
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up previous wrapper: {cleanup_error}")
    
    def unload_current_file_from_gimp(self, gimp_exe_loc: str) -> None:
        """
        Attempt to unload the current file from GIMP by calling it with no arguments.
        This helps release file handles before processing previous wrapper completion.
        
        Args:
            gimp_exe_loc: Path to GIMP executable
        """
        logger.info("Attempting to unload current file from GIMP")
        try:
            # Call GIMP with no arguments to potentially unload current file
            unload_command = ["set", "LANG=en", "&&", gimp_exe_loc]
            subprocess.call(unload_command, shell=True, timeout=5)
            logger.info("GIMP unload command completed")
        except subprocess.TimeoutExpired:
            logger.warning("GIMP unload command timed out, proceeding anyway")
        except Exception as e:
            logger.warning(f"GIMP unload command failed: {e}, proceeding anyway")
        
    def open_image_in_gimp(self, filepath: str, gimp_exe_loc: str) -> None:
        """
        Open an image in GIMP, using temporary directory if needed.
        Handles concurrent requests by letting GIMP manage multiple files in same process.
        
        Args:
            filepath: Path to the image file to open
            gimp_exe_loc: Path to GIMP executable
        """
        global _gimp_process, _current_filepath, _is_gimp_running, _current_wrapper
        
        with _gimp_process_lock:
            # Check if same file is already being processed
            if _current_filepath == filepath and _is_gimp_running:
                logger.info(f"Ignoring request to open same file: {filepath}")
                self.app_actions.toast(_("File is already being opened in GIMP"))
                return
            
            # If GIMP is already running with a different file, handle wrapper transition
            if _is_gimp_running and _current_filepath != filepath:
                logger.info(f"Opening different file in existing GIMP process: {filepath}")
                
                # Try to unload current file from GIMP before processing previous wrapper
                self.unload_current_file_from_gimp(gimp_exe_loc)
                
                # Handle the previous wrapper properly to avoid data loss
                if _current_wrapper is not None:
                    logger.info("Handling previous wrapper completion before opening new file")
                    self._handle_previous_wrapper_completion(_current_wrapper)
            
            # Use the same logic for both new processes and existing processes
            if self._should_use_temp_directory():
                self._open_with_temp_directory(filepath, gimp_exe_loc)
            else:
                # The method will handle the appropriate toast message and process management
                self._open_directly(filepath, gimp_exe_loc)
    
    def _should_use_temp_directory(self) -> bool:
        """
        Determine if we should use a temporary directory based on file count.
        
        Returns:
            True if the directory is slow (many files), False otherwise
        """
        try:
            return self.files_threshold_reached()
        except Exception as e:
            logger.warning(f"Error checking if directory is slow: {e}")
            return False
    
    def _open_directly(self, filepath: str, gimp_exe_loc: str) -> None:
        """
        Open file directly in GIMP without temporary directory.
        
        Args:
            filepath: Path to the image file
            gimp_exe_loc: Path to GIMP executable
        """
        global _gimp_process, _current_filepath, _is_gimp_running, _current_wrapper
        
        # Show appropriate toast message based on GIMP state
        filename = os.path.basename(filepath)
        if _is_gimp_running and _current_filepath != filepath:
            self.app_actions.toast(_("Opening {0} in existing GIMP process").format(filename))
        else:
            self.app_actions.toast(_("Opening file in GIMP: {0}").format(filename))
        
        # Track current process
        _current_filepath = filepath
        _current_wrapper = self  # Set this instance as the current wrapper
        
        def gimp_process():
            global _gimp_process, _current_filepath, _is_gimp_running, _current_wrapper
            try:
                start_time = time.time()
                command = ["set", "LANG=en", "&&", gimp_exe_loc, filepath]
                process = subprocess.Popen(command, shell=True)
                _gimp_process = process
                _is_gimp_running = True
                
                # Use non-blocking monitoring instead of process.wait()
                while True:
                    # Check if this wrapper is still active
                    if _current_wrapper != self:
                        logger.info("This wrapper is no longer active, stopping monitoring")
                        return
                    
                    # Check if process has terminated (non-blocking)
                    if process.poll() is not None:
                        elapsed_time = time.time() - start_time
                        minutes = int(elapsed_time // 60)
                        seconds = elapsed_time % 60
                        if minutes > 0:
                            logger.info(f"GIMP process completed (took {minutes}m {seconds:.1f}s)")
                        else:
                            logger.info(f"GIMP process completed (took {seconds:.1f}s)")
                        break
                    
                    # Wait a bit before checking again
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in direct GIMP process: {e}")
                traceback.print_exc()
                self.app_actions.alert(
                    _("GIMP Error"), 
                    _("Failed to open file in GIMP: {0}").format(str(e)),
                    kind="error"
                )
            finally:
                # Clear process tracking
                with _gimp_process_lock:
                    _gimp_process = None
                    _current_filepath = None
                    _is_gimp_running = False
                    _current_wrapper = None
        
        start_thread(gimp_process)
    
    def _open_with_temp_directory(self, filepath: str, gimp_exe_loc: str) -> None:
        """
        Open file in GIMP using temporary directory for slow directories.
        
        Args:
            filepath: Path to the image file
            gimp_exe_loc: Path to GIMP executable
        """
        try:
            # Set up temporary directory and copy file
            self._setup_temp_directory(filepath)
            
            # Show informative toast for temp directory usage
            filename = os.path.basename(filepath)
            self.app_actions.toast(_("Opening {0} in GIMP (using temporary directory for performance)").format(filename))
            
            global _gimp_process, _current_filepath, _is_gimp_running, _current_wrapper
            
            # Track current process
            _current_filepath = filepath
            _current_wrapper = self  # Set this instance as the current wrapper
            
            # Start GIMP process in a thread
            def gimp_process():
                global _gimp_process, _current_filepath, _is_gimp_running, _current_wrapper
                try:
                    # Use subprocess.Popen for better process control
                    process = subprocess.Popen(
                        ["set", "LANG=en", "&&", gimp_exe_loc, self._temp_file_path],
                        shell=True
                    )
                    
                    # Store process reference for potential termination
                    _gimp_process = process
                    _is_gimp_running = True
                    
                    # Monitor the process properly
                    self._monitor_gimp_process(process)
                    
                except Exception as e:
                    logger.error(f"Error in GIMP process: {e}")
                    traceback.print_exc()
                    
                    # Show error to user via alert
                    self.app_actions.alert(
                        _("GIMP Error"), 
                        _("Failed to open file in GIMP: {0}\n\nCheck that GIMP is properly installed and configured.").format(str(e)),
                        kind="error"
                    )
                    self._cleanup_temp_directory()
                finally:
                    # Clear process tracking
                    with _gimp_process_lock:
                        _gimp_process = None
                        _current_filepath = None
                        _is_gimp_running = False
                        _current_wrapper = None
            
            start_thread(gimp_process)
            
        except Exception as e:
            logger.error(f"Error setting up temporary directory: {e}")
            traceback.print_exc()
            self.app_actions.alert(
                _("Setup Error"),
                _("Failed to set up temporary directory: {0}").format(str(e)),
                kind="error"
            )
            self._cleanup_temp_directory()
            raise
    
    def _setup_temp_directory(self, filepath: str) -> None:
        """
        Set up temporary directory and copy the file.
        
        Args:
            filepath: Path to the original image file
        """
        # Create temporary directory
        self._temp_dir = tempfile.mkdtemp(prefix="sic_gimp_wrapper_")
        
        # Store original file path
        self._original_file_path = filepath
        
        # Create temporary file path
        filename = os.path.basename(filepath)
        self._temp_file_path = os.path.join(self._temp_dir, filename)
        
        # Copy file to temporary directory
        shutil.copy2(filepath, self._temp_file_path)
        
        # Calculate hash of original file for comparison
        self._original_file_hash = self._safe_calculate_hash(filepath)
        if self._original_file_hash is None:
            raise Exception(f"Failed to calculate hash for original file: {filepath}")
        
        logger.info(f"Temporary directory created: {self._temp_dir}")
        logger.info(f"File copied to temp: {os.path.basename(filepath)}")
    
    def _handle_gimp_completion(self) -> None:
        """
        Handle the completion of GIMP process and manage file changes.
        """
        try:
            # Check what happened to the files
            temp_files = self._get_temp_directory_files()
            
            # Determine the states and handle accordingly
            states = self._determine_state(temp_files)
            self._handle_scenarios(states, temp_files)
            
        except Exception as e:
            logger.error(f"Error handling GIMP completion: {e}")
            traceback.print_exc()
            
            # Show error to user
            self.app_actions.alert(
                _("File Processing Error"),
                _("Error processing files after GIMP completion: {0}").format(str(e)),
                kind="error"
            )
        finally:
            # Always clean up temporary directory
            self._cleanup_temp_directory()
    
    def _get_temp_directory_files(self) -> List[str]:
        """
        Get list of files in the temporary directory.
        
        Returns:
            List of file paths in the temporary directory
        """
        if not self._temp_dir or not os.path.exists(self._temp_dir):
            return []
        
        files = []
        for item in os.listdir(self._temp_dir):
            item_path = os.path.join(self._temp_dir, item)
            if os.path.isfile(item_path):
                files.append(item_path)
        
        return files
    
    def _determine_state(self, temp_files: List[str]) -> Dict[FileGroup, FileState]:
        """
        Determine the state of original file and new files based on temp directory contents.
        Multiple cases can occur simultaneously.
        
        Args:
            temp_files: List of files in temporary directory
            
        Returns:
            Dictionary with state flags: {
                FileGroup.ORIGINAL_FILE: FileState,
                FileGroup.NEW_FILES: FileState
            }
        """
        original_filename = os.path.basename(self._original_file_path)
        temp_original_path = os.path.join(self._temp_dir, original_filename)
        
        states = {}
        
        # Determine original file state
        if os.path.exists(temp_original_path):
            current_hash = self._safe_calculate_hash(temp_original_path)
            if current_hash is None:
                logger.error(f"Failed to calculate hash for temp file: {temp_original_path}")
                # Assume file was modified if we can't calculate hash
                states[FileGroup.ORIGINAL_FILE] = FileState.CONFLICT
            elif current_hash == self._original_file_hash:
                states[FileGroup.ORIGINAL_FILE] = FileState.NO_CONFLICT  # No changes
            else:
                states[FileGroup.ORIGINAL_FILE] = FileState.CONFLICT  # Modified
        else:
            states[FileGroup.ORIGINAL_FILE] = FileState.NO_EXIST  # File was deleted/renamed
        
        # Determine new files state (excluding the original file)
        new_files = [f for f in temp_files if os.path.basename(f) != original_filename]
        
        if new_files:
            source_dir = os.path.dirname(self._original_file_path)
            existing_files = set(os.listdir(source_dir)) if os.path.exists(source_dir) else set()
            
            new_filenames = [os.path.basename(f) for f in new_files]
            conflicts = [f for f in new_filenames if f in existing_files]
            
            if conflicts:
                states[FileGroup.NEW_FILES] = FileState.CONFLICT
            else:
                states[FileGroup.NEW_FILES] = FileState.NO_CONFLICT
        else:
            states[FileGroup.NEW_FILES] = FileState.NO_EXIST
        
        return states
    
    def _handle_scenarios(self, states: Dict[FileGroup, FileState], temp_files: List[str]) -> None:
        """
        Handle multiple scenarios that can occur simultaneously.
        
        Args:
            states: Dictionary with file state flags
            temp_files: List of files in temporary directory
        """
        logger.info(f"Processing file changes: original={states.get(FileGroup.ORIGINAL_FILE).value}, new_files={states.get(FileGroup.NEW_FILES).value}")
        
        # Handle original file changes first
        original_state = states.get(FileGroup.ORIGINAL_FILE)
        if original_state == FileState.CONFLICT:
            self._handle_original_modified(temp_files)
        elif original_state == FileState.NO_CONFLICT:
            self._handle_original_unchanged()
        # NO_EXIST case for original file is ignored
        
        # Handle new files (excluding the original file)
        original_filename = os.path.basename(self._original_file_path)
        new_files = [f for f in temp_files if os.path.basename(f) != original_filename]
        
        new_files_state = states.get(FileGroup.NEW_FILES)
        if new_files_state == FileState.CONFLICT:
            self._handle_new_files_conflict(new_files)
        elif new_files_state == FileState.NO_CONFLICT:
            self._handle_new_files_no_conflict(new_files)
        # NO_EXIST case for new files means no new files were created
    
    def _handle_original_unchanged(self) -> None:
        """Handle case where original file was unchanged."""
        message = _("Original file unchanged - no modifications detected")
        self.app_actions.warn(message)
        # Files will be cleaned up in _cleanup_temp_directory()
    
    def _handle_original_modified(self, temp_files: List[str]) -> None:
        """
        Handle case where original file was modified.
        
        Args:
            temp_files: List of files in temporary directory
        """
        logger.info("Original file was modified - updating source file")
        
        original_filename = os.path.basename(self._original_file_path)
        temp_original_path = os.path.join(self._temp_dir, original_filename)
        
        if os.path.exists(temp_original_path):
            try:
                # Move the modified file back to original location
                shutil.move(temp_original_path, self._original_file_path)
                logger.info(f"Moving modified file back: {os.path.basename(self._original_file_path)}")
                
                # Show toast notification
                self.app_actions.toast(_("File updated: {0}").format(original_filename))
                
            except Exception as e:
                logger.error(f"Error moving file back: {e}")
                self.app_actions.alert(_("Error"), _("Failed to update file: {0}").format(str(e)))
    
    def _handle_new_files_no_conflict(self, temp_files: List[str]) -> None:
        """
        Handle case where new files were created with no conflicts.
        
        Args:
            temp_files: List of files in temporary directory
        """
        logger.info(f"New files detected: {len(temp_files)} files to move")
        
        source_dir = os.path.dirname(self._original_file_path)
        
        try:
            for temp_file in temp_files:
                filename = os.path.basename(temp_file)
                dest_path = os.path.join(source_dir, filename)
                shutil.move(temp_file, dest_path)
                logger.info(f"Moving new file: {os.path.basename(temp_file)}")
            
            # Show toast notification
            file_count = len(temp_files)
            if file_count == 1:
                self.app_actions.success(_("New file created: {0}").format(os.path.basename(temp_files[0])))
            else:
                self.app_actions.success(_("{0} new files created").format(file_count))
                
        except Exception as e:
            logger.error(f"Error moving new files: {e}")
            self.app_actions.alert(_("Error"), _("Failed to move new files: {0}").format(str(e)))
    
    def _handle_new_files_conflict(self, temp_files: List[str]) -> None:
        """
        Handle case where new files were created with filename conflicts.
        
        Args:
            temp_files: List of files in temporary directory
        """
        source_dir = os.path.dirname(self._original_file_path)
        existing_files = set(os.listdir(source_dir)) if os.path.exists(source_dir) else set()
        
        # Find conflicting files
        conflicting_files = []
        for temp_file in temp_files:
            filename = os.path.basename(temp_file)
            if filename in existing_files:
                conflicting_files.append(filename)
        
        logger.info(f"New files with conflicts detected: {len(conflicting_files)} files need confirmation")
        
        if conflicting_files:
            # Ask user for confirmation
            conflict_list = "\n".join(conflicting_files)
            message = _("The following files already exist and will be overwritten:\n\n{0}\n\nDo you want to proceed?").format(conflict_list)
            
            # Use the app's alert method for consistency
            result = self.app_actions.alert(_("Confirm Overwrite"), message, kind="askokcancel")
            if result:
                # User confirmed, proceed with moving files
                self._handle_new_files_no_conflict(temp_files)
            else:
                logger.info("User cancelled file overwrite")
                self.app_actions.toast(_("File overwrite cancelled"))
    
    def _cleanup_temp_directory(self) -> None:
        """Clean up the temporary directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")
        
        # Reset instance variables
        self._temp_dir = None
        self._original_file_path = None
        self._temp_file_path = None
        self._original_file_hash = None


def open_image_in_gimp_wrapper(filepath: str, gimp_exe_loc: str, files_threshold_reached: Callable[[], bool], app_actions) -> None:
    """
    Convenience function to open an image in GIMP using the wrapper.
    
    Args:
        filepath: Path to the image file to open
        gimp_exe_loc: Path to GIMP executable
        files_threshold_reached: Callback function that returns True if directory has many files
        app_actions: App actions instance
    """
    wrapper = GimpWrapper(files_threshold_reached, app_actions)
    wrapper.open_image_in_gimp(filepath, gimp_exe_loc)
