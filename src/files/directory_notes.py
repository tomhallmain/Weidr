import json
import os
from typing import Dict, List, Optional, Tuple

from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("directory_notes")


class DirectoryNotes:
    """
    Manages notes and marked files for individual base directories.
    This is separate from the runtime marked files used for moving files.
    """
    
    MARKED_FILES_KEY = "directory_notes_marked_files"
    FILE_NOTES_KEY = "directory_notes_file_notes"
    
    @staticmethod
    def normalize_directory(directory: str) -> str:
        """Normalize directory path for consistent storage."""
        return app_info_cache.normalize_directory_key(directory)
    
    @staticmethod
    def get_marked_files(base_dir: str) -> List[str]:
        """Get the list of marked files for a directory."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        return list(app_info_cache.get(normalized_dir, DirectoryNotes.MARKED_FILES_KEY, default_val=[]))
    
    @staticmethod
    def add_marked_file(base_dir: str, filepath: str) -> bool:
        """
        Add a file to the marked files list for a directory.
        Returns True if added, False if already present.
        """
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        marked_files = DirectoryNotes.get_marked_files(base_dir)
        if filepath not in marked_files:
            marked_files.append(filepath)
            app_info_cache.set(normalized_dir, DirectoryNotes.MARKED_FILES_KEY, marked_files)
            app_info_cache.store()
            return True
        return False
    
    @staticmethod
    def remove_marked_file(base_dir: str, filepath: str) -> bool:
        """
        Remove a file from the marked files list for a directory.
        Returns True if removed, False if not present.
        """
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        marked_files = DirectoryNotes.get_marked_files(base_dir)
        if filepath in marked_files:
            marked_files.remove(filepath)
            app_info_cache.set(normalized_dir, DirectoryNotes.MARKED_FILES_KEY, marked_files)
            app_info_cache.store()
            return True
        return False
    
    @staticmethod
    def clear_marked_files(base_dir: str) -> None:
        """Clear all marked files for a directory."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        app_info_cache.set(normalized_dir, DirectoryNotes.MARKED_FILES_KEY, [])
        app_info_cache.store()
    
    @staticmethod
    def is_marked_file(base_dir: str, filepath: str) -> bool:
        """Check if a file is marked for a directory."""
        marked_files = DirectoryNotes.get_marked_files(base_dir)
        return filepath in marked_files
    
    @staticmethod
    def get_file_note(base_dir: str, filepath: str) -> Optional[str]:
        """Get the note for a specific file."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        file_notes = app_info_cache.get(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, default_val={})
        return file_notes.get(filepath)
    
    @staticmethod
    def set_file_note(base_dir: str, filepath: str, note: str) -> None:
        """Set or update the note for a specific file."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        file_notes = app_info_cache.get(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, default_val={})
        if note.strip():
            file_notes[filepath] = note.strip()
        else:
            # Remove note if empty
            file_notes.pop(filepath, None)
        app_info_cache.set(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, file_notes)
        app_info_cache.store()
    
    @staticmethod
    def remove_file_note(base_dir: str, filepath: str) -> bool:
        """Remove the note for a specific file. Returns True if removed."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        file_notes = app_info_cache.get(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, default_val={})
        if filepath in file_notes:
            del file_notes[filepath]
            app_info_cache.set(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, file_notes)
            app_info_cache.store()
            return True
        return False
    
    @staticmethod
    def get_all_file_notes(base_dir: str) -> Dict[str, str]:
        """Get all file notes for a directory as a dictionary."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        return dict(app_info_cache.get(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, default_val={}))
    
    @staticmethod
    def clear_all_notes(base_dir: str) -> None:
        """Clear all notes for a directory."""
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        app_info_cache.set(normalized_dir, DirectoryNotes.FILE_NOTES_KEY, {})
        app_info_cache.store()
    
    @staticmethod
    def export_to_text(base_dir: str, output_path: Optional[str] = None) -> str:
        """
        Export marked files and notes to a text file.
        Returns the path to the exported file.
        """
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        marked_files = DirectoryNotes.get_marked_files(base_dir)
        file_notes = DirectoryNotes.get_all_file_notes(base_dir)
        
        if output_path is None:
            # Generate default filename based on directory name
            dir_name = os.path.basename(normalized_dir.rstrip(os.sep)) or "root"
            safe_dir_name = "".join(c for c in dir_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = os.path.join(normalized_dir, f"{safe_dir_name}_notes.txt")
        
        lines = []
        lines.append(f"Directory Notes Export")
        lines.append(f"Directory: {normalized_dir}")
        lines.append("=" * 80)
        lines.append("")
        
        # Marked files section
        lines.append("MARKED FILES")
        lines.append("-" * 80)
        if marked_files:
            for i, filepath in enumerate(marked_files, 1):
                basename = os.path.basename(filepath)
                lines.append(f"{i}. {basename}")
                lines.append(f"   {filepath}")
                lines.append("")
        else:
            lines.append("(No marked files)")
            lines.append("")
        
        # File notes section
        lines.append("FILE NOTES")
        lines.append("-" * 80)
        if file_notes:
            for filepath, note in sorted(file_notes.items()):
                basename = os.path.basename(filepath)
                lines.append(f"{basename}")
                lines.append(f"Path: {filepath}")
                lines.append(f"Note: {note}")
                lines.append("")
        else:
            lines.append("(No file notes)")
            lines.append("")
        
        # Write to file
        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Exported directory notes to {output_path}")
        return output_path
    
    @staticmethod
    def has_any_notes(base_dir: Optional[str] = None) -> bool:
        """
        Check if there are any notes or marked files.
        If base_dir is None, checks all directories.
        """
        if base_dir:
            normalized_dir = DirectoryNotes.normalize_directory(base_dir)
            marked_files = DirectoryNotes.get_marked_files(base_dir)
            file_notes = DirectoryNotes.get_all_file_notes(base_dir)
            return len(marked_files) > 0 or len(file_notes) > 0
        
        # Check all directories
        directory_info = app_info_cache._get_directory_info()
        for directory in directory_info.keys():
            marked_files = DirectoryNotes.get_marked_files(directory)
            file_notes = DirectoryNotes.get_all_file_notes(directory)
            if len(marked_files) > 0 or len(file_notes) > 0:
                return True
        return False
    
    @staticmethod
    def _find_files_by_basename(base_dir: str, basename: str, recursive: bool = True) -> List[str]:
        """
        Find all files in base_dir with the given basename.
        Returns a list of full file paths.
        """
        found_files = []
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        
        if not os.path.isdir(normalized_dir):
            return found_files
        
        def search_directory(directory: str):
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            if entry.name == basename:
                                found_files.append(entry.path)
                        elif entry.is_dir(follow_symlinks=False) and recursive:
                            search_directory(entry.path)
            except PermissionError:
                logger.warning(f"Permission denied: {directory}")
            except Exception as e:
                logger.error(f"Error searching directory {directory}: {e}")
        
        search_directory(normalized_dir)
        return found_files
    
    @staticmethod
    def import_from_text_file(base_dir: str, file_path: str, recursive: bool = True) -> Tuple[int, int, List[str]]:
        """
        Import marked files from a text file where each line is a filename.
        
        Args:
            base_dir: Base directory to search for files
            file_path: Path to the text file to import
            recursive: Whether to search recursively in subdirectories
        
        Returns:
            Tuple of (added_count, not_found_count, not_found_filenames)
        """
        normalized_dir = DirectoryNotes.normalize_directory(base_dir)
        added_count = 0
        not_found_filenames = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Import file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            filename = line.strip()
            if not filename:  # Skip empty lines
                continue
            
            # Find files matching this basename
            found_files = DirectoryNotes._find_files_by_basename(normalized_dir, filename, recursive)
            
            if found_files:
                # Add all found files (in case there are duplicates with same basename)
                for filepath in found_files:
                    if DirectoryNotes.add_marked_file(base_dir, filepath):
                        added_count += 1
            else:
                not_found_filenames.append(filename)
        
        logger.info(f"Imported {added_count} files from text file, {len(not_found_filenames)} not found")
        return added_count, len(not_found_filenames), not_found_filenames
    
    @staticmethod
    def generate_file_list(base_dir: str, output_path: Optional[str] = None) -> str:
        """
        Generate a JSON file list of marked files (list of absolute paths, indent=4).
        Defaults to config.file_paths_json_path when output_path is not given.
        Returns the path to the generated file.
        """
        from utils.config import config
        marked_files = DirectoryNotes.get_marked_files(base_dir)

        if output_path is None:
            output_path = config.file_paths_json_path

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(marked_files, f, indent=4)

        logger.info(f"Generated file list ({len(marked_files)} entries) to {output_path}")
        return output_path

    @staticmethod
    def import_from_json_file(base_dir: str, file_path: str) -> Tuple[int, int, List[str]]:
        """
        Import marked files from a JSON file containing a list of file paths.
        
        Args:
            base_dir: Base directory (for validation, but file paths in JSON should be absolute)
            file_path: Path to the JSON file to import
        
        Returns:
            Tuple of (added_count, invalid_count, invalid_paths)
        """
        added_count = 0
        invalid_paths = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Import file not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of file paths")
        
        for filepath in data:
            if not isinstance(filepath, str):
                invalid_paths.append(str(filepath))
                continue
            
            # Normalize the path
            filepath = os.path.normpath(filepath)
            
            # Check if file exists
            if os.path.isfile(filepath):
                if DirectoryNotes.add_marked_file(base_dir, filepath):
                    added_count += 1
            else:
                invalid_paths.append(filepath)
        
        logger.info(f"Imported {added_count} files from JSON file, {len(invalid_paths)} invalid/missing")
        return added_count, len(invalid_paths), invalid_paths

