from typing import List, Optional

from utils.logging_setup import get_logger
from utils.translations import _
logger = get_logger("directory_profile")


class DirectoryProfile:
    """Represents a profile that groups multiple directories together."""

    directory_profiles: List['DirectoryProfile'] = []
    
    def __init__(self, name="", directories=None):
        self.name = name  # Unique name to identify this profile
        self.directories = directories if directories is not None else []  # List of directory paths
    
    def __eq__(self, other):
        """Check equality based on name and directories (order-insensitive, duplicate-insensitive)."""
        if not isinstance(other, DirectoryProfile):
            return False
        return self.name == other.name and set(self.directories) == set(other.directories)
    
    def __hash__(self):
        """Hash based on name and sorted directories tuple."""
        return hash((self.name, tuple(sorted(set(self.directories)))))
    
    def to_dict(self):
        """Serialize to dictionary for caching."""
        return {
            "name": self.name,
            "directories": self.directories,
        }
    
    @staticmethod
    def from_dict(d):
        """Deserialize from dictionary."""
        return DirectoryProfile(
            name=d.get("name", ""),
            directories=d.get("directories", [])
        )
    
    @staticmethod
    def add_profile(profile: 'DirectoryProfile') -> bool:
        """
        Add a profile to the list.
        
        Args:
            profile: The DirectoryProfile to add
            
        Returns:
            True if added successfully, False if profile with same name already exists
        """
        # Check for duplicate name
        existing = DirectoryProfile.get_profile_by_name(profile.name)
        if existing is not None:
            logger.error(f"Profile with name {profile.name} already exists")
            return False
        
        DirectoryProfile.directory_profiles.append(profile)
        logger.info(f"Added profile: {profile.name}")
        return True
    
    @staticmethod
    def update_profile(old_name: str, new_profile: 'DirectoryProfile') -> bool:
        """
        Update an existing profile.
        
        Args:
            old_name: The old name of the profile
            new_profile: The updated profile
            
        Returns:
            True if updated successfully, False if new name conflicts with existing profile
        """
        # Find the old profile
        old_profile = DirectoryProfile.get_profile_by_name(old_name)
        if old_profile is None:
            logger.error(f"Profile {old_name} not found")
            return False
        
        # Check if new name conflicts (unless it's the same profile)
        if new_profile.name != old_name:
            existing = DirectoryProfile.get_profile_by_name(new_profile.name)
            if existing is not None and existing != old_profile:
                logger.error(f"Profile with name {new_profile.name} already exists")
                return False
        
        # Update the profile in place
        old_profile.name = new_profile.name
        old_profile.directories = new_profile.directories
        
        logger.info(f"Updated profile: {old_name} -> {new_profile.name}")
        return True

    @staticmethod
    def get_profile_by_name(name: str) -> Optional['DirectoryProfile']:
        """Get a profile by name. Returns None if not found."""
        for profile in DirectoryProfile.directory_profiles:
            if name == profile.name:
                return profile
        return None