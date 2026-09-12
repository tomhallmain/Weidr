from datetime import datetime
import os

from PIL import Image

from files.file_metadata_cache import file_metadata_cache
from files.related_image import extract_filename_base_stem, get_origin_basename
from image.frame_cache import FrameCache
from image.image_data_extractor import image_data_extractor

class SortableFile:
    def __init__(self, full_file_path, dir_entry=None):
        """*dir_entry* is the ``os.DirEntry`` from the scan that discovered this
        file, when the caller has one.

        Windows fills a DirEntry's size and timestamps from the directory
        enumeration itself, so ``dir_entry.stat()`` costs no syscall there; on
        other platforms it issues the same one ``os.stat`` would have. Callers
        pass it only when they are about to build a file that is not already
        cached, so the lazy ``stat()`` never runs for a file that needed no
        metadata.
        """
        self.full_file_path = full_file_path
        self.basename = os.path.basename(full_file_path)
        self.name_length = len(self.basename)
        self.root, self.extension = os.path.splitext(self.basename)
        self.suffix = None
        self.related_image_path = None
        self.related_image_path_key = self.full_file_path
        self._image_dimensions = None
        cached = file_metadata_cache.get(full_file_path)
        if cached is None or not self._apply_cached_metadata(cached):
            try:
                stat_obj = dir_entry.stat() if dir_entry is not None else os.stat(full_file_path)
                self.ctime = datetime.fromtimestamp(stat_obj.st_ctime)
                self.mtime = datetime.fromtimestamp(stat_obj.st_mtime)
                self.size = stat_obj.st_size
            except Exception:
                self.ctime = datetime.fromtimestamp(0)
                self.mtime = datetime.fromtimestamp(0)
                self.size = 0
            else:
                # Only a stat that succeeded is worth keeping; the zeroed
                # fallback above would otherwise be served for every later
                # session as though it were real.
                file_metadata_cache.update_stat(
                    full_file_path, stat_obj.st_ctime, stat_obj.st_mtime, stat_obj.st_size
                )
        self.tags = self.get_tags()

    def _apply_cached_metadata(self, cached) -> bool:
        """Adopt a cache entry's metadata, reporting whether it was usable.

        A False return sends the caller to a real stat. The entry comes from a
        blob that persists across sessions, so a truncated or out-of-range
        value in it must not take the whole directory load down.
        """
        if "size" not in cached:
            return False
        try:
            self.ctime = datetime.fromtimestamp(cached["ctime"])
            self.mtime = datetime.fromtimestamp(cached["mtime"])
            self.size = cached["size"]
        except Exception:
            return False
        if cached.get("image_width") is not None:
            self._image_dimensions = (cached["image_width"], cached["image_height"])
        if cached.get("related_image_path") is not None:
            self.related_image_path = cached["related_image_path"]
        return True

    def get_tags(self):
        tags = []

        # TODO
        # try:
        #     pass
        # except Exception:
        #     pass

        return tags

    def set_suffix(self):
        base_stem = extract_filename_base_stem(self.basename)
        if base_stem and len(base_stem) < len(self.root):
            self.suffix = self.root[len(base_stem):]
        else:
            self.suffix = ""

    def get_suffix(self):
        if self.suffix is None:
            self.set_suffix()
        return self.suffix

    def set_related_image_path(self):
        self.related_image_path = image_data_extractor.get_related_image_path(self.full_file_path)
        if self.related_image_path is None:
            self.related_image_path = ""
        file_metadata_cache.update_related_image_path(
            self.full_file_path, self.related_image_path
        )

    def get_related_image_or_self(self):
        if self.related_image_path is None:
            self.set_related_image_path()
            return self.get_related_image_or_self()
        elif len(self.related_image_path) > 0:
            return os.path.basename(self.related_image_path)
        else:
            return self.basename

    def get_origin_image_or_self(self, basename_lookup, visited=None):
        return get_origin_basename(self, basename_lookup, visited)

    def __eq__(self, other):
        if not isinstance(other, SortableFile):
            return False
        return (
            self.full_file_path == other.full_file_path
            and self.ctime == other.ctime
            and self.mtime == other.mtime
            and self.size == other.size
            )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.full_file_path, self.ctime, self.mtime, self.size))

    def get_image_dimensions(self):
        """
        Get the image dimensions (width, height) for this file.
        Returns (width, height) tuple or None if not an image or dimensions can't be determined.
        """
        if self._image_dimensions is None:
            try:
                self._image_dimensions = Image.open(self.full_file_path).size
            except Exception:
                try:
                    image_path = FrameCache.get_image_path(self.full_file_path)
                    self._image_dimensions = Image.open(image_path).size
                except Exception:
                    self._image_dimensions = (0, 0)
            width, height = self._image_dimensions
            if width > 0 and height > 0:
                # (0, 0) means neither the file nor a rendered frame could be
                # read. Leaving that uncached lets a later session retry rather
                # than sorting the file as zero-pixel for good.
                file_metadata_cache.update_dimensions(self.full_file_path, width, height)
        return self._image_dimensions

    def get_image_pixels(self):
        """
        Get the total number of pixels in the image (width * height).
        Returns 0 if not an image or dimensions can't be determined.
        """
        dimensions = self.get_image_dimensions()
        if dimensions is None:
            return 0
        return dimensions[0] * dimensions[1]

    def get_image_height(self):
        """
        Get the image height.
        Returns 0 if not an image or dimensions can't be determined.
        """
        dimensions = self.get_image_dimensions()
        if dimensions is None:
            return 0
        return dimensions[1]

    def get_image_width(self):
        """
        Get the image width.
        Returns 0 if not an image or dimensions can't be determined.
        """
        dimensions = self.get_image_dimensions()
        if dimensions is None:
            return 0
        return dimensions[0]