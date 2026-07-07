from datetime import datetime
import os

from PIL import Image

from files.related_image import extract_filename_base_stem, get_origin_basename
from image.frame_cache import FrameCache
from image.image_data_extractor import image_data_extractor

class SortableFile:
    def __init__(self, full_file_path):
        self.full_file_path = full_file_path
        self.basename = os.path.basename(full_file_path)
        self.name_length = len(self.basename)
        self.root, self.extension = os.path.splitext(self.basename)
        self.suffix = None
        self.related_image_path = None
        self.related_image_path_key = self.full_file_path
        self._image_dimensions = None
        try:
            stat_obj = os.stat(full_file_path)
            self.ctime = datetime.fromtimestamp(stat_obj.st_ctime)
            self.mtime = datetime.fromtimestamp(stat_obj.st_mtime)
            self.size = stat_obj.st_size
        except Exception:
            self.ctime = datetime.fromtimestamp(0)
            self.mtime = datetime.fromtimestamp(0)
            self.size = 0
        self.tags = self.get_tags()

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