import os
import random
import sys
import time
import math

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageEnhance
import PIL.Image

from utils.config import config
from utils.logging_setup import get_logger
from extensions.gimp.gimp_gegl_client import GimpGeglClient

logger = get_logger("image_ops")

from utils.pillow_plugins import ensure_pillow_plugins_registered
ensure_pillow_plugins_registered()


class ImageOps:
    COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white", "gray", "pink", "brown"]
    TEXTURE_DRAW_TYPES = ["perlin", "gaussian", "gradient", "cellular"]
    
    # Class-level cache for GEGL validation
    _gegl_validation_cache = None
    
    # Texture tile cache: {(texture_type, bg_color_key): (tile_texture, timestamp, use_count)}
    # Uses fixed-size tiles that can be tiled across any dimensions
    # bg_color_key is None for default colors, or the RGB tuple for custom colors
    _texture_tile_cache = {}
    _texture_tile_size = 512  # Fixed tile size for caching
    _texture_cache_max_age = 300  # Cache for 5 minutes
    _texture_cache_max_uses = 50  # Regenerate after 50 uses
    
    # Cache of previously used random colors to increase cache hit probability
    _used_random_colors = []
    _max_cached_colors = 50  # Maximum number of random colors to cache

    @staticmethod
    def new_filepath(image_path: str, new_filename: str = "", append_part: str | None = None) -> str:
        dirname = os.path.dirname(image_path)
        if new_filename == "" or append_part is not None:
            basename = os.path.basename(image_path)
            filename_part, ext = os.path.splitext(basename)
            if append_part is not None:
                new_filename = filename_part + append_part + ext
            else:
                new_filename = filename_part + "_crop" + ext
        new_filepath = os.path.join(dirname, new_filename)
        return new_filepath

    @staticmethod
    def crop_image_to_rect(image_path: str, left: int, upper: int, right: int, lower: int) -> str:
        """Crop image to the given pixel bounding box and save as a new sibling file.

        Animated GIFs are handled frame-by-frame with original durations preserved.
        """
        try:
            from PIL import ImageSequence
            from utils.utils import Utils
            new_path = Utils.unique_sibling_path(image_path, "_crop")
            with Image.open(image_path) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames > 1:
                    loop = img.info.get("loop", 0)
                    frames, durations = [], []
                    for frame in ImageSequence.Iterator(img):
                        frames.append(frame.convert("RGBA").crop((left, upper, right, lower)))
                        durations.append(frame.info.get("duration", 100))
                    frames[0].save(
                        new_path,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=loop,
                        optimize=False,
                    )
                else:
                    img.crop((left, upper, right, lower)).save(new_path)
            return new_path
        except Exception as e:
            logger.error("Error in crop_image_to_rect: %s", e)
            return ""

    @staticmethod
    def generate_box_fill_image(width: int, height: int, use_texture: bool | None = None):
        """Return an opaque RGB fill for a box overlay: solid random color or a
        random noise pattern, chosen the same way as the random-draw texture feature.

        *use_texture* forces the branch (True: pattern, False: solid color); when
        None (default) it is chosen randomly via ``config.image_edit_configuration.texture_draw_probability``.
        """
        if use_texture is None:
            use_texture = random.random() < config.image_edit_configuration.texture_draw_probability
        if use_texture:
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture = ImageOps.generate_noise_texture(width, height, texture_type, use_random_color=True)
            return PIL.Image.fromarray(texture)
        return PIL.Image.new("RGB", (width, height), ImageOps.get_random_color())

    @staticmethod
    def draw_box_at_rect(image_path: str, left: int, upper: int, right: int, lower: int) -> str:
        """Paint a random color/pattern box over the given pixel bounding box and save
        as a new sibling file. Does not modify *image_path*.

        Animated GIFs are handled frame-by-frame with original durations preserved.
        """
        try:
            from PIL import ImageSequence
            from utils.utils import Utils
            new_path = Utils.unique_sibling_path(image_path, "_box")
            width, height = right - left, lower - upper
            with Image.open(image_path) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames > 1:
                    loop = img.info.get("loop", 0)
                    frames, durations = [], []
                    for frame in ImageSequence.Iterator(img):
                        frame_rgba = frame.convert("RGBA")
                        frame_rgba.paste(ImageOps.generate_box_fill_image(width, height), (left, upper))
                        frames.append(frame_rgba)
                        durations.append(frame.info.get("duration", 100))
                    frames[0].save(
                        new_path,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=loop,
                        optimize=False,
                    )
                else:
                    out = img.convert("RGB")
                    out.paste(ImageOps.generate_box_fill_image(width, height), (left, upper))
                    out.save(new_path)
            return new_path
        except Exception as e:
            logger.error("Error in draw_box_at_rect: %s", e)
            return ""

    @staticmethod
    def draw_background_box_at_rect(image_path: str, left: int, upper: int, right: int, lower: int) -> str:
        """Paint a random color/pattern fill over everything *outside* the given
        pixel bounding box (the "background"), leaving the box's interior
        unchanged, and save as a new sibling file. Does not modify *image_path*.

        Animated GIFs are handled frame-by-frame with original durations preserved.
        """
        try:
            from PIL import ImageSequence
            from utils.utils import Utils
            new_path = Utils.unique_sibling_path(image_path, "_bgbox")
            with Image.open(image_path) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames > 1:
                    loop = img.info.get("loop", 0)
                    frames, durations = [], []
                    for frame in ImageSequence.Iterator(img):
                        frame_rgba = frame.convert("RGBA")
                        w, h = frame_rgba.size
                        out = ImageOps.generate_box_fill_image(w, h).convert("RGBA")
                        out.paste(frame_rgba.crop((left, upper, right, lower)), (left, upper))
                        frames.append(out)
                        durations.append(frame.info.get("duration", 100))
                    frames[0].save(
                        new_path,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=loop,
                        optimize=False,
                    )
                else:
                    src = img.convert("RGB")
                    w, h = src.size
                    out = ImageOps.generate_box_fill_image(w, h)
                    out.paste(src.crop((left, upper, right, lower)), (left, upper))
                    out.save(new_path)
            return new_path
        except Exception as e:
            logger.error("Error in draw_background_box_at_rect: %s", e)
            return ""

    @staticmethod
    def rotate_image(image_path, right=False):
        try:
            #loading the image into a numpy array 
            img = cv2.imread(image_path)
            #rotating the image
            if right:
                rotated = np.rot90(img, k=-1)
            else:
                rotated = np.rot90(img, k=1)

            current_extension = os.path.splitext(image_path)[-1]
            new_filepath = ImageOps.new_filepath(image_path, append_part="_rot")
            cv2.imwrite(new_filepath, rotated)
            return new_filepath
        except Exception as e:
            logger.error(f'Error in rotate image: {e}')
            return None

    @staticmethod
    def rotate_image_partial(image_path, angle=90, center=None, scale=1.0, texture_probability=0.75):
        """
        Rotate image with random texture or solid color background.
        
        Args:
            image_path: Path to the image file
            angle: Rotation angle in degrees
            center: Rotation center point (defaults to image center)
            scale: Scale factor for rotation
            texture_probability: Probability of using texture (0.0-1.0, default 0.75)
        """
        image = cv2.imread(image_path)
        
        # Randomly decide whether to use texture based on probability
        use_texture = random.random() < texture_probability
        
        if image is None:
            logger.error("rotate_image_partial: could not read %s", image_path)
            return None
        rotated = ImageOps._rotate_image_partial(image, angle=angle, center=center, scale=scale, use_texture=use_texture)
        new_filepath = ImageOps.new_filepath(image_path, append_part="_rot")
        cv2.imwrite(new_filepath, rotated)
        return new_filepath

    @staticmethod
    def get_random_color(true_random_chance=0.75):
        if random.random() < true_random_chance:
            return tuple([random.randint(0, 255) for i in range(3)])
        else:
            return (0, 0, 0) if random.random() > 0.5 else (255, 255, 255)

    @staticmethod
    def generate_noise_texture(width, height, texture_type="perlin", background_color=None, use_random_color=False):
        """
        Generate various types of random textures for background filling.
        Uses tiling-based caching: generates a fixed-size tile once and tiles it across the required dimensions.
        Tiles are cached separately for each background color to maximize reuse.
        
        Args:
            width: Width of the texture
            height: Height of the texture
            texture_type: Type of texture ("perlin", "gaussian", "gradient", "cellular")
            background_color: RGB tuple for background color. If None, uses default based on texture type.
            use_random_color: If True, generates a random background color instead of using default.
        """
        # Generate random color if requested, with 40% chance to reuse a cached color
        if use_random_color:
            if ImageOps._used_random_colors and random.random() < 0.4:
                # 40% chance to reuse a previously used color
                background_color = random.choice(ImageOps._used_random_colors)
            else:
                # 60% chance to generate a new random color
                background_color = ImageOps.get_random_color()
                # Add to cache (remove oldest if cache is full)
                ImageOps._used_random_colors.append(background_color)
                if len(ImageOps._used_random_colors) > ImageOps._max_cached_colors:
                    ImageOps._used_random_colors.pop(0)
        
        current_time = time.time()
        
        # Get or generate a cached tile for this texture type and background color
        tile = ImageOps._get_cached_texture_tile(texture_type, background_color, current_time)
        
        # Tile the cached texture across the required dimensions
        texture = ImageOps._tile_texture(tile, width, height)
        
        return texture
    
    @staticmethod
    def _get_cached_texture_tile(texture_type, background_color, current_time):
        """Get a cached texture tile, generating a new one if needed."""
        # Create cache key: use None for default colors, or the RGB tuple for custom colors
        cache_key = (texture_type, background_color)
        
        # Check if we have a cached tile for this texture type and background color
        if cache_key in ImageOps._texture_tile_cache:
            cached_tile, timestamp, use_count = ImageOps._texture_tile_cache[cache_key]
            age = current_time - timestamp
            
            # Use cached tile if it's still fresh and hasn't been used too many times
            if age < ImageOps._texture_cache_max_age and use_count < ImageOps._texture_cache_max_uses:
                ImageOps._texture_tile_cache[cache_key] = (cached_tile, timestamp, use_count + 1)
                return cached_tile.copy()
            else:
                # Cache expired or overused, remove it
                del ImageOps._texture_tile_cache[cache_key]
        
        # Generate new tile with the specified background color (None = use defaults in generation functions)
        tile_size = ImageOps._texture_tile_size
        if texture_type == "perlin":
            tile = ImageOps._generate_perlin_noise(tile_size, tile_size, background_color)
        elif texture_type == "gaussian":
            tile = ImageOps._generate_gaussian_noise(tile_size, tile_size, background_color)
        elif texture_type == "gradient":
            tile = ImageOps._generate_gradient_texture(tile_size, tile_size, background_color)
        elif texture_type == "cellular":
            tile = ImageOps._generate_cellular_texture(tile_size, tile_size, background_color)
        else:
            # Default to solid color if unknown type
            tile = ImageOps._generate_solid_color(tile_size, tile_size)
        
        # Cache the new tile
        ImageOps._texture_tile_cache[cache_key] = (tile.copy(), current_time, 1)
        return tile
    
    @staticmethod
    def _tile_texture(tile, width, height):
        """Tile a texture tile across the specified dimensions."""
        tile_h, tile_w = tile.shape[:2]
        
        # Calculate how many tiles we need in each direction
        tiles_x = (width + tile_w - 1) // tile_w  # Ceiling division
        tiles_y = (height + tile_h - 1) // tile_h  # Ceiling division
        
        # Create tiled texture
        tiled = np.tile(tile, (tiles_y, tiles_x, 1))
        
        # Crop to exact dimensions if needed
        if tiled.shape[0] > height or tiled.shape[1] > width:
            tiled = tiled[:height, :width]
        
        return tiled

    @staticmethod
    def _apply_background_color(texture, background_color):
        """Apply a background color to a texture, blending with existing colors."""
        if background_color is None:
            return texture
        bg = np.array(background_color, dtype=np.uint8)
        # Blend background with texture (70% texture, 30% background)
        for i in range(3):
            texture[:, :, i] = (texture[:, :, i] * 0.7 + bg[i] * 0.3).astype(np.uint8)
        return texture

    @staticmethod
    def _generate_perlin_noise(width, height, background_color=None):
        """Generate Perlin-like noise texture using OpenCV."""
        # Default to light gray background for perlin (more visible than black)
        if background_color is None:
            background_color = (180, 180, 180)
        
        # Create multiple octaves of noise
        texture = np.full((height, width, 3), background_color, dtype=np.uint8)
        
        for octave in range(3):
            scale = 2 ** octave
            noise = np.random.rand(height // scale, width // scale, 3) * 255
            noise_resized = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
            texture = cv2.addWeighted(texture, 0.7, noise_resized.astype(np.uint8), 0.3, 0)
        
        return texture

    @staticmethod
    def _generate_gaussian_noise(width, height, background_color=None):
        """Generate Gaussian noise texture."""
        # Default to light gray background for gaussian (more visible than black)
        if background_color is None:
            background_color = (180, 180, 180)
        
        # Generate noise with controlled variance around the background color
        bg_array = np.array(background_color, dtype=np.float32)
        noise = np.random.normal(bg_array, 50, (height, width, 3))
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return noise

    @staticmethod
    def _generate_gradient_texture(width, height, background_color=None):
        """Generate radial or linear gradient texture."""
        # Gradient textures don't need a specific background color as they fill the whole space
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        if random.random() > 0.5:
            # Radial gradient
            center_x, center_y = width // 2, height // 2
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            gradient = (distance / max_distance * 255).astype(np.uint8)
            
            # Apply random color tinting
            color = ImageOps.get_random_color()
            for i in range(3):
                texture[:, :, i] = (gradient * color[i] / 255).astype(np.uint8)
        else:
            # Linear gradient
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            if direction == 'horizontal':
                gradient = np.linspace(0, 255, width)
                gradient = np.tile(gradient, (height, 1))
            elif direction == 'vertical':
                gradient = np.linspace(0, 255, height)
                gradient = np.tile(gradient, (width, 1)).T
            else:  # diagonal - optimized to avoid nested loops
                # Use vectorized operations instead of nested loops
                i, j = np.ogrid[:height, :width]
                gradient = ((i + j) / (height + width)) * 255
            
            # Apply random color tinting
            color = ImageOps.get_random_color()
            for i in range(3):
                texture[:, :, i] = (gradient * color[i] / 255).astype(np.uint8)
        
        return texture

    @staticmethod
    def _generate_cellular_texture(width, height, background_color=None):
        """Generate cellular/Voronoi-like texture using optimized vectorized operations."""
        # Default to black background for cellular (pattern is more prominent)
        if background_color is None:
            background_color = (0, 0, 0)
        
        # Create random points
        num_points = random.randint(5, 15)
        points = np.random.rand(num_points, 2) * [width, height]
        
        # Create distance map using vectorized operations (much faster than nested loops)
        y, x = np.ogrid[:height, :width]
        texture = np.full((height, width, 3), background_color, dtype=np.uint8)
        
        # Vectorized distance calculation - calculate distances from all points to all pixels at once
        # This is much faster than nested loops
        min_distances = np.full((height, width), np.inf)
        for point_idx in range(num_points):
            px, py = points[point_idx]
            # Calculate distance from this point to all pixels using vectorized operations
            distances = np.sqrt((x - px)**2 + (y - py)**2)
            min_distances = np.minimum(min_distances, distances)
        
        # Normalize and apply color
        max_dist = max(width, height)
        intensity = (min_distances / max_dist * 255).astype(np.uint8)
        color = ImageOps.get_random_color()
        
        # Blend cellular pattern with background
        for i in range(3):
            texture[:, :, i] = (background_color[i] * 0.3 + color[i] * intensity / 255 * 0.7).astype(np.uint8)
        
        return texture

    @staticmethod
    def _generate_solid_color(width, height):
        """Generate solid color texture (fallback)."""
        color = ImageOps.get_random_color()
        texture = np.full((height, width, 3), color, dtype=np.uint8)
        return texture

    @staticmethod
    def get_random_texture_type():
        """Get a random texture type for variety."""
        texture_types = ["perlin", "gaussian", "gradient", "cellular"]
        return random.choice(texture_types)

    @staticmethod
    def _rotate_image_partial(image, angle=90, center=None, scale=1.0, use_texture=True):
        # grab the dimensions of the image
        (h, w) = image.shape[:2]
        # if the center is None, initialize it as the center of the image
        if center is None:
            center = (w // 2, h // 2)
        
        if use_texture:
            # Create a background texture
            texture_type = ImageOps.get_random_texture_type()
            background_texture = ImageOps.generate_noise_texture(w, h, texture_type, use_random_color=True)
            
            # Perform rotation with transparent border (we'll handle the background ourselves)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h),
                                   borderMode=cv2.BORDER_TRANSPARENT)
            
            # Create a mask to identify the transparent areas (where the original image wasn't)
            # We'll use a different approach: rotate a white image to create a mask
            white_image = np.ones((h, w, 3), dtype=np.uint8) * 255
            rotated_mask = cv2.warpAffine(white_image, M, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
            
            # Convert mask to grayscale and normalize
            mask_gray = cv2.cvtColor(rotated_mask, cv2.COLOR_BGR2GRAY)
            mask_normalized = mask_gray.astype(np.float32) / 255.0
            
            # Composite the rotated image with the background texture
            # Where mask is 1 (original image), use rotated image
            # Where mask is 0 (background), use texture
            result = np.zeros_like(image)
            for i in range(3):  # For each color channel
                result[:, :, i] = (rotated[:, :, i] * mask_normalized + 
                                 background_texture[:, :, i] * (1 - mask_normalized)).astype(np.uint8)
            
            return result
        else:
            # Original behavior with solid color
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=ImageOps.get_random_color())
            return rotated

    @staticmethod
    def enhance_image(image_path):
        dirname = os.path.dirname(image_path)
        filename_parts = os.path.splitext(os.path.basename(image_path))
        new_filename = filename_parts[0] + "_b" + filename_parts[1]
        new_file = os.path.join(dirname, new_filename)
        if os.path.exists(new_file):
            raise Exception("File already exists: " + new_filename) # TODO maybe remove this
        try:
            with PIL.Image.open(image_path) as image:
                brightness_enhancer = ImageEnhance.Brightness(image)
                brightened = brightness_enhancer.enhance(1.3)
            contrast_enhancer = ImageEnhance.Contrast(brightened)
            contrasted = contrast_enhancer.enhance(1.1)
            brightened.close()
            contrasted.save(new_file)
            contrasted.close()
            return new_file
        except Exception as e:
            logger.warning("enhance_image failed for %s: %s", image_path, e)
            raise

    @staticmethod
    def flip_image(image_path, top_bottom=False):
        try:
            with PIL.Image.open(image_path) as original_img:
                mod_img, _ = ImageOps._flip_image(original_img, top_bottom=top_bottom)
            new_filepath = ImageOps.new_filepath(image_path, append_part="_flip")
            mod_img.save(new_filepath)
            mod_img.close()
            return new_filepath
        except Exception as e:
            logger.warning("flip_image failed for %s: %s", image_path, e)
            raise

    @staticmethod
    def convert_to_jpg(image_path, output_path=None, quality=85):
        """
        Convert lossless image formats to JPG to reduce file size.
        Preserves original dimensions and removes EXIF data.
        
        Args:
            image_path: Path to the source image
            output_path: Optional explicit output JPG filepath
            quality: JPG quality (1-100, default 85)
        """
        try:
            # Open the image
            image = PIL.Image.open(image_path)
            source_ext = os.path.splitext(image_path)[1].lower()

            # Fast path: JPG/JPEG source with no special EXIF needs no conversion.
            if source_ext in (".jpg", ".jpeg"):
                has_exif = False
                try:
                    exif_data = image.getexif()
                    has_exif = bool(exif_data) and len(exif_data) > 0
                except Exception:
                    has_exif = False
                jfif_container_keys = {
                    "jfif",
                    "jfif_version",
                    "jfif_unit",
                    "jfif_density",
                }
                info_keys = set(image.info.keys())
                has_non_jfif_info = len(info_keys - jfif_container_keys) > 0
                if not has_exif and not has_non_jfif_info:
                    image.close()
                    logger.info(f"Skipped JPG conversion (no EXIF to strip): {image_path}")
                    return image_path
            
            # Convert to RGB if necessary (JPG doesn't support alpha channel)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create a white background for transparent images
                background = PIL.Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create new filepath with .jpg extension unless caller provided one.
            if output_path is not None and str(output_path).strip() != "":
                new_filepath = output_path
            else:
                new_filepath = ImageOps.new_filepath(image_path, append_part="")
                base_path = os.path.splitext(new_filepath)[0]
                target_ext = ".jpeg" if source_ext == ".jpeg" else ".jpg"
                new_filepath = base_path + target_ext
            
            # Save as JPG without EXIF data
            image.save(new_filepath, 'JPEG', quality=quality, optimize=True)
            image.close()
            
            logger.info(f"Converted {image_path} to JPG: {new_filepath}")
            return new_filepath
            
        except Exception as e:
            logger.error(f"Error converting image to JPG: {e}")
            raise

    @staticmethod
    def scale_image_to_equivalent_pixels(
        image_path: str,
        target_side: int,
        output_path: str | None = None,
    ) -> tuple[str, bool]:
        """Scale an image so its total pixel count matches *target_side*².

        Preserves the exact aspect ratio — the image is scaled by
        ``sqrt(target_side² / (width × height))``, so a non-square image will
        not have either dimension equal to *target_side*.  Only images whose
        pixel count exceeds the target are scaled down; smaller images are left
        unchanged.

        For animated images (GIF, animated WebP, etc.) all frames are resized.

        Args:
            image_path:  Source image file path.
            target_side: One side of the equivalent square (e.g. 320 → 102,400
                         target pixels).  Must be >= 1.
            output_path: Destination path.  Defaults to *image_path* (in-place).

        Returns:
            (path, scaled) — output file path and whether resizing occurred.

        Raises:
            ValueError if target_side < 1.
            Exception if the file cannot be opened or written.
        """
        if target_side < 1:
            raise ValueError(f"target_side must be >= 1, got {target_side}")

        dest = output_path if output_path else image_path
        target_pixels = target_side * target_side

        img = PIL.Image.open(image_path)
        try:
            orig_w, orig_h = img.size
            current_pixels = orig_w * orig_h

            if current_pixels <= target_pixels:
                img.close()
                return dest, False

            scale = math.sqrt(target_pixels / current_pixels)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))

            n_frames = getattr(img, "n_frames", 1)
            if n_frames > 1:
                from PIL import ImageSequence
                frames = []
                durations = []
                for frame in ImageSequence.Iterator(img):
                    durations.append(frame.info.get("duration", 100))
                    frames.append(
                        frame.copy().resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                    )
                frames[0].save(
                    dest,
                    save_all=True,
                    append_images=frames[1:],
                    loop=img.info.get("loop", 0),
                    duration=durations,
                )
            else:
                dest_ext = os.path.splitext(dest)[1].lower()
                resized = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                save_kwargs: dict = {}
                if dest_ext in (".jpg", ".jpeg"):
                    if resized.mode not in ("RGB", "L"):
                        resized = resized.convert("RGB")
                    save_kwargs = {"quality": 90, "optimize": True}
                resized.save(dest, **save_kwargs)
        finally:
            img.close()

        logger.info(
            "Scaled %s: %dx%d → %dx%d (target %d²=%d px)",
            image_path, orig_w, orig_h, new_w, new_h, target_side, target_pixels,
        )
        return dest, True

    @staticmethod
    def _flip_image(im, top_bottom=False):
        return im.transpose(method=PIL.Image.FLIP_TOP_BOTTOM if top_bottom else PIL.Image.FLIP_LEFT_RIGHT), im

    @staticmethod
    def _parse_aspect_ratio(target_aspect_ratio) -> float:
        """
        Parse an aspect ratio from formats like:
        - float/int (e.g. 1.777)
        - "16:9"
        - "4/3"
        - "1.5"
        """
        if isinstance(target_aspect_ratio, (int, float)):
            ratio = float(target_aspect_ratio)
        else:
            ratio_text = str(target_aspect_ratio).strip()
            if ratio_text == "":
                raise ValueError("Aspect ratio cannot be empty")
            ratio_text = ratio_text.replace(" ", "")
            if ":" in ratio_text:
                lhs, rhs = ratio_text.split(":", 1)
                ratio = float(lhs) / float(rhs)
            elif "/" in ratio_text:
                lhs, rhs = ratio_text.split("/", 1)
                ratio = float(lhs) / float(rhs)
            else:
                ratio = float(ratio_text)
        if ratio <= 0:
            raise ValueError("Aspect ratio must be greater than zero")
        return ratio

    @staticmethod
    def change_aspect_ratio(image_path: str, target_aspect_ratio, anchor: str = "center") -> str:
        """
        Convert an image to a target aspect ratio using interpolation-based
        non-uniform scaling (no cropping).

        Args:
            image_path: Path to source image
            target_aspect_ratio: Desired ratio (e.g. "16:9", "4/3", 1.777...)
            anchor: Reserved for future behavior variants
        """
        target_ratio = ImageOps._parse_aspect_ratio(target_aspect_ratio)
        with PIL.Image.open(image_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("Image dimensions must be greater than zero")

            current_ratio = width / height
            if math.isclose(current_ratio, target_ratio, rel_tol=0.0, abs_tol=1e-6):
                converted = image.copy()
            else:
                # Keep image area approximately constant, then warp into target ratio.
                area = width * height
                target_height = max(1, int(round(math.sqrt(area / target_ratio))))
                target_width = max(1, int(round(target_height * target_ratio)))

                if target_width == width and target_height == height:
                    converted = image.copy()
                else:
                    resampling = (
                        PIL.Image.Resampling.BICUBIC
                        if target_width > width or target_height > height
                        else PIL.Image.Resampling.LANCZOS
                    )
                    converted = image.resize((target_width, target_height), resampling)

            new_filepath = ImageOps.new_filepath(image_path, append_part="_aspect")
            converted.save(new_filepath)
            converted.close()
            return new_filepath

    @staticmethod
    def _upscale(im, shortest_side=1200):
        width, height = im.size
        if width > height:
            if height < shortest_side:
                im = im.resize((int(width * shortest_side / height), shortest_side))
        elif width < shortest_side:
            im = im.resize((shortest_side, int(height * shortest_side / width)))
        return im

    @staticmethod
    def _random_crop_and_upscale(im, allowable_proportions=0.4, shortest_side=1200):
        width, height = im.size
        landscape = width > height
        midpoint_x = int(width / 2)
        midpoint_y = int(height / 2)
        if landscape:
            allowable_range_x = midpoint_x * allowable_proportions * 2
            allowable_range_y = midpoint_y * allowable_proportions
        else:
            allowable_range_x = midpoint_x * allowable_proportions
            allowable_range_y = midpoint_y * allowable_proportions * 2
        x0 = int(allowable_range_x * random.random())
        x1 = int(width - (allowable_range_x * random.random()))
        y0 = int(allowable_range_y * random.random())
        y1 = int(height - (allowable_range_y * random.random()))
        bbox = (x0, y0, x1, y1)
        cropped = im.crop(bbox)
        return ImageOps._upscale(cropped, shortest_side)

    @staticmethod
    def random_crop_and_upscale(image_path, allowable_proportions=0.4, shortest_side=1200):
        try:
            with PIL.Image.open(image_path) as im:
                cropped = ImageOps._random_crop_and_upscale(im, allowable_proportions, shortest_side)
            new_filepath = ImageOps.new_filepath(image_path, append_part="_crop")
            cropped.save(new_filepath)
            return new_filepath
        except Exception as e:
            logger.warning("random_crop_and_upscale failed for %s: %s", image_path, e)
            raise

    @staticmethod
    def random_rotate_and_crop():
        pass

    @staticmethod
    def rotate_and_shear_image(im, angle=30, x_shear=.7, y_shear=.7):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        change from https://stackoverflow.com/a/51109152
        https://gist.github.com/hsuRush/b2def27c98ce7ba3eb84a42e6d01328c
        """
        #logger.debug(im.shape)
        height, width = im.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_im = cv2.getRotationMatrix2D(image_center, angle, 1.)
        rotation_im[0,1] += x_shear
        rotation_im[1,0] += y_shear

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos_x = abs(rotation_im[0,0]) 
        abs_sin_x = abs(rotation_im[0,1])
        abs_cos_y = abs(rotation_im[1,0]) 
        abs_sin_y = abs(rotation_im[1,1])

        # find the new width and height bounds
        bound_w = int(width * abs_cos_x +  height *  abs_sin_x )
        bound_h = int(height * abs_sin_y  + width * abs_cos_y )

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_im[0, 2] += bound_w/2 - image_center[0] - image_center[1] * x_shear
        rotation_im[1, 2] += bound_h/2 - image_center[1] - image_center[0] * y_shear

        # rotate image with the new bounds and translated rotation imrix
        im = cv2.warpAffine(im, rotation_im, (bound_w, bound_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=ImageOps.get_random_color())
        im = cv2.resize(im, (width, height))

        #logger.debug(rotated_im.shape)
        return im

    @staticmethod
    def upscale(image_path):
        from cv2 import dnn_superres
        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()
        # Read image
        image = cv2.imread(image_path)
        # Read the desired model
        path = "EDSR_x3.pb"
        sr.readModel(path)
        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 3)
        # Upscale the image
        result = sr.upsample(image)
        new_filepath = ImageOps.new_filepath(image_path, append_part="_up")
        # Save the image
        cv2.imwrite(new_filepath, result)

    @staticmethod
    def randomly_modify_image(image_path):
        try:
            im = PIL.Image.open(image_path)
        except Exception as e:
            logger.warning("randomly_modify_image could not open %s: %s", image_path, e)
            raise
        try:
            cfg = config.image_edit_configuration
            # Each entry: (weight, stable_sort_index, name)
            candidates = [
                (cfg.random_rotation_chance, 0, "rotate"),
                (cfg.random_flip_chance,     1, "flip"),
                (cfg.random_shear_chance,    2, "shear"),
                (cfg.random_draw_chance,     3, "draw"),
                (cfg.random_crop_chance,     4, "crop"),
            ]
            available = [(w, idx, name) for w, idx, name in candidates if w > 0]
            if not available:
                raise ValueError("No modifications enabled in image_edit_configuration.")

            k = random.randint(1, min(4, len(available)))

            # Weighted sampling without replacement
            selected = []
            pool = list(available)
            for _ in range(k):
                weights = [w for w, _, _ in pool]
                (chosen,) = random.choices(pool, weights=weights, k=1)
                selected.append(chosen)
                pool.remove(chosen)

            # Apply in original declaration order (operations are order-sensitive)
            selected.sort(key=lambda x: x[1])

            for _, _, name in selected:
                if name == "rotate":
                    cv2_image = ImageOps.pil_to_cv2(im)
                    angle_diff = int(random.random() * 55)
                    angle = angle_diff if random.random() > 0.5 else 360 - angle_diff
                    cv2_image = ImageOps._rotate_image_partial(cv2_image, angle=angle)
                    im.close()
                    im = ImageOps.cv2_to_pil(cv2_image)
                elif name == "flip":
                    im, original_im = ImageOps._flip_image(im)
                    original_im.close()
                elif name == "shear":
                    cv2_image = ImageOps.pil_to_cv2(im)
                    angle = random.randint(0, 25)
                    x_shear = int(random.random() * 2 - 1)
                    y_shear = int(random.random() * 2 - 1)
                    cv2_image = ImageOps.rotate_and_shear_image(cv2_image, angle=angle, x_shear=x_shear, y_shear=y_shear)
                    im.close()
                    im = ImageOps.cv2_to_pil(cv2_image)
                    temp_im = ImageOps._random_crop_and_upscale(im)
                    im.close()
                    im = temp_im
                elif name == "draw":
                    ImageOps._random_draw(im)
                elif name == "crop":
                    temp_im = ImageOps._random_crop_and_upscale(im)
                    im.close()
                    im = temp_im

            new_filepath = ImageOps.new_filepath(image_path, append_part="_edit")
            im.save(new_filepath)
            im.close()
            return new_filepath
        except Exception as e:
            logger.warning("randomly_modify_image failed for %s: %s", image_path, e)
            try:
                im.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _block_shuffle_array(arr: np.ndarray, block_size: int) -> np.ndarray:
        h, w = arr.shape[:2]
        bh, bw = h // block_size, w // block_size
        if bh < 2 or bw < 2:
            return arr
        blocks = [
            arr[i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size].copy()
            for i in range(bh) for j in range(bw)
        ]
        coords = [(i * block_size, j * block_size) for i in range(bh) for j in range(bw)]
        random.shuffle(blocks)
        result = arr.copy()
        for k, (y0, x0) in enumerate(coords):
            result[y0:y0 + block_size, x0:x0 + block_size] = blocks[k]
        return result

    @staticmethod
    def _phase_scramble_array(arr: np.ndarray) -> np.ndarray:
        result = np.empty_like(arr)
        for c in range(arr.shape[2]):
            channel = arr[:, :, c].astype(np.float32)
            f = np.fft.fft2(channel)
            rng_phase = np.random.uniform(-np.pi, np.pi, channel.shape)
            new_f = np.abs(f) * np.exp(1j * rng_phase)
            result[:, :, c] = np.clip(np.real(np.fft.ifft2(new_f)), 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _swap_patches_array(
        arr: np.ndarray, n_swaps: int,
        min_frac: float = 0.25, max_frac: float = 0.55,
    ) -> np.ndarray:
        h, w = arr.shape[:2]
        result = arr.copy()
        for _ in range(n_swaps):
            ph = random.randint(max(1, int(h * min_frac)), max(2, int(h * max_frac)))
            pw = random.randint(max(1, int(w * min_frac)), max(2, int(w * max_frac)))
            if ph >= h or pw >= w:
                continue
            y1 = random.randint(0, h - ph)
            x1 = random.randint(0, w - pw)
            y2 = random.randint(0, h - ph)
            x2 = random.randint(0, w - pw)
            p1 = result[y1:y1 + ph, x1:x1 + pw].copy()
            result[y1:y1 + ph, x1:x1 + pw] = result[y2:y2 + ph, x2:x2 + pw]
            result[y2:y2 + ph, x2:x2 + pw] = p1
        return result

    @staticmethod
    def _strip_shuffle_array(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        result = arr.copy()
        if random.random() > 0.5:
            stripe = max(1, random.randint(h // 15, h // 6))
            n = h // stripe
            if n < 2:
                return result
            strips = [result[i * stripe:(i + 1) * stripe].copy() for i in range(n)]
            random.shuffle(strips)
            for i, strip in enumerate(strips):
                result[i * stripe:(i + 1) * stripe] = strip
        else:
            stripe = max(1, random.randint(w // 15, w // 6))
            n = w // stripe
            if n < 2:
                return result
            strips = [result[:, j * stripe:(j + 1) * stripe].copy() for j in range(n)]
            random.shuffle(strips)
            for j, strip in enumerate(strips):
                result[:, j * stripe:(j + 1) * stripe] = strip
        return result

    @staticmethod
    def scramble_image(image_path: str, output_path: str | None = None) -> str:
        """Apply heavy random distortion to produce a visually incoherent image.

        Unlike randomly_modify_image() which preserves legibility, this is
        designed for coherence-detection training data: the output retains some
        colour/frequency characteristics of the source but loses all meaningful
        spatial structure.

        Randomly combines 1–3 of:
        - block_shuffle  : permutes NxN pixel blocks, destroying spatial layout
        - phase_scramble : randomises FFT phase, eliminating all spatial relations
        - patch_swap     : transplants large regions to distant positions (swabbing)
        - strip_shuffle  : shuffles horizontal or vertical strips

        Returns the output file path.
        """
        _STRATEGIES = ["block_shuffle", "phase_scramble", "patch_swap", "strip_shuffle"]
        chosen = random.sample(_STRATEGIES, random.randint(1, min(3, len(_STRATEGIES))))

        img = None
        try:
            img = PIL.Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)
            img.close()
            img = None

            for strategy in chosen:
                if strategy == "block_shuffle":
                    h, w = arr.shape[:2]
                    block_size = random.randint(
                        max(8, min(h, w) // 20),
                        max(16, min(h, w) // 8),
                    )
                    arr = ImageOps._block_shuffle_array(arr, block_size)
                elif strategy == "phase_scramble":
                    arr = ImageOps._phase_scramble_array(arr)
                elif strategy == "patch_swap":
                    arr = ImageOps._swap_patches_array(arr, n_swaps=random.randint(3, 8))
                elif strategy == "strip_shuffle":
                    arr = ImageOps._strip_shuffle_array(arr)

            dest = output_path or ImageOps.new_filepath(image_path, append_part="_scramble")
            result_img = PIL.Image.fromarray(arr)
            save_kwargs: dict = {}
            if os.path.splitext(dest)[1].lower() in (".jpg", ".jpeg"):
                save_kwargs = {"quality": 90, "optimize": True}
            result_img.save(dest, **save_kwargs)
            result_img.close()

            logger.info("scramble_image: strategies=%s → %s", chosen, dest)
            return dest
        except Exception as e:
            logger.warning("scramble_image failed for %s: %s", image_path, e)
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass
            raise

    @staticmethod
    def _apply_local_gaussian_blur(
        arr: np.ndarray, y0: int, x0: int, y1: int, x1: int, sigma: float
    ) -> np.ndarray:
        ksize = max(3, int(sigma * 6) | 1)  # must be odd and >= 3
        result = arr.copy()
        result[y0:y1, x0:x1] = cv2.GaussianBlur(
            result[y0:y1, x0:x1], (ksize, ksize), sigma
        )
        return result

    @staticmethod
    def _apply_local_smear(
        arr: np.ndarray, y0: int, x0: int, y1: int, x1: int,
        length: int, angle_deg: float,
    ) -> np.ndarray:
        length = max(3, length | 1)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((length // 2, length // 2), angle_deg, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        total = kernel.sum()
        if total > 0:
            kernel /= total
        result = arr.copy()
        result[y0:y1, x0:x1] = cv2.filter2D(
            result[y0:y1, x0:x1], -1, kernel, borderType=cv2.BORDER_REFLECT
        )
        return result

    @staticmethod
    def _apply_local_pixel_shuffle(
        arr: np.ndarray, y0: int, x0: int, y1: int, x1: int
    ) -> np.ndarray:
        result = arr.copy()
        region = result[y0:y1, x0:x1].copy()
        orig_shape = region.shape
        flat = region.reshape(-1, orig_shape[2])
        np.random.shuffle(flat)  # shuffles pixel rows in-place
        result[y0:y1, x0:x1] = flat.reshape(orig_shape)
        return result

    @staticmethod
    def _apply_local_pixelation(
        arr: np.ndarray, y0: int, x0: int, y1: int, x1: int, block_size: int
    ) -> np.ndarray:
        result = arr.copy()
        region = result[y0:y1, x0:x1]
        rh, rw = region.shape[:2]
        small_h = max(1, rh // block_size)
        small_w = max(1, rw // block_size)
        # INTER_AREA = averaging; INTER_NEAREST = blocky upscale (preserves mosaic look)
        small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
        result[y0:y1, x0:x1] = cv2.resize(
            small, (rw, rh), interpolation=cv2.INTER_NEAREST
        )
        return result

    @staticmethod
    def semi_scramble_image(image_path: str, output_path: str | None = None) -> str:
        """Apply local distortions to a random subset of the image, leaving the rest coherent.

        Selects 1–4 random regions and applies one of the following per region:
        - gaussian_blur  : heavy Gaussian blur — destroys fine detail and edges
        - smear          : directional motion blur — drags content in a random direction
        - pixel_shuffle  : randomly reorders every pixel — noise-like scrambling
        - pixelate       : mosaic / block-average (censor-style) — preserves coarse colour

        Designed for "semi-incoherent" training examples: the distorted regions lose
        local coherence while the untouched areas remain fully readable.

        Returns the output file path (suffix ``_semi_scramble`` unless output_path given).
        """
        _EFFECTS = ["gaussian_blur", "smear", "pixel_shuffle", "pixelate"]

        img = None
        try:
            img = PIL.Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)
            img.close()
            img = None

            h, w = arr.shape[:2]
            n_regions = random.randint(1, 4)
            applied: list[str] = []

            for _ in range(n_regions):
                rh = max(16, random.randint(int(h * 0.20), int(h * 0.60)))
                rw = max(16, random.randint(int(w * 0.20), int(w * 0.60)))
                if rh >= h or rw >= w:
                    continue
                y0 = random.randint(0, h - rh)
                x0 = random.randint(0, w - rw)
                y1, x1 = y0 + rh, x0 + rw

                effect = random.choice(_EFFECTS)
                applied.append(effect)

                if effect == "gaussian_blur":
                    sigma = random.uniform(8.0, max(8.0, min(rh, rw) * 0.15))
                    arr = ImageOps._apply_local_gaussian_blur(arr, y0, x0, y1, x1, sigma)
                elif effect == "smear":
                    min_len = max(15, min(rh, rw) // 8)
                    max_len = max(min_len + 2, min(rh, rw) // 3)
                    length = random.randint(min_len, max_len)
                    arr = ImageOps._apply_local_smear(
                        arr, y0, x0, y1, x1, length, random.uniform(0.0, 360.0)
                    )
                elif effect == "pixel_shuffle":
                    arr = ImageOps._apply_local_pixel_shuffle(arr, y0, x0, y1, x1)
                elif effect == "pixelate":
                    min_blk = max(8, min(rh, rw) // 20)
                    max_blk = max(min_blk + 2, min(rh, rw) // 8)
                    arr = ImageOps._apply_local_pixelation(
                        arr, y0, x0, y1, x1, random.randint(min_blk, max_blk)
                    )

            dest = output_path or ImageOps.new_filepath(
                image_path, append_part="_semi_scramble"
            )
            result_img = PIL.Image.fromarray(arr)
            save_kwargs: dict = {}
            if os.path.splitext(dest)[1].lower() in (".jpg", ".jpeg"):
                save_kwargs = {"quality": 90, "optimize": True}
            result_img.save(dest, **save_kwargs)
            result_img.close()

            logger.info("semi_scramble_image: effects=%s → %s", applied, dest)
            return dest
        except Exception as e:
            logger.warning("semi_scramble_image failed for %s: %s", image_path, e)
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass
            raise

    @staticmethod
    def random_draw(image_path):
        try:
            with PIL.Image.open(image_path) as image:
                ImageOps._random_draw(image)
                new_filepath = ImageOps.new_filepath(image_path, append_part="_drawn")
                image.save(new_filepath)
            return new_filepath
        except Exception as e:
            logger.warning("random_draw failed for %s: %s", image_path, e)
            raise

    @staticmethod
    def _random_draw(image):
        draw = ImageDraw.Draw(image)
        for i in range(random.randint(1, 5)):
            # Decide whether to use texture-based drawing based on configuration
            use_texture = random.random() < config.image_edit_configuration.texture_draw_probability
            
            if use_texture:
                # Use texture-based drawing methods
                choice = random.randint(0, 3)
                if choice == 0:
                    ImageOps._texture_arc(image, draw)
                elif choice == 1:
                    ImageOps._texture_line(image, draw)
                elif choice == 2:
                    ImageOps._texture_chord(image, draw)
                elif choice == 3:
                    ImageOps._texture_shape(image, draw)
            else:
                # Use traditional solid color drawing methods
                choice = random.randint(0, 2)
                if choice == 0:
                    ImageOps._arc(image, draw)
                elif choice == 1:
                    ImageOps._line(image, draw)
                elif choice == 2:
                    ImageOps._chord(image, draw)

    @staticmethod
    def _color():
        return random.choice(ImageOps.COLORS)

    @staticmethod
    def _opacity():
        return random.uniform(0.5, 1.0)

    @staticmethod
    def _line(image, image_draw):
        if random.randint(0, 1) == 0:
            for i in range(0, 100, 20):
                image_draw.line((i, 0) + image.size, width=random.randint(2, 20), fill=ImageOps._color())
        else:
            points = []
            for i in range(random.randint(2, 4)):
                points.append((random.randint(0, image.size[0]), random.randint(0, image.size[1])))
            image_draw.line(points, width=random.randint(2, 40), fill=ImageOps._color(), joint="curve")

    @staticmethod
    def _arc(image, image_draw):
        for i in range(random.randint(1, 5)):
            start = random.randint(0, image.size[0])
            end = random.randint(start, image.size[0])            
            start0 = random.randint(0, image.size[0])
            end0 = random.randint(start0, image.size[0])
            start1 = random.randint(0, image.size[1])
            end1 = random.randint(start1, image.size[1])
            bounds = (start0, start1, end0, end1)
            image_draw.arc(bounds, start=start, end=end, fill=ImageOps._color(), width=random.randint(2, 40))

    @staticmethod
    def _chord(image, image_draw):
        for i in range(random.randint(1, 5)):
            start = random.randint(0, image.size[0])
            end = random.randint(start, image.size[0])            
            start0 = random.randint(0, image.size[0])
            end0 = random.randint(start0, image.size[0])
            start1 = random.randint(0, image.size[1])
            end1 = random.randint(start1, image.size[1])
            bounds = (start0, start1, end0, end1)
            image_draw.chord(bounds, start=start, end=end, fill=ImageOps._color(), outline=ImageOps._color(), width=random.randint(2, 40))

    @staticmethod
    def _texture_line(image, image_draw):
        """Draw lines with texture-based patterns instead of solid colors."""
        if random.randint(0, 1) == 0:
            # Parallel lines with texture variation
            for i in range(0, 100, 20):
                # Generate a small texture patch for this line segment
                texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
                line_width = random.randint(2, 20)
                texture_patch = ImageOps.generate_noise_texture(line_width * 2, line_width * 2, texture_type)
                
                # Convert texture to PIL Image and resize
                texture_img = PIL.Image.fromarray(texture_patch)
                texture_img = texture_img.resize((line_width * 2, line_width * 2))
                
                # Create a temporary image for the line with texture
                temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                temp_draw.line((i, 0) + image.size, width=line_width, fill=(255, 255, 255, 255))
                
                # Apply texture to the line area
                ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
                temp_img.close()
                texture_img.close()
        else:
            # Curved lines with texture
            points = []
            for i in range(random.randint(2, 4)):
                points.append((random.randint(0, image.size[0]), random.randint(0, image.size[1])))
            
            line_width = random.randint(2, 40)
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture_patch = ImageOps.generate_noise_texture(line_width * 3, line_width * 3, texture_type)
            
            # Convert texture to PIL Image
            texture_img = PIL.Image.fromarray(texture_patch)
            texture_img = texture_img.resize((line_width * 3, line_width * 3))
            
            # Create temporary image for curved line
            temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.line(points, width=line_width, fill=(255, 255, 255, 255), joint="curve")
            
            # Apply texture to the line area
            ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
            temp_img.close()
            texture_img.close()

    @staticmethod
    def _texture_arc(image, image_draw):
        """Draw arcs with texture-based patterns."""
        for i in range(random.randint(1, 5)):
            start = random.randint(0, image.size[0])
            end = random.randint(start, image.size[0])            
            start0 = random.randint(0, image.size[0])
            end0 = random.randint(start0, image.size[0])
            start1 = random.randint(0, image.size[1])
            end1 = random.randint(start1, image.size[1])
            bounds = (start0, start1, end0, end1)
            
            arc_width = random.randint(2, 40)
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture_patch = ImageOps.generate_noise_texture(arc_width * 2, arc_width * 2, texture_type)
            
            # Convert texture to PIL Image
            texture_img = PIL.Image.fromarray(texture_patch)
            texture_img = texture_img.resize((arc_width * 2, arc_width * 2))
            
            # Create temporary image for arc
            temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.arc(bounds, start=start, end=end, fill=(255, 255, 255, 255), width=arc_width)
            
            # Apply texture to the arc area
            ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
            temp_img.close()
            texture_img.close()

    @staticmethod
    def _texture_chord(image, image_draw):
        """Draw chords with texture-based patterns."""
        for i in range(random.randint(1, 5)):
            start = random.randint(0, image.size[0])
            end = random.randint(start, image.size[0])            
            start0 = random.randint(0, image.size[0])
            end0 = random.randint(start0, image.size[0])
            start1 = random.randint(0, image.size[1])
            end1 = random.randint(start1, image.size[1])
            bounds = (start0, start1, end0, end1)
            
            chord_width = random.randint(2, 40)
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture_patch = ImageOps.generate_noise_texture(chord_width * 2, chord_width * 2, texture_type)
            
            # Convert texture to PIL Image
            texture_img = PIL.Image.fromarray(texture_patch)
            texture_img = texture_img.resize((chord_width * 2, chord_width * 2))
            
            # Create temporary image for chord
            temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.chord(bounds, start=start, end=end, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255), width=chord_width)
            
            # Apply texture to the chord area
            ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
            temp_img.close()
            texture_img.close()

    @staticmethod
    def _texture_shape(image, image_draw):
        """Draw various shapes with texture-based patterns."""
        shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
        
        if shape_type == 'rectangle':
            # Random rectangle
            x0 = random.randint(0, image.size[0] // 2)
            y0 = random.randint(0, image.size[1] // 2)
            x1 = random.randint(x0, image.size[0])
            y1 = random.randint(y0, image.size[1])
            bounds = (x0, y0, x1, y1)
            
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture_patch = ImageOps.generate_noise_texture(x1-x0, y1-y0, texture_type)
            
            # Convert texture to PIL Image
            texture_img = PIL.Image.fromarray(texture_patch)
            texture_img = texture_img.resize((x1-x0, y1-y0))
            
            # Create temporary image for rectangle
            temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.rectangle(bounds, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
            
            # Apply texture to the rectangle area
            ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
            temp_img.close()
            texture_img.close()
            
        elif shape_type == 'ellipse':
            # Random ellipse
            x0 = random.randint(0, image.size[0] // 2)
            y0 = random.randint(0, image.size[1] // 2)
            x1 = random.randint(x0, image.size[0])
            y1 = random.randint(y0, image.size[1])
            bounds = (x0, y0, x1, y1)
            
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture_patch = ImageOps.generate_noise_texture(x1-x0, y1-y0, texture_type)
            
            # Convert texture to PIL Image
            texture_img = PIL.Image.fromarray(texture_patch)
            texture_img = texture_img.resize((x1-x0, y1-y0))
            
            # Create temporary image for ellipse
            temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.ellipse(bounds, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
            
            # Apply texture to the ellipse area
            ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
            temp_img.close()
            texture_img.close()
            
        else:  # polygon
            # Random polygon
            num_points = random.randint(3, 8)
            points = []
            for i in range(num_points):
                points.append((random.randint(0, image.size[0]), random.randint(0, image.size[1])))
            
            # Calculate bounding box for texture sizing
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
            
            texture_type = random.choice(ImageOps.TEXTURE_DRAW_TYPES)
            texture_patch = ImageOps.generate_noise_texture(max_x-min_x, max_y-min_y, texture_type)
            
            # Convert texture to PIL Image
            texture_img = PIL.Image.fromarray(texture_patch)
            texture_img = texture_img.resize((max_x-min_x, max_y-min_y))
            
            # Create temporary image for polygon
            temp_img = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.polygon(points, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
            
            # Apply texture to the polygon area
            ImageOps._apply_texture_to_mask(image, temp_img, texture_img)
            temp_img.close()
            texture_img.close()

    @staticmethod
    def _apply_texture_to_mask(target_image, mask_image, texture_image):
        """Apply texture to areas defined by a mask on the target image."""
        # Convert images to numpy arrays for processing
        target_array = np.array(target_image)
        mask_array = np.array(mask_image)
        texture_array = np.array(texture_image)
        
        # Get mask alpha channel
        if mask_array.shape[2] == 4:  # RGBA
            mask_alpha = mask_array[:, :, 3] / 255.0
        else:  # RGB
            mask_alpha = np.ones((mask_array.shape[0], mask_array.shape[1]))
        
        # Find bounding box of non-zero mask area
        coords = np.where(mask_alpha > 0)
        if len(coords[0]) == 0:
            return  # No mask area to process
        
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        # Resize texture to match mask area
        mask_height = max_y - min_y + 1
        mask_width = max_x - min_x + 1
        
        if texture_array.shape[0] != mask_height or texture_array.shape[1] != mask_width:
            texture_resized = cv2.resize(texture_array, (mask_width, mask_height))
        else:
            texture_resized = texture_array
        
        # Apply texture to masked area
        mask_region = mask_alpha[min_y:max_y+1, min_x:max_x+1]
        
        for i in range(3):  # RGB channels
            target_region = target_array[min_y:max_y+1, min_x:max_x+1, i]
            texture_region = texture_resized[:, :, i]
            
            # Blend texture with existing image based on mask
            blended = (target_region * (1 - mask_region) + 
                      texture_region * mask_region).astype(np.uint8)
            
            target_array[min_y:max_y+1, min_x:max_x+1, i] = blended
        
        # Convert back to PIL Image
        result_image = PIL.Image.fromarray(target_array)
        
        # Paste the modified region back to the original image
        target_image.paste(result_image, (0, 0))

    @staticmethod
    def pil_to_cv2(pil_image):
        # use numpy to convert the pil_image into a numpy array
        numpy_image = np.array(pil_image)
        # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
        # the color is converted from RGB to BGR format
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return opencv_image

    @staticmethod
    def cv2_to_pil(opencv_image):
        # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
        # the color is converted from BGR to RGB
        color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_converted)
        return pil_image

    # GIMP GEGL Integration Methods
    @staticmethod
    def is_gimp_gegl_available():
        """Check if GIMP GEGL integration is available."""
        # Check if we've already validated this session
        if ImageOps._gegl_validation_cache is not None:
            return ImageOps._gegl_validation_cache
        
        # Trigger lazy GIMP validation if not already done
        config.validate_and_find_gimp()
        
        # If GIMP is not available, GEGL is not available
        if not config.gimp_gegl_enabled:
            ImageOps._gegl_validation_cache = False
            return False
        
        # Perform comprehensive GEGL validation
        try:
            from extensions.gimp.gimp_gegl_validator import GimpGeglValidator
            validator = GimpGeglValidator()
            is_valid, errors = validator.validate_complete_setup()
            
            if is_valid:
                logger.debug("GEGL setup validation successful")
                ImageOps._gegl_validation_cache = True
                return True
            else:
                logger.warning(f"GEGL setup validation failed: {', '.join(errors)}")
                # Update config to disable GEGL for this session
                config.gimp_gegl_enabled = False
                ImageOps._gegl_validation_cache = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate GEGL setup: {e}")
            # Update config to disable GEGL for this session
            config.gimp_gegl_enabled = False
            ImageOps._gegl_validation_cache = False
            return False

    @staticmethod
    def clear_gegl_validation_cache():
        """Clear the GEGL validation cache to force re-validation on next check."""
        ImageOps._gegl_validation_cache = None
        logger.debug("GEGL validation cache cleared")

    @staticmethod
    def apply_gegl_operation(image_path: str, operation_name: str, parameters: dict, output_path: str = None, 
                           opacity: float = None, blend_mode = None) -> str:
        """
        Apply a GEGL operation to an image using GIMP 3.
        
        Args:
            image_path: Path to the input image
            operation_name: Name of the GEGL operation (e.g., "gegl:brightness-contrast")
            parameters: Dictionary of parameters for the operation
            output_path: Optional output path. If None, generates one automatically
            opacity: Opacity for the operation (0.0 to 1.0). If None, uses default (1.0)
            blend_mode: Blend mode for the operation. If None, uses default (normal)
            
        Returns:
            Path to the processed image
            
        Raises:
            RuntimeError: If GIMP GEGL is not available or operation fails
        """
        if not ImageOps.is_gimp_gegl_available():
            raise RuntimeError("GIMP GEGL integration is not available. Check GIMP installation and configuration.")
        
        try:
            with GimpGeglClient() as client:
                return client.apply_gegl_operation(image_path, operation_name, parameters, output_path, opacity, blend_mode)
        except Exception as e:
            logger.error(f"GEGL operation failed: {e}")
            raise RuntimeError(f"GEGL operation '{operation_name}' failed: {e}")

    @staticmethod
    def gegl_brightness_contrast(image_path: str, brightness: float = 0.0, contrast: float = 0.0, output_path: str = None) -> str:
        """
        Apply brightness and contrast adjustment using GEGL.
        
        Args:
            image_path: Path to the input image
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast adjustment (-1.0 to 1.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path, 
            "gegl:brightness-contrast", 
            {"brightness": brightness, "contrast": contrast},
            output_path
        )

    @staticmethod
    def gegl_color_balance(image_path: str, cyan_red: float = 0.0, magenta_green: float = 0.0, 
                          yellow_blue: float = 0.0, output_path: str = None) -> str:
        """
        Apply color balance adjustment using GEGL.
        
        Args:
            image_path: Path to the input image
            cyan_red: Cyan-Red balance (-1.0 to 1.0)
            magenta_green: Magenta-Green balance (-1.0 to 1.0)
            yellow_blue: Yellow-Blue balance (-1.0 to 1.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:color-balance",
            {"cyan-red": cyan_red, "magenta-green": magenta_green, "yellow-blue": yellow_blue},
            output_path
        )

    @staticmethod
    def gegl_hue_saturation(image_path: str, hue: float = 0.0, saturation: float = 0.0, 
                           lightness: float = 0.0, output_path: str = None) -> str:
        """
        Apply hue, saturation, and lightness adjustment using GEGL.
        
        Args:
            image_path: Path to the input image
            hue: Hue shift (-180.0 to 180.0 degrees)
            saturation: Saturation adjustment (-100.0 to 100.0)
            lightness: Lightness adjustment (-100.0 to 100.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:hue-saturation",
            {"hue": hue, "saturation": saturation, "lightness": lightness},
            output_path
        )

    @staticmethod
    def gegl_gaussian_blur(image_path: str, std_dev_x: float = 1.0, std_dev_y: float = 1.0, 
                          output_path: str = None) -> str:
        """
        Apply Gaussian blur using GEGL.
        
        Args:
            image_path: Path to the input image
            std_dev_x: Standard deviation in X direction (0.0 to 100.0)
            std_dev_y: Standard deviation in Y direction (0.0 to 100.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:gaussian-blur",
            {"std-dev-x": std_dev_x, "std-dev-y": std_dev_y},
            output_path
        )

    @staticmethod
    def gegl_unsharp_mask(image_path: str, std_dev: float = 1.0, scale: float = 0.5, 
                         output_path: str = None) -> str:
        """
        Apply unsharp mask using GEGL.
        
        Args:
            image_path: Path to the input image
            std_dev: Standard deviation (0.0 to 10.0)
            scale: Scale factor (0.0 to 10.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:unsharp-mask",
            {"std-dev": std_dev, "scale": scale},
            output_path
        )

    @staticmethod
    def gegl_noise_reduce(image_path: str, iterations: int = 1, spatial_radius: float = 1.0, 
                         temporal_radius: float = 0.0, output_path: str = None) -> str:
        """
        Apply noise reduction using GEGL.
        
        Args:
            image_path: Path to the input image
            iterations: Number of iterations (1 to 10)
            spatial_radius: Spatial radius (0.0 to 10.0)
            temporal_radius: Temporal radius (0.0 to 10.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:noise-reduce",
            {"iterations": iterations, "spatial-radius": spatial_radius, "temporal-radius": temporal_radius},
            output_path
        )

    @staticmethod
    def gegl_levels(image_path: str, in_low: float = 0.0, in_high: float = 1.0, 
                   out_low: float = 0.0, out_high: float = 1.0, gamma: float = 1.0, 
                   output_path: str = None) -> str:
        """
        Apply levels adjustment using GEGL.
        
        Args:
            image_path: Path to the input image
            in_low: Input low level (0.0 to 1.0)
            in_high: Input high level (0.0 to 1.0)
            out_low: Output low level (0.0 to 1.0)
            out_high: Output high level (0.0 to 1.0)
            gamma: Gamma correction (0.0 to 10.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:levels",
            {"in-low": in_low, "in-high": in_high, "out-low": out_low, "out-high": out_high, "gamma": gamma},
            output_path
        )

    @staticmethod
    def gegl_exposure(image_path: str, black: float = 0.0, exposure: float = 0.0, 
                     gamma: float = 1.0, output_path: str = None) -> str:
        """
        Apply exposure adjustment using GEGL.
        
        Args:
            image_path: Path to the input image
            black: Black point (0.0 to 1.0)
            exposure: Exposure adjustment (-10.0 to 10.0)
            gamma: Gamma correction (0.0 to 10.0)
            output_path: Optional output path
            
        Returns:
            Path to the processed image
        """
        return ImageOps.apply_gegl_operation(
            image_path,
            "gegl:exposure",
            {"black": black, "exposure": exposure, "gamma": gamma},
            output_path
        )

    @staticmethod
    def get_available_gegl_operations():
        """Get list of available GEGL operations."""
        if not ImageOps.is_gimp_gegl_available():
            return []
        
        try:
            with GimpGeglClient() as client:
                return client.get_available_operations()
        except Exception as e:
            logger.error(f"Failed to get GEGL operations: {e}")
            return []

    @staticmethod
    def get_gegl_operation_schema(operation_name: str):
        """Get parameter schema for a GEGL operation."""
        if not ImageOps.is_gimp_gegl_available():
            return {}
        
        try:
            with GimpGeglClient() as client:
                return client.get_operation_schema(operation_name)
        except Exception as e:
            logger.error(f"Failed to get GEGL operation schema: {e}")
            return {}

    @staticmethod
    def get_gegl_validation_report():
        """
        Get a comprehensive validation report for GIMP GEGL setup.
        
        Returns:
            Formatted validation report string
        """
        try:
            from extensions.gimp.gimp_gegl_validator import get_validation_report
            return get_validation_report()
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return f"Error generating validation report: {e}"

    @staticmethod
    def get_available_blend_modes():
        """Get list of available GIMP blend modes."""
        if not ImageOps.is_gimp_gegl_available():
            return []
        
        try:
            from extensions.gimp.gimp_gegl_client import GimpBlendMode
            return [mode.value for mode in GimpBlendMode]
        except Exception as e:
            logger.error(f"Failed to get blend modes: {e}")
            return []

    @staticmethod
    def create_gegl_client_with_settings(default_opacity: float = 1.0, default_blend_mode = None):
        """
        Create a GimpGeglClient with custom default settings.
        
        Args:
            default_opacity: Default opacity for operations (0.0 to 1.0)
            default_blend_mode: Default blend mode for operations
            
        Returns:
            Configured GimpGeglClient instance
        """
        if not ImageOps.is_gimp_gegl_available():
            raise RuntimeError("GIMP GEGL integration is not available")
        
        try:
            return GimpGeglClient(default_opacity=default_opacity, default_blend_mode=default_blend_mode)
        except Exception as e:
            logger.error(f"Failed to create GEGL client: {e}")
            raise RuntimeError(f"Failed to create GEGL client: {e}")

    # ── Surya OCR ──────────────────────────────────────────────────

    @staticmethod
    def is_surya_ocr_available() -> bool:
        """Return True if the Surya OCR extension is usable."""
        try:
            from extensions.surya_ocr_client import is_surya_available
            return is_surya_available()
        except Exception:
            return False

    @staticmethod
    def run_ocr(image_path: str):
        """
        Run OCR on *image_path* using Surya and return a
        :class:`~extensions.surya_ocr_client.SuryaOCRResult`.

        Raises ``RuntimeError`` if Surya is not installed.
        """
        from extensions.surya_ocr_client import run_ocr as _run_ocr
        return _run_ocr(image_path)

    # ── Image analysis ───────────────────────────────────────────

    @staticmethod
    def color_diversity_score(image_path: str) -> float:
        """Return a Shannon-entropy diversity score in [0, 1] for the image.

        1.0 means all 64 quantized palette slots are equally populated
        (maximum diversity); 0.0 means the image is effectively monochrome.
        """
        MAX_DIM = 200
        N_COLORS = 64

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.NEAREST)
            quantized = img.quantize(colors=N_COLORS)

        counts = quantized.getcolors()
        if not counts:
            return 0.0

        total = sum(c for c, _ in counts)
        probs = np.array([c / total for c, _ in counts], dtype=np.float64)
        entropy = -float(np.sum(probs * np.log2(probs)))
        return min(entropy / np.log2(N_COLORS), 1.0)

    @staticmethod
    def hue_breadth_score(image_path: str) -> float:
        """Return a hue-breadth score in [0, 1].

        Measures how widely chromatic pixels are distributed around the colour
        wheel, ignoring lightness and saturation variation. A muted image with
        many distinct hues scores higher than a trichromatic image with a full
        range of shades. Near-achromatic pixels (very low HSV saturation) are
        excluded so that grey/black/white regions don't dilute the result.
        """
        MAX_DIM = 200
        N_BINS = 36    # 10° per bin
        MIN_SAT = 0.08  # below this HSV saturation a pixel is treated as achromatic

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.NEAREST)
            rgb = np.asarray(img, dtype=np.float32) / 255.0

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        sat = np.where(cmax > 0, delta / cmax, 0.0)
        chromatic = sat > MIN_SAT
        if not np.any(chromatic):
            return 0.0

        hue = np.zeros_like(r)
        eps = 1e-9
        m_r = (cmax == r) & (delta > 0)
        m_g = (cmax == g) & (delta > 0)
        m_b = (cmax == b) & (delta > 0)
        hue[m_r] = ((g[m_r] - b[m_r]) / (delta[m_r] + eps)) % 6
        hue[m_g] = (b[m_g] - r[m_g]) / (delta[m_g] + eps) + 2
        hue[m_b] = (r[m_b] - g[m_b]) / (delta[m_b] + eps) + 4
        hue /= 6.0

        counts, _ = np.histogram(hue[chromatic], bins=N_BINS, range=(0.0, 1.0))
        total = counts.sum()
        probs = counts[counts > 0] / total
        entropy = -float(np.sum(probs * np.log2(probs)))
        return min(entropy / np.log2(N_BINS), 1.0)

    @staticmethod
    def color_histogram(
        image_path: str,
        h_bins: int = 36,
        s_bins: int = 16,
        v_bins: int = 16,
    ) -> "np.ndarray":
        """Return a normalized HSV colour histogram for the image.

        The result is a 1-D float64 array of length
        ``h_bins + s_bins + v_bins`` (default 68).  Each channel's histogram
        is independently normalized to sum to 1, then concatenated.

        Intended as a colour-space signature for dataset diversity analysis:
        aggregate across a corpus to find under-represented colour regions,
        or compare pairs with a distance metric (Earth Mover's Distance /
        cosine similarity / histogram intersection).
        """
        MAX_DIM = 200

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.NEAREST)
            rgb = np.asarray(img, dtype=np.float32) / 255.0

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        val = cmax
        sat = np.where(cmax > 0, delta / cmax, 0.0)

        hue = np.zeros_like(r)
        eps = 1e-9
        m_r = (cmax == r) & (delta > 0)
        m_g = (cmax == g) & (delta > 0)
        m_b = (cmax == b) & (delta > 0)
        hue[m_r] = ((g[m_r] - b[m_r]) / (delta[m_r] + eps)) % 6
        hue[m_g] = (b[m_g] - r[m_g]) / (delta[m_g] + eps) + 2
        hue[m_b] = (r[m_b] - g[m_b]) / (delta[m_b] + eps) + 4
        hue /= 6.0

        def _norm(data: "np.ndarray", bins: int, lo: float, hi: float) -> "np.ndarray":
            counts, _ = np.histogram(data.ravel(), bins=bins, range=(lo, hi))
            total = counts.sum()
            return (counts / total).astype(np.float64) if total > 0 else counts.astype(np.float64)

        return np.concatenate([
            _norm(hue, h_bins, 0.0, 1.0),
            _norm(sat, s_bins, 0.0, 1.0),
            _norm(val, v_bins, 0.0, 1.0),
        ])

    # ── Image content comparison ─────────────────────────────────

    @staticmethod
    def compare_image_content_without_exif(filepath1, filepath2):
        """
        Compare two images for exact pixel content equality without considering EXIF data.
        This is useful when images have identical visual content but different metadata.
        
        Args:
            filepath1: Path to first image
            filepath2: Path to second image
            
        Returns:
            bool: True if images have identical pixel content, False otherwise
        """
        ext = os.path.splitext(filepath1)[1].lower()
        if ext not in Image.registered_extensions():
            return False
        try:
            # Open images and convert to RGB to normalize format
            img1 = Image.open(filepath1).convert('RGB')
            img2 = Image.open(filepath2).convert('RGB')
            
            # Check if dimensions are identical
            if img1.size != img2.size:
                img1.close()
                img2.close()
                return False
            
            # Convert to numpy arrays for pixel-by-pixel comparison
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Close images to free memory
            img1.close()
            img2.close()
            
            # Check for exact pixel equality
            return np.array_equal(arr1, arr2)
            
        except Exception as e:
            logger.error(f"Error comparing image content: {e}")
            return False

    @staticmethod
    def create_interpolation_gif(
        image_path_a: str,
        image_path_b: str,
        n_frames: int = 20,
        fps: int = 12,
        output_path: str = None,
    ) -> str:
        """Create a smooth interpolation GIF between two images using bidirectional optical flow.

        Warps A forward and B backward using Farneback dense optical flow, blending at each
        intermediate step.  The result is a ping-pong GIF that loops seamlessly.
        """
        from image.frame_cache import FrameCache

        def _load(path):
            # FrameCache handles video/GIF/PDF/SVG → temp JPEG; returns path unchanged for images.
            readable = FrameCache.get_image_path(path)
            img = cv2.imread(readable)
            if img is None:
                # cv2 can't decode some formats (AVIF, HEIC, JXL, …); fall back to Pillow,
                # which supports them via the installed plugin extensions.
                pil_img = Image.open(readable).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img

        img_a = _load(image_path_a)
        img_b = _load(image_path_b)

        h, w = img_a.shape[:2]
        if img_b.shape[:2] != (h, w):
            img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        farneback_params = dict(pyr_scale=0.5, levels=3, winsize=15,
                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **farneback_params)
        flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, **farneback_params)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                      np.arange(h, dtype=np.float32))
        fa = img_a.astype(np.float32)
        fb = img_b.astype(np.float32)

        pil_frames = []
        for t in np.linspace(0.0, 1.0, n_frames):
            map_x_ab = (grid_x + t * flow_ab[..., 0]).astype(np.float32)
            map_y_ab = (grid_y + t * flow_ab[..., 1]).astype(np.float32)
            map_x_ba = (grid_x + (1.0 - t) * flow_ba[..., 0]).astype(np.float32)
            map_y_ba = (grid_y + (1.0 - t) * flow_ba[..., 1]).astype(np.float32)

            warp_a = cv2.remap(fa, map_x_ab, map_y_ab, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            warp_b = cv2.remap(fb, map_x_ba, map_y_ba, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

            blended = ((1.0 - t) * warp_a + t * warp_b).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)))

        # Ping-pong: forward then reverse (interior frames only) for seamless loop
        all_frames = pil_frames + pil_frames[-2:0:-1]

        if output_path is None:
            stem = os.path.splitext(image_path_a)[0]
            output_path = stem + "_interpolation.gif"

        duration_ms = max(1, int(1000 / fps))
        all_frames[0].save(
            output_path,
            save_all=True,
            append_images=all_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return output_path

if __name__ == "__main__":
   ImageOps.randomly_modify_image(sys.argv[1])
