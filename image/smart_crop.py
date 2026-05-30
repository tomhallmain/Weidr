import os
import sys
from typing import Tuple, Union, Dict, List
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from PIL import ImageChops
from PIL import Image
from PIL.Image import Image as PilImage

from image.image_ops import ImageOps
from utils.config import config
from utils.logging_setup import get_logger

logger = get_logger("smart_crop")


def detect_edges_sobel(im: PilImage) -> PilImage:
    """
    Use Sobel edge detection to identify potential division lines.
    This would help catch gradient borders and more subtle divisions.
    """
    # Convert to grayscale
    gray = im.convert('L')
    gray_array = np.array(gray)
    
    # Sobel edge detection
    sobelx = ndimage.sobel(gray_array, axis=0)
    sobely = ndimage.sobel(gray_array, axis=1)
    
    # Combine edges
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = magnitude.astype(np.uint8)
    
    return Image.fromarray(magnitude)

def detect_contrast_regions(im: PilImage, window_size: int = 10) -> Dict[Tuple[int, int], float]:
    """
    Analyze local contrast in sliding windows to identify potential divisions.
    This helps catch divisions that might not have sharp edges but have
    significant contrast differences.
    """
    width, height = im.size
    contrast_map = {}
    
    for x in range(0, width - window_size, window_size//2):
        for y in range(0, height - window_size, window_size//2):
            region = im.crop((x, y, x + window_size, y + window_size))
            contrast = region.entropy()  # or use custom contrast calculation
            contrast_map[(x, y)] = contrast
            
    return contrast_map

def analyze_color_clusters(im: PilImage) -> np.ndarray:
    """
    Use K-means clustering to identify dominant color regions and potential
    divisions between them. This helps catch divisions based on color
    composition rather than just edge detection.
    """
    # Convert image to array of RGB values
    img_array = np.array(im)
    pixels = img_array.reshape(-1, 3)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(pixels)
    
    # Analyze cluster boundaries
    return kmeans.labels_.reshape(img_array.shape[:2])

def smart_consolidate_diffs(diffs: Dict[int, float], image_size: int, min_gap: int = 20) -> Dict[int, float]:
    """
    Enhanced consolidation that considers:
    - Minimum gap between divisions
    - Strength of the division (contrast/edge strength)
    - Proximity to image edges
    - Symmetry considerations
    """
    consolidated = {}
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[0])
    
    for i, (pos, strength) in enumerate(sorted_diffs):
        if i > 0 and pos - sorted_diffs[i-1][0] < min_gap:
            # Merge with previous if closer than min_gap
            prev_pos, prev_strength = sorted_diffs[i-1]
            if strength > prev_strength:
                consolidated[pos] = strength
                if prev_pos in consolidated:
                    del consolidated[prev_pos]
            else:
                consolidated[prev_pos] = prev_strength
        else:
            consolidated[pos] = strength
            
    return consolidated

def validate_division(im: PilImage, division_pos: int, is_horizontal: bool) -> bool:
    """
    Validate a potential division by checking:
    - Length of the division line
    - Consistency of the division across the image
    - Whether it creates reasonable subimage sizes
    - Whether it aligns with other detected divisions
    """
    width, height = im.size
    min_size = 30  # Minimum subimage size
    
    if is_horizontal:
        if division_pos < min_size or division_pos > height - min_size:
            return False
    else:
        if division_pos < min_size or division_pos > width - min_size:
            return False
            
    # Add more validation logic here
    return True

class Cropper:
    @staticmethod
    def smart_crop_simple(image_path: str, new_filename: str) -> None:
        new_filepath = ImageOps.new_filepath(image_path, new_filename)
        if os.path.exists(new_filepath):
            logger.info("Skipping crop already run: " + new_filepath)
            return
        im = Image.open(image_path)
        cropped_image, is_cropped = Cropper.remove_borders(im)
        im.close()
        if is_cropped:
            cropped_image.save(new_filepath)
            cropped_image.close()
            logger.info("Cropped image: " + new_filepath)
        else:
            logger.info("No cropping")

    @staticmethod
    def smart_crop_multi_detect(image_path: str, new_filename: str) -> list[str]:
        '''
        The image file may contain multiple divisions, that is, multiple valid images.
        The challenge is to determine whether the parts separating a division on the exteriors
        of the image are simply borders or images by themselves. Depending on the answer
        we can crop the image and save valid images accordingly.
        '''
        saved_files = []
        new_filepath = ImageOps.new_filepath(image_path, new_filename)
        if os.path.exists(new_filepath):
            logger.info("Skipping crop already run: " + new_filepath)
            return saved_files
        im = Image.open(image_path)
        cropped_images, is_cropped = Cropper.remove_borders_by_division_detection(im)
        im.close()
        if is_cropped:
            index_filepaths = len(cropped_images) > 1
            for i in range(len(cropped_images)):
                cropped_image = cropped_images[i]
                if index_filepaths:
                    new_filepath = ImageOps.new_filepath(image_path, new_filename, append_part="_" + str(i))
                cropped_image.save(new_filepath)
                cropped_image.close()
            logger.info("Cropped image: " + new_filepath)
        else:
            logger.info("No cropping")
        return saved_files

    @staticmethod
    def is_in_color_range(px: Tuple[int, int, int], minimal_color: int) -> bool:
        return px[0] >= minimal_color and px[1] >= minimal_color and px[2] >= minimal_color

    @staticmethod
    def is_close_color(px: Tuple[int, int, int], color: Tuple[int, int, int], tolerance=5) -> bool:
        return abs(px[0] - color[0]) <= tolerance and \
                abs(px[1] - color[1]) <= tolerance and \
                abs(px[2] - color[2]) <= tolerance

    @staticmethod
    def remove_borders(im: PilImage) -> Tuple[PilImage, bool]:
        '''
        A crude way to remove borders from an image using the top left pixel color.
        '''
        width, height = im.size
        left = 0
        top = 0
        right = width - 1
        bottom = height - 1
        top_left_color = im.getpixel((0, 0))
    #    top_right_color = im.getpixel((width - 1, 0))
    #    bottom_left_color = im.getpixel((0, height - 1))
    #    bottom_right_color = im.getpixel((width - 1, height - 1))

        while left < right and Cropper.is_column_color(im, left, top_left_color):
            if config.debug:
                logger.debug(f"LEFT: {left} RIGHT: {right}")
            left += 1
        while right > left and Cropper.is_column_color(im, right, top_left_color):
            if config.debug:
                logger.debug(f"RIGHT: {right} LEFT: {left}")
            right -= 1
        while top < bottom and Cropper.is_line_color(im, top, top_left_color):
            if config.debug:
                logger.debug(f"TOP: {top} BOTTOM: {bottom}")
            top += 1
        while bottom > top and Cropper.is_line_color(im, bottom, top_left_color):
            if config.debug:
                logger.debug(f"BOTTOM: {bottom} TOP: {top}")
            bottom -= 1

        if top == 0 and left == 0 and right == width - 1 and bottom == height - 1:
            logger.info('no borders detected')
            return im, False

        # Crop based on found borders
        bbox = (left, top, right, bottom)
        if config.debug:
            logger.debug(f"ORIGINAL IMAGE BOX: 0, 0, {width}, {height}")
            logger.debug(f"CROPPED IMAGE BOX: {left}, {top}, {right}, {bottom}")
        return im.crop(bbox), True

    @staticmethod
    def find_standard_deviation_of_pixel_color_in_image(im: PilImage):
        '''
        Returns the standard deviation of the pixel color in the image.
        '''
        width, height = im.size
#        total_pixels = width * height
#        pixel_colors = im.getcolors(total_pixels)

    @staticmethod
    def remove_borders_by_division_detection(im: PilImage, tolerance: int = 100) -> Tuple[list[PilImage], bool]:
        '''
        Find vertical and horizontal divisions in an image and remove borders or split the image based on these.
        Uses multiple detection strategies for better accuracy.
        '''
        width, height = im.size
        midpoint_x, midpoint_y = int(width / 2), int(height / 2)
        
        logger.info("Starting multi-strategy division detection...")
        logger.info(f"Image dimensions: {width}x{height}")

        # Get edge detection results
        logger.info("Running Sobel edge detection...")
        edge_image = detect_edges_sobel(im)
        edge_array = np.array(edge_image)
        
        # Get contrast analysis
        logger.info("Analyzing contrast regions...")
        contrast_map = detect_contrast_regions(im)
        
        # Get color clustering results
        logger.info("Performing color clustering analysis...")
        color_clusters = analyze_color_clusters(im)
        
        # Initialize division dictionaries
        horizontal_diffs = {}
        vertical_diffs = {}
        
        # Process edge detection results for horizontal divisions
        logger.info("Processing horizontal edge detection results...")
        for y in range(1, height):
            edge_strength = np.mean(edge_array[y, :])
            if edge_strength > tolerance:
                horizontal_diffs[y] = edge_strength
                
        # Process edge detection results for vertical divisions
        logger.info("Processing vertical edge detection results...")
        for x in range(1, width):
            edge_strength = np.mean(edge_array[:, x])
            if edge_strength > tolerance:
                vertical_diffs[x] = edge_strength
        
        # Process contrast map for additional divisions
        logger.info("Processing contrast map for additional divisions...")
        for (x, y), contrast in contrast_map.items():
            if contrast > tolerance:
                if x % 10 == 0:  # Vertical division
                    vertical_diffs[x] = max(vertical_diffs.get(x, 0), contrast)
                if y % 10 == 0:  # Horizontal division
                    horizontal_diffs[y] = max(horizontal_diffs.get(y, 0), contrast)
        
        # Process color clusters for additional divisions
        logger.info("Processing color cluster boundaries...")
        cluster_changes_x = np.diff(color_clusters, axis=1)
        cluster_changes_y = np.diff(color_clusters, axis=0)
        
        for x in range(1, width-1):
            if np.any(cluster_changes_x[:, x] != 0):
                vertical_diffs[x] = max(vertical_diffs.get(x, 0), tolerance)
                
        for y in range(1, height-1):
            if np.any(cluster_changes_y[y, :] != 0):
                horizontal_diffs[y] = max(horizontal_diffs.get(y, 0), tolerance)

        logger.info(f"Initial detection found {len(horizontal_diffs)} horizontal and {len(vertical_diffs)} vertical potential divisions")

        # Smart consolidation of all detected divisions
        logger.info("Consolidating close divisions...")
        horizontal_diffs = smart_consolidate_diffs(horizontal_diffs, height)
        vertical_diffs = smart_consolidate_diffs(vertical_diffs, width)
        
        logger.info(f"After consolidation: {len(horizontal_diffs)} horizontal and {len(vertical_diffs)} vertical divisions")
        
        # Validate divisions
        logger.info("Validating detected divisions...")
        validated_horizontal = {pos: strength for pos, strength in horizontal_diffs.items() 
                              if validate_division(im, pos, True)}
        validated_vertical = {pos: strength for pos, strength in vertical_diffs.items() 
                            if validate_division(im, pos, False)}

        logger.info(f"After validation: {len(validated_horizontal)} horizontal and {len(validated_vertical)} vertical valid divisions")

        if len(validated_horizontal) == 0 and len(validated_vertical) == 0:
            logger.info('No borders or subimages detected')
            return [im], False

        if config.debug:
            logger.info(f'Found horizontal diffs: {validated_horizontal}')
            logger.info(f'Found vertical diffs: {validated_vertical}')

        # If the image is divided down the middle, test both the left and right images for entropy.
        logger.info("Checking for middle divisions...")
        if len(validated_horizontal) == 1 and \
                abs(max(validated_horizontal.keys()) - midpoint_y) < int(height/10):
            logger.info("Detected middle horizontal division")
            validated_horizontal[0] = 0
            validated_horizontal[height] = height
        elif len(validated_vertical) == 1 and \
                abs(max(validated_vertical.keys()) - midpoint_x) < int(width/10):
            logger.info("Detected middle vertical division")
            validated_vertical[0] = 0
            validated_vertical[width] = width

        if len(validated_horizontal) > 2 or len(validated_vertical) > 2:
            logger.info('Multiple subimages detected!')
            logger.info(f"Horizontal diffs: {validated_horizontal}")
            logger.info(f"Vertical diffs: {validated_vertical}")
            return Cropper.split_image(im, validated_horizontal, validated_vertical), True
        else:
            logger.info("Processing single division case...")
            if len(validated_vertical) == 0 or (len(validated_vertical) == 1 and min(validated_vertical.keys()) > midpoint_x):
                left = 0
            else:
                left = min(validated_vertical.keys())
            if len(validated_vertical) == 0 or (len(validated_vertical) == 1 and max(validated_vertical.keys()) < midpoint_x):
                right = width - 1
            else:
                right = max(validated_vertical.keys())
            if len(validated_horizontal) == 0 or (len(validated_horizontal) == 1 and min(validated_horizontal.keys()) > midpoint_y):
                top = 0
            else:
                top = min(validated_horizontal.keys())
            if len(validated_horizontal) == 0 or (len(validated_horizontal) == 1 and max(validated_horizontal.keys()) < midpoint_y):
                bottom = height - 1
            else:
                bottom = max(validated_horizontal.keys())
            bbox = (left, top, right, bottom)
            if config.debug:
                logger.info(f"Original image box: 0, 0, {width}, {height}")
                logger.info(f"Cropped image box: {left}, {top}, {right}, {bottom}")
            return [im.crop(bbox)], True

    @staticmethod
    def split_image(im, horizontal_diffs, vertical_diffs) -> list[PilImage]:
        '''
        Splits the image into a list of images based on known horizontal and vertical divisions.
        '''
        width, height = im.size
        logger.debug(f"{width}x{height}")
        subimages = []
        xs = list(vertical_diffs.keys())
        ys = list(horizontal_diffs.keys())
        xs.sort()
        ys.sort()
        if len(xs) == 0 and len(ys) == 0:
            return [im]
        if 0 not in xs:
            xs.insert(0, 0)
        if width not in xs:
            xs.append(width)
        if 0 not in ys:
            ys.insert(0, 0)
        if height not in ys:
            ys.append(height)
        if config.debug:
            logger.debug(f"SUBIMAGE CROP Xs: {xs}")
            logger.debug(f"SUBIMAGE CROP Ys: {ys}")
        for x in range(len(xs) - 1):
            for y in range(len(ys) - 1):
                subimages.append(im.crop((xs[x], ys[y], xs[x + 1], ys[y + 1])))
        i = 0
        subimage_count = 0
        while i < len(subimages):
            subimage = subimages[i]
            if Cropper.is_small(subimage):
                logger.info(f"Subimage {subimage_count} is invalid due to being too small.")
                del subimages[i]
            elif Cropper.is_low_entropy(subimage):
                logger.info(f"Subimage {subimage_count} is invalid due to low entropy.")
                del subimages[i]
            else:
                i += 1
            subimage_count += 1
        return subimages

    @staticmethod
    def is_low_entropy(im):
        '''
        Checks the image to see if it is low entropy.
        '''
        entropy = im.entropy()
        if config.debug:
            logger.debug(f"Entropy of {im} is {entropy}.")
        return entropy < 5

    @staticmethod
    def is_small(im):
        width, height = im.size
        return width < 30 or height < 30

    @staticmethod
    def is_line_color(im: PilImage, y: int, color: Tuple[int, int, int]) -> bool:
        width, _unused = im.size
        print_counts = 0
        if config.debug:
            logger.debug("Comparison for line: " + str(y))
        for x in range(0, width - 1):
            px = im.getpixel((x,y))
            if not Cropper.is_close_color(px, color):
                if config.debug:
                    logger.debug(f"{px} <> {color} (unmatched on x {x})")
                return False
            if config.debug and print_counts < 10:
                logger.debug(f"{px} <> {color}")
                print_counts += 1
        return True

    @staticmethod
    def is_column_color(im: PilImage, x: int, color: Tuple[int, int, int]) -> bool:
        _unused, height = im.size
        print_counts = 0
        if config.debug:
            logger.debug("Comparison for column: " + str(x))
        for y in range(0, height-1):
            px = im.getpixel((x,y))
            if not Cropper.is_close_color(px, color):
                if config.debug:
                    logger.debug(f"{px} <> {color} (unmatched on y {y})")
                return False
            if config.debug and print_counts < 10:
                logger.debug(f"{px} <> {color}")
                print_counts += 1
        return True

    @staticmethod
    def get_crop_box_by_px_color(
            im: PilImage,
            px: Tuple[int, int],
            scale: float,
            offset: int) -> Tuple[int, int, int, int]:
        bg = Image.new(im.mode, im.size, px)
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, scale, offset)
        return diff.getbbox()

    @staticmethod
    def crop_by_background(
            im: PilImage,
            minimal_light_background_color_value: int) -> Tuple[int, int, int, int]:
        width, height = im.size
        original_box = (0, 0, width, height)

        # crop by top left pixel color
        px = im.getpixel((0, height-1))
        if Cropper.is_in_color_range(px, minimal_light_background_color_value):
            bbox1 = Cropper.get_crop_box_by_px_color(im, px, 2.0, -100)
        else:
            bbox1 = original_box
        
        # crop by bottom right pixel color
        px = im.getpixel((width-1, height-1))
        if Cropper.is_in_color_range(px, minimal_light_background_color_value):
            bbox2 = Cropper.get_crop_box_by_px_color(im, px, 2.0, -100)
        else:
            bbox2 = original_box

        crop = (
            max(bbox1[0], bbox2[0]),
            max(bbox1[1], bbox2[1]),
            min(bbox1[2], bbox2[2]),
            min(bbox1[3], bbox2[3])
        )

        return crop

    def calculate_optimal_crop(
            self,
            im_width: int,
            im_height: int,
            inner_rect: Tuple[int, int, int, int],
            ratio: float
    ) -> Union[Tuple[int, int, int, int], None]:
        im_ratio = im_width / im_height

        # not all images have to be cropped
        if im_ratio == ratio:
            return None
        
        left, upper, right, bottom = inner_rect

        # calculate with max height
        height = im_height
        width = int(im_height * ratio)

        # crop width
        if width <= im_width:
            c_left, c_right = Cropper.expand(left, right, width, im_width-1)
            c_upper = 0; c_bottom = height-1

        # crop height
        else:
            width = im_width
            height = int(im_width / ratio)
            c_upper, c_bottom = Cropper.expand(upper, bottom, height, im_height-1)
            c_left = 0; c_right = im_width-1

        return (c_left, c_upper, c_right, c_bottom)

    @staticmethod
    def expand(m1: int, m2: int, value: int, max_size: int) -> Tuple[int, int]:
        value = int((value - (m2-m1)) / 2)
        m1 -= value; m2 += value

        if m1 < 0:
            m2 += abs(m1); m1 = 0
        elif m2 > max_size:
            m1 -= m2 - max_size; m2 = max_size
        
        return m1, m2

    @staticmethod
    def detect_perfectly_vertical_divisions(im: PilImage, tolerance: int = 100, diffs: dict = {}) -> Tuple[dict, dict]:
        '''
        Some images have borders that use gradient colors, making a simple
        test for the presence of a vertical line not matching a color insufficient.
        This function detects the presence of such a vertical division.
        Finds the average difference between left and right pixels.
        '''
        width, height = im.size
        x = 1
        if len(diffs) == 0:
            while x < width:
                diffs_for_x = []
                y = 0
                while y < height:
                    px1 = im.getpixel((x-1, y))
                    px2 = im.getpixel((x, y))
                    diff = abs(px2[0] - px1[0]) + \
                        abs(px2[1] - px1[1]) + \
                        abs(px2[2] - px1[2])
                    y += 1
                    diffs_for_x.append(diff)
                if len(diffs_for_x) == 0:
                    raise Exception(f"Failed to parse diffs for x={x}")
                diffs[x] = sum(diffs_for_x) / len(diffs_for_x)
                x += 1
        diffs_copy = {k:v for k, v in diffs.items()}
        for x, avg_diff in sorted(diffs.items(), key=lambda x:x[1], reverse=True):
            if config.debug and avg_diff > int(tolerance / 2):
                logger.debug(f"x = {x}, avg diff = {avg_diff}, tolerance = {tolerance}")
            if avg_diff < tolerance:
                del diffs[x]
        Cropper.consolidate_close_diffs(width, diffs, tolerance=max(10, int(width/10)))
        for x, avg_diff in diffs.items():
            logger.debug(f"FINAL x = {x}, avg diff = {avg_diff}")
        return diffs, diffs_copy

    @staticmethod
    def detect_perfectly_horizontal_divisions(im: PilImage, tolerance: int = 100, diffs: dict = {}) -> Tuple[dict, dict]:
        '''
        Some images have borders that use gradient colors, making a simple
        test for the presence of a horizontal line matching a color insufficient.
        This function detects the presence of such a horizontal division.
        Finds the average difference between top and bottom pixels.
        '''
        width, height = im.size
        y = 1
        diffs = {}
        if len(diffs) == 0:
            while y < height:
                diffs_for_y = []
                x = 0
                while x < width:
                    px1 = im.getpixel((x, y-1))
                    px2 = im.getpixel((x, y))
                    diff = abs(px2[0] - px1[0]) + \
                        abs(px2[1] - px1[1]) + \
                        abs(px2[2] - px1[2])
                    x += 1
                    diffs_for_y.append(diff)
                if len(diffs_for_y) == 0:
                    raise Exception(f"Failed to parse diffs for y={y}")
                diffs[y] = sum(diffs_for_y) / len(diffs_for_y)
                y += 1
        diffs_copy = {k:v for k, v in diffs.items()}
        for y, avg_diff in sorted(diffs.items(), key=lambda y:y[1], reverse=True):
            if avg_diff > int(tolerance / 2):
                logger.debug(f"y = {y}, avg diff = {avg_diff}")
            if avg_diff < tolerance:
                del diffs[y]
        Cropper.consolidate_close_diffs(height, diffs, tolerance=max(10, int(height/10)))
        for y, avg_diff in diffs.items():
            logger.debug(f"FINAL y = {y}, avg diff = {avg_diff}")
        return diffs, diffs_copy

    @staticmethod
    def consolidate_close_diffs(_max, diffs, tolerance=10):
        '''
        Consolidate the diffs dict by merging close entries. Identifies which 
        entries are close enough to each other and then selects the most likely
        candidate based on its proximity to the edges of the image (the max).
        If the diff position is close to 0, select the higher value, and if it's
        close to the max, select the lower value -- We only want to preserve 
        the valid part of the image, which is probably not inclusive of small
        slivers that the interior of these grouped diff positions would represent.
        '''
        midpoint = int(_max / 2)
        keys = list(diffs.keys())
        keys.sort()
        matches = {}
        def is_close_to_existing_match(key):
            if config.debug:
                logger.debug(f"key = {key}, match values = {matches}")
            for match_id, match_values in matches.items():
                for match_value in match_values:
                    if key == match_value:
                        return -2 # The key is already in a group.
            for match_id, match_values in matches.items():
                for match_value in match_values:
                    if abs(key - match_value) < tolerance:
                        return match_id
            return -1
        match_id = -1
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                existing_match = False
                test_match_id = is_close_to_existing_match(keys[i])
                if test_match_id != -1:
                    if test_match_id != -2:
                        matches[test_match_id].append(keys[i])
                    existing_match = True
                test_match_id = is_close_to_existing_match(keys[j])
                if test_match_id > -1:
                    if test_match_id != -2:
                        matches[test_match_id].append(keys[j])
                    existing_match = True
                if not existing_match and abs(keys[j] - keys[i]) < tolerance:
                    match_id += 1
                    matches[match_id] = [keys[i], keys[j]]
        for match in matches.values():
            match.sort()
        match_keys = list(matches.keys())
        match_keys.sort()
        for match_values in matches.values():
            avg_value = sum(match_values)/len(match_values)
            winning_value = min(match_values) if avg_value > midpoint else max(match_values)
            for val in match_values:
                if val != winning_value:
                    if config.debug:
                        logger.debug(f'Consolidated value = {val} Winning value = {winning_value} Max = {_max}')
                    del diffs[val]
        return diffs


if __name__ == '__main__':


#   Cropper.smart_crop_simple(sys.argv[1], "")
    Cropper.smart_crop_multi_detect(sys.argv[1], "")
    exit()

    extensions = [".jpg", ".jpeg", ".png", ".webp", ".tiff"]
    directory_to_process = sys.argv[1]
    if not os.path.isdir(directory_to_process):
        logger.error('not a directory: "' + directory_to_process + '"')
        exit()
    files_to_crop = []
    for f in os.listdir(directory_to_process):
        for ext in extensions:
            if f[-len(ext):] == ext:
                files_to_crop.append(os.path.join(directory_to_process, f))

    for f in files_to_crop:
        try:
            Cropper.smart_crop_multi_detect(f, "")
        except Exception as e:
            logger.error("Error processing file " + f)
            logger.error(e)

