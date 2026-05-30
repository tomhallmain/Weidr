#!/usr/bin/env python3
"""
Image Dimension Search Tool

Finds image files matching specified dimensions in a directory tree.
Supports direct dimension input or reading dimensions from a reference image.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math
import time
from fractions import Fraction

# Supported image extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.heic', '.heif', '.ico', '.jfif', '.pjpeg', '.pjp'
}

def calculate_aspect_ratio(width: int, height: int) -> Tuple[int, int]:
    """Calculate simplified aspect ratio (e.g., 16:9)."""
    if width <= 0 or height <= 0:
        return (0, 0)
    
    # Use fractions module for accurate ratio calculation
    ratio = Fraction(width, height)
    return (ratio.numerator, ratio.denominator)

def aspect_ratios_match(ratio1: Tuple[int, int], ratio2: Tuple[int, int], 
                       tolerance: float = 0.01) -> bool:
    """Check if two aspect ratios match within tolerance."""
    if ratio1[0] == 0 or ratio2[0] == 0:
        return False
    
    # Compare as floating point with tolerance
    ratio1_val = ratio1[0] / ratio1[1]
    ratio2_val = ratio2[0] / ratio2[1]
    return abs(ratio1_val - ratio2_val) <= tolerance

def validate_dimensions(width: int, height: int) -> None:
    """Validate dimension inputs."""
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive integers")

def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image dimensions without loading entire image into memory.
    Uses minimal file reading for common formats.
    """
    try:
        with open(image_path, 'rb') as f:
            # PNG
            if image_path.suffix.lower() in ('.png',):
                f.seek(16)
                return int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
            
            # JPEG
            elif image_path.suffix.lower() in ('.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp'):
                f.seek(0)
                data = f.read(2)
                while data:
                    while data and data[0] != 0xFF:
                        data = f.read(1)
                    while data and data[0] == 0xFF:
                        data = f.read(1)
                    
                    if data and data[0] >= 0xC0 and data[0] <= 0xCF and data[0] != 0xC4:
                        f.read(3)
                        height = int.from_bytes(f.read(2), 'big')
                        width = int.from_bytes(f.read(2), 'big')
                        return width, height
                    elif data:
                        size = int.from_bytes(f.read(2), 'big')
                        f.read(size - 2)
                        data = f.read(2)
            
            # GIF
            elif image_path.suffix.lower() == '.gif':
                f.seek(6)
                return int.from_bytes(f.read(2), 'little'), int.from_bytes(f.read(2), 'little')
            
            # BMP
            elif image_path.suffix.lower() == '.bmp':
                f.seek(18)
                width = int.from_bytes(f.read(4), 'little')
                height = int.from_bytes(f.read(4), 'little')
                # Handle negative height (top-down DIB)
                return width, abs(height)
            
            # TIFF
            elif image_path.suffix.lower() in ('.tiff', '.tif'):
                f.seek(0)
                byte_order = f.read(2)
                big_endian = byte_order == b'MM'
                if byte_order not in (b'II', b'MM'):
                    raise ValueError("Not a valid TIFF file")
                
                f.read(2)  # Skip magic number
                ifd_offset = int.from_bytes(f.read(4), 'little' if not big_endian else 'big')
                f.seek(ifd_offset)
                
                num_entries = int.from_bytes(f.read(2), 'little' if not big_endian else 'big')
                width = height = None
                
                for _unused in range(num_entries):
                    tag = int.from_bytes(f.read(2), 'little' if not big_endian else 'big')
                    f.read(6)  # Skip type and count
                    value = int.from_bytes(f.read(4), 'little' if not big_endian else 'big')
                    
                    if tag == 256:  # ImageWidth
                        width = value
                    elif tag == 257:  # ImageLength
                        height = value
                    
                    if width and height:
                        return width, height
        
        # Fallback to PIL for formats not handled above
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            pass
        
        raise ValueError(f"Unsupported image format or corrupted file: {image_path}")
        
    except (IOError, ValueError, IndexError) as e:
        raise ValueError(f"Cannot read image dimensions: {e}")

def build_dimension_map(directory: Path, extensions: list[str]) -> Dict[Tuple[int, int], List[Path]]:
    """Build a mapping of dimensions to file paths."""
    return build_dimension_and_aspect_map(directory, extensions)[0]

def build_dimension_and_aspect_map(
    directory: Path,
    extensions: list[str],
    progress_interval: int = 500,
    quiet: bool = False
) -> Tuple[Dict[Tuple[int, int], List[Path]], Dict[Tuple[int, int], List[Path]]]:
    """
    Build mappings for dimensions and aspect ratios.

    Returns:
        Tuple of (dimension_map, aspect_ratio_map)
    """
    dimension_map = defaultdict(list)  # (width, height) -> [paths]
    aspect_ratio_map = defaultdict(list)  # (numerator, denominator) -> [paths]

    # Use os.scandir for better performance than Path.rglob
    image_count = 0
    processed = 0
    errors = 0
    start_time = time.time()
    
    # First, count total images for progress reporting
    if not quiet:
        if not quiet:
            print(f"Scanning directory for images...", file=sys.stderr)
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if Path(filename).suffix.lower() in extensions:
                    image_count += 1
        if not quiet:
            print(f"Found {image_count} potential image files", file=sys.stderr)
    
    # Now process each image
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()
            
            if ext in extensions:
                if not quiet:
                    processed += 1
                    if processed % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(
                            f"Processed {processed}/{image_count} images "
                            f"({processed/image_count*100:.1f}%, "
                            f"{rate:.1f} images/sec)",
                            file=sys.stderr
                        )

                try:
                    dimensions = get_image_dimensions(filepath)
                    dimension_map[dimensions].append(filepath)
                    
                    # Also calculate and store aspect ratio
                    aspect_ratio = calculate_aspect_ratio(*dimensions)
                    if aspect_ratio != (0, 0):
                        aspect_ratio_map[aspect_ratio].append(filepath)
          
                except (ValueError, IOError):
                    errors += 1
                    continue
    
    if not quiet:
        elapsed = time.time() - start_time
        print(
            f"Done. Processed {processed} images in {elapsed:.1f} seconds "
            f"({processed/elapsed:.1f} images/sec), {errors} errors",
            file=sys.stderr
        )
   
    return dimension_map, aspect_ratio_map

def main():
    parser = argparse.ArgumentParser(
        description="Find images by dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mutually exclusive group for dimension specification
    dimension_group = parser.add_mutually_exclusive_group(required=True)
    dimension_group.add_argument(
        '-d', '--dimensions',
        nargs=2,
        type=int,
        metavar=('WIDTH', 'HEIGHT'),
        help="Target dimensions (width height)"
    )
    dimension_group.add_argument(
        '-i', '--image',
        type=Path,
        metavar='IMAGE_FILE',
        help="Reference image to get dimensions from"
    )
    
    parser.add_argument(
        'directory',
        type=Path,
        help="Directory to search for images"
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        default=True,
        help="Search recursively (default: True)"
    )

    parser.add_argument(
        '-e', '--extensions',
        type=str,
        default=",".join(IMAGE_EXTENSIONS),
        help="Set allowed extensions"
    )

    # Aspect ratio matching
    parser.add_argument(
        '-a', '--aspect-ratio',
        action='store_true',
        help="Also include images with matching aspect ratio"
    )

    parser.add_argument(
        '--aspect-tolerance',
        type=float,
        default=0.01,
        metavar='TOLERANCE',
        help="Tolerance for aspect ratio matching (default: 0.01)"
    )

    # Progress logging
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Suppress progress output"
    )


    args = parser.parse_args()
    
    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    if not args.extensions:
        print(f"Error: '{args.extensions}' is not a valid extension list")
        sys.exit(1)

    extensions = []
    try:
        extensions = args.extensions.split("(\w)*,(\w)*")
    except Exception:
        pass
    if len(extensions) < 1:
        print(f"Error: '{args.extensions}' is not a valid extension list")
        sys.exit(1)

    # Get target dimensions
    target_dimensions = None
    try:
        if args.image:
            if not args.image.exists():
                print(f"Error: Image '{args.image}' does not exist", file=sys.stderr)
                sys.exit(1)
            target_dimensions = get_image_dimensions(args.image)
        else:
            width, height = args.dimensions
            validate_dimensions(width, height)
            target_dimensions = (width, height)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Build dimension map and find matches
    try:
        dimension_map, aspect_ratio_map = build_dimension_and_aspect_map(
            args.directory,
            args.extensions,
            quiet=args.quiet
        )
        
        # Get exact dimension matches
        matching_files = dimension_map.get(target_dimensions, [])

        # Get aspect ratio matches if requested
        aspect_files = []
        if args.aspect_ratio:
            target_aspect = calculate_aspect_ratio(*target_dimensions)
            if target_aspect != (0, 0):
                for aspect_ratio, files in aspect_ratio_map.items():
                    if aspect_ratios_match(
                        target_aspect,
                        aspect_ratio,
                        args.aspect_tolerance
                    ):
                        for filepath in files:
                            # Don't add files that are already in exact matches
                            if filepath not in matching_files:
                                aspect_files.append(filepath)

        total_found = len(matching_files) + len(aspect_files)
        
        # Output results
        if total_found > 0:
            print(f"Found {len(matching_files)} image(s) with dimensions {target_dimensions[0]}x{target_dimensions[1]}:")
            for filepath in sorted(matching_files):
                print(filepath.resolve())
            
            if args.aspect_ratio and aspect_files:
                target_aspect = calculate_aspect_ratio(*target_dimensions)
                print(f"\nFound {len(aspect_files)} additional image(s) with matching aspect ratio {target_aspect[0]}:{target_aspect[1]}:")
                for filepath in sorted(aspect_files):
                    # Get dimensions for display
                    try:
                        dim = get_image_dimensions(filepath)
                        print(f"{filepath.resolve()} ({dim[0]}x{dim[1]})")
                    except:
                        print(filepath.resolve())
        else:
            print(f"No images found with dimensions {target_dimensions[0]}x{target_dimensions[1]}")
            if args.aspect_ratio:
                target_aspect = calculate_aspect_ratio(*target_dimensions)
                print(f"No images found with aspect ratio {target_aspect[0]}:{target_aspect[1]} either")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nSearch interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
