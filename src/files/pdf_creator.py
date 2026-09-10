import os
import shutil
import subprocess
import tempfile

from PIL import Image
import pypdfium2 as pdfium

from image.frame_cache import FrameCache
from utils.logging_setup import get_logger
from utils.translations import _
logger = get_logger("pdf_creator")


class PDFCreator:
    """
    Handles creation of PDFs from various file types.
    Each file will be added as a page in the PDF.
    Images will be added directly, while other file types will be converted to images first.
    """
    
    @staticmethod
    def create_pdf_from_files(file_paths, app_actions, output_path=None, options=None,
                              save_file_callback=None, notify_success=True):
        """
        Create a PDF from a list of files. Each file will be added as a page in the PDF.
        
        Args:
            file_paths: List of file paths to include in the PDF
            app_actions: AppActions instance for UI feedback
            output_path: Optional path to save the PDF. If None, will prompt user
                         via *save_file_callback*.
            options: Dictionary of PDF creation options:
                - preserve_quality: If True, maintain original image quality and format
                                  If False, compress images to reduce PDF size
                - filename: Base filename for the PDF
            save_file_callback: A callable(default_dir, default_name) -> str|None
                that shows a "Save As" dialog and returns the chosen path, or
                None if the user cancelled.  Required when *output_path* is None.
            notify_success: If True, show success toast on completion.
            
        Returns:
            bool: True if PDF was created successfully, False otherwise
        """
        if not file_paths:
            app_actions.toast(_("No files to create PDF"))
            return False

        if output_path is None:
            # Get the directory of the first file
            default_dir = os.path.dirname(file_paths[0])
            
            # Use the filename from options if provided, otherwise use default
            combined_images = _('combined_images')
            default_name = options.get('filename', combined_images) if options else combined_images
            
            if save_file_callback is None:
                logger.error("save_file_callback is required when output_path is not provided")
                return False
            output_path = save_file_callback(default_dir, default_name)
            if not output_path:
                return False

        try:
            # Create a new PDF document
            pdf = pdfium.PdfDocument.new()
            successful_pages = 0
            
            for file_path in file_paths:
                try:
                    # Get image path (handles conversion of various file types)
                    image_path = FrameCache.get_image_path(file_path)
                    
                    # Create a new image object in the PDF
                    image_obj = pdfium.PdfImage.new(pdf)
                    
                    # Check if it's a JPEG
                    if image_path.lower().endswith(('.jpg', '.jpeg')):
                        # For JPEGs, we can load directly
                        image_obj.load_jpeg(image_path, inline=True)
                    else:
                        # For other formats, convert using PIL
                        with Image.open(image_path) as img:
                            if options and not options.get('preserve_quality', True):
                                # Compressed mode - convert to JPEG with reduced quality
                                if img.mode in ('RGBA', 'LA'):
                                    background = Image.new('RGB', img.size, (255, 255, 255))
                                    background.paste(img, mask=img.split()[-1])
                                    img = background
                                elif img.mode != 'RGB':
                                    img = img.convert('RGB')
                                # Create a temporary file for the compressed image
                                temp_path = os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(image_path)}")
                                try:
                                    img.save(temp_path, format='JPEG', quality=85)
                                    image_obj.load_jpeg(temp_path, inline=True)
                                finally:
                                    # Ensure temp file is cleaned up even if an error occurs
                                    if os.path.exists(temp_path):
                                        try:
                                            os.remove(temp_path)
                                        except Exception as e:
                                            logger.error(f"Error cleaning up temp file {temp_path}: {str(e)}")
                            else:
                                # Quality preservation mode - use PIL to convert to bitmap
                                bitmap = pdfium.PdfBitmap.from_pil(img)
                                image_obj.set_bitmap(bitmap)
                                bitmap.close()
                    
                    # Get image dimensions and create page
                    w, h = image_obj.get_size()
                    image_obj.set_matrix(pdfium.PdfMatrix().scale(w, h))
                    page = pdf.new_page(w, h)
                    page.insert_obj(image_obj)
                    page.gen_content()
                    
                    # Clean up
                    image_obj.close()
                    page.close()
                    successful_pages += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

            if successful_pages > 0:
                # Save the PDF
                pdf.save(output_path)
                pdf.close()
                mode = _("compressed") if options and not options.get('preserve_quality', True) else _("high quality")
                if notify_success:
                    app_actions.success(_("Created {0} PDF with {1} pages").format(mode, successful_pages))
                return True
            else:
                app_actions.alert(_("Error"), _("No pages could be added to the PDF"), kind="error")
                return False
            
        except ImportError:
            app_actions.alert(_("Error"), _("PDF creation requires pypdfium2 package"), kind="error")
            return False
        except Exception as e:
            app_actions.alert(_("Error"), str(e), kind="error")
            return False 

    @staticmethod
    def get_default_diff_filename(file_paths) -> str:
        """Build a readable default basename for a 2-file diff PDF."""
        if not file_paths or len(file_paths) != 2:
            return "marked_files_diff"
        file_a, file_b = file_paths
        base_a = os.path.splitext(os.path.basename(file_a))[0]
        base_b = os.path.splitext(os.path.basename(file_b))[0]
        return f"{base_a}__VS__{base_b}__diff"

    @staticmethod
    def create_diff_pdf_from_files(
        file_paths,
        app_actions,
        options=None,
        output_path=None,
        save_file_callback=None,
    ) -> bool:
        """
        Create a visual diff PDF from exactly two marked files.

        Non-PDF files are converted into temporary one-page PDFs first.
        """
        if not file_paths or len(file_paths) != 2:
            app_actions.alert(
                _("Diff PDFs"),
                _("Please mark exactly 2 files."),
                kind="warning",
            )
            return False

        file_a, file_b = file_paths
        if not os.path.isfile(file_a) or not os.path.isfile(file_b):
            app_actions.alert(
                _("Invalid file"),
                _("One or more marked files do not exist on disk."),
                kind="warning",
            )
            return False

        diff_pdf_exe = shutil.which("diff-pdf") or shutil.which("diff-pdf.exe")
        if not diff_pdf_exe:
            app_actions.alert(
                _("diff-pdf not found"),
                _("The diff-pdf executable was not found on PATH."),
                kind="error",
            )
            return False

        if output_path is None:
            default_name = PDFCreator.get_default_diff_filename(file_paths)
            default_dir = os.path.dirname(file_a)
            selected_name = (
                (options or {}).get("filename", "") if options is not None else ""
            )
            selected_name = str(selected_name or "").strip()
            if selected_name:
                default_name = selected_name

            # For now, always write diff output next to the first marked file.
            # TODO: Allow custom save locations again once temp media windows
            # support all media types used by absolute-path go_to_file fallback.
            output_path = os.path.join(default_dir, default_name)

        # Ensure extension
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"

        with tempfile.TemporaryDirectory(prefix="weidr_diffpdf_") as tmp_dir:
            pdf_inputs = []
            for idx, path in enumerate((file_a, file_b), start=1):
                if path.lower().endswith(".pdf"):
                    pdf_inputs.append(path)
                    continue

                temp_pdf = os.path.join(tmp_dir, f"converted_{idx}.pdf")
                ok = PDFCreator.create_pdf_from_files(
                    [path],
                    app_actions,
                    output_path=temp_pdf,
                    options=options,
                    notify_success=False,
                )
                if not ok or not os.path.isfile(temp_pdf):
                    app_actions.alert(
                        _("Error"),
                        _("Failed to convert marked file to PDF for diff: {0}").format(path),
                        kind="error",
                    )
                    return False
                pdf_inputs.append(temp_pdf)

            cmd = [diff_pdf_exe, "--output-diff", output_path, pdf_inputs[0], pdf_inputs[1]]
            logger.info("Running diff-pdf command: %s", cmd)
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                app_actions.alert(
                    _("Error"),
                    _("Failed to run diff-pdf: {0}").format(str(e)),
                    kind="error",
                )
                return False

            app_actions.refresh()
            if os.path.isfile(output_path):
                try:
                    # Preferred behavior: navigate via absolute path.
                    app_actions.go_to_file(
                        search_text=output_path,
                        exact_match=True,
                    )
                except Exception as e:
                    # TODO: Remove fallback once temp image/media windows fully
                    # support non-image absolute-path navigation (e.g. PDFs).
                    logger.warning(f"Absolute-path navigation failed for diff PDF: {e}")
                    try:
                        app_actions.go_to_file(
                            search_text=os.path.basename(output_path),
                            exact_match=True,
                        )
                    except Exception as nested:
                        logger.warning(f"Basename fallback navigation failed: {nested}")
                app_actions.toast(_("Created diff PDF: {0}").format(os.path.basename(output_path)))
                return True

            stderr_text = (result.stderr or "").strip()
            if stderr_text:
                logger.warning(f"diff-pdf stderr: {stderr_text}")
            app_actions.alert(
                _("Error"),
                _("diff-pdf did not create an output file."),
                kind="error",
            )
            return False