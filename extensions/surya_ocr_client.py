"""
Surya OCR client -- wraps the Surya OCR library to extract text from images.

Uses Surya's FoundationPredictor, DetectionPredictor, and RecognitionPredictor
to perform OCR with bounding-box-based reading-order sorting.

Surya is an optional heavy dependency (PyTorch + model downloads on first use).
The module gracefully degrades when `surya` is not installed.
"""

from __future__ import annotations

import os

from utils.logging_setup import get_logger

logger = get_logger("surya_ocr_client")

# ---------------------------------------------------------------------------
# Lazy availability flag
# ---------------------------------------------------------------------------
_surya_available: bool | None = None  # None = not yet checked


def is_surya_available() -> bool:
    """Return *True* if the surya package can be imported."""
    global _surya_available
    if _surya_available is None:
        try:
            from surya.recognition import RecognitionPredictor  # noqa: F401
            from surya.detection import DetectionPredictor       # noqa: F401
            from surya.foundation import FoundationPredictor     # noqa: F401
            _surya_available = True
        except ImportError:
            _surya_available = False
            logger.info("Surya OCR is not installed -- OCR features disabled")
    return _surya_available


# ---------------------------------------------------------------------------
# Predictor singletons (initialised on first call)
# ---------------------------------------------------------------------------
_foundation_predictor = None
_det_predictor = None
_rec_predictor = None


def _ensure_predictors():
    """Lazily initialise the heavy Surya model objects (downloads on first use)."""
    global _foundation_predictor, _det_predictor, _rec_predictor

    if _rec_predictor is not None:
        return

    from surya.recognition import RecognitionPredictor
    from surya.foundation import FoundationPredictor
    from surya.detection import DetectionPredictor

    logger.info("Initialising Surya OCR predictors (may download models on first use)...")
    _foundation_predictor = FoundationPredictor()
    _det_predictor = DetectionPredictor()
    _rec_predictor = RecognitionPredictor(_foundation_predictor)
    logger.info("Surya OCR predictors ready")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SuryaOCRResult:
    """Lightweight container for OCR results."""

    __slots__ = ("text", "text_lines", "line_count", "has_text",
                 "confidence_scores", "avg_confidence")

    def __init__(
        self,
        text: str,
        text_lines: list[str],
        line_count: int,
        has_text: bool,
        confidence_scores: list[float] | None = None,
        avg_confidence: float | None = None,
    ) -> None:
        self.text = text
        self.text_lines = text_lines
        self.line_count = line_count
        self.has_text = has_text
        self.confidence_scores = confidence_scores
        self.avg_confidence = avg_confidence


def run_ocr(image_path: str) -> SuryaOCRResult:
    """
    Run Surya OCR on *image_path* and return a :class:`SuryaOCRResult`.

    Text lines are sorted in visual reading order (top-to-bottom,
    left-to-right) using bounding-box coordinates with adaptive
    Y-tolerance grouping.

    Raises
    ------
    RuntimeError
        If Surya is not installed.
    FileNotFoundError
        If *image_path* does not exist.
    """
    if not is_surya_available():
        raise RuntimeError(
            "Surya OCR is not installed. "
            "Install with: pip install surya-ocr"
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    from PIL import Image
    from surya.common.surya.schema import TaskNames

    _ensure_predictors()

    logger.info(f"Running OCR on: {image_path}")
    with Image.open(image_path) as image:
        predictions = _rec_predictor(
            [image],
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=_det_predictor,
            math_mode=False,
        )

    # ------------------------------------------------------------------
    # Extract text + position data
    # ------------------------------------------------------------------
    line_data: list[tuple[float, float, str, float | None]] = []

    for page_pred in predictions:
        bboxes = None
        if hasattr(page_pred, "bboxes") and page_pred.bboxes is not None:
            bboxes = page_pred.bboxes
        elif hasattr(page_pred, "line_bboxes") and page_pred.line_bboxes is not None:
            bboxes = page_pred.line_bboxes

        for line_idx, line in enumerate(page_pred.text_lines):
            if not (line.text and line.text.strip()):
                continue

            text = line.text.strip()
            confidence = getattr(line, "confidence", None)

            y_pos, x_pos = 0.0, 0.0
            bbox_found = False

            # Method 1: direct bbox on the line object
            if hasattr(line, "bbox") and line.bbox is not None:
                bbox = line.bbox
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x_pos, y_pos = float(bbox[0]), float(bbox[1])
                    bbox_found = True

            # Method 2: page-level bboxes array
            if not bbox_found and bboxes is not None and line_idx < len(bboxes):
                bbox = bboxes[line_idx]
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x_pos, y_pos = float(bbox[0]), float(bbox[1])
                    bbox_found = True

            # Method 3: geometry / polygon fallbacks
            if not bbox_found:
                if hasattr(line, "geometry") and line.geometry is not None:
                    geom = line.geometry
                    if hasattr(geom, "bbox"):
                        bbox = geom.bbox
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            x_pos, y_pos = float(bbox[0]), float(bbox[1])
                            bbox_found = True
                elif hasattr(line, "polygon") and line.polygon is not None:
                    poly = line.polygon
                    if isinstance(poly, (list, tuple)) and poly:
                        pt = poly[0]
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            x_pos, y_pos = float(pt[0]), float(pt[1])
                            bbox_found = True

            line_data.append((y_pos, x_pos, text, confidence))

    image.close()

    # ------------------------------------------------------------------
    # Sort in reading order
    # ------------------------------------------------------------------
    Y_TOLERANCE = 10.0
    valid_ys = [y for y, _, _, _ in line_data if y > 0]
    if valid_ys and len(valid_ys) > 1:
        y_range = max(valid_ys) - min(valid_ys)
        Y_TOLERANCE = max(10.0, y_range / 50.0)

    def _sort_key(item):
        y, x, _t, _c = item
        if y == 0 and x == 0:
            return (999999.0, 999999.0)
        y_group = round(y / Y_TOLERANCE) * Y_TOLERANCE
        return (y_group, x)

    line_data.sort(key=_sort_key)

    text_lines = [t for _, _, t, _ in line_data]
    confidence_scores = [c for _, _, _, c in line_data if c is not None]

    avg_conf = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores
        else None
    )

    result = SuryaOCRResult(
        text="\n".join(text_lines) if text_lines else "",
        text_lines=text_lines,
        line_count=len(text_lines),
        has_text=len(text_lines) > 0,
        confidence_scores=confidence_scores or None,
        avg_confidence=avg_conf,
    )

    logger.info(f"OCR complete -- {result.line_count} line(s) extracted")
    return result
