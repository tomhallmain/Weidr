"""
Ordered interceptor-rule list, persistence, matching, and transform execution.

This is the only module the marks transfer loop imports for interception:
``apply()`` returns an :class:`InterceptResult` describing what the caller
should do, and ``cleanup_after_move()`` handles the post-transfer original
deletion, so no rule logic lives in the transfer loop itself.

Rules are evaluated in list order and the first match wins, matching how
prevalidations and file-action history are already resolved elsewhere in the
app. List position is therefore the user's precedence control: a media-type
rule placed above a target-directory rule wins for files that match both.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from files.file_interceptor_rule import (
    FileInterceptorRule,
    InterceptorBehavior,
    InterceptorTransformOp,
)
from image.image_ops import ImageOps
from utils.app_info_cache import app_info_cache
from utils.constants import CompareMediaType
from utils.logging_setup import get_logger
from utils.media_utils import get_media_type_for_path, resolve_rendered_frame_source
from utils.translations import _
from utils.utils import Utils

logger = get_logger("file_interceptor_rules_manager")


@dataclass
class InterceptResult:
    """What the transfer loop should do with one file after interception.

    The default instance (no rule matched) is deliberately inert: not blocked,
    no substituted source, nothing to clean up.
    """

    blocked: bool = False
    #: Reason text recorded for a blocked file; already shown to the user by apply().
    block_message: str = ""
    #: Transformed file to transfer in place of the original, when a transform ran.
    transformed_source: Optional[str] = None
    rule_name: str = ""
    #: Whether the original should be deleted after the transformed file transfers.
    delete_original: bool = False


class FileInterceptorRulesManager:
    """Class-level rule storage plus the evaluation entry points."""

    RULES_KEY = "file_interceptor_rules"

    rules: list[FileInterceptorRule] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def load_rules() -> None:
        loaded: list[FileInterceptorRule] = []
        for rule_dict in list(app_info_cache.get_meta(
            FileInterceptorRulesManager.RULES_KEY, default_val=[]
        )):
            if not isinstance(rule_dict, dict):
                continue
            try:
                loaded.append(FileInterceptorRule.from_dict(rule_dict))
            except Exception as e:
                logger.error(f"Failed to load file interceptor rule: {e}")
        FileInterceptorRulesManager.rules = loaded

    @staticmethod
    def store_rules() -> None:
        app_info_cache.set_meta(
            FileInterceptorRulesManager.RULES_KEY,
            [rule.to_dict() for rule in FileInterceptorRulesManager.rules],
        )

    @staticmethod
    def set_rules(rules: list) -> None:
        FileInterceptorRulesManager.rules = list(rules)
        FileInterceptorRulesManager.store_rules()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_dir(dirpath: Optional[str]) -> str:
        if not dirpath:
            return ""
        return os.path.normcase(os.path.normpath(os.path.abspath(dirpath)))

    @staticmethod
    def _target_dir_matches(rule: FileInterceptorRule, target_dir: str) -> bool:
        if not rule.match_target_dirs:
            return True
        normalized_target = FileInterceptorRulesManager._normalize_dir(target_dir)
        if normalized_target == "":
            return False
        for configured in rule.match_target_dirs:
            normalized_rule_dir = FileInterceptorRulesManager._normalize_dir(configured)
            if normalized_rule_dir == "":
                continue
            if normalized_target == normalized_rule_dir:
                return True
            if rule.include_subdirectories and normalized_target.startswith(
                normalized_rule_dir + os.sep
            ):
                return True
        return False

    @staticmethod
    def _filename_matches(rule: FileInterceptorRule, filepath: str) -> bool:
        if not rule.match_filename_patterns:
            return True
        filename = os.path.basename(filepath)
        if not rule.filename_case_sensitive:
            filename = filename.lower()
        for pattern in rule.match_filename_patterns:
            candidate = pattern if rule.filename_case_sensitive else pattern.lower()
            if candidate and candidate in filename:
                return True
        return False

    @staticmethod
    def _media_type_matches(rule: FileInterceptorRule, filepath: str) -> bool:
        if not rule.match_media_types:
            return True
        return get_media_type_for_path(filepath) in rule.match_media_types

    @staticmethod
    def rule_matches(
        rule: FileInterceptorRule, filepath: str, target_dir: str, is_moving: bool
    ) -> bool:
        if not rule.is_active:
            return False
        if not rule.applies_to.allows(is_moving):
            return False
        return (
            FileInterceptorRulesManager._target_dir_matches(rule, target_dir)
            and FileInterceptorRulesManager._filename_matches(rule, filepath)
            and FileInterceptorRulesManager._media_type_matches(rule, filepath)
        )

    @staticmethod
    def find_matching_rule(
        filepath: str, target_dir: str, is_moving: bool
    ) -> Optional[FileInterceptorRule]:
        """First active rule in list order matching all its set conditions."""
        for rule in FileInterceptorRulesManager.rules:
            try:
                if FileInterceptorRulesManager.rule_matches(
                    rule, filepath, target_dir, is_moving
                ):
                    return rule
            except Exception as e:
                logger.error(f"Error evaluating interceptor rule {rule.name!r}: {e}")
        return None

    # ------------------------------------------------------------------
    # Entry points used by the transfer loop
    # ------------------------------------------------------------------
    @staticmethod
    def apply(
        filepath: str, target_dir: str, is_moving: bool, app_actions=None
    ) -> InterceptResult:
        """Evaluate the rule list against one file about to be transferred.

        Warns the user directly for a BLOCK match. A transform that fails is
        logged and downgraded to "no interception" so the untransformed file
        still transfers -- losing the transform is recoverable, silently
        dropping the file from the transfer is not.
        """
        rule = FileInterceptorRulesManager.find_matching_rule(
            filepath, target_dir, is_moving
        )
        if rule is None:
            return InterceptResult()

        if rule.behavior == InterceptorBehavior.BLOCK:
            message = rule.block_message.strip() or _(
                "Blocked by interceptor rule: {0}"
            ).format(rule.name)
            logger.warning(
                f"Interceptor rule {rule.name!r} blocked transfer of {filepath}"
            )
            if app_actions is not None:
                app_actions.warn(
                    _("{0}\nBlocked: {1}").format(message, os.path.basename(filepath))
                )
            return InterceptResult(
                blocked=True, block_message=message, rule_name=rule.name
            )

        if not rule.is_transform():
            return InterceptResult()

        try:
            transformed = FileInterceptorRulesManager.run_transform(rule, filepath)
        except Exception as e:
            logger.error(
                f"Interceptor rule {rule.name!r} transform failed for {filepath}: {e}"
            )
            return InterceptResult()
        if not transformed:
            return InterceptResult()

        return InterceptResult(
            transformed_source=transformed,
            rule_name=rule.name,
            # Only a move can delete the original: on copy the source must survive.
            delete_original=bool(is_moving and rule.delete_original_after_transform),
        )

    @staticmethod
    def cleanup_after_move(
        result: InterceptResult, filepath: str, app_actions=None
    ) -> None:
        """Delete the untransformed original after its transformed file transferred.

        The rendered-frame cache is evicted first: deleting the file does not
        clear it, so an SVG/PDF/HTML original would otherwise leave an entry
        pointing at a path that no longer exists.
        """
        if not result.delete_original or not result.transformed_source:
            return
        try:
            from image.frame_cache import FrameCache

            if FrameCache.get_cached_path(filepath):
                FrameCache.remove_from_cache(filepath, delete_temp_file=True)
        except Exception as e:
            logger.warning(f"Failed to evict render cache for {filepath}: {e}")
        try:
            if app_actions is not None:
                app_actions.delete(filepath, toast=False, manual_delete=False)
            elif os.path.isfile(filepath):
                os.remove(filepath)
            logger.info(
                f"Interceptor rule {result.rule_name!r} removed original after "
                f"transformed transfer: {filepath}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to remove original after interceptor transform: {filepath} - {e}"
            )

    # ------------------------------------------------------------------
    # Transform execution
    # ------------------------------------------------------------------
    @staticmethod
    def run_transform(rule: FileInterceptorRule, filepath: str) -> Optional[str]:
        """Run the rule's op and return the new file, or None when nothing was produced.

        Returning None (rather than raising) covers the legitimate no-op cases --
        a JPG that is already a JPG, a media type the op cannot read -- and
        leaves the caller transferring the original unchanged.
        """
        if rule.transform_op == InterceptorTransformOp.CONVERT_TO_JPG:
            return FileInterceptorRulesManager._convert_to_jpg(filepath)
        if rule.transform_op == InterceptorTransformOp.ENHANCE:
            return FileInterceptorRulesManager._enhance(filepath)
        return None

    @staticmethod
    def _unique_jpg_sibling(source_path: str) -> str:
        """Collision-safe ``<stem>.jpg`` next to *source_path*."""
        jpg_target = os.path.splitext(os.path.abspath(source_path))[0] + ".jpg"
        return Utils.unique_sibling_path(jpg_target, "")

    @staticmethod
    def _convert_to_jpg(filepath: str) -> Optional[str]:
        """Write a JPG next to *filepath* and return it.

        SVG/HTML/PDF cannot be read as rasters directly, so the already-rendered
        frame is converted instead and the result written beside the real source
        rather than beside the temp render. Video is not convertible here.
        An already-JPG source is a no-op: the rule exists to produce a JPG, and
        one already exists.
        """
        media_type = get_media_type_for_path(filepath)
        if media_type == CompareMediaType.VIDEO:
            logger.info(f"Skipping JPG conversion for video: {filepath}")
            return None
        if os.path.splitext(filepath)[1].lower() in (".jpg", ".jpeg"):
            logger.info(f"Skipping JPG conversion, already a JPG: {filepath}")
            return None

        source_for_raster = filepath
        output_base = filepath
        if media_type in (CompareMediaType.SVG, CompareMediaType.HTML, CompareMediaType.PDF):
            true_source_path, frame_path = resolve_rendered_frame_source(filepath)
            if true_source_path is None or frame_path is None:
                logger.warning(f"Could not render a raster frame to convert: {filepath}")
                return None
            source_for_raster = frame_path
            output_base = true_source_path

        output_path = FileInterceptorRulesManager._unique_jpg_sibling(output_base)
        result = ImageOps.convert_to_jpg(source_for_raster, output_path=output_path)
        if not result or os.path.abspath(result) == os.path.abspath(filepath):
            return None
        return result

    @staticmethod
    def _enhance(filepath: str) -> Optional[str]:
        """Write an improved copy next to *filepath* and return it.

        The output path is uniquified rather than left to the default ``_b``
        sibling, which raises when that name is already taken -- a rule that
        runs repeatedly over the same directory would otherwise fail on every
        file it had already enhanced once.
        """
        media_type = get_media_type_for_path(filepath)
        if media_type not in (CompareMediaType.IMAGE, CompareMediaType.GIF):
            logger.info(f"Skipping enhance for unsupported media type: {filepath}")
            return None
        output_path = Utils.unique_sibling_path(filepath, "_b")
        return ImageOps.enhance_image(filepath, output_path=output_path)
