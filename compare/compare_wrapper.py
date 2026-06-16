from copy import deepcopy
import os
import pprint
from typing import Optional

from compare.compare_args import CompareArgs
from compare.compare_result import CompareResult
from compare.compare_colors import CompareColors
from compare.compare_embeddings_align import CompareEmbeddingAlign
from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.compare_embeddings_flava import CompareEmbeddingFlava
from compare.compare_embeddings_siglip import CompareEmbeddingSiglip
from compare.compare_embeddings_xvlm import CompareEmbeddingXVLM
from compare.compare_embeddings_laion import CompareEmbeddingLaion
from compare.compare_embeddings_eva_clip import CompareEmbeddingEvaClip
from compare.compare_embeddings_metaclip import CompareEmbeddingMetaClip
from compare.compare_embeddings_vjepa2 import CompareEmbeddingVJepa2
from compare.compare_embeddings_face import CompareEmbeddingFace
from compare.compare_prompts import ComparePrompts
from compare.compare_prompts_exact import ComparePromptsExact
from compare.compare_size import CompareSize
from compare.compare_models import CompareModels
from compare.classifier_actions_manager import ClassifierActionsManager
from files.marked_files import MarkedFiles
from utils.audio_media import is_audio_for_display
from utils.media_utils import get_image_dimensions
from utils.config import config
from utils.constants import Mode, CompareMode, Direction, ClassifierActionType, Sort
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils
logger = get_logger("compare_wrapper")


class CompareWrapper:
    def __init__(self, master, compare_mode, app_actions):
        self._master = master
        self._compare = None
        self.compare_mode = compare_mode
        self._app_actions = app_actions

        self.files_grouped = {}
        self.file_groups = {}
        self.files_matched = []
        self.search_media_path = None
        self.negative_search_media_path = None
        self.has_media_matches = False
        self.current_group = None
        self.current_group_index = 0
        self.current_supergroup_index = 0
        self.match_index = 0
        self.group_indexes = []
        self.max_group_index = 0
        self.hidden_media = []
        self.label_suffix = ""  # appended to every group label (e.g. " (composite)")

    def clear_compare(self):
        self._compare = None

    def share_data_from(self, other: 'CompareWrapper') -> None:
        """Reuse the already-loaded compare instance from a same-mode wrapper.

        Sets self._compare to other's compare object so run() will skip the
        data-loading phase when args are otherwise compatible, avoiding
        redundant disk I/O and GPU inference. Only call when both wrappers
        have the same compare_mode and the same base_dir.
        """
        if other._compare is not None:
            self._compare = other._compare

    def has_compare(self):
        return self._compare is not None

    def cancel(self):
        """Cancel any running compare operation."""
        if self._compare is not None:
            self._compare.cancel()

    def get_args(self):
        if self.has_compare():
            return self._compare.args.clone()
        return CompareArgs()

    def compare(self):
        if self._compare is None:
            raise Exception("No compare object created")
        return self._compare

    def toggle_search_only_return_closest(self):
        config.search_only_return_closest = not config.search_only_return_closest

    def validate_compare_mode(self, required_compare_mode, error_text):
        if type(required_compare_mode) == list:
            if self.compare_mode not in required_compare_mode:
                self._app_actions.alert(_("Invalid mode"), error_text, kind="warning")
                raise Exception(f"Invalid mode: {self.compare_mode}")
        elif required_compare_mode != self.compare_mode:
            self._app_actions.alert(_("Invalid mode"), error_text, kind="warning")
            raise Exception(f"Invalid mode: {self.compare_mode}")

    def current_match(self):
        return self.files_matched[self.match_index]

    def actual_group_index(self):
        return self.group_indexes[self.current_group_index]

    def _get_prev_media(self):
        if self.match_index > 0:
            self.match_index -= 1
        else:
            self.match_index = len(self.files_matched) - 1

        return self.current_match()

    def _get_next_media(self):
        if len(self.files_matched) > self.match_index + 1:
            self.match_index += 1
        else:
            self.match_index = 0

        return self.current_match()

    def show_prev_media(self, show_alert=True):
        if self.files_matched is None:
            return False
        elif len(self.files_matched) == 0:
            if show_alert:
                self._app_actions.alert(_("Search required"), _("No matches found. Search again to find potential matches."))
            return False

        self._app_actions._set_toggled_view_matches()
        prev_media = self._get_prev_media()
        start_media = prev_media
        # Skip media that should be skipped, but break if we've gone through all media
        while self.skip_media(prev_media):
            prev_media = self._get_prev_media()
            if prev_media == start_media:
                # We've gone through all media and they all need to be skipped (TODO: show an alert)
                break
        self._master.update()
        self._app_actions.create_media(prev_media)
        return True

    def show_next_media(self, show_alert=True):
        if self.files_matched is None:
            return False
        elif len(self.files_matched) == 0:
            if show_alert:
                self._app_actions.alert(_("Search required"), _("No matches found. Search again to find potential matches."))
            return False

        self._app_actions._set_toggled_view_matches()
        next_file = self._get_next_media()
        start_media = next_file
        # Skip media that should be skipped, but break if we've gone through all media
        while self.skip_media(next_file):
            next_file = self._get_next_media()
            if next_file == start_media:
                # We've gone through all media and they all need to be skipped (TODO: show an alert)
                break
        self._master.update()
        self._app_actions.create_media(next_file)
        return True

    def skip_media(self, media_path):
        if media_path in self.hidden_media:
            return True
        if config.enable_prevalidations:
            if is_audio_for_display(media_path):
                return False
            if ClassifierActionsManager.is_dynamic_prevalidation_media(media_path):
                prevalidation_action = self._run_dynamic_prevalidation_with_spinner(media_path)
            else:
                prevalidation_action = ClassifierActionsManager.prevalidate_media(
                    media_path,
                    self._app_actions.get_base_dir,
                    self._app_actions.prevalidation_callbacks_with_mark,
                )
            if prevalidation_action is not None:
                if prevalidation_action == ClassifierActionType.BLUR:
                    self._app_actions.request_media_blur(media_path)
                return prevalidation_action not in (
                    ClassifierActionType.NOTIFY,
                    ClassifierActionType.BLUR,
                    ClassifierActionType.GENERATE,
                )
        return False

    def _run_dynamic_prevalidation_with_spinner(self, media_path):
        """
        Run prevalidation for dynamic media (video/GIF/PDF) on a worker QThread so
        the main-thread event loop keeps processing — allowing the spinner badge
        to animate during the (potentially expensive) frame-sampling loop.

        `hide_current_media` is already wrapped with BlockingQueuedConnection via
        ts(), so it is safe to call from the worker thread while the main-thread
        event loop is running.  `title_notify` uses Qt signals and is likewise
        thread-safe.  `add_mark_if_not_present` writes shared state and must run
        on the main thread; collected paths are applied after the thread joins.
        """
        from PySide6.QtCore import QEventLoop, QThread

        result = [None]
        deferred_marks = []

        def _collect_mark(path):
            deferred_marks.append(path)

        app_actions = self._app_actions
        callbacks = self._app_actions.make_prevalidation_callbacks(_collect_mark)

        class _Worker(QThread):
            def run(self_inner):
                result[0] = ClassifierActionsManager.prevalidate_media(
                    media_path,
                    app_actions.get_base_dir,
                    callbacks,
                )

        loop = QEventLoop()
        worker = _Worker()
        worker.finished.connect(loop.quit)
        app_actions.start_loading_spinner(force=True)
        worker.start()
        loop.exec()
        app_actions.stop_loading_spinner()
        for path in deferred_marks:
            MarkedFiles.add_mark_if_not_present(path, app_actions=app_actions)
        return result[0]

    def find_next_unrelated_media(self, file_browser, forward=True):
        found_unrelated_media = False
        previous_media = file_browser.current_file()
        original_media = str(previous_media)
        skip_count = 0
        if previous_media is None or len(previous_media) == 0:
            return
        while not found_unrelated_media:
            next_media = file_browser.next_file() if forward else file_browser.previous_file()
            if (self.compare_mode == CompareMode.COLOR_MATCHING and not CompareColors.is_related(previous_media, next_media)) or \
                    (self.compare_mode != CompareMode.COLOR_MATCHING and not CompareEmbeddingClip.is_related(previous_media, next_media)):
                found_unrelated_media = True
                self._app_actions.create_media(next_media)
                self._app_actions.toast(_("Skipped {0} media.").format(skip_count))
                return
            skip_count += 1
            previous_media = str(next_media)
            if original_media == previous_media:
                # Looped around and couldn't find unrelated media
                self._app_actions.alert(_("No Unrelated Media"), _("No unrelated media found."))
                break

    def show_prev_group(self, event=None, file_browser=None) -> None:
        '''
        While in group mode, navigate to the previous group.
        '''
        if file_browser:
            self.find_next_unrelated_media(file_browser, forward=False)
            return
        if (self.file_groups is None or len(self.group_indexes) == 0):
            self.current_group_index = 0
        elif self.current_group_index == 0:
            self.current_group_index = len(self.group_indexes) - 1
        else:
            self.current_group_index -= 1
        self.set_current_group()

    def show_next_group(self, event=None, file_browser=None) -> None:
        '''
        While in group mode, navigate to the next group.
        '''
        if file_browser:
            self.find_next_unrelated_media(file_browser, forward=True)
            return
        if (self.file_groups is None or len(self.group_indexes) == 0
                or self.current_group_index + 1 == len(self.group_indexes)):
            self.current_group_index = 0
        else:
            self.current_group_index += 1
        self.set_current_group()

    def _get_supergroups(self) -> list:
        '''
        Partition of group_index values into supergroups -- clusters of
        related groups based on group mean-embedding similarity, computed by
        BaseCompareEmbedding.compute_supergroups() and stored on
        compare_result.supergroups (list of lists of group_index). Returns []
        when unavailable: no compare instance yet, compare mode has no
        per-file embedding to average, or fewer than 2 groups survived
        clustering. getattr default also covers a CompareResult unpickled
        from before this feature existed, which won't have the attribute.
        '''
        if self._compare is None:
            return []
        compare_result = getattr(self._compare, "compare_result", None)
        if compare_result is None:
            return []
        return getattr(compare_result, "supergroups", [])

    def show_prev_supergroup(self, event=None) -> None:
        '''While in group mode, navigate to the previous supergroup.'''
        self._show_adjacent_supergroup(forward=False)

    def show_next_supergroup(self, event=None) -> None:
        '''While in group mode, navigate to the next supergroup.'''
        self._show_adjacent_supergroup(forward=True)

    def _show_adjacent_supergroup(self, forward: bool) -> None:
        supergroups = self._get_supergroups()
        if not supergroups:
            self._app_actions.toast(_("No Supergroups Found"))
            return
        step = 1 if forward else -1
        self.current_supergroup_index = (self.current_supergroup_index + step) % len(supergroups)
        # Defensive: supergroups are computed once by compute_supergroups() and not
        # re-clustered after later group-mutating operations (random purge, single-file
        # removal, composite-filter rebuild), so a member group_index may no longer exist
        # in group_indexes -- skip any that don't, and pick whichever surviving candidate
        # sorts first per the existing within-group ordering.
        candidates = [g for g in supergroups[self.current_supergroup_index] if g in self.group_indexes]
        if not candidates:
            self._app_actions.toast(_("Supergroup no longer available"))
            return
        target_group = min(candidates, key=self.group_indexes.index)
        self.current_group_index = self.group_indexes.index(target_group)
        self.set_current_group()

    def random_purge_groups(self, event=None) -> None:
        """Delete all but one randomly-chosen file from every similarity group.

        Presents a confirmation dialog showing the number of files that will be
        deleted.  On confirm, iterates every group, picks one survivor at random,
        and deletes the rest via app_actions.delete.  Resets group state and
        returns to browse mode when complete.
        """
        import random
        from PySide6.QtWidgets import QApplication

        if not self.file_groups:
            self._app_actions.warn(_("No groups available to purge."))
            return

        group_count = len(self.file_groups)
        files_in_groups = sum(len(g) for g in self.file_groups.values())
        files_to_delete = files_in_groups - group_count

        if files_to_delete <= 0:
            self._app_actions.warn(_("All groups already have only one file."))
            return

        ok = self._app_actions.alert(
            _("Random Purge"),
            _(
                "This will permanently delete {delete_count} of {total_count} files "
                "across {group_count} groups, keeping one random file per group.\n\n"
                "This action cannot be undone. Continue?"
            ).format(
                delete_count=files_to_delete,
                total_count=files_in_groups,
                group_count=group_count,
            ),
            kind="askokcancel",
        )
        if not ok:
            return

        # Snapshot before the loop so live mutations to file_groups don't
        # corrupt iteration.
        groups_snapshot = {
            idx: list(group.keys()) for idx, group in self.file_groups.items()
        }

        deleted = 0
        errors = 0

        self._app_actions.release_media_canvas()
        self._app_actions.start_loading_spinner(force=True)
        self._app_actions.start_progress_bar()
        try:
            for group_num, (group_index, filepaths) in enumerate(
                groups_snapshot.items(), start=1
            ):
                if len(filepaths) <= 1:
                    continue
                qualified = [
                    p for p in filepaths
                    if (dims := get_image_dimensions(p)) is None
                    or (dims[0] >= 120 and dims[1] >= 120)
                ]
                keeper = random.choice(qualified or filepaths)
                for filepath in filepaths:
                    if filepath == keeper:
                        continue
                    try:
                        MarkedFiles.handle_file_removal(filepath)
                        self._app_actions.delete(
                            filepath, toast=False, manual_delete=False
                        )
                        deleted += 1
                    except Exception as exc:
                        errors += 1
                        logger.error(
                            "Random purge: failed to delete %s: %s", filepath, exc
                        )

                self._app_actions._set_label_state(
                    _("Purging: group {current} / {total}").format(
                        current=group_num, total=group_count
                    )
                )
                QApplication.processEvents()
        finally:
            self._app_actions.stop_loading_spinner()
            self._app_actions.stop_progress_bar()

        # Reset group state — same fields cleared by _update_groups_for_removed_file
        # when the last group is removed.
        self.file_groups = {}
        self.files_grouped = {}
        self.group_indexes = []
        self.files_matched = []
        self.match_index = 0
        self.current_group_index = 0
        self.current_supergroup_index = 0
        self.has_media_matches = False

        compare_result = getattr(self._compare, "compare_result", None) if self._compare else None
        if compare_result is not None:
            compare_result.clear_supergroups()

        self._remove_stored_result()
        self._app_actions.set_mode(Mode.BROWSE)
        self._app_actions._set_label_state(_("Set a directory to run comparison."))
        self._app_actions.refresh()

        if errors:
            msg = _(
                "Random purge complete: {deleted} files deleted, {errors} error(s)."
            ).format(deleted=deleted, errors=errors)
        else:
            msg = _(
                "Random purge complete: {deleted} files deleted across {groups} groups."
            ).format(deleted=deleted, groups=group_count)
        self._app_actions.toast(msg, time_in_seconds=8)

    def set_current_group(self, start_match_index=0) -> None:
        '''
        While in group mode, navigate between the groups.
        '''
        if self.file_groups is None or len(self.file_groups) == 0:
            self._app_actions.toast(_("No Groups Found"))
            return

        actual_group_index = self.actual_group_index()
        self.current_group = self.file_groups[actual_group_index]
        self.match_index = start_match_index
        self.files_matched = []

        for f in sorted(self.current_group, key=lambda f: self.current_group[f]):
            self.files_matched.append(f)

        self._app_actions._set_label_state(group_number=self.current_group_index, size=len(self.files_matched),
                                            suffix=self.label_suffix + self._supergroup_label_suffix(actual_group_index))
        self._master.update()
        self._app_actions.create_media(self.current_match())

    def _supergroup_label_suffix(self, actual_group_index: int) -> str:
        '''
        " | Supergroup X/Y" when actual_group_index belongs to a known
        supergroup, else "". Looked up by membership rather than
        current_supergroup_index, since plain group navigation (Shift+Left/Right)
        can move into a different supergroup without updating that cursor.
        '''
        supergroups = self._get_supergroups()
        for i, cluster in enumerate(supergroups):
            if actual_group_index in cluster:
                return _(" | Supergroup {0}/{1}").format(i + 1, len(supergroups))
        return ""

    def page_down(self, half_length=False):
        paging_length = self._get_paging_length(half_length=half_length)
        test_cursor = self.match_index + paging_length
        if test_cursor >= len(self.files_matched):
            test_cursor = 0
        self.match_index = test_cursor
        return self.current_match()

    def page_up(self, half_length=False):
        paging_length = self._get_paging_length(half_length=half_length)
        test_cursor = self.match_index - paging_length
        if test_cursor < 0:
            test_cursor = -1
        self.match_index = test_cursor
        return self.current_match()

    def _get_paging_length(self, half_length=False):
        divisor = 20 if half_length else 10
        paging_length = int(len(self.files_matched) / divisor)
        if paging_length > 200:
            return 200
        if paging_length == 0:
            return 1
        return paging_length

    def select_series(self, start_media, end_file):
        if start_media not in self.files_matched:
            raise Exception('Start file not in list of matches')
        if end_file not in self.files_matched:
            raise Exception('End file not in list of matches')
        start_index = self.files_matched.index(start_media)
        end_index = self.files_matched.index(end_file)
        if start_index > end_index:
            selected = self.files_matched[end_index:start_index+1]
        else:
            selected = self.files_matched[start_index:end_index+1]
        return selected

    def _requires_new_compare(self, base_dir, is_group=False):
        if self._compare is None:
            return True
        if self._compare.base_dir != base_dir:
            return True
        if self.compare_mode != self._compare.COMPARE_MODE:
            return True
        if is_group:
            result = getattr(self._compare, "compare_result", None)
            applied = getattr(result, "applied_group_sort", None)
            if applied is not None and applied != config.compare_group_sort:
                return True
        return False

    def run(self, args=CompareArgs()):
        get_new_data = True
        self.current_group_index = 0
        self.current_group = None
        self.max_group_index = 0
        self.group_indexes = []
        self.files_matched = []
        self.match_index = 0
        self.search_media_path = args.search_media_path

        if self._requires_new_compare(args.base_dir, is_group=args.not_searching()):
            self._app_actions._set_label_state(Utils._wrap_text_to_fit_length(
                _("Gathering media data... setup may take a while depending on number of files involved."), 30))
            self.new_compare(args)
        else:
            assert self._compare is not None
            get_new_data = self._compare.args._is_new_data_request_required(args)
            self._compare.args = args
            self._compare.sync_search_state()
            self._compare.set_similarity_threshold(args.threshold)
            self._compare.print_settings()

        if self._compare is None:
            raise Exception("No compare object created")
        
        if not self._compare.is_runnable():
            raise Exception(f"Compare object of type {type(self._compare)} is not runnable, please see log and validate configuration.")

        if self._compare.is_run_search:
            self._app_actions.set_mode(Mode.SEARCH, do_update=False)
            self._app_actions._set_toggled_view_matches()
        else:
            if args.mode == Mode.SEARCH:
                res = self._app_actions.alert(_("Confirm group run"),
                                 _("Search mode detected, please confirm switch to group mode before run. Group mode will take longer as all media in the base directory are compared."),
                                 kind="askokcancel")
                if not res:
                    return
            self._app_actions.set_mode(Mode.GROUP, do_update=False)

        if get_new_data:
            self._app_actions.toast(_("Gathering media data for comparison"))
            self._compare.get_files()
            self._compare.get_data()

        if args.not_searching():
            self.run_group(args)
        else:
            self.run_search()

    def new_compare(self, args):
        args.compare_mode = self.compare_mode
        if self.compare_mode == CompareMode.CLIP_EMBEDDING:
            self._compare = CompareEmbeddingClip(args)
        elif self.compare_mode == CompareMode.COLOR_MATCHING:
            self._compare = CompareColors(args, use_thumb=True)
        elif self.compare_mode == CompareMode.SIGLIP_EMBEDDING:
            self._compare = CompareEmbeddingSiglip(args)
        elif self.compare_mode == CompareMode.FLAVA_EMBEDDING:
            self._compare = CompareEmbeddingFlava(args)
        elif self.compare_mode == CompareMode.ALIGN_EMBEDDING:
            self._compare = CompareEmbeddingAlign(args)
        elif self.compare_mode == CompareMode.XVLM_EMBEDDING:
            self._compare = CompareEmbeddingXVLM(args)
        elif self.compare_mode == CompareMode.LAION_EMBEDDING:
            self._compare = CompareEmbeddingLaion(args)
        elif self.compare_mode == CompareMode.EVA_CLIP_EMBEDDING:
            self._compare = CompareEmbeddingEvaClip(args)
        elif self.compare_mode == CompareMode.METACLIP_EMBEDDING:
            self._compare = CompareEmbeddingMetaClip(args)
        elif self.compare_mode == CompareMode.VJEPA2_EMBEDDING:
            self._compare = CompareEmbeddingVJepa2(args)
        elif self.compare_mode == CompareMode.FACE_EMBEDDING:
            self._compare = CompareEmbeddingFace(args)
        elif self.compare_mode == CompareMode.PROMPTS:
            self._compare = ComparePrompts(args)
        elif self.compare_mode == CompareMode.PROMPTS_EXACT:
            self._compare = ComparePromptsExact(args)
        elif self.compare_mode == CompareMode.SIZE:
            self._compare = CompareSize(args)
        elif self.compare_mode == CompareMode.MODELS:
            self._compare = CompareModels(args)
        else:
            raise Exception(f"Unhandled compare mode: {self.compare_mode}")

    def run_search(self) -> None:
        assert self._compare is not None
        self._app_actions._set_label_state(Utils._wrap_text_to_fit_length(_("Searching media..."), 30))
        self.files_grouped = self._compare.run_search()
        self.file_groups = deepcopy(self.files_grouped)

        if len(self.files_grouped[0]) == 0:
            self.has_media_matches = False
            self._app_actions._set_label_state(_("Set a directory and search file or search text."))
            self._app_actions.alert(_("No Match Found"), _("None of the files match the search filters with current settings."))
            self.group_indexes = []
            self._app_actions.refresh_masonry()
            return

        reverse = self.compare_mode.is_embedding()
        for f in sorted(self.files_grouped[0], key=lambda f: self.files_grouped[0][f], reverse=reverse):
            self.files_matched.append(f)

        self.group_indexes = [0]
        self.current_group_index = 0
        self.max_group_index = 0
        self.match_index = 0
        self.has_media_matches = True
        self._app_actions._set_label_state(Utils._wrap_text_to_fit_length(
            _("{0} possibly related media found.").format(len(self.files_matched)), 30))

        self._app_actions._add_buttons_for_mode()
        self._app_actions.create_media(self.files_matched[self.match_index])
        self._app_actions.refresh_masonry()

    def run_group(self, args=CompareArgs()) -> None:
        assert self._compare is not None
        self._app_actions._set_label_state(Utils._wrap_text_to_fit_length(
            _("Running media comparisons..."), 30))
        self.files_grouped, self.file_groups = self._compare.run(store_checkpoints=args.store_checkpoints)

        if len(self.files_grouped) == 0:
            self.has_media_matches = False
            self._app_actions._set_label_state(_("Set a directory and search file."))
            self._app_actions.alert(_("No Groups Found"), _("None of the files can be grouped with current settings."))
            self.group_indexes = []
            self._app_actions.refresh_masonry()
            return

        self.group_indexes = self._compare.compare_result.build_sorted_group_indexes(
            self.file_groups, reverse=(config.compare_group_sort == Sort.DESC)
        )
        self._compare.compare_result.applied_group_sort = config.compare_group_sort
        self.max_group_index = max(self.file_groups.keys())
        self._app_actions._add_buttons_for_mode()
        self.current_group_index = 0

        if args.find_duplicates:
            self.file_groups = {}
            self.group_indexes = []
            duplicates = self._compare.get_probable_duplicates()
            if len(duplicates) == 0:
                self.has_media_matches = False
                self._app_actions._set_label_state(_("Set a directory and search file."))
                self._app_actions.alert(_("No Duplicates Found"), _("None of the files appear to be duplicates based on the current settings."))
                # group_indexes was already cleared above; refresh keeps masonry visible but empty
                self._app_actions.refresh_masonry()
                return
            self._app_actions.set_mode(Mode.DUPLICATES, do_update=True)
            logger.info("Probable duplicates:")
            pprint.pprint(duplicates, width=160)
            duplicate_group_count = 0
            for file1, file2 in duplicates:
                self.file_groups[duplicate_group_count] = {
                    file1: 0,
                    file2: 0
                }
                self.group_indexes.append(duplicate_group_count)
                duplicate_group_count += 1
            self.max_group_index = duplicate_group_count
            self.set_current_group()
            self._app_actions.refresh_masonry()
        else:
            has_found_stranded_group_members = False

            while len(self.file_groups[self.actual_group_index()]) == 1:
                has_found_stranded_group_members = True
                self.current_group_index += 1

            self.set_current_group()
            self._app_actions.refresh_masonry()
            if has_found_stranded_group_members:
                self._app_actions.alert(_("Stranded Group Members Found"), _("Some group members were left stranded by the grouping process."))

    def find_file_after_comparison(self, app_mode, search_text="", exact_match=False):
        if not search_text or search_text.strip() == "":
            return None, None
        file_group_map = self._get_file_group_map(app_mode)
        for file, group_indexes in file_group_map.items():
            if search_text == os.path.basename(file):
                return file, group_indexes
        if exact_match:
            return None, None
        search_text = search_text.lower()
        for file, group_indexes in file_group_map.items():
            if os.path.basename(file).lower().startswith(search_text):
                return file, group_indexes
        for file, group_indexes in file_group_map.items():
            if search_text in os.path.basename(file).lower():
                return file, group_indexes
        return None, None

    def _update_groups_for_removed_file(self, app_mode, group_index, match_index, set_group=True, show_next_media=None):
        '''
        After a file has been removed, delete the cached file path for it and
        remove the group if only one file remains in that group.

        NOTE: This would be more complex if there was not a guarantee groups are disjoint.
        '''
        if config.debug:
            logger.debug(f"Updating groups for removed file {match_index} in group {group_index}")
        actual_index = self.group_indexes[group_index]
        if set_group or group_index == self.current_group_index:
            files_matched = self.files_matched
            set_group = True
            if config.debug and app_mode != Mode.SEARCH:
                logger.debug("setting group")
        else:
            files_matched = []
            group = self.file_groups[actual_index]
            for f in self._get_sorted_file_matches(group, app_mode):
                files_matched.append(f)

        if len(files_matched) < 3:
            if app_mode not in (Mode.GROUP, Mode.DUPLICATES):
                return

            # remove this group as it will only have one file
            if app_mode != Mode.SEARCH:
                self.files_grouped = {
                    k: v for k, v in self.files_grouped.items() if v[0] != actual_index}
            del self.file_groups[actual_index]
            del self.group_indexes[group_index]
            compare_result = getattr(getattr(self, "_compare", None), "compare_result", None)
            if compare_result is not None:
                compare_result.prune_stale_supergroups(active_group_indexes=set(self.file_groups.keys()))
            if group_index < self.current_group_index:
                self.current_group_index -= 1

            if len(self.file_groups) == 0:
                self._app_actions.alert(_("No More Groups"),
                           _("There are no more media groups remaining for this directory and current filter settings."))
                self.current_group_index = 0
                self.current_supergroup_index = 0
                self.files_grouped = {}
                self.file_groups = {}
                self.match_index = 0
                self.files_matched = []
                self.group_indexes = []
                self._app_actions.set_mode(Mode.BROWSE)
                self._app_actions._set_label_state(_("Set a directory to run comparison."))
                self._app_actions.show_next_media()
                return
            elif group_index == len(self.file_groups):
                self.current_group_index = 0

            if set_group:
                self.set_current_group()
        else:
            filepath = files_matched[match_index]
            # logger.debug(f"Filepath from update_groups: {filepath}")
            if app_mode != Mode.SEARCH:
                self.files_grouped = {
                    k: v for k, v in self.files_grouped.items() if v[0] != actual_index}
            del files_matched[match_index]
            del self.file_groups[actual_index][filepath]

            if set_group:
                if self.match_index == len(self.files_matched):
                    self.match_index = 0
                elif self.match_index > match_index:
                    self.match_index -= 1

                if show_next_media is not None:
                    self._master.update()
                    self._app_actions.release_media_canvas()
                    media = self._get_prev_media() if show_next_media == Direction.BACKWARD else self.current_match()
                    self._app_actions.create_media(media)

    def update_compare_for_readded_file(self, readded_file):
        self._compare.readd_files([readded_file])

    def _sync_result_after_deletion(self, filepath: str) -> None:
        """Update the on-disk CompareResult checkpoint to reflect a deleted file.

        Silently no-ops when no compare object exists, no checkpoint has been
        written to disk, or any part of the update fails.
        """
        if self._compare is None:
            return
        compare_result = getattr(self._compare, "compare_result", None)
        if compare_result is None:
            return
        cache_path = CompareResult.cache_path(
            self._compare.base_dir,
            getattr(self._compare, "COMPARE_MODE", None),
        )
        if not os.path.exists(cache_path):
            return

        compare_result.file_groups = {
            idx: dict(group) for idx, group in self.file_groups.items()
        }
        try:
            compare_result._dir_files_hash.remove(filepath)
        except (ValueError, AttributeError):
            pass
        active_group_indexes = set(self.file_groups.keys())
        compare_result.files_grouped = {
            k: v for k, v in compare_result.files_grouped.items()
            if v[0] in active_group_indexes
        }
        compare_result.prune_stale_supergroups()
        try:
            compare_result.store()
        except Exception as exc:
            logger.error(
                "Failed to update compare result after deletion of %s: %s", filepath, exc
            )

    def _remove_stored_result(self) -> None:
        """Delete the on-disk CompareResult checkpoint for this compare session."""
        if self._compare is None:
            return
        cache_path = CompareResult.cache_path(
            self._compare.base_dir,
            getattr(self._compare, "COMPARE_MODE", None),
        )
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception as exc:
            logger.error("Failed to remove stored compare result: %s", exc)

    def get_grouped_filepaths(self, app_mode) -> list:
        """Return all comparison files ordered group-by-group.

        Files are ordered by group (in group_indexes order), then by similarity
        score within each group. This flat list is the correct display order for
        the masonry grid in compare mode.
        """
        return list(self._get_file_group_map(app_mode).keys())

    def get_file_group_for_filepath(self, filepath: str, app_mode) -> Optional[tuple]:
        """Return (group_display_idx, file_idx_within_group) for a filepath, or None."""
        return self._get_file_group_map(app_mode).get(filepath)

    def _get_file_group_map(self, app_mode):
        if app_mode == Mode.BROWSE:
            raise Exception("Cannot get file group map in Browse mode")
        group_map = {}
        for group_count in range(len(self.group_indexes)):
            group_index = self.group_indexes[group_count]
            group = self.file_groups[group_index]
            group_file_count = 0
            for f in self._get_sorted_file_matches(group, app_mode):
                group_map[f] = (group_count, group_file_count)
                group_file_count += 1
        return group_map

    def _get_sorted_file_matches(self, group, app_mode):
        if app_mode == Mode.SEARCH and self.compare_mode.is_embedding():
            return sorted(group, key=lambda f: group[f], reverse=True)
        else:
            return sorted(group, key=lambda f: group[f])
