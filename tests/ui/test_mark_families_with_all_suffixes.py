"""End-to-end tests for "Mark families with all suffixes" against real files.

Unlike the mocked controller tests in test_related_images_window.py, these
load a purpose-built directory into an AppWindow and run the real grouping,
so they pin what actually gets marked: every member of a qualifying family
(seed, unsuffixed metadata-linked derivatives, multi-hop chains, members
with suffixes outside the required set, repeats) and nothing else.

Directory layout built by the family_window fixture (required set in most
tests: edit + upscale):

  alpha  -- qualifies. Seed alpha_seed.png plus:
            alpha_seed_edit.png            required suffix
            alpha_seed_edit_2.png          repeat of a required suffix
            alpha_seed_edit_upscale.png    chained via alpha_seed_edit; carries upscale
            alpha_seed_other.png           suffix outside the required set
            ComfyUI_00001_.png             no suffix, metadata -> alpha_seed.png
            ComfyUI_00002_.png             no suffix, metadata -> ComfyUI_00001_.png
  beta   -- has edit only, never upscale: not marked.
  gamma  -- seed lives outside the directory; derivatives point straight to it
            by metadata. Families are scoped to seeds inside the directory, so
            each derivative is its own seed and none are marked, even though
            edit and upscale are both present.
  theta  -- theta_seed.png points outside by metadata, but has its own
            derivatives in the directory (edit, upscale, other, and an
            unsuffixed metadata link). The chain stops at the boundary, so
            theta_seed.png is the seed and the whole family is marked.
  zeta   -- seed is named zeta_edit; its own name must not satisfy "edit".
  epsilon.png -- unrelated singleton.
"""

import os
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin
from PySide6.QtWidgets import QApplication, QPushButton

from files.marked_files import MarkedFiles
from tests.ui.app_window_fixtures import _teardown_app_window
from ui.app_window.app_window import AppWindow
from ui.files.related_images_window_qt import RelatedImagesWindow
from utils.translations import _


ALPHA = [
    "alpha_seed.png",
    "alpha_seed_edit.png",
    "alpha_seed_edit_2.png",
    "alpha_seed_edit_upscale.png",
    "alpha_seed_other.png",
    "ComfyUI_00001_.png",
    "ComfyUI_00002_.png",
]
BETA = ["beta_seed.png", "beta_seed_edit.png", "beta_seed_other.png"]
GAMMA = ["gamma_source_edit.png", "gamma_source_upscale.png", "gamma_source_misc.png"]
THETA = [
    "theta_seed.png",
    "theta_seed_edit.png",
    "theta_seed_upscale.png",
    "theta_seed_other.png",
    "ComfyUI_00003_.png",
]
ZETA = ["zeta_edit.png", "zeta_edit_upscale.png"]
UNRELATED = ["epsilon.png"]
ALL_FILES = ALPHA + BETA + GAMMA + THETA + ZETA + UNRELATED


def _write_png(path: str, related_image: str | None = None) -> None:
    img = Image.new("RGB", (8, 8), (90, 120, 150))
    if related_image is None:
        img.save(path, format="PNG")
        return
    info = PngImagePlugin.PngInfo()
    info.add_text("related_image", related_image)
    img.save(path, format="PNG", pnginfo=info)


def _build_family_dir(root) -> SimpleNamespace:
    family_dir = root / "families"
    family_dir.mkdir()
    external_gamma = str(root / "elsewhere" / "gamma_source.png")
    external_theta = str(root / "elsewhere" / "theta_origin.png")

    def p(name: str) -> str:
        return str(family_dir / name)

    metadata = {
        "ComfyUI_00001_.png": p("alpha_seed.png"),
        "ComfyUI_00002_.png": p("ComfyUI_00001_.png"),
        **{name: external_gamma for name in GAMMA},
        "theta_seed.png": external_theta,
        "ComfyUI_00003_.png": p("theta_seed.png"),
    }
    for name in ALL_FILES:
        _write_png(p(name), metadata.get(name))
    return SimpleNamespace(dir=str(family_dir), path=p)


@pytest.fixture
def family_window(qtbot, tmp_path):
    """AppWindow loaded with the family layout described in the module docstring."""
    layout = _build_family_dir(tmp_path)
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.set_base_dir(layout.dir)
    qtbot.waitUntil(lambda: win.base_dir == layout.dir, timeout=2000)
    qtbot.waitUntil(
        lambda: len(win.file_browser.get_files()) == len(ALL_FILES), timeout=5000)
    MarkedFiles.file_marks = []
    yield win, layout
    MarkedFiles.file_marks = []
    for w in list(QApplication.topLevelWidgets()):
        if isinstance(w, RelatedImagesWindow):
            w.close()
            w.deleteLater()
    _teardown_app_window(win)


def _run(win, text: str) -> list[str]:
    win.file_marks_ctrl.mark_families_with_all_suffixes_in_dir(suffixes_text=text)
    return MarkedFiles.file_marks[:]


def _names(paths) -> set[str]:
    return {os.path.basename(p) for p in paths}


class TestMarkFamiliesWithAllSuffixesRealFiles:
    def test_marks_entire_qualifying_families_only(self, family_window):
        win, _layout = family_window
        marks = _run(win, "edit, upscale")
        assert _names(marks) == set(ALPHA + THETA)

    def test_unsuffixed_and_out_of_set_members_are_included(self, family_window):
        win, _layout = family_window
        names = _names(_run(win, "edit upscale"))
        # Seeds, metadata-only derivatives (one and two hops), out-of-set suffixes.
        for name in ("alpha_seed.png", "ComfyUI_00001_.png", "ComfyUI_00002_.png",
                     "alpha_seed_other.png", "theta_seed.png", "theta_seed_other.png",
                     "ComfyUI_00003_.png"):
            assert name in names

    def test_only_files_inside_the_directory_are_marked(self, family_window):
        win, layout = family_window
        marks = _run(win, "edit, upscale")
        assert all(os.path.dirname(p) == layout.dir for p in marks)
        assert set(marks) <= set(win.file_browser.get_files())

    def test_seed_outside_directory_forms_no_family(self, family_window):
        """gamma carries both suffixes, but only via a seed outside the directory."""
        win, _layout = family_window
        assert _names(_run(win, "edit, upscale")).isdisjoint(GAMMA)
        assert _names(_run(win, "edit")).isdisjoint(GAMMA)

    def test_chain_leaving_directory_makes_boundary_image_the_seed(self, family_window):
        """theta_seed.png points outside, so it is the seed; its own name
        cannot satisfy a suffix, but its derivatives do."""
        win, _layout = family_window
        assert set(THETA) <= _names(_run(win, "edit, upscale"))
        assert _names(_run(win, "edit, other")) == set(ALPHA + BETA + THETA)

    def test_no_duplicate_marks_and_browser_order(self, family_window):
        win, _layout = family_window
        marks = _run(win, "edit, upscale")
        assert len(marks) == len(set(marks))
        expected_order = [p for p in win.file_browser.get_files() if p in set(marks)]
        assert marks == expected_order

    def test_family_missing_one_suffix_is_untouched(self, family_window):
        win, _layout = family_window
        names = _names(_run(win, "edit, upscale"))
        assert names.isdisjoint(BETA)
        assert names.isdisjoint(UNRELATED)

    def test_seed_name_does_not_satisfy_its_own_suffix(self, family_window):
        win, _layout = family_window
        assert _names(_run(win, "edit, upscale")).isdisjoint(ZETA)
        # zeta does carry upscale on a derivative, so it qualifies for that alone.
        assert set(ZETA) <= _names(_run(win, "upscale"))

    def test_single_suffix_marks_every_family_carrying_it(self, family_window):
        win, _layout = family_window
        assert _names(_run(win, "edit")) == set(ALPHA + BETA + THETA)

    def test_input_normalisation_gives_same_result(self, family_window):
        win, _layout = family_window
        assert _names(_run(win, "_EDIT,  Upscale edit")) == set(ALPHA + THETA)

    def test_existing_marks_are_replaced(self, family_window):
        win, layout = family_window
        MarkedFiles.file_marks = [layout.path("epsilon.png")]
        names = _names(_run(win, "edit, upscale"))
        assert "epsilon.png" not in names
        assert names == set(ALPHA + THETA)

    def test_no_qualifying_family_leaves_existing_marks(self, family_window):
        win, layout = family_window
        existing = [layout.path("epsilon.png")]
        MarkedFiles.file_marks = existing[:]
        assert _run(win, "edit, nomatch") == existing

    def test_window_entry_and_button_mark_families(self, family_window, qtbot):
        win, _layout = family_window
        win.window_launcher.open_related_images_window()
        qtbot.waitUntil(lambda: any(
            isinstance(w, RelatedImagesWindow) and w.isVisible()
            for w in QApplication.topLevelWidgets()), timeout=5000)
        rw = next(w for w in QApplication.topLevelWidgets()
                  if isinstance(w, RelatedImagesWindow) and w.isVisible())
        rw._suffix_entry.setText("edit, upscale")
        btn = next(b for b in rw.findChildren(QPushButton)
                   if b.text() == _("Mark families with all suffixes"))
        btn.click()
        names = _names(MarkedFiles.file_marks)
        result_text = rw._result_label.text()
        rw.close()
        assert names == set(ALPHA + THETA)
        assert str(len(ALPHA + THETA)) in result_text
