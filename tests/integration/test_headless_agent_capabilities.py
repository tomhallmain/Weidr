"""Creating, persisting and navigating -- all without Qt.

An automated caller must be able to do what the GUI can, not only run
comparisons. Executing a pipeline is covered in test_pipeline_batch_headless.py;
what remains is the rest of the loop: build a pipeline or prevalidation, persist
it, read it back in a later session, and move a browsing cursor around.

None of this needed new production code -- ClassifierPipelines, Prevalidation
and FileBrowser are already Qt-free. These tests exist so that stays true and
so the Qt-free entry points are named somewhere executable.

The pipelines here are inactive with no nodes: nothing is evaluated, because
what is under test is the lifecycle, not classification.
"""

import os

import pytest
from PIL import Image

from compare.classifier_action import Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_pipeline import ClassifierPipeline, ClassifierPipelines
from files.file_browser import FileBrowser
from utils.config import config


def _png(path) -> None:
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path, format="PNG")


class TestNoQtInvolved:
    def test_no_qapplication_exists(self):
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is not None:
            pytest.skip(
                "a QApplication already exists -- another package's tests ran "
                "first in this process, so the premise cannot be checked here"
            )


class TestPipelineLifecycle:
    """E-1 for pipelines: build, modify, persist, read back."""

    def test_pipeline_survives_a_store_load_round_trip(self):
        pipeline = ClassifierPipeline(name="AgentBuilt", is_active=False)
        pipeline.nodes = []
        pipeline.category_map = {"apple": "_app"}
        ClassifierPipelines.add_pipeline(pipeline)
        ClassifierPipelines.store()

        # Drop the in-memory copy so the reload proves persistence, not identity.
        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()

        restored = ClassifierPipelines.get_pipeline_by_name("AgentBuilt")
        assert restored is not None
        assert restored.is_active is False
        assert restored.category_map == {"apple": "_app"}

    def test_modification_is_persisted(self):
        pipeline = ClassifierPipeline(name="Editable", is_active=False)
        pipeline.nodes = []
        ClassifierPipelines.add_pipeline(pipeline)
        ClassifierPipelines.store()

        pipeline.category_map = {"changed": "_chg"}
        ClassifierPipelines.store()

        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()
        assert ClassifierPipelines.get_pipeline_by_name("Editable").category_map == {
            "changed": "_chg"
        }

    def test_removal_is_persisted(self):
        pipeline = ClassifierPipeline(name="Doomed", is_active=False)
        pipeline.nodes = []
        ClassifierPipelines.add_pipeline(pipeline)
        ClassifierPipelines.store()

        ClassifierPipelines.remove_pipeline("Doomed")
        ClassifierPipelines.store()

        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()
        assert ClassifierPipelines.get_pipeline_by_name("Doomed") is None


class TestPrevalidationLifecycle:
    """E-1 for prevalidations."""

    def test_prevalidation_survives_a_store_load_round_trip(self):
        pv = Prevalidation(name="AgentPrevalidation", positives=["cat"])
        ClassifierActionsManager.prevalidations.append(pv)
        ClassifierActionsManager.store_prevalidations()

        # load_prevalidations() appends only what is absent (Prevalidation
        # compares by name), so clear first or the reload proves nothing.
        ClassifierActionsManager.prevalidations = []
        ClassifierActionsManager.load_prevalidations()

        names = [p.name for p in ClassifierActionsManager.prevalidations]
        assert "AgentPrevalidation" in names

    def test_prevalidation_fields_round_trip_through_dict(self):
        pv = Prevalidation(name="Fields", positives=["cat"], negatives=["dog"])
        restored = Prevalidation.from_dict(pv.to_dict())
        assert restored.name == "Fields"
        assert restored.positives == ["cat"]
        assert restored.negatives == ["dog"]


@pytest.fixture
def browser(tmp_path, monkeypatch):
    """A FileBrowser over four name-sorted PNGs: f0 f1 f2 f3."""
    monkeypatch.setattr(config, "file_types", [".png"])
    for i in range(4):
        _png(tmp_path / f"f{i}.png")
    fb = FileBrowser(str(tmp_path))
    fb.set_directory(str(tmp_path))
    return fb


class TestNavigation:
    """E-3: navigation headless runs through FileBrowser.

    The presentation port's go_to_file/show_next_media answer False without a
    screen, so this is the interface an automated caller uses instead.
    """

    def test_directory_is_listed_in_sorted_order(self, browser):
        assert [os.path.basename(p) for p in browser.get_files()] == [
            "f0.png", "f1.png", "f2.png", "f3.png"
        ]

    def test_next_file_walks_forward_from_the_start(self, browser):
        # A freshly set directory leaves the cursor at the -1 pre-position
        # sentinel, so the first next_file() lands on the first file.
        assert os.path.basename(browser.next_file()) == "f0.png"
        assert os.path.basename(browser.next_file()) == "f1.png"

    def test_next_file_wraps_at_the_end(self, browser):
        browser.last_file()
        assert os.path.basename(browser.next_file()) == "f0.png"

    def test_previous_file_walks_backwards(self, browser):
        browser.go_to_index(3)  # f2
        assert os.path.basename(browser.previous_file()) == "f1.png"

    def test_previous_file_wraps_at_the_start(self, browser):
        browser.go_to_index(1)  # f0
        assert os.path.basename(browser.previous_file()) == "f3.png"

    def test_go_to_file_moves_the_cursor(self, browser):
        target = browser.get_files()[2]
        browser.go_to_file(target)
        assert browser.current_file() == target
        assert browser.get_cursor() == 2

    def test_go_to_index_is_one_based(self, browser):
        # Not zero-based: index 1 is the first file. A caller assuming
        # zero-based would silently land one file off.
        assert os.path.basename(browser.go_to_index(1)) == "f0.png"
        assert os.path.basename(browser.go_to_index(4)) == "f3.png"

    def test_go_to_index_rejects_out_of_range(self, browser):
        with pytest.raises(ValueError):
            browser.go_to_index(5)
        with pytest.raises(ValueError):
            browser.go_to_index(0)

    def test_random_file_stays_within_the_directory(self, browser):
        assert browser.random_file() in browser.get_files()

    def test_last_file_lands_on_the_end(self, browser):
        assert os.path.basename(browser.last_file()) == "f3.png"
        assert browser.get_cursor() == 3
