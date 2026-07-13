"""
Unit tests for ClassifierAction.run() outside the prototype-only case.

Regression coverage: run() previously only did anything when use_prototype
was the (implicit) validation type — any other enabled validation type
(image classifier, embedding, prompts, filename-contains, base-stem) caused
run() to load the classifier, log a line, and silently do nothing else. Now
run() requires pre-gathered media_paths for that case and sweeps them via
run_on_media_path in a background thread, patched here to run inline.
"""
from unittest.mock import MagicMock

import pytest

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction
from utils.constants import ClassifierActionType

_NOOP = lambda *a, **kw: None


def _action(**kwargs) -> ClassifierAction:
    defaults = dict(
        name="test",
        action=ClassifierActionType.ADD_MARK,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_prototype=False,
    )
    defaults.update(kwargs)
    return ClassifierAction(**defaults)


def _run_sync(monkeypatch, ca, directory_paths, media_paths=None, **run_kwargs):
    """Run ca.run(...) with the background sweep thread patched to run inline,
    so assertions can be made immediately without waiting/polling."""
    monkeypatch.setattr(
        "compare.classifier_action.start_thread",
        lambda fn, *a, **kw: fn(),
    )
    marked = []
    callbacks = ActionCallbacks(
        notify_callback=_NOOP,
        add_mark_callback=lambda p: marked.append(p),
    )
    ca.run(directory_paths, callbacks, media_paths=media_paths, **run_kwargs)
    return marked


# ---------------------------------------------------------------------------
# has_non_prototype_validation
# ---------------------------------------------------------------------------

class TestHasNonPrototypeValidation:
    def test_false_when_only_prototype(self):
        assert _action(use_prototype=True).has_non_prototype_validation() is False

    def test_false_when_nothing_enabled(self):
        assert _action().has_non_prototype_validation() is False

    def test_true_for_image_classifier(self):
        assert _action(use_image_classifier=True).has_non_prototype_validation() is True

    def test_true_for_embedding(self):
        assert _action(use_embedding=True).has_non_prototype_validation() is True

    def test_true_for_prompts(self):
        assert _action(use_prompts=True).has_non_prototype_validation() is True

    def test_true_for_filename_contains(self):
        assert _action(use_filename_contains=True).has_non_prototype_validation() is True

    def test_true_for_base_stem_match(self):
        assert _action(use_base_stem_match=True).has_non_prototype_validation() is True

    def test_true_when_prototype_plus_image_classifier_both_enabled(self):
        # Prototype AND another type enabled together must not take the
        # prototype-only fast path (which would silently ignore the other type).
        ca = _action(use_prototype=True, use_image_classifier=True)
        assert ca.has_non_prototype_validation() is True


# ---------------------------------------------------------------------------
# run() with an image-classifier-based action (the reported regression)
# ---------------------------------------------------------------------------

class TestRunImageClassifierSweep:
    def test_matching_media_path_runs_action(self, monkeypatch, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")

        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = True
        fake_classifier.classify_image.return_value = "positive"

        ca = _action(use_image_classifier=True, image_classifier_selected_categories=["positive"])
        ca.image_classifier = fake_classifier  # bypass real model loading

        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img)])
        assert marked == [str(img)]

    def test_non_matching_media_path_takes_no_action(self, monkeypatch, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")

        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = False

        ca = _action(use_image_classifier=True, image_classifier_selected_categories=["positive"])
        ca.image_classifier = fake_classifier

        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img)])
        assert marked == []

    def test_only_matching_paths_act_among_several(self, monkeypatch, tmp_path):
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"1")
        img2.write_bytes(b"2")

        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.side_effect = (
            lambda path, categories: path == str(img2)
        )
        fake_classifier.classify_image.return_value = "positive"

        ca = _action(use_image_classifier=True, image_classifier_selected_categories=["positive"])
        ca.image_classifier = fake_classifier

        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img1), str(img2)])
        assert marked == [str(img2)]

    def test_missing_media_paths_does_nothing_rather_than_silently_no_op_wrong(self, monkeypatch, tmp_path):
        """The exact reported bug: a non-prototype action given no media_paths
        must not proceed as if everything matched — it should do nothing."""
        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = True
        fake_classifier.classify_image.return_value = "positive"

        ca = _action(use_image_classifier=True, image_classifier_selected_categories=["positive"])
        ca.image_classifier = fake_classifier

        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=None)
        assert marked == []

    def test_disabled_action_does_not_run(self, monkeypatch, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = True
        fake_classifier.classify_image.return_value = "positive"

        ca = _action(
            use_image_classifier=True,
            image_classifier_selected_categories=["positive"],
            is_active=False,
        )
        ca.image_classifier = fake_classifier

        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img)])
        assert marked == []


# ---------------------------------------------------------------------------
# run() with an embedding-based (non-classifier, non-prototype) action
# ---------------------------------------------------------------------------

class TestRunEmbeddingSweep:
    def test_matching_media_path_runs_action(self, monkeypatch, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")

        monkeypatch.setattr(
            "compare.classifier_action.CompareEmbeddingClip.multi_text_compare",
            lambda image_path, positives, negatives, threshold: image_path == str(img),
        )

        ca = _action(use_embedding=True, positives=["cat"])
        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img)])
        assert marked == [str(img)]

    def test_non_matching_media_path_takes_no_action(self, monkeypatch, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")

        monkeypatch.setattr(
            "compare.classifier_action.CompareEmbeddingClip.multi_text_compare",
            lambda *a, **kw: False,
        )

        ca = _action(use_embedding=True, positives=["cat"])
        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img)])
        assert marked == []


# ---------------------------------------------------------------------------
# The prototype-only fast path must be unaffected by the media_paths change
# ---------------------------------------------------------------------------

class TestRunPrototypeOnlyUnaffected:
    def test_prototype_only_runs_via_directory_paths_without_media_paths(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            ClassifierAction,
            "_run_with_batch_prototype_validation",
            lambda self, directory_paths, callbacks, max_images_per_batch: calls.append(directory_paths),
        )
        ca = _action(use_prototype=True, prototype_directory=str(tmp_path))
        callbacks = ActionCallbacks(notify_callback=_NOOP)
        ca.run([str(tmp_path)], callbacks, media_paths=None)
        assert calls == [[str(tmp_path)]]

    def test_prototype_plus_classifier_uses_general_sweep_not_batch_path(self, monkeypatch, tmp_path):
        """Enabling prototype alongside another type must not take the
        prototype-only fast path, since that path ignores the other type."""
        batch_calls = []
        monkeypatch.setattr(
            ClassifierAction,
            "_run_with_batch_prototype_validation",
            lambda self, directory_paths, callbacks, max_images_per_batch: batch_calls.append(True),
        )
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = True
        fake_classifier.classify_image.return_value = "positive"

        ca = _action(
            use_prototype=True,
            prototype_directory=str(tmp_path),
            use_image_classifier=True,
            image_classifier_selected_categories=["positive"],
        )
        ca.image_classifier = fake_classifier
        marked = _run_sync(monkeypatch, ca, [str(tmp_path)], media_paths=[str(img)])
        assert batch_calls == []
        assert marked == [str(img)]
