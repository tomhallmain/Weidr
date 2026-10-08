"""
UI tests for object detection support in the HF Model Manager: search-tab
presets for object-detection repos, the detection install path (background
category inferred from the downloaded model's labels), the edit dialog's
detection fields and model_kwargs handling, and detections in the Test result.
"""

import json
from unittest.mock import MagicMock

from PySide6.QtWidgets import QTreeWidgetItem

import ui.compare.hf_model_manager_window_qt as hf_window
from ui.compare.hf_model_manager_window_qt import (
    HfModelManagerWindow,
    _ClassifierTestWorker,
    _format_detection_lines,
    _InstalledModelEditDialog,
)
from utils.translations import _


def _make_window(qtbot, monkeypatch, snapshot_dir=None):
    win = HfModelManagerWindow(parent=None, app_actions=MagicMock())
    qtbot.addWidget(win)
    api_mock = MagicMock()
    api_mock.list_model_files.return_value = ["config.json", "model.safetensors", "preprocessor_config.json"]
    if snapshot_dir is not None:
        api_mock.download_snapshot.return_value = str(snapshot_dir)
    monkeypatch.setattr(win, "_api", lambda: api_mock)
    persist_mock = MagicMock(return_value=True)
    monkeypatch.setattr(win, "_persist_model_details", persist_mock)
    return win, persist_mock


def _select_row(win, repo_id, task):
    win._search_tree.clearSelection()
    item = QTreeWidgetItem(win._search_tree, [repo_id, task, "0", "0", "unknown", "no"])
    item.setSelected(True)


def _snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"stub")
    (root / "config.json").write_text(json.dumps({"id2label": {"0": "person", "1": "dog"}}))
    return root


class TestSearchTabPresets:
    def test_object_detection_repo_presets_detection(self, qtbot, monkeypatch):
        win, _persist = _make_window(qtbot, monkeypatch)
        _select_row(win, "org/detector", "object-detection")
        assert win._backend_combo.currentText() == "detection"
        assert win._categories_edit.text() == "person,no person"
        assert not win._use_transformers_auto_model_cb.isEnabled()

    def test_switching_to_classification_repo_resets_detection_backend(self, qtbot, monkeypatch):
        win, _persist = _make_window(qtbot, monkeypatch)
        _select_row(win, "org/detector", "object-detection")
        _select_row(win, "org/classifier", "image-classification")
        assert win._backend_combo.currentText() == "auto"
        assert win._use_transformers_auto_model_cb.isEnabled()


class TestDetectionInstall:
    def test_background_category_inferred_from_model_labels(self, qtbot, monkeypatch, tmp_path):
        snapshot = _snapshot(tmp_path)
        win, persist_mock = _make_window(qtbot, monkeypatch, snapshot)
        _select_row(win, "org/detector", "object-detection")
        win._filename_combo.setEditText("model.safetensors")
        win._download_and_install_selected()

        persist_mock.assert_called_once()
        details = persist_mock.call_args.args[0]
        assert details["backend"] == "detection"
        assert details["model_categories"] == ["person", "no person"]
        assert details["model_kwargs"] == {
            "hf_pretrained_path": str(snapshot),
            "background_category": "no person",
        }

    def test_several_non_label_categories_are_rejected(self, qtbot, monkeypatch, tmp_path):
        win, persist_mock = _make_window(qtbot, monkeypatch, _snapshot(tmp_path))
        _select_row(win, "org/detector", "object-detection")
        win._categories_edit.setText("persn,no person")
        win._filename_combo.setEditText("model.safetensors")
        win._download_and_install_selected()

        persist_mock.assert_not_called()
        assert win._app_actions.alert.call_args.args[0] == _("Invalid Model Configuration")

    def test_transformers_auto_model_is_not_applied_to_detection(self, qtbot, monkeypatch, tmp_path):
        win, persist_mock = _make_window(qtbot, monkeypatch, _snapshot(tmp_path))
        _select_row(win, "org/detector", "object-detection")
        win._use_transformers_auto_model_cb.setChecked(True)
        win._filename_combo.setEditText("model.safetensors")
        win._download_and_install_selected()

        details = persist_mock.call_args.args[0]
        assert details["backend"] == "detection"
        assert "use_transformers_auto_model" not in details["model_kwargs"]


def _edit_dialog(qtbot, initial_model, monkeypatch):
    saved = MagicMock()
    alert_mock = MagicMock()
    monkeypatch.setattr("ui.compare.hf_model_manager_window_qt.qt_alert", alert_mock)
    dialog = _InstalledModelEditDialog(
        parent=None,
        title="edit",
        initial_model=initial_model,
        api_getter=MagicMock(),
        save_callback=saved,
        model_file_extensions=HfModelManagerWindow._MODEL_FILE_EXTENSIONS,
    )
    qtbot.addWidget(dialog)
    return dialog, saved, alert_mock


def _detection_model(model_location, **model_kwargs):
    return {
        "model_name": "detector",
        "model_location": model_location,
        "model_categories": ["animal", "no animal"],
        "backend": "detection",
        "model_kwargs": {
            "background_category": "no animal",
            "score_threshold": 0.4,
            "min_box_area_ratio": 0.02,
            "label_aliases": {"animal": ["dog", "cat"]},
            **model_kwargs,
        },
    }


class TestEditDialogDetectionFields:
    def test_fields_shown_only_for_detection_backend(self, qtbot, monkeypatch):
        dialog, _saved, _alert = _edit_dialog(qtbot, _detection_model(__file__), monkeypatch)
        assert not dialog._detection_row.isHidden()
        assert not dialog._use_transformers_cb.isEnabled()
        dialog._backend_combo.setCurrentText("onnx")
        assert dialog._detection_row.isHidden()
        assert dialog._use_transformers_cb.isEnabled()

    def test_save_keeps_detection_settings_and_unmanaged_kwargs(self, qtbot, monkeypatch):
        dialog, saved, _alert = _edit_dialog(qtbot, _detection_model(__file__), monkeypatch)
        dialog._score_threshold_spin.setValue(0.5)
        dialog._save()

        saved.assert_called_once()
        model_kwargs = saved.call_args.args[0]["model_kwargs"]
        assert model_kwargs["score_threshold"] == 0.5
        assert model_kwargs["min_box_area_ratio"] == 0.02
        assert model_kwargs["background_category"] == "no animal"
        assert model_kwargs["label_aliases"] == {"animal": ["dog", "cat"]}

    def test_background_category_must_be_a_category(self, qtbot, monkeypatch):
        dialog, saved, alert_mock = _edit_dialog(qtbot, _detection_model(__file__), monkeypatch)
        dialog._background_category_edit.setText("nothing")
        dialog._save()

        saved.assert_not_called()
        assert alert_mock.call_args.args[1] == _("Invalid background category")

    def test_switching_away_from_detection_drops_detection_kwargs(self, qtbot, monkeypatch):
        dialog, saved, _alert = _edit_dialog(qtbot, _detection_model(__file__), monkeypatch)
        dialog._backend_combo.setCurrentText("onnx")
        dialog._save()

        model_kwargs = saved.call_args.args[0].get("model_kwargs", {})
        assert "background_category" not in model_kwargs
        assert "score_threshold" not in model_kwargs
        assert "label_aliases" not in model_kwargs

    def test_unmanaged_kwargs_of_other_backends_are_kept(self, qtbot, monkeypatch):
        initial = {
            "model_name": "onnx_model",
            "model_location": __file__,
            "model_categories": ["a", "b"],
            "backend": "onnx",
            "model_kwargs": {"channels_first": False, "rescale": False},
        }
        dialog, saved, _alert = _edit_dialog(qtbot, initial, monkeypatch)
        dialog._save()

        assert saved.call_args.args[0]["model_kwargs"] == {"channels_first": False, "rescale": False}

    def test_switching_backend_drops_kwargs_the_new_backend_rejects(self, qtbot, monkeypatch):
        initial = {
            "model_name": "onnx_model",
            "model_location": __file__,
            "model_categories": ["a", "b"],
            "backend": "onnx",
            "model_kwargs": {"channels_first": False, "device": "cpu"},
        }
        dialog, saved, _alert = _edit_dialog(qtbot, initial, monkeypatch)
        dialog._backend_combo.setCurrentText("pytorch")
        dialog._save()

        assert saved.call_args.args[0]["model_kwargs"] == {"device": "cpu"}


class _FakeDetectionBackend:
    def __init__(self, last=None):
        self.last = last
        self.detect_calls = 0
        self.detect_paths = []

    def last_detections(self, image_path):
        return self.last

    def detect(self, image_path):
        self.detect_calls += 1
        self.detect_paths.append(image_path)
        return [
            {"label": "dog", "label_id": 2, "score": 0.6, "box": (0, 0, 1, 1), "area_ratio": 0.1},
            {"label": "person", "label_id": 0, "score": 0.9, "box": (0, 0, 1, 1), "area_ratio": 0.25},
        ]

    def categories_for_label_id(self, label_id):
        return ["person"] if label_id == 0 else []


class _FakeWrapper:
    can_run = True

    def __init__(self, backend):
        self.classifier = backend
        self.paths = []

    @staticmethod
    def input_image_path(media_path):
        return media_path.replace(".svg", "_render.png")

    def discard_cached_prediction(self, path):
        self.paths.append(path)

    def predict_image_ranked(self, path):
        self.paths.append(path)
        return [("person", 0.9), ("no person", 0.1)]

    def classify_image(self, path):
        self.paths.append(path)
        return "person"


def _run_test_worker(monkeypatch, backend, media_path="img.png", wrappers=None):
    def get_classifier(name):
        wrapper = _FakeWrapper(backend)
        if wrappers is not None:
            wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(hf_window.image_classifier_manager, "get_classifier", get_classifier)
    worker = _ClassifierTestWorker("detector", media_path)
    results = []
    worker.finished.connect(lambda name, path, result: results.append(result))
    worker.run()
    return results[0]


class TestClassifierTestDetections:
    def test_worker_adds_detections_with_their_categories(self, qtbot, monkeypatch):
        result = _run_test_worker(monkeypatch, _FakeDetectionBackend())
        assert [d["categories"] for d in result["detections"]] == [[], ["person"]]
        assert result["classification"] == "person"

    def test_worker_reuses_detections_from_the_prediction(self, qtbot, monkeypatch):
        reused = [{"label": "cat", "label_id": 3, "score": 0.7, "box": (0, 0, 1, 1), "area_ratio": 0.5}]
        backend = _FakeDetectionBackend(last=reused)
        result = _run_test_worker(monkeypatch, backend)
        assert [d["label"] for d in result["detections"]] == ["cat"]
        assert backend.detect_calls == 0

    def test_worker_classifies_and_detects_on_the_rendered_image(self, qtbot, monkeypatch):
        backend = _FakeDetectionBackend()
        wrappers = []
        _run_test_worker(monkeypatch, backend, media_path="drawing.svg", wrappers=wrappers)
        assert set(wrappers[0].paths) == {"drawing_render.png"}
        assert backend.detect_paths == ["drawing_render.png"]

    def test_worker_omits_detections_for_classifier_backends(self, qtbot, monkeypatch):
        result = _run_test_worker(monkeypatch, object())
        assert "detections" not in result

    def test_lines_sorted_by_score_with_categories(self):
        detections = [dict(d, categories=_FakeDetectionBackend().categories_for_label_id(d["label_id"]))
                      for d in _FakeDetectionBackend().detect("img.png")]
        lines = _format_detection_lines(detections)
        assert lines[0] == _("Detections (after the threshold and box-area filters):")
        assert lines[1] == "  " + _("{0}: {1:.2%} confidence, box covers {2:.1%} of the image").format(
            "person", 0.9, 0.25) + " " + _("(counts toward: {0})").format("person")
        assert lines[2] == "  " + _("{0}: {1:.2%} confidence, box covers {2:.1%} of the image").format(
            "dog", 0.6, 0.1)

    def test_no_detections_line(self):
        assert _format_detection_lines([])[1] == "  " + _("(none)")

    def test_long_lists_are_truncated(self):
        detections = [{"label": "person", "label_id": 0, "score": 0.5, "area_ratio": 0.1}] * 60
        lines = _format_detection_lines(detections)
        assert lines[-1] == "  " + _("... and {0} more").format(10)

    def test_result_dialog_includes_detections(self, qtbot, monkeypatch):
        win, _persist = _make_window(qtbot, monkeypatch)
        dialog_cls = MagicMock()
        monkeypatch.setattr(hf_window, "_TextPreviewDialog", dialog_cls)
        result = _run_test_worker(monkeypatch, _FakeDetectionBackend())
        win._on_classifier_test_finished("detector", "img.png", result)
        text = dialog_cls.call_args.kwargs["text"]
        assert _("Detections (after the threshold and box-area filters):") in text
        assert "person" in text and "dog" in text
