"""
Unit tests for ClassifierAction/Prevalidation's classifier_domain support
(compare/classifier_action.py) -- the consolidated approach where image and
audio classifiers share one set of fields (image_classifier_name,
image_classifier_selected_categories, classification_mode) and only the
discrete per-file dispatch (which manager, which wrapper method names) differs
by domain. See the dispatch helpers: _classifier_domain_manager,
_classify_with_classifier, _predict_ranked_with_classifier,
_test_categories_with_classifier.
"""

from unittest.mock import MagicMock

import pytest

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction, ClassifierClassificationMode, Prevalidation
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


# ---------------------------------------------------------------------------
# classifier_domain field: defaults, normalization, serialization
# ---------------------------------------------------------------------------

class TestClassifierDomainField:
    def test_defaults_to_image(self):
        assert _action().classifier_domain == "image"

    def test_accepts_audio(self):
        assert _action(classifier_domain="audio").classifier_domain == "audio"

    def test_garbage_value_normalizes_to_image(self):
        assert _action(classifier_domain="video").classifier_domain == "image"
        assert _action(classifier_domain="").classifier_domain == "image"
        assert _action(classifier_domain=None).classifier_domain == "image"

    def test_to_dict_includes_classifier_domain(self):
        ca = _action(classifier_domain="audio")
        assert ca.to_dict()["classifier_domain"] == "audio"

    def test_from_dict_round_trips_audio(self):
        ca = _action(classifier_domain="audio")
        d = ca.to_dict()
        d.pop("can_run", None)
        d.pop("initialization_error", None)
        ca2 = ClassifierAction.from_dict(d)
        assert ca2.classifier_domain == "audio"

    def test_from_dict_defaults_to_image_when_key_absent(self):
        """Backward compat: every ClassifierAction/Prevalidation saved before
        classifier_domain existed must keep behaving as an image classifier."""
        ca = _action()
        d = ca.to_dict()
        d.pop("classifier_domain", None)
        d.pop("can_run", None)
        d.pop("initialization_error", None)
        ca2 = ClassifierAction.from_dict(d)
        assert ca2.classifier_domain == "image"

    def test_prevalidation_from_dict_defaults_to_image_when_key_absent(self):
        """Prevalidation.from_dict has its own duplicated backward-compat
        defaulting (doesn't delegate to ClassifierAction.from_dict) -- must be
        covered independently."""
        pv = Prevalidation(name="pv", profile_name=None)
        d = pv.to_dict()
        d.pop("classifier_domain", None)
        d.pop("can_run", None)
        d.pop("initialization_error", None)
        pv2 = Prevalidation.from_dict(d)
        assert pv2.classifier_domain == "image"


# ---------------------------------------------------------------------------
# _classifier_domain_manager dispatch
# ---------------------------------------------------------------------------

class TestClassifierDomainManagerDispatch:
    def test_image_domain_uses_image_manager(self):
        import image.image_classifier_manager as img_mgr_mod

        ca = _action(classifier_domain="image")
        assert ca._classifier_domain_manager() is img_mgr_mod.image_classifier_manager

    def test_audio_domain_uses_audio_manager(self):
        import image.audio_classifier_manager as audio_mgr_mod

        ca = _action(classifier_domain="audio")
        assert ca._classifier_domain_manager() is audio_mgr_mod.audio_classifier_manager


# ---------------------------------------------------------------------------
# Per-file dispatch helpers
# ---------------------------------------------------------------------------

class TestPerFileDispatchHelpers:
    def test_classify_dispatches_to_classify_audio(self):
        ca = _action(classifier_domain="audio")
        fake = MagicMock()
        fake.classify_audio.return_value = "explicit"
        ca.image_classifier = fake
        assert ca._classify_with_classifier("/fake/audio.mp3") == "explicit"
        fake.classify_audio.assert_called_once_with("/fake/audio.mp3")
        fake.classify_image.assert_not_called()

    def test_classify_dispatches_to_classify_image_by_default(self):
        ca = _action(classifier_domain="image")
        fake = MagicMock()
        fake.classify_image.return_value = "cat"
        ca.image_classifier = fake
        assert ca._classify_with_classifier("/fake/photo.jpg") == "cat"
        fake.classify_image.assert_called_once_with("/fake/photo.jpg")
        fake.classify_audio.assert_not_called()

    def test_predict_ranked_dispatches_by_domain(self):
        ca = _action(classifier_domain="audio")
        fake = MagicMock()
        fake.predict_audio_ranked.return_value = [("explicit", 0.9)]
        ca.image_classifier = fake
        assert ca._predict_ranked_with_classifier("/fake/audio.mp3") == [("explicit", 0.9)]
        fake.predict_image_ranked.assert_not_called()

    def test_test_categories_dispatches_by_domain(self):
        ca = _action(classifier_domain="audio")
        fake = MagicMock()
        fake.test_audio_for_categories.return_value = True
        ca.image_classifier = fake
        assert ca._test_categories_with_classifier("/fake/audio.mp3", ["explicit"]) is True
        fake.test_audio_for_categories.assert_called_once_with("/fake/audio.mp3", ["explicit"])
        fake.test_image_for_categories.assert_not_called()

    def test_helpers_return_falsy_when_no_classifier_loaded(self):
        ca = _action(classifier_domain="audio")
        assert ca.image_classifier is None
        assert ca._classify_with_classifier("/fake.mp3") is None
        assert ca._predict_ranked_with_classifier("/fake.mp3") is None
        assert ca._test_categories_with_classifier("/fake.mp3", ["x"]) is False


# ---------------------------------------------------------------------------
# End-to-end: run_on_image_path with an audio classifier match
# ---------------------------------------------------------------------------

class TestRunOnImagePathAudioClassifier:
    """Mirrors TestRunImageClassifierSweep in test_classifier_action_run_media_paths.py,
    but for classifier_domain="audio" -- same ca.image_classifier = fake_classifier
    bypass-real-loading pattern."""

    def test_matching_audio_file_triggers_action(self, tmp_path):
        audio_path = tmp_path / "clip.mp3"
        audio_path.write_bytes(b"fake")

        fake_classifier = MagicMock()
        fake_classifier.test_audio_for_categories.return_value = True
        fake_classifier.classify_audio.return_value = "explicit"

        marked = []
        callbacks = ActionCallbacks(notify_callback=_NOOP, add_mark_callback=lambda p: marked.append(p))

        ca = _action(
            use_image_classifier=True,
            classifier_domain="audio",
            image_classifier_selected_categories=["explicit"],
        )
        ca.image_classifier = fake_classifier

        ca.run_on_image_path(str(audio_path), callbacks, base_directory=str(tmp_path))
        assert marked == [str(audio_path)]

    def test_non_matching_audio_file_takes_no_action(self, tmp_path):
        audio_path = tmp_path / "clip.mp3"
        audio_path.write_bytes(b"fake")

        fake_classifier = MagicMock()
        fake_classifier.test_audio_for_categories.return_value = False

        marked = []
        callbacks = ActionCallbacks(notify_callback=_NOOP, add_mark_callback=lambda p: marked.append(p))

        ca = _action(
            use_image_classifier=True,
            classifier_domain="audio",
            image_classifier_selected_categories=["explicit"],
        )
        ca.image_classifier = fake_classifier

        ca.run_on_image_path(str(audio_path), callbacks, base_directory=str(tmp_path))
        assert marked == []
        # Only the audio-domain method should ever be probed, never the image one.
        fake_classifier.test_image_for_categories.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_model_strategy_positive_categories: domain-aware manager + errors
# ---------------------------------------------------------------------------

class TestModelStrategyPositiveCategoriesAudioDomain:
    def test_raises_with_audio_wording_when_name_unset(self):
        ca = _action(classifier_domain="audio", classification_mode=ClassifierClassificationMode.MODEL_STRATEGY)
        with pytest.raises(Exception, match="Audio classifier name must be set"):
            ca._resolve_model_strategy_positive_categories()

    def test_resolves_positive_categories_from_audio_manager(self, monkeypatch):
        import compare.classifier_action as ca_mod

        fake_cfg = MagicMock()
        fake_cfg.positive_groups = [["explicit", "suggestive"]]

        class FakeManager:
            def resolve_registered_model_name(self, name):
                return "resolved_name"

            classifier_metadata = {"resolved_name": fake_cfg}

        # classifier_action.py does `from image.audio_classifier_manager import
        # audio_classifier_manager`, which binds a separate name into its own
        # module namespace -- patching image.audio_classifier_manager itself
        # would not affect that already-bound reference, so the patch target
        # must be compare.classifier_action's own name.
        monkeypatch.setattr(ca_mod, "audio_classifier_manager", FakeManager())

        ca = _action(
            classifier_domain="audio",
            image_classifier_name="nsfw_audio_v1",
            classification_mode=ClassifierClassificationMode.MODEL_STRATEGY,
        )
        positives = ca._resolve_model_strategy_positive_categories()
        assert positives == frozenset({"explicit", "suggestive"})

    def test_cache_key_distinguishes_domain(self, monkeypatch):
        """Switching classifier_domain without re-picking a name (possible via the
        UI's domain toggle) must not serve a stale cached result from the other
        domain's manager, even if the resolved name string happens to collide."""
        import compare.classifier_action as ca_mod

        image_cfg = MagicMock()
        image_cfg.positive_groups = [["nsfw_image"]]
        audio_cfg = MagicMock()
        audio_cfg.positive_groups = [["nsfw_audio"]]

        class FakeImageManager:
            def resolve_registered_model_name(self, name):
                return "shared_name"
            classifier_metadata = {"shared_name": image_cfg}

        class FakeAudioManager:
            def resolve_registered_model_name(self, name):
                return "shared_name"
            classifier_metadata = {"shared_name": audio_cfg}

        # Patch the names as bound inside compare.classifier_action itself (see
        # the comment in test_resolves_positive_categories_from_audio_manager).
        monkeypatch.setattr(ca_mod, "image_classifier_manager", FakeImageManager())
        monkeypatch.setattr(ca_mod, "audio_classifier_manager", FakeAudioManager())

        ca = _action(
            classifier_domain="image",
            image_classifier_name="shared_name",
            classification_mode=ClassifierClassificationMode.MODEL_STRATEGY,
        )
        assert ca._resolve_model_strategy_positive_categories() == frozenset({"nsfw_image"})

        ca.classifier_domain = "audio"
        assert ca._resolve_model_strategy_positive_categories() == frozenset({"nsfw_audio"})


# ---------------------------------------------------------------------------
# _build_classifier_detail: trigger_type reflects domain
# ---------------------------------------------------------------------------

class TestBuildClassifierDetailDomain:
    def test_audio_domain_sets_audio_trigger_type(self):
        ca = _action(classifier_domain="audio")
        fake = MagicMock()
        fake.predict_audio_ranked.return_value = [("explicit", 0.9)]
        ca.image_classifier = fake
        detail = ca._build_classifier_detail("/fake/audio.mp3", "explicit")
        assert detail.trigger_type == "audio_classifier"

    def test_image_domain_sets_image_trigger_type(self):
        ca = _action(classifier_domain="image")
        fake = MagicMock()
        fake.predict_image_ranked.return_value = [("cat", 0.9)]
        ca.image_classifier = fake
        detail = ca._build_classifier_detail("/fake/photo.jpg", "cat")
        assert detail.trigger_type == "image_classifier"


# ---------------------------------------------------------------------------
# validate(): domain-aware manager lookup + error wording
# ---------------------------------------------------------------------------

class TestValidateAudioDomain:
    def test_unregistered_audio_classifier_name_raises_with_audio_wording(self, monkeypatch):
        import compare.classifier_action as ca_mod

        class FakeManager:
            def resolve_registered_model_name(self, name):
                return None
            classifier_metadata = {}

        # Patch the name as bound inside compare.classifier_action itself (see
        # the comment in TestModelStrategyPositiveCategoriesAudioDomain).
        monkeypatch.setattr(ca_mod, "audio_classifier_manager", FakeManager())

        ca = _action(
            use_image_classifier=True,
            classifier_domain="audio",
            image_classifier_name="not_registered",
        )
        with pytest.raises(Exception, match="audio classifier \"not_registered\" was not found"):
            ca.validate()

    def test_does_not_consult_image_manager_when_domain_is_audio(self, monkeypatch):
        """A name that only exists in the image registry must still fail
        validation for an audio-domain action -- the two registries are not
        interchangeable fallbacks for each other."""
        import compare.classifier_action as ca_mod

        class FakeAudioManager:
            def resolve_registered_model_name(self, name):
                return None
            classifier_metadata = {}

        class FakeImageManager:
            def resolve_registered_model_name(self, name):
                return "found_in_image_registry"
            classifier_metadata = {"found_in_image_registry": MagicMock(model_categories=["a"])}

        monkeypatch.setattr(ca_mod, "audio_classifier_manager", FakeAudioManager())
        monkeypatch.setattr(ca_mod, "image_classifier_manager", FakeImageManager())

        ca = _action(
            use_image_classifier=True,
            classifier_domain="audio",
            image_classifier_name="found_in_image_registry",
        )
        with pytest.raises(Exception, match="was not found"):
            ca.validate()

    def test_selected_categories_validated_against_audio_model_categories(self, monkeypatch):
        import compare.classifier_action as ca_mod

        class FakeManager:
            def resolve_registered_model_name(self, name):
                return "nsfw_audio_v1"
            classifier_metadata = {"nsfw_audio_v1": MagicMock(model_categories=["safe", "explicit"])}

        monkeypatch.setattr(ca_mod, "audio_classifier_manager", FakeManager())

        ca = _action(
            use_image_classifier=True,
            classifier_domain="audio",
            image_classifier_name="nsfw_audio_v1",
            image_classifier_selected_categories=["not_a_real_category"],
        )
        with pytest.raises(Exception, match="not found in the audio classifier's"):
            ca.validate()
