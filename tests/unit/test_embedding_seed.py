"""Unit tests for EmbeddingSeed (compare/embedding_seed.py) -- Phase 1 CRUD/persistence."""

import json

import numpy as np

from compare.embedding_seed import EmbeddingSeed


class TestEmbeddingSeedNormalization:
    def test_positive_vector_is_l2_normalized_on_construction(self):
        seed = EmbeddingSeed(name="test", positive=np.array([3.0, 4.0]))
        assert np.isclose(np.linalg.norm(seed.positive), 1.0)
        assert np.allclose(seed.positive, [0.6, 0.8])

    def test_zero_vector_is_left_as_is(self):
        seed = EmbeddingSeed(name="zero", positive=np.array([0.0, 0.0]))
        assert np.allclose(seed.positive, [0.0, 0.0])

    def test_negative_vector_is_also_normalized(self):
        seed = EmbeddingSeed(
            name="test", positive=np.array([1.0, 0.0]), negative=np.array([0.0, 5.0])
        )
        assert np.allclose(seed.negative, [0.0, 1.0])

    def test_none_vectors_remain_none(self):
        seed = EmbeddingSeed(name="no-vectors")
        assert seed.positive is None
        assert seed.negative is None


class TestEmbeddingSeedRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        seed = EmbeddingSeed(
            name="Warm indoor portraits",
            description="test seed",
            tags=["portrait", "warm"],
            positive=np.array([1.0, 2.0, 2.0]),
            negative=np.array([1.0, 0.0, 0.0]),
            embedding_model="CLIP_EMBEDDING",
            embedding_dim=3,
            source={"kind": "supergroup_centroid", "member_count": 12},
        )
        restored = EmbeddingSeed.from_dict(seed.to_dict())

        assert restored.id == seed.id
        assert restored.name == seed.name
        assert restored.description == seed.description
        assert restored.tags == seed.tags
        assert np.allclose(restored.positive, seed.positive)
        assert np.allclose(restored.negative, seed.negative)
        assert restored.embedding_model == seed.embedding_model
        assert restored.embedding_dim == seed.embedding_dim
        assert restored.source == seed.source
        assert restored.captured_at == seed.captured_at

    def test_round_trip_is_json_safe(self):
        seed = EmbeddingSeed(name="json-safe", positive=np.array([1.0, 1.0]))
        encoded = json.dumps(seed.to_dict())
        restored = EmbeddingSeed.from_dict(json.loads(encoded))

        assert restored.name == seed.name
        assert np.allclose(restored.positive, seed.positive)

    def test_from_dict_tolerates_missing_optional_fields(self):
        restored = EmbeddingSeed.from_dict({"name": "minimal"})

        assert restored.name == "minimal"
        assert restored.tags == []
        assert restored.positive is None
        assert restored.deprecated is False

    def test_from_dict_swallows_invalid_timestamp(self):
        restored = EmbeddingSeed.from_dict({"name": "bad-ts", "captured_at": "not-a-date"})

        assert restored.captured_at is None


class TestEmbeddingSeedModelCompatibility:
    def test_matching_model_is_compatible(self):
        seed = EmbeddingSeed(name="clip-seed", embedding_model="CLIP_EMBEDDING")
        assert seed.is_compatible_with("CLIP_EMBEDDING")

    def test_mismatched_model_is_not_compatible(self):
        seed = EmbeddingSeed(name="clip-seed", embedding_model="CLIP_EMBEDDING")
        assert not seed.is_compatible_with("FLAVA_EMBEDDING")

    def test_list_seeds_filters_by_compatible_model(self):
        EmbeddingSeed.create_seed(EmbeddingSeed(name="a", embedding_model="CLIP_EMBEDDING"))
        EmbeddingSeed.create_seed(EmbeddingSeed(name="b", embedding_model="FLAVA_EMBEDDING"))

        result = EmbeddingSeed.list_seeds(compatible_with="CLIP_EMBEDDING")

        assert [s.name for s in result] == ["a"]


class TestEmbeddingSeedCrud:
    def test_create_seed_rejects_duplicate_name(self):
        EmbeddingSeed.create_seed(EmbeddingSeed(name="dup"))

        assert EmbeddingSeed.create_seed(EmbeddingSeed(name="dup")) is False
        assert len(EmbeddingSeed.list_seeds()) == 1

    def test_update_seed_renames(self):
        seed = EmbeddingSeed(name="old-name")
        EmbeddingSeed.create_seed(seed)

        assert EmbeddingSeed.update_seed(seed.id, name="new-name") is True
        assert EmbeddingSeed.get_seed(seed.id).name == "new-name"

    def test_update_seed_rejects_rename_to_existing_name(self):
        a = EmbeddingSeed(name="a")
        b = EmbeddingSeed(name="b")
        EmbeddingSeed.create_seed(a)
        EmbeddingSeed.create_seed(b)

        assert EmbeddingSeed.update_seed(b.id, name="a") is False
        assert EmbeddingSeed.get_seed(b.id).name == "b"

    def test_update_seed_allows_no_op_rename(self):
        seed = EmbeddingSeed(name="same")
        EmbeddingSeed.create_seed(seed)

        assert EmbeddingSeed.update_seed(seed.id, name="same", description="updated") is True
        assert EmbeddingSeed.get_seed(seed.id).description == "updated"

    def test_delete_seed_removes_entry(self):
        seed = EmbeddingSeed(name="to-delete")
        EmbeddingSeed.create_seed(seed)

        assert EmbeddingSeed.delete_seed(seed.id) is True
        assert EmbeddingSeed.get_seed(seed.id) is None

    def test_list_seeds_excludes_deprecated_when_asked(self):
        EmbeddingSeed.create_seed(EmbeddingSeed(name="active"))
        EmbeddingSeed.create_seed(EmbeddingSeed(name="retired", deprecated=True))

        assert {s.name for s in EmbeddingSeed.list_seeds()} == {"active", "retired"}
        assert {s.name for s in EmbeddingSeed.list_seeds(include_deprecated=False)} == {"active"}

    def test_increment_use_updates_count_and_timestamp(self):
        seed = EmbeddingSeed(name="used")

        assert seed.use_count == 0
        assert seed.last_used_at is None
        seed.increment_use()
        assert seed.use_count == 1
        assert seed.last_used_at is not None


class TestEmbeddingSeedPersistence:
    def test_store_then_load_round_trips_through_app_info_cache(self):
        seed = EmbeddingSeed(
            name="persisted",
            positive=np.array([1.0, 0.0, 0.0]),
            embedding_model="CLIP_EMBEDDING",
        )
        EmbeddingSeed.create_seed(seed)
        EmbeddingSeed.store_seeds()

        EmbeddingSeed.seeds = []  # simulate a fresh process
        EmbeddingSeed.load_seeds()

        restored = EmbeddingSeed.get_seed_by_name("persisted")
        assert restored is not None
        assert np.allclose(restored.positive, [1.0, 0.0, 0.0])
        assert restored.embedding_model == "CLIP_EMBEDDING"
