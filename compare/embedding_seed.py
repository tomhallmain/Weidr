"""
EmbeddingSeed -- named, persisted embedding vectors reusable as search seeds
across compare runs and directories, independent of the source files that
produced them. See docs/embedding-seed-library.md for the design.

Phase 1 only (CRUD + persistence): capture paths (supergroup context action)
and compare-search consumption are wired elsewhere; nurturing features
(merge, suggested promotions) are deferred.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np

from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("embedding_seed")

_CACHE_KEY = "embedding_seed_library"

# Distinguishes "caller didn't pass captured_at" (default to now -- a fresh
# capture) from "caller explicitly passed captured_at=None" (from_dict, when
# the stored timestamp was missing/corrupted -- stay None, don't relabel a
# corrupted record as captured just now).
_UNSET = object()


class EmbeddingSeed:
    """
    A detached prototype embedding vector plus metadata.

    The design doc's illustrative schema nests the vectors under a
    ``vectors: {positive, negative}`` dict; here they're flattened to
    ``positive``/``negative`` attributes directly, matching this codebase's
    flat-field convention for persisted records (Lookahead, DirectoryProfile).
    """

    seeds: List['EmbeddingSeed'] = []

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        positive: Optional[np.ndarray] = None,
        negative: Optional[np.ndarray] = None,
        embedding_model: str = "",
        embedding_dim: int = 0,
        source: Optional[dict] = None,
        use_count: int = 0,
        last_used_at: Optional[datetime] = None,
        captured_at=_UNSET,
        deprecated: bool = False,
        supersedes: Optional[str] = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tags = list(tags) if tags is not None else []
        self.positive = EmbeddingSeed._normalize(positive) if positive is not None else None
        self.negative = EmbeddingSeed._normalize(negative) if negative is not None else None
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.source = source if source is not None else {}
        self.use_count = use_count
        self.last_used_at = last_used_at
        self.captured_at = datetime.now() if captured_at is _UNSET else captured_at
        self.deprecated = deprecated
        self.supersedes = supersedes

    @staticmethod
    def _normalize(vector) -> np.ndarray:
        """L2-normalize, matching EmbeddingPrototype/compute_group_centroids."""
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def is_compatible_with(self, embedding_model: str) -> bool:
        """Model/dim binding check -- seeds captured under a different
        embedding model are not valid search input for the active mode."""
        return self.embedding_model == embedding_model

    def increment_use(self) -> None:
        self.use_count += 1
        self.last_used_at = datetime.now()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "positive": self.positive.tolist() if self.positive is not None else None,
            "negative": self.negative.tolist() if self.negative is not None else None,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "source": self.source,
            "use_count": self.use_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "deprecated": self.deprecated,
            "supersedes": self.supersedes,
        }

    @staticmethod
    def from_dict(d: dict) -> 'EmbeddingSeed':
        def _parse_dt(s):
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None

        positive = d.get("positive")
        negative = d.get("negative")
        return EmbeddingSeed(
            id=d.get("id"),
            name=d.get("name", ""),
            description=d.get("description", ""),
            tags=list(d.get("tags", [])),
            positive=np.array(positive) if positive is not None else None,
            negative=np.array(negative) if negative is not None else None,
            embedding_model=d.get("embedding_model", ""),
            embedding_dim=d.get("embedding_dim", 0),
            source=d.get("source", {}),
            use_count=d.get("use_count", 0),
            last_used_at=_parse_dt(d.get("last_used_at")),
            captured_at=_parse_dt(d.get("captured_at")),
            deprecated=d.get("deprecated", False),
            supersedes=d.get("supersedes"),
        )

    # ------------------------------------------------------------------
    # Persistence (global meta key, same pattern as embedding_prototypes)
    # ------------------------------------------------------------------
    @staticmethod
    def load_seeds() -> None:
        EmbeddingSeed.seeds = [
            EmbeddingSeed.from_dict(d)
            for d in app_info_cache.get_meta(_CACHE_KEY, default_val=[])
        ]

    @staticmethod
    def store_seeds() -> None:
        app_info_cache.set_meta(_CACHE_KEY, [s.to_dict() for s in EmbeddingSeed.seeds])

    # ------------------------------------------------------------------
    # CRUD API
    # ------------------------------------------------------------------
    @staticmethod
    def create_seed(seed: 'EmbeddingSeed') -> bool:
        if EmbeddingSeed.get_seed_by_name(seed.name) is not None:
            logger.error(f"Embedding seed with name {seed.name} already exists")
            return False
        EmbeddingSeed.seeds.append(seed)
        logger.info(f"Added embedding seed: {seed.name}")
        return True

    @staticmethod
    def update_seed(
        seed_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        deprecated: Optional[bool] = None,
    ) -> bool:
        seed = EmbeddingSeed.get_seed(seed_id)
        if seed is None:
            logger.error(f"Embedding seed {seed_id} not found")
            return False
        if name is not None and name != seed.name:
            existing = EmbeddingSeed.get_seed_by_name(name)
            if existing is not None and existing.id != seed.id:
                logger.error(f"Embedding seed with name {name} already exists")
                return False
            seed.name = name
        if description is not None:
            seed.description = description
        if tags is not None:
            seed.tags = list(tags)
        if deprecated is not None:
            seed.deprecated = deprecated
        return True

    @staticmethod
    def delete_seed(seed_id: str) -> bool:
        seed = EmbeddingSeed.get_seed(seed_id)
        if seed is None:
            return False
        EmbeddingSeed.seeds.remove(seed)
        return True

    @staticmethod
    def list_seeds(
        include_deprecated: bool = True,
        compatible_with: Optional[str] = None,
    ) -> List['EmbeddingSeed']:
        result = list(EmbeddingSeed.seeds)
        if not include_deprecated:
            result = [s for s in result if not s.deprecated]
        if compatible_with is not None:
            result = [s for s in result if s.is_compatible_with(compatible_with)]
        return result

    @staticmethod
    def get_seed(seed_id: str) -> Optional['EmbeddingSeed']:
        for s in EmbeddingSeed.seeds:
            if s.id == seed_id:
                return s
        return None

    @staticmethod
    def get_seed_by_name(name: str) -> Optional['EmbeddingSeed']:
        for s in EmbeddingSeed.seeds:
            if s.name == name:
                return s
        return None
