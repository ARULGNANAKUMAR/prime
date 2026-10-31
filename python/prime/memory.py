"""
prime.memory — Public Memory Engine API for PRIME Phase 3.

Provides a structured memory subsystem with three tiers:

    Working Memory   — small, temporary, current-task context
    Short-Term Memory— recent items with automatic eviction
    Long-Term Memory — persistent items (SQLite-backed when db_path given)

Typical usage::

    from prime import PRIME

    prime = PRIME()
    mem = prime.memory()

    mid = mem.store("Python is a programming language.", memory_type="long_term",
                    importance=0.8)

    results = mem.search("Python programming")
    for item in results:
        print(item.id, item.content)

    mem.update(mid, importance=0.9)
    mem.delete(mid)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import _prime_memory


class MemoryError(Exception):
    """Raised when a PRIME memory operation fails."""


@dataclass
class MemoryItem:
    """A single memory item returned by the engine.

    This is the Python-facing counterpart to the C++ MemoryItem struct.
    All fields are read-only after creation.
    """
    id:            int
    type:          str       # "working" | "short_term" | "long_term"
    content:       str
    importance:    float
    confidence:    float
    created_at:    float
    last_accessed: float
    access_count:  int
    expires_at:    float
    metadata:      Dict[str, str] = field(default_factory=dict)

    @classmethod
    def _from_c(cls, c_item: Any) -> "MemoryItem":
        return cls(
            id            = c_item.id,
            type          = c_item.type,
            content       = c_item.content,
            importance    = c_item.importance,
            confidence    = c_item.confidence,
            created_at    = c_item.created_at,
            last_accessed = c_item.last_accessed,
            access_count  = c_item.access_count,
            expires_at    = c_item.expires_at,
            metadata      = dict(c_item.metadata),
        )

    def __repr__(self) -> str:
        snippet = self.content[:40].replace("\n", " ")
        if len(self.content) > 40:
            snippet += "…"
        return (f"<MemoryItem id={self.id} type={self.type!r} "
                f"importance={self.importance:.2f} content={snippet!r}>")


@dataclass
class MemoryConfig:
    """Configuration for the PRIME Memory Engine.

    Attributes:
        db_path: Path to the SQLite database for long-term persistence.
            Empty string (default) uses in-memory storage only.
        max_working: Maximum working memory items (0 = unlimited).
        max_short_term: Maximum short-term memory items (0 = unlimited).
        max_long_term: Maximum long-term memory items (0 = unlimited).
        recency_half_life: Half-life in seconds for recency decay during search.
        auto_consolidate: If True, automatically promote high-importance STM
            items to LTM on store.
        consolidation_threshold: Minimum importance for auto-consolidation.
        eviction_policy: One of "fifo", "lru", "importance_asc".
    """
    db_path:                 str   = ""
    max_working:             int   = 16
    max_short_term:          int   = 128
    max_long_term:           int   = 0
    recency_half_life:       float = 86400.0   # 1 day
    auto_consolidate:        bool  = False
    consolidation_threshold: float = 0.8
    eviction_policy:         str   = "lru"

    def validate(self) -> None:
        if self.max_working < 0:
            raise ValueError("max_working must be >= 0")
        if self.max_short_term < 0:
            raise ValueError("max_short_term must be >= 0")
        if self.max_long_term < 0:
            raise ValueError("max_long_term must be >= 0")
        if self.recency_half_life < 0:
            raise ValueError("recency_half_life must be >= 0")
        if not 0.0 <= self.consolidation_threshold <= 1.0:
            raise ValueError("consolidation_threshold must be in [0, 1]")
        if self.eviction_policy not in ("fifo", "lru", "importance_asc"):
            raise ValueError(
                f"eviction_policy must be 'fifo', 'lru', or 'importance_asc', "
                f"got {self.eviction_policy!r}"
            )


class MemoryEngine:
    """The PRIME Memory Engine.

    Manages three tiers of memory for the PRIME runtime:

    - **Working** — small, ephemeral; use for current-task context.
    - **Short-term** — recent items; automatically evicted when at capacity.
    - **Long-term** — persistent; survives between sessions when db_path is set.

    Obtain via :meth:`prime.PRIME.memory`:

        mem = prime.memory()

    or construct directly for testing:

        mem = MemoryEngine(MemoryConfig(db_path="/tmp/mem.db"))
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        if config is None:
            config = MemoryConfig()
        config.validate()
        self._config = config
        try:
            self._engine = _prime_memory._MemoryEngine(
                db_path              = config.db_path,
                max_working          = config.max_working,
                max_short_term       = config.max_short_term,
                max_long_term        = config.max_long_term,
                recency_half_life    = config.recency_half_life,
                auto_consolidate     = config.auto_consolidate,
                consolidation_threshold = config.consolidation_threshold,
                eviction_policy      = config.eviction_policy,
            )
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def store(self,
              content:     str,
              memory_type: str   = "long_term",
              importance:  float = 0.5,
              confidence:  float = 1.0,
              metadata:    Optional[Dict[str, str]] = None,
              expires_at:  float = 0.0) -> int:
        """Store a new memory item.

        Args:
            content:     The text content to remember.
            memory_type: "working", "short_term", or "long_term".
            importance:  Score in [0.0, 1.0].
            confidence:  Confidence in the content, in [0.0, 1.0].
            metadata:    Optional string key/value pairs.
            expires_at:  Unix timestamp after which the item expires (0 = never).

        Returns:
            The integer ID assigned to the new memory item.

        Raises:
            MemoryError: On storage failure or invalid parameters.
            ValueError:  On invalid memory_type, importance, or confidence.
        """
        try:
            return self._engine.store(
                content,
                memory_type,
                importance,
                confidence,
                metadata or {},
                expires_at,
            )
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def get(self, memory_id: int) -> MemoryItem:
        """Retrieve a memory item by ID.

        Raises:
            MemoryError: If the ID does not exist.
        """
        try:
            return MemoryItem._from_c(self._engine.get(memory_id))
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def update(self,
               memory_id:  int,
               content:    Optional[str]              = None,
               importance: Optional[float]             = None,
               confidence: Optional[float]             = None,
               metadata:   Optional[Dict[str, str]]    = None,
               memory_type:Optional[str]               = None,
               expires_at: Optional[float]             = None) -> None:
        """Update one or more fields on an existing memory item.

        Only fields that are not None are changed.

        Raises:
            MemoryError: If the ID does not exist or parameters are invalid.
        """
        try:
            self._engine.update(
                memory_id,
                content,
                importance,
                confidence,
                metadata,
                memory_type,
                expires_at,
            )
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def delete(self, memory_id: int) -> None:
        """Remove a memory item by ID.

        Raises:
            MemoryError: If the ID does not exist.
        """
        try:
            self._engine.delete(memory_id)
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Search & retrieval
    # ------------------------------------------------------------------

    def search(self,
               query:           str  = "",
               limit:           int  = 10,
               memory_type:     Optional[str]              = None,
               metadata_filter: Optional[Dict[str, str]]   = None) -> List[MemoryItem]:
        """Search memory for items relevant to *query*.

        Uses a lightweight bag-of-words Jaccard similarity combined with
        recency, importance, and access-frequency signals.

        Args:
            query:           Text query.  Empty string returns all items
                             ranked by composite score.
            limit:           Maximum number of results.
            memory_type:     Restrict to "working", "short_term", or "long_term".
            metadata_filter: All key/value pairs must match.

        Returns:
            List of :class:`MemoryItem` sorted by relevance (highest first).
        """
        try:
            raw = self._engine.search(
                query, limit, memory_type, metadata_filter or {}
            )
            return [MemoryItem._from_c(r) for r in raw]
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def retrieve_context(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Return the most relevant memory items for *query*.

        This is the integration point for future reasoning/generation phases:
        call this before an LLM generation step to inject relevant context.

        Returns:
            List of :class:`MemoryItem` sorted by relevance.
        """
        try:
            raw = self._engine.retrieve_context(query, limit)
            return [MemoryItem._from_c(r) for r in raw]
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Working memory helpers
    # ------------------------------------------------------------------

    def working_add(self,
                    content:    str,
                    importance: float = 0.5,
                    metadata:   Optional[Dict[str, str]] = None) -> int:
        """Add an item to working memory (convenience wrapper)."""
        try:
            return self._engine.working_add(content, importance, metadata or {})
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def working_clear(self) -> None:
        """Remove all items from working memory."""
        try:
            self._engine.working_clear()
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def count(self, memory_type: Optional[str] = None) -> int:
        """Return the total number of stored items.

        Args:
            memory_type: If given, count only items of that tier.
        """
        try:
            return self._engine.count(memory_type)
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def exists(self, memory_id: int) -> bool:
        """Return True if *memory_id* is present in the store."""
        try:
            return bool(self._engine.exists(memory_id))
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def clear(self, memory_type: Optional[str] = None) -> None:
        """Delete all memory items (or only those of *memory_type*)."""
        try:
            self._engine.clear(memory_type)
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def expire_all(self) -> int:
        """Remove all expired memory items.

        Returns:
            Number of items removed.
        """
        try:
            return self._engine.expire_all()
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    def consolidate(self, importance_threshold: float = -1.0) -> int:
        """Promote short-term items above *importance_threshold* to long-term.

        Args:
            importance_threshold: Defaults to the engine's configured threshold.

        Returns:
            Number of items promoted.
        """
        try:
            return self._engine.consolidate(importance_threshold)
        except _prime_memory.MemoryError as exc:
            raise MemoryError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def __repr__(self) -> str:
        return (f"<MemoryEngine backend={'sqlite' if self._config.db_path else 'in-memory'} "
                f"items={self.count()}>")
