"""
tests/python/test_memory.py — Python unit tests for PRIME Phase 3 Memory Engine.

Run:
    PYTHONPATH=python python3 -m unittest tests/python/test_memory.py -v
"""

import os
import sys
import time
import tempfile
import unittest

# Ensure the prime package is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from prime import (
    PRIME,
    MemoryEngine,
    MemoryConfig,
    MemoryItem,
    MemoryError,
)


# ---------------------------------------------------------------------------
# MemoryConfig tests
# ---------------------------------------------------------------------------

class TestMemoryConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = MemoryConfig()
        self.assertEqual(cfg.db_path, "")
        self.assertEqual(cfg.max_working, 16)
        self.assertEqual(cfg.max_short_term, 128)
        self.assertEqual(cfg.max_long_term, 0)
        self.assertAlmostEqual(cfg.recency_half_life, 86400.0)
        self.assertFalse(cfg.auto_consolidate)
        self.assertAlmostEqual(cfg.consolidation_threshold, 0.8)
        self.assertEqual(cfg.eviction_policy, "lru")

    def test_validate_ok(self):
        cfg = MemoryConfig()
        cfg.validate()  # should not raise

    def test_validate_invalid_eviction_policy(self):
        cfg = MemoryConfig(eviction_policy="bad_policy")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_negative_max_working(self):
        cfg = MemoryConfig(max_working=-1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_consolidation_threshold(self):
        cfg = MemoryConfig(consolidation_threshold=1.5)
        with self.assertRaises(ValueError):
            cfg.validate()


# ---------------------------------------------------------------------------
# MemoryEngine — initialization
# ---------------------------------------------------------------------------

class TestMemoryEngineInit(unittest.TestCase):

    def test_create_default(self):
        mem = MemoryEngine()
        self.assertIsInstance(mem, MemoryEngine)

    def test_create_with_config(self):
        cfg = MemoryConfig(max_working=8, max_short_term=64)
        mem = MemoryEngine(cfg)
        self.assertEqual(mem.config.max_working, 8)

    def test_repr(self):
        mem = MemoryEngine()
        r = repr(mem)
        self.assertIn("MemoryEngine", r)
        self.assertIn("in-memory", r)

    def test_count_empty(self):
        mem = MemoryEngine()
        self.assertEqual(mem.count(), 0)


# ---------------------------------------------------------------------------
# MemoryEngine — store / get
# ---------------------------------------------------------------------------

class TestMemoryEngineStoreCRUD(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine(MemoryConfig())

    def test_store_returns_id(self):
        mid = self.mem.store("hello", memory_type="long_term")
        self.assertIsInstance(mid, int)
        self.assertGreater(mid, 0)

    def test_store_get_roundtrip(self):
        mid = self.mem.store("Python is great", memory_type="long_term",
                              importance=0.8)
        item = self.mem.get(mid)
        self.assertEqual(item.content, "Python is great")
        self.assertAlmostEqual(item.importance, 0.8)
        self.assertEqual(item.type, "long_term")

    def test_get_returns_memory_item(self):
        mid = self.mem.store("test", memory_type="long_term")
        item = self.mem.get(mid)
        self.assertIsInstance(item, MemoryItem)

    def test_get_missing_raises(self):
        with self.assertRaises(MemoryError):
            self.mem.get(99999)

    def test_store_empty_content_raises(self):
        with self.assertRaises(MemoryError):
            self.mem.store("", memory_type="long_term")

    def test_store_invalid_importance_raises(self):
        with self.assertRaises(MemoryError):
            self.mem.store("x", importance=1.5)
        with self.assertRaises(MemoryError):
            self.mem.store("x", importance=-0.1)

    def test_store_invalid_type_raises(self):
        with self.assertRaises((MemoryError, ValueError)):
            self.mem.store("x", memory_type="invalid_tier")

    def test_store_all_types(self):
        id_w  = self.mem.store("working item",    memory_type="working")
        id_st = self.mem.store("short term item", memory_type="short_term")
        id_lt = self.mem.store("long term item",  memory_type="long_term")
        self.assertEqual(self.mem.get(id_w).type,  "working")
        self.assertEqual(self.mem.get(id_st).type, "short_term")
        self.assertEqual(self.mem.get(id_lt).type, "long_term")

    def test_store_with_metadata(self):
        mid = self.mem.store("ML paper", metadata={"topic": "AI", "lang": "en"})
        item = self.mem.get(mid)
        self.assertEqual(item.metadata["topic"], "AI")
        self.assertEqual(item.metadata["lang"],  "en")

    def test_store_with_expiry(self):
        future = time.time() + 3600
        mid = self.mem.store("expires later", expires_at=future)
        item = self.mem.get(mid)
        self.assertAlmostEqual(item.expires_at, future, delta=1.0)


# ---------------------------------------------------------------------------
# MemoryEngine — update
# ---------------------------------------------------------------------------

class TestMemoryEngineUpdate(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine()

    def test_update_content(self):
        mid = self.mem.store("original")
        self.mem.update(mid, content="updated")
        self.assertEqual(self.mem.get(mid).content, "updated")

    def test_update_importance(self):
        mid = self.mem.store("test", importance=0.3)
        self.mem.update(mid, importance=0.9)
        self.assertAlmostEqual(self.mem.get(mid).importance, 0.9)

    def test_update_metadata(self):
        mid = self.mem.store("test")
        self.mem.update(mid, metadata={"source": "user"})
        self.assertEqual(self.mem.get(mid).metadata["source"], "user")

    def test_update_type(self):
        mid = self.mem.store("test", memory_type="short_term")
        self.assertEqual(self.mem.count("short_term"), 1)
        self.mem.update(mid, memory_type="long_term")
        self.assertEqual(self.mem.count("short_term"), 0)
        self.assertEqual(self.mem.count("long_term"),  1)

    def test_update_missing_raises(self):
        with self.assertRaises(MemoryError):
            self.mem.update(99999, content="x")

    def test_update_partial(self):
        mid = self.mem.store("original", importance=0.5)
        self.mem.update(mid, importance=0.9)
        item = self.mem.get(mid)
        self.assertEqual(item.content, "original")  # unchanged
        self.assertAlmostEqual(item.importance, 0.9)


# ---------------------------------------------------------------------------
# MemoryEngine — delete
# ---------------------------------------------------------------------------

class TestMemoryEngineDelete(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine()

    def test_delete_removes_item(self):
        mid = self.mem.store("bye")
        self.mem.delete(mid)
        self.assertFalse(self.mem.exists(mid))

    def test_delete_then_get_raises(self):
        mid = self.mem.store("bye")
        self.mem.delete(mid)
        with self.assertRaises(MemoryError):
            self.mem.get(mid)

    def test_delete_missing_raises(self):
        with self.assertRaises(MemoryError):
            self.mem.delete(99999)


# ---------------------------------------------------------------------------
# MemoryEngine — count / exists / clear
# ---------------------------------------------------------------------------

class TestMemoryEngineBookkeeping(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine()

    def test_count_all(self):
        self.assertEqual(self.mem.count(), 0)
        self.mem.store("a", memory_type="long_term")
        self.mem.store("b", memory_type="short_term")
        self.mem.store("c", memory_type="working")
        self.assertEqual(self.mem.count(), 3)

    def test_count_by_type(self):
        self.mem.store("a", memory_type="long_term")
        self.mem.store("b", memory_type="long_term")
        self.mem.store("c", memory_type="short_term")
        self.assertEqual(self.mem.count("long_term"),  2)
        self.assertEqual(self.mem.count("short_term"), 1)
        self.assertEqual(self.mem.count("working"),    0)

    def test_exists_true(self):
        mid = self.mem.store("x")
        self.assertTrue(self.mem.exists(mid))

    def test_exists_false(self):
        self.assertFalse(self.mem.exists(99999))

    def test_clear_all(self):
        self.mem.store("a")
        self.mem.store("b")
        self.mem.clear()
        self.assertEqual(self.mem.count(), 0)

    def test_clear_by_type(self):
        self.mem.store("a", memory_type="long_term")
        self.mem.store("b", memory_type="short_term")
        self.mem.clear("short_term")
        self.assertEqual(self.mem.count(), 1)
        self.assertEqual(self.mem.count("long_term"), 1)
        self.assertEqual(self.mem.count("short_term"), 0)


# ---------------------------------------------------------------------------
# MemoryEngine — working memory helpers
# ---------------------------------------------------------------------------

class TestWorkingMemory(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine()

    def test_working_add(self):
        mid = self.mem.working_add("current task")
        item = self.mem.get(mid)
        self.assertEqual(item.type, "working")
        self.assertEqual(item.content, "current task")

    def test_working_clear(self):
        self.mem.working_add("a")
        self.mem.working_add("b")
        self.assertEqual(self.mem.count("working"), 2)
        self.mem.working_clear()
        self.assertEqual(self.mem.count("working"), 0)

    def test_working_clear_does_not_affect_other_tiers(self):
        self.mem.store("lt", memory_type="long_term")
        self.mem.working_add("w")
        self.mem.working_clear()
        self.assertEqual(self.mem.count("long_term"), 1)


# ---------------------------------------------------------------------------
# MemoryEngine — search
# ---------------------------------------------------------------------------

class TestMemoryEngineSearch(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine(MemoryConfig(recency_half_life=0))

    def test_search_returns_list(self):
        self.mem.store("Python programming")
        results = self.mem.search("Python")
        self.assertIsInstance(results, list)

    def test_search_returns_memory_items(self):
        self.mem.store("Python programming")
        results = self.mem.search("Python")
        for item in results:
            self.assertIsInstance(item, MemoryItem)

    def test_search_relevance_ordering(self):
        self.mem.store("Python is a programming language", importance=0.8)
        self.mem.store("Rust is a systems language",       importance=0.7)
        self.mem.store("The sky is blue",                  importance=0.1)
        results = self.mem.search("Python programming", limit=3)
        self.assertGreater(len(results), 0)
        self.assertIn("Python", results[0].content)

    def test_search_by_type_filter(self):
        self.mem.store("Python long_term",  memory_type="long_term")
        self.mem.store("Python short_term", memory_type="short_term")
        results = self.mem.search("Python", memory_type="long_term")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].type, "long_term")

    def test_search_metadata_filter(self):
        self.mem.store("ML paper",     metadata={"topic": "AI"})
        self.mem.store("recipe",       metadata={"topic": "cooking"})
        results = self.mem.search("", metadata_filter={"topic": "AI"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["topic"], "AI")

    def test_search_empty_query_returns_all(self):
        self.mem.store("a")
        self.mem.store("b")
        self.mem.store("c")
        results = self.mem.search("", limit=100)
        self.assertEqual(len(results), 3)

    def test_search_limit_respected(self):
        for i in range(10):
            self.mem.store(f"item {i}")
        results = self.mem.search("", limit=3)
        self.assertEqual(len(results), 3)

    def test_search_case_insensitive(self):
        self.mem.store("Machine Learning is important")
        results = self.mem.search("machine learning")
        self.assertGreater(len(results), 0)
        self.assertIn("Machine Learning", results[0].content)


# ---------------------------------------------------------------------------
# MemoryEngine — retrieve_context
# ---------------------------------------------------------------------------

class TestRetrieveContext(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine()

    def test_retrieve_returns_list(self):
        self.mem.store("deep learning uses neural networks", importance=0.9)
        ctx = self.mem.retrieve_context("neural networks")
        self.assertIsInstance(ctx, list)

    def test_retrieve_top_relevant(self):
        id1 = self.mem.store("deep learning uses neural networks", importance=0.9)
        self.mem.store("the weather is sunny today", importance=0.3)
        ctx = self.mem.retrieve_context("neural networks", limit=1)
        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx[0].id, id1)


# ---------------------------------------------------------------------------
# MemoryEngine — eviction
# ---------------------------------------------------------------------------

class TestEviction(unittest.TestCase):

    def test_working_fifo_eviction(self):
        cfg = MemoryConfig(max_working=3, eviction_policy="fifo")
        mem = MemoryEngine(cfg)
        id1 = mem.working_add("a")
        id2 = mem.working_add("b")
        id3 = mem.working_add("c")
        mem.working_add("d")  # triggers eviction of id1 (FIFO)
        self.assertEqual(mem.count("working"), 3)
        self.assertFalse(mem.exists(id1))
        self.assertTrue(mem.exists(id2))
        self.assertTrue(mem.exists(id3))

    def test_short_term_lru_eviction(self):
        cfg = MemoryConfig(max_short_term=3, eviction_policy="lru",
                           recency_half_life=0)
        mem = MemoryEngine(cfg)
        id1 = mem.store("a", memory_type="short_term")
        id2 = mem.store("b", memory_type="short_term")
        id3 = mem.store("c", memory_type="short_term")
        # Touch id1 to make it recently used
        mem.get(id1)
        # Add 4th: should evict id2 (LRU)
        mem.store("d", memory_type="short_term")
        self.assertEqual(mem.count("short_term"), 3)
        self.assertTrue(mem.exists(id1))
        self.assertTrue(mem.exists(id3))

    def test_importance_eviction(self):
        cfg = MemoryConfig(max_working=3, eviction_policy="importance_asc")
        mem = MemoryEngine(cfg)
        id_low  = mem.working_add("low",  importance=0.1)
        id_high = mem.working_add("high", importance=0.9)
        mem.working_add("mid",  importance=0.5)
        mem.working_add("new",  importance=0.5)
        self.assertEqual(mem.count("working"), 3)
        self.assertFalse(mem.exists(id_low))
        self.assertTrue(mem.exists(id_high))


# ---------------------------------------------------------------------------
# MemoryEngine — expiration
# ---------------------------------------------------------------------------

class TestExpiration(unittest.TestCase):

    def test_expire_all_removes_expired(self):
        mem = MemoryEngine()
        past = time.time() - 1.0
        exp_id = mem.store("expired content", expires_at=past)
        mem.store("valid content")
        self.assertEqual(mem.count(), 2)
        removed = mem.expire_all()
        self.assertEqual(removed, 1)
        self.assertEqual(mem.count(), 1)
        self.assertFalse(mem.exists(exp_id))

    def test_future_expiry_not_removed(self):
        mem = MemoryEngine()
        future = time.time() + 3600
        mem.store("still valid", expires_at=future)
        removed = mem.expire_all()
        self.assertEqual(removed, 0)
        self.assertEqual(mem.count(), 1)

    def test_no_expiry_never_removed(self):
        mem = MemoryEngine()
        mem.store("no expiry")  # expires_at defaults to 0.0
        removed = mem.expire_all()
        self.assertEqual(removed, 0)


# ---------------------------------------------------------------------------
# MemoryEngine — consolidation
# ---------------------------------------------------------------------------

class TestConsolidation(unittest.TestCase):

    def test_consolidate_promotes_high_importance(self):
        mem = MemoryEngine()
        mem.store("important fact",   memory_type="short_term", importance=0.9)
        mem.store("less important",   memory_type="short_term", importance=0.3)
        promoted = mem.consolidate(importance_threshold=0.7)
        self.assertEqual(promoted, 1)
        self.assertEqual(mem.count("long_term"), 1)

    def test_consolidate_default_threshold(self):
        cfg = MemoryConfig(consolidation_threshold=0.8)
        mem = MemoryEngine(cfg)
        mem.store("x", memory_type="short_term", importance=0.9)
        promoted = mem.consolidate()  # uses cfg threshold
        self.assertEqual(promoted, 1)

    def test_auto_consolidate(self):
        cfg = MemoryConfig(auto_consolidate=True, consolidation_threshold=0.7)
        mem = MemoryEngine(cfg)
        mem.store("auto-promote me", memory_type="short_term", importance=0.9)
        # Should have been auto-copied to LTM on store
        self.assertEqual(mem.count("long_term"), 1)


# ---------------------------------------------------------------------------
# MemoryEngine — access tracking
# ---------------------------------------------------------------------------

class TestAccessTracking(unittest.TestCase):

    def test_access_count_increments(self):
        mem = MemoryEngine()
        mid = mem.store("tracked")
        mem.get(mid)
        mem.get(mid)
        item = mem.get(mid)
        self.assertGreaterEqual(item.access_count, 2)

    def test_last_accessed_updated(self):
        mem = MemoryEngine()
        mid = mem.store("tracked")
        item_before = mem.get(mid)
        time.sleep(0.01)
        item_after = mem.get(mid)
        self.assertGreaterEqual(item_after.last_accessed, item_before.last_accessed)


# ---------------------------------------------------------------------------
# MemoryItem — field validation
# ---------------------------------------------------------------------------

class TestMemoryItemFields(unittest.TestCase):

    def setUp(self):
        self.mem = MemoryEngine()

    def test_timestamps_set_on_store(self):
        before = time.time()
        mid = self.mem.store("timestamped")
        item = self.mem.get(mid)
        self.assertGreaterEqual(item.created_at, before)

    def test_importance_range(self):
        mid = self.mem.store("test", importance=0.0)
        self.assertEqual(self.mem.get(mid).importance, 0.0)
        mid2 = self.mem.store("test2", importance=1.0)
        self.assertAlmostEqual(self.mem.get(mid2).importance, 1.0)

    def test_confidence_stored(self):
        mid = self.mem.store("test", confidence=0.75)
        self.assertAlmostEqual(self.mem.get(mid).confidence, 0.75)

    def test_memory_item_repr(self):
        mid = self.mem.store("hello repr")
        item = self.mem.get(mid)
        r = repr(item)
        self.assertIn("MemoryItem", r)
        self.assertIn(str(mid), r)


# ---------------------------------------------------------------------------
# PRIME.memory() integration
# ---------------------------------------------------------------------------

class TestPRIMEMemoryIntegration(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_prime_memory_returns_engine(self):
        mem = self.prime.memory()
        self.assertIsInstance(mem, MemoryEngine)

    def test_prime_memory_full_workflow(self):
        mem = self.prime.memory()
        mid = mem.store("Python is a programming language.",
                        memory_type="long_term",
                        importance=0.8)
        results = mem.search("Python programming")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].id, mid)

        mem.update(mid, importance=0.9)
        self.assertAlmostEqual(mem.get(mid).importance, 0.9)

        mem.delete(mid)
        self.assertFalse(mem.exists(mid))

    def test_prime_memory_after_shutdown_raises(self):
        self.prime.shutdown()
        with self.assertRaises(Exception):
            self.prime.memory()


# ---------------------------------------------------------------------------
# Persistence test (uses SQLite if available; skips otherwise)
# ---------------------------------------------------------------------------

class TestPersistence(unittest.TestCase):

    def _has_sqlite_support(self):
        """Check if the _prime_memory extension was built with SQLite."""
        try:
            import _prime_memory
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
                db_path = f.name
            os.unlink(db_path)
            cfg = MemoryConfig(db_path=db_path)
            mem = MemoryEngine(cfg)
            mid = mem.store("probe", memory_type="long_term")
            # If it silently fell back to in-memory, persistence won't work.
            # Check by trying a second instance on the same path.
            mem2 = MemoryEngine(cfg)
            has = mem2.count() > 0
            try:
                os.unlink(db_path)
            except OSError:
                pass
            return has
        except Exception:
            return False

    def test_persistence_roundtrip(self):
        if not self._has_sqlite_support():
            self.skipTest("SQLite backend not available (PRIME_HAVE_SQLITE not compiled in)")

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            # Session 1: store
            cfg = MemoryConfig(db_path=db_path)
            mem1 = MemoryEngine(cfg)
            mid = mem1.store("Machine learning is a subset of AI.",
                              memory_type="long_term", importance=0.9)
            stored_content = mem1.get(mid).content
            del mem1  # triggers shutdown / close

            # Session 2: reload
            mem2 = MemoryEngine(MemoryConfig(db_path=db_path))
            self.assertEqual(mem2.count(), 1)
            results = mem2.search("machine learning")
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0].content, stored_content)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Benchmark (not a test — run manually)
# ---------------------------------------------------------------------------

def run_benchmark():
    """Minimal benchmark for Phase 3 memory operations."""
    import time as _time

    print("\n=== Phase 3 Memory Engine Benchmark ===")
    cfg = MemoryConfig(max_working=0, max_short_term=0, max_long_term=0,
                       recency_half_life=0)
    mem = MemoryEngine(cfg)

    # --- Insert benchmark ---
    N = 1000
    t0 = _time.perf_counter()
    for i in range(N):
        mem.store(f"Memory item {i}: The quick brown fox jumps over the lazy dog.",
                  memory_type="long_term", importance=round((i % 10) / 10, 1))
    elapsed = _time.perf_counter() - t0
    print(f"  Insert {N} items: {elapsed*1000:.1f} ms  "
          f"({N/elapsed:.0f} writes/sec)")

    # --- Search benchmark ---
    queries = ["quick fox", "lazy dog", "memory item", "brown fox jumps"]
    t0 = _time.perf_counter()
    for q in queries * 25:  # 100 queries
        mem.search(q, limit=10)
    elapsed = _time.perf_counter() - t0
    print(f"  Search 100 queries over {N} items: {elapsed*1000:.1f} ms  "
          f"({100/elapsed:.0f} searches/sec)")

    # --- retrieve_context benchmark ---
    t0 = _time.perf_counter()
    for _ in range(100):
        mem.retrieve_context("quick brown fox", limit=5)
    elapsed = _time.perf_counter() - t0
    print(f"  retrieve_context x100: {elapsed*1000:.1f} ms")

    # --- Memory footprint (rough) ---
    import tracemalloc
    tracemalloc.start()
    cfg2 = MemoryConfig(max_long_term=0, recency_half_life=0)
    mem2 = MemoryEngine(cfg2)
    for i in range(1000):
        mem2.store(f"item {i}" * 10, memory_type="long_term")
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    total = sum(stat.size for stat in snapshot.statistics('lineno'))
    print(f"  Approx RSS for 1K items: {total/1024:.1f} KiB (tracemalloc)")

    print("=== Benchmark complete ===\n")


if __name__ == '__main__':
    unittest.main()
