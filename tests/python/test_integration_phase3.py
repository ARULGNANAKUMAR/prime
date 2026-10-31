"""
tests/python/test_integration_phase3.py — Phase 3 integration test.

Verifies that the LLM Engine and Memory Engine can communicate within
a single PRIME runtime:

    PRIME runtime
        ↓
    MemoryEngine.store(information)
        ↓
    MemoryEngine.retrieve_context(query)
        ↓
    Retrieved text prepended to LLM prompt
        ↓
    LLMModel.generate(augmented_prompt)

This test does NOT implement intelligent memory extraction or RAG;
it only verifies that the modules can be plumbed together.

Run:
    PYTHONPATH=python python3 -m unittest tests/python/test_integration_phase3.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'fixtures', 'tiny_llama.gguf'
)

from prime import PRIME, MemoryEngine, MemoryConfig, GenerationConfig


class TestPhase3Integration(unittest.TestCase):
    """Integration tests: PRIME + Memory Engine + LLM Engine."""

    @classmethod
    def setUpClass(cls):
        cls.prime = PRIME()
        cls.mem = cls.prime.memory()
        if not os.path.isfile(MODEL_PATH):
            cls.model = None
        else:
            cls.model = cls.prime.load_model(MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.prime.shutdown()

    # ------------------------------------------------------------------
    # Memory Engine available from PRIME
    # ------------------------------------------------------------------

    def setUp(self):
        """Reset shared memory before each test."""
        self.mem.clear()

    def test_memory_available_from_prime(self):
        """PRIME.memory() returns a working MemoryEngine."""
        self.assertIsNotNone(self.mem)
        self.assertEqual(self.mem.count(), 0)

    def test_store_and_retrieve(self):
        """Memory can store and retrieve a fact."""
        self.mem.clear()
        mid = self.mem.store(
            "The Eiffel Tower is located in Paris, France.",
            memory_type="long_term",
            importance=0.9,
        )
        results = self.mem.retrieve_context("Eiffel Tower Paris", limit=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].id, mid)
        self.assertIn("Eiffel Tower", results[0].content)

    def test_working_memory_lifecycle(self):
        """Working memory can be populated and cleared cleanly."""
        self.mem.working_clear()
        self.mem.working_add("current task: analyse text")
        self.mem.working_add("current input: 'Hello world'")
        self.assertEqual(self.mem.count("working"), 2)
        self.mem.working_clear()
        self.assertEqual(self.mem.count("working"), 0)

    def test_multi_tier_coexistence(self):
        """Working, short-term, and long-term memory coexist independently."""
        self.mem.clear()
        self.mem.store("long term fact",  memory_type="long_term")
        self.mem.store("short term note", memory_type="short_term")
        self.mem.working_add("working note")
        self.assertEqual(self.mem.count("long_term"),  1)
        self.assertEqual(self.mem.count("short_term"), 1)
        self.assertEqual(self.mem.count("working"),    1)
        self.assertEqual(self.mem.count(),             3)

    def test_consolidation_workflow(self):
        """Short-term items can be promoted to long-term."""
        self.mem.clear()
        self.mem.store("important observation", memory_type="short_term", importance=0.95)
        self.mem.store("trivial note",          memory_type="short_term", importance=0.1)
        self.assertEqual(self.mem.count("long_term"), 0)
        promoted = self.mem.consolidate(importance_threshold=0.8)
        self.assertEqual(promoted, 1)
        self.assertEqual(self.mem.count("long_term"), 1)

    # ------------------------------------------------------------------
    # LLM + Memory integration (requires tiny_llama.gguf fixture)
    # ------------------------------------------------------------------

    def _require_model(self):
        if self.model is None:
            self.skipTest("LLM fixture not available: " + MODEL_PATH)

    def test_llm_still_works(self):
        """Phase 2 LLM engine still generates text after Phase 3 changes."""
        self._require_model()
        cfg = GenerationConfig(max_new_tokens=8, temperature=0.0, seed=42)
        output = self.model.generate("hello", config=cfg)
        self.assertIsInstance(output, str)

    def test_memory_augmented_prompt(self):
        """
        Memory context can be prepended to an LLM prompt.

        This is the minimal integration proof:
            store fact → retrieve context → prepend to prompt → generate
        No intelligent memory agent is implemented.
        """
        self._require_model()

        # Store a short fact (tiny model context = 64 tokens total)
        self.mem.store("Paris France", memory_type="long_term", importance=0.9)

        # Retrieve context
        context_items = self.mem.retrieve_context("France", limit=1)
        self.assertGreater(len(context_items), 0)

        # Build a short augmented prompt that fits the tiny model's 64-token context
        context_text = context_items[0].content
        augmented_prompt = f"ctx:{context_text} q:capital?"

        # Generate from the augmented prompt
        cfg = GenerationConfig(max_new_tokens=4, temperature=0.0, seed=42)
        output = self.model.generate(augmented_prompt, config=cfg)
        self.assertIsInstance(output, str)
        # The tiny fixture produces arbitrary tokens; we only verify the pipeline.

    def test_memory_search_filters_by_type(self):
        """Memory search type filter correctly restricts results."""
        self.mem.clear()
        self.mem.store("long term info",  memory_type="long_term")
        self.mem.store("short term info", memory_type="short_term")
        long_results  = self.mem.search("info", memory_type="long_term")
        short_results = self.mem.search("info", memory_type="short_term")
        self.assertEqual(len(long_results),  1)
        self.assertEqual(len(short_results), 1)
        self.assertEqual(long_results[0].type,  "long_term")
        self.assertEqual(short_results[0].type, "short_term")


if __name__ == '__main__':
    unittest.main()
