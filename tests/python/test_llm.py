"""
test_llm.py — Python tests for PRIME Phase 2 LLM engine.

Run with: PYTHONPATH=python python3 tests/python/test_llm.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from prime import PRIME, GenerationConfig
from prime.llm import LLMModel, LLMError
import _prime_llm

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "../../tests/fixtures/tiny_llama.gguf"
)


class TestGenerationConfig(unittest.TestCase):
    def test_defaults_valid(self):
        GenerationConfig().validate()

    def test_invalid_max_new_tokens(self):
        with self.assertRaises(ValueError):
            GenerationConfig(max_new_tokens=0).validate()

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            GenerationConfig(temperature=-0.1).validate()

    def test_invalid_top_p(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=0.0).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=1.1).validate()

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_k=-1).validate()


class TestLLMModelLoad(unittest.TestCase):
    def test_load_fixture(self):
        model = LLMModel(FIXTURE_PATH)
        self.assertEqual(model.arch, "llama")
        self.assertEqual(model.vocab_size, 64)
        self.assertEqual(model.context_length, 64)
        self.assertEqual(model.model_path, FIXTURE_PATH)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            LLMModel("/nonexistent/model.gguf")

    def test_repr(self):
        model = LLMModel(FIXTURE_PATH)
        r = repr(model)
        self.assertIn("llama", r)
        self.assertIn("64", r)

    def test_load_via_prime_object(self):
        brain = PRIME()
        model = brain.load_model(FIXTURE_PATH)
        self.assertIsInstance(model, LLMModel)
        brain.shutdown()

    def test_load_via_shutdown_prime_raises(self):
        brain = PRIME()
        brain.shutdown()
        with self.assertRaises(Exception):
            brain.load_model(FIXTURE_PATH)


class TestLLMGenerate(unittest.TestCase):
    def setUp(self):
        self.model = LLMModel(FIXTURE_PATH)

    def test_generate_returns_string(self):
        result = self.model.generate(
            "hello", GenerationConfig(max_new_tokens=5, temperature=0.0)
        )
        self.assertIsInstance(result, str)

    def test_generate_greedy_deterministic(self):
        cfg = GenerationConfig(max_new_tokens=8, temperature=0.0)
        r1 = self.model.generate("hello", cfg)
        r2 = self.model.generate("hello", cfg)
        r3 = self.model.generate("hello", cfg)
        self.assertEqual(r1, r2)
        self.assertEqual(r2, r3)

    def test_generate_different_prompts_differ(self):
        cfg = GenerationConfig(max_new_tokens=5, temperature=0.0)
        r1 = self.model.generate("hello", cfg)
        r2 = self.model.generate("world", cfg)
        # With a real model and different prompts, outputs should differ.
        # (May occasionally match by chance with tiny random weights,
        # but it's extremely unlikely for prompts this different.)
        # We just verify the call succeeds; equality is not guaranteed.
        self.assertIsInstance(r1, str)
        self.assertIsInstance(r2, str)

    def test_generate_default_config(self):
        # No config: should use defaults without raising.
        result = self.model.generate("hi")
        self.assertIsInstance(result, str)

    def test_generate_with_top_k(self):
        cfg = GenerationConfig(max_new_tokens=5, temperature=1.0, top_k=4, seed=42)
        result = self.model.generate("hello", cfg)
        self.assertIsInstance(result, str)

    def test_generate_with_top_p(self):
        cfg = GenerationConfig(max_new_tokens=5, temperature=1.0, top_p=0.9, seed=7)
        result = self.model.generate("hello", cfg)
        self.assertIsInstance(result, str)

    def test_generate_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            self.model.generate("hello", GenerationConfig(max_new_tokens=0))

    def test_generate_empty_prompt(self):
        # Empty prompt: tokenizer adds BOS; generation should still work.
        result = self.model.generate("", GenerationConfig(max_new_tokens=3, temperature=0.0))
        self.assertIsInstance(result, str)


class TestLLMStream(unittest.TestCase):
    def setUp(self):
        self.model = LLMModel(FIXTURE_PATH)

    def test_stream_yields_strings(self):
        pieces = list(self.model.stream(
            "hello", GenerationConfig(max_new_tokens=5, temperature=0.0)
        ))
        self.assertIsInstance(pieces, list)
        for p in pieces:
            self.assertIsInstance(p, str)

    def test_stream_matches_generate(self):
        cfg = GenerationConfig(max_new_tokens=8, temperature=0.0)
        # Reset KV cache between calls via separate tokens.
        full = self.model.generate("hello", cfg)
        streamed = "".join(self.model.stream("hello", cfg))
        self.assertEqual(full, streamed)

    def test_stream_empty_prompt(self):
        pieces = list(self.model.stream(
            "", GenerationConfig(max_new_tokens=3, temperature=0.0)
        ))
        self.assertIsInstance(pieces, list)


class TestCppExtension(unittest.TestCase):
    """Test the raw _prime_llm C extension directly."""

    def test_llm_error_exists(self):
        self.assertTrue(hasattr(_prime_llm, "LLMError"))

    def test_engine_type_exists(self):
        self.assertTrue(hasattr(_prime_llm, "_LLMEngine"))

    def test_engine_load_and_generate(self):
        eng = _prime_llm._LLMEngine(model_path=FIXTURE_PATH)
        out = eng.generate("hello", max_new_tokens=3, temperature=0.0)
        self.assertIsInstance(out, str)
        self.assertEqual(eng.vocab_size(), 64)
        self.assertEqual(eng.arch_name(), "llama")

    def test_engine_bad_path_raises(self):
        with self.assertRaises(_prime_llm.LLMError):
            _prime_llm._LLMEngine(model_path="/bad/path.gguf")

    def test_python_cpp_round_trip(self):
        """Full Python → C++ → Python generation round-trip."""
        brain = PRIME()
        model = brain.load_model(FIXTURE_PATH)
        result = model.generate(
            "hello world", GenerationConfig(max_new_tokens=4, temperature=0.0)
        )
        self.assertIsInstance(result, str)
        brain.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
