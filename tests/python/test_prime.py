"""
test_prime.py — Python unit tests for PRIME Phase 1.

Run with: python -m pytest tests/python/ -v
or:        python tests/python/test_prime.py
"""

import sys
import os
import unittest

# Allow running from the project root without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from prime import PRIME, PrimeError, RuntimeConfig
import _prime


class TestCExtension(unittest.TestCase):
    """Test the raw _prime C++ extension directly."""

    def test_version_returns_string(self):
        v = _prime.version()
        self.assertIsInstance(v, str)
        self.assertIn("0.1.0", v)

    def test_log_constants_exist(self):
        self.assertEqual(_prime.LOG_SILENT, 0)
        self.assertEqual(_prime.LOG_ERROR,  1)
        self.assertEqual(_prime.LOG_WARN,   2)
        self.assertEqual(_prime.LOG_INFO,   3)
        self.assertEqual(_prime.LOG_DEBUG,  4)

    def test_state_constants_exist(self):
        self.assertEqual(_prime.STATE_UNINITIALIZED, 0)
        self.assertEqual(_prime.STATE_RUNNING,       1)
        self.assertEqual(_prime.STATE_SHUTDOWN,      2)

    def test_runtime_object_lifecycle(self):
        rt = _prime._Runtime(log_level=_prime.LOG_SILENT)
        self.assertEqual(rt.get_state(), _prime.STATE_UNINITIALIZED)
        rt.initialize()
        self.assertEqual(rt.get_state(), _prime.STATE_RUNNING)
        status = rt.status_string()
        self.assertIn("RUNNING", status)
        rt.shutdown()
        self.assertEqual(rt.get_state(), _prime.STATE_SHUTDOWN)

    def test_double_init_raises(self):
        rt = _prime._Runtime(log_level=_prime.LOG_SILENT)
        rt.initialize()
        with self.assertRaises(_prime.PrimeError):
            rt.initialize()
        rt.shutdown()

    def test_shutdown_before_init_raises(self):
        rt = _prime._Runtime(log_level=_prime.LOG_SILENT)
        with self.assertRaises(_prime.PrimeError):
            rt.shutdown()

    def test_invalid_backend_raises(self):
        with self.assertRaises(_prime.PrimeError):
            _prime._Runtime(compute_backend="invalid_backend")


class TestRuntimeConfig(unittest.TestCase):
    """Test Python-side RuntimeConfig validation."""

    def test_default_config_valid(self):
        cfg = RuntimeConfig()
        cfg.validate()  # should not raise

    def test_invalid_log_level(self):
        cfg = RuntimeConfig(log_level="verbose")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_backend(self):
        cfg = RuntimeConfig(compute_backend="gpu")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_negative_memory_raises(self):
        cfg = RuntimeConfig(max_memory_bytes=-1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_log_level_int_mapping(self):
        cfg = RuntimeConfig(log_level="debug")
        self.assertEqual(cfg._log_level_int(), _prime.LOG_DEBUG)
        cfg.log_level = "silent"
        self.assertEqual(cfg._log_level_int(), _prime.LOG_SILENT)


class TestPRIMEObject(unittest.TestCase):
    """Test the public PRIME Python API."""

    def test_basic_init_and_shutdown(self):
        brain = PRIME(RuntimeConfig(log_level="silent"))
        self.assertTrue(brain.is_running())
        brain.shutdown()
        self.assertFalse(brain.is_running())

    def test_status_returns_string(self):
        brain = PRIME(RuntimeConfig(log_level="silent"))
        s = brain.status()
        self.assertIsInstance(s, str)
        self.assertIn("RUNNING", s)
        brain.shutdown()

    def test_status_after_shutdown_raises(self):
        brain = PRIME(RuntimeConfig(log_level="silent"))
        brain.shutdown()
        with self.assertRaises(PrimeError):
            brain.status()

    def test_double_shutdown_is_noop(self):
        brain = PRIME(RuntimeConfig(log_level="silent"))
        brain.shutdown()
        brain.shutdown()  # should not raise

    def test_repr(self):
        brain = PRIME(RuntimeConfig(log_level="silent"))
        r = repr(brain)
        self.assertIn("running", r)
        brain.shutdown()
        r = repr(brain)
        self.assertIn("shutdown", r)

    def test_context_manager(self):
        with PRIME(RuntimeConfig(log_level="silent")) as brain:
            self.assertTrue(brain.is_running())
        self.assertFalse(brain.is_running())

    def test_config_accessible(self):
        cfg = RuntimeConfig(log_level="warn", compute_backend="cpu")
        brain = PRIME(cfg)
        self.assertEqual(brain.config.log_level, "warn")
        brain.shutdown()

    def test_default_config(self):
        brain = PRIME()
        self.assertTrue(brain.is_running())
        brain.shutdown()

    def test_invalid_config_raises_on_create(self):
        with self.assertRaises(ValueError):
            PRIME(RuntimeConfig(log_level="invalid"))

    def test_cxx_communication(self):
        """Verify Python → C++ → Python round-trip produces a meaningful result."""
        brain = PRIME(RuntimeConfig(log_level="silent"))
        s = brain.status()
        self.assertIn("PRIME Runtime", s)
        self.assertIn("cpu", s)
        brain.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
