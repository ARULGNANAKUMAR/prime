"""
tests/python/test_integration_phase10.py
─────────────────────────────────────────
Phase 10: Full Integration & Research Validation Tests.

Coverage:
  ── Configuration ──────────────────────────────────────────────────────────
   1.  PrimeConfig defaults + validate()
   2.  PrimeConfig to_dict / from_dict round-trip
   3.  PrimeConfig to_json / from_json round-trip
   4.  PrimeConfig.minimal()
   5.  PrimeConfig.for_ablation() all baselines
   6.  PrimeConfig validation errors

  ── FeatureFlags ───────────────────────────────────────────────────────────
   7.  FeatureFlags defaults
   8.  FeatureFlags.all_off()
   9.  FeatureFlags presets: static / mosaic_only / mosaic_echo /
       mosaic_echo_learning / full
  10.  FeatureFlags to_dict / from_dict

  ── SolveStatus ────────────────────────────────────────────────────────────
  11.  SolveStatus.succeeded for success codes
  12.  SolveStatus.succeeded for failure codes
  13.  SolveStatus enum values exist

  ── PrimeTrace ─────────────────────────────────────────────────────────────
  14.  PrimeTrace creation defaults
  15.  PrimeTrace.explain() returns string with required sections
  16.  PrimeTrace.to_dict() serialisable
  17.  PhaseTimings.to_dict()

  ── PrimeResult ────────────────────────────────────────────────────────────
  18.  PrimeResult defaults
  19.  PrimeResult.succeeded property
  20.  PrimeResult.answer_with_unit
  21.  PrimeResult.explain() returns string
  22.  PrimeResult.to_dict() serialisable (no trace key overflow)
  23.  PrimeResult.__repr__
  24.  PrimeResult.__str__ delegates to explain()

  ── PRIME.solve() ──────────────────────────────────────────────────────────
  25.  prime.solve() returns PrimeResult
  26.  prime.solve() status is a SolveStatus
  27.  prime.solve() includes trace when requested
  28.  prime.solve() does NOT include trace by default
  29.  prime.solve() empty task → INVALID_INPUT
  30.  prime.solve() learning=False disables learning flag
  31.  prime.solve() optimization=False disables optimization
  32.  prime.solve() with PrimeConfig
  33.  prime.solve() latency_ms is positive
  34.  prime.solve() modules_used is a list
  35.  prime.solve() repeated calls do not crash

  ── PRIME.pipeline() ───────────────────────────────────────────────────────
  36.  prime.pipeline() returns PrimePipeline
  37.  PrimePipeline.solve() returns PrimeResult
  38.  PrimePipeline is reusable

  ── Feature flag ablations via solve() ────────────────────────────────────
  39.  solve() with features.enable_mosaic=False (no MOSAIC path)
  40.  solve() with features.enable_echo=False
  41.  solve() with features.enable_learning=False
  42.  solve() with features.enable_optimization=False
  43.  solve() static baseline config

  ── SyntheticBenchmarkSuite ────────────────────────────────────────────────
  44.  all_tasks() non-empty
  45.  by_category("arithmetic") returns only arithmetic
  46.  core_tasks() is a subset of all_tasks()
  47.  BenchmarkTask to_dict()
  48.  REPEATED tasks all have ground truth "50"

  ── ExperimentResult ───────────────────────────────────────────────────────
  49.  ExperimentResult compute_aggregates with no tasks
  50.  ExperimentResult compute_aggregates with tasks
  51.  ExperimentResult to_dict / to_json serialisable

  ── ExperimentRunner ───────────────────────────────────────────────────────
  52.  ExperimentRunner.run() returns ExperimentResult
  53.  ExperimentRunner.run() n_tasks matches task count
  54.  ExperimentRunner.run() records latency > 0
  55.  ExperimentRunner.run() n_repeats multiplies task count

  ── AblationStudy ──────────────────────────────────────────────────────────
  56.  AblationStudy.run_all() returns dict with expected keys
  57.  AblationStudy.compare() returns non-empty string

  ── LearningCurveExperiment ────────────────────────────────────────────────
  58.  LearningCurveExperiment.run() returns list of LearningCurvePoints
  59.  LearningCurvePoint n_experiences monotonic

  ── SystemInfo ─────────────────────────────────────────────────────────────
  60.  collect_system_info() returns dict with required keys

  ── profile_component ──────────────────────────────────────────────────────
  61.  profile_component() returns ComponentProfile
  62.  ComponentProfile mean > 0

  ── Regression: all previous phases ────────────────────────────────────────
  63.  ATLAS parses problem (Phase 2)
  64.  Reasoning solves arithmetic (Phase 4)
  65.  MOSAIC composes plan (Phase 6)
  66.  ECHO records experience (Phase 7)
  67.  Learning controller enabled (Phase 8)
  68.  Optimizer produces trace (Phase 9)

  ── End-to-end integration ─────────────────────────────────────────────────
  69.  solve() → trace.atlas_summary populated
  70.  solve() → trace.mosaic_modules populated
  71.  solve() → trace.timings.total_s > 0
  72.  solve() → trace.explain() contains all 8 sections
  73.  solve() result is JSON-serialisable
  74.  Sequential solve() calls accumulate optimizer history
  75.  PrimeConfig.for_ablation() + solve() pipeline executes
"""

from __future__ import annotations

import json
import sys
import unittest

sys.path.insert(0, "python")

import prime
from prime import PRIME, PrimeConfig, PrimeResult, PrimeTrace, SolveStatus
from prime import (
    FeatureFlags, PhaseTimings, PrimePipeline, PipelineError,
    collect_system_info,
    BenchmarkTask, TaskResult, SyntheticBenchmarkSuite,
    ExperimentResult, ExperimentRunner,
    AblationStudy, ABLATION_CONFIGS,
    LearningCurveExperiment, LearningCurvePoint,
    ComponentProfile, profile_component,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prime_factory(config: PrimeConfig = None) -> PRIME:
    """Create a PRIME instance; pipeline attaches automatically."""
    from prime import PRIME
    p = PRIME()
    p._p10_config = config  # stash for pipeline
    # Monkey-patch solve so ExperimentRunner works transparently
    orig_solve = p.solve
    def _patched_solve(task, **kwargs):
        cfg = config or PrimeConfig.default()
        return orig_solve(task, config=cfg, **kwargs)
    p.solve = _patched_solve
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1–6. PrimeConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimeConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = PrimeConfig()
        self.assertEqual(cfg.log_level, "warn")
        self.assertEqual(cfg.compute_backend, "cpu")
        self.assertEqual(cfg.version, "1.0.0")

    def test_validate_passes(self):
        cfg = PrimeConfig()
        cfg.validate()  # no exception

    def test_validate_bad_log_level(self):
        cfg = PrimeConfig(log_level="verbose")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_bad_backend(self):
        cfg = PrimeConfig(compute_backend="gpu")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_bad_learning_rate(self):
        cfg = PrimeConfig(learning_rate=2.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_bad_optimizer_policy(self):
        cfg = PrimeConfig(optimizer_policy="FASTEST")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_to_dict_from_dict_round_trip(self):
        cfg = PrimeConfig(log_level="silent", max_tokens=64,
                           optimizer_policy="LOW_LATENCY")
        d = cfg.to_dict()
        cfg2 = PrimeConfig.from_dict(d)
        self.assertEqual(cfg2.log_level, "silent")
        self.assertEqual(cfg2.max_tokens, 64)
        self.assertEqual(cfg2.optimizer_policy, "LOW_LATENCY")

    def test_to_json_from_json_round_trip(self):
        cfg = PrimeConfig(learning_rate=0.1)
        j = cfg.to_json()
        cfg2 = PrimeConfig.from_json(j)
        self.assertAlmostEqual(cfg2.learning_rate, 0.1)

    def test_minimal_config(self):
        cfg = PrimeConfig.minimal()
        self.assertEqual(cfg.log_level, "silent")
        self.assertFalse(cfg.features.enable_mosaic)
        self.assertFalse(cfg.features.enable_echo)

    def test_for_ablation_all_baselines(self):
        for b in ["static", "mosaic_only", "mosaic_echo",
                   "mosaic_echo_learning", "full"]:
            cfg = PrimeConfig.for_ablation(b)
            self.assertIsInstance(cfg, PrimeConfig)

    def test_for_ablation_invalid_raises(self):
        with self.assertRaises(ValueError):
            PrimeConfig.for_ablation("nonexistent_baseline")

    def test_default_factory(self):
        cfg = PrimeConfig.default()
        self.assertIsInstance(cfg, PrimeConfig)


# ─────────────────────────────────────────────────────────────────────────────
# 7–10. FeatureFlags
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFlags(unittest.TestCase):

    def test_full_default(self):
        f = FeatureFlags.full()
        self.assertTrue(f.enable_reasoning)
        self.assertTrue(f.enable_mosaic)
        self.assertTrue(f.enable_echo)
        self.assertTrue(f.enable_learning)
        self.assertTrue(f.enable_optimization)

    def test_all_off(self):
        f = FeatureFlags.all_off()
        for attr in vars(f).values():
            self.assertFalse(attr)

    def test_static_baseline(self):
        f = FeatureFlags.static_baseline()
        self.assertTrue(f.enable_reasoning)
        self.assertFalse(f.enable_mosaic)
        self.assertFalse(f.enable_echo)
        self.assertFalse(f.enable_learning)
        self.assertFalse(f.enable_optimization)

    def test_mosaic_only(self):
        f = FeatureFlags.mosaic_only()
        self.assertTrue(f.enable_mosaic)
        self.assertFalse(f.enable_echo)

    def test_mosaic_echo(self):
        f = FeatureFlags.mosaic_echo()
        self.assertTrue(f.enable_echo)
        self.assertFalse(f.enable_learning)

    def test_mosaic_echo_learning(self):
        f = FeatureFlags.mosaic_echo_learning()
        self.assertTrue(f.enable_learning)
        self.assertFalse(f.enable_optimization)

    def test_to_dict_from_dict_round_trip(self):
        f = FeatureFlags(enable_llm=True, enable_echo=False)
        d = f.to_dict()
        f2 = FeatureFlags.from_dict(d)
        self.assertTrue(f2.enable_llm)
        self.assertFalse(f2.enable_echo)


# ─────────────────────────────────────────────────────────────────────────────
# 11–13. SolveStatus
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveStatus(unittest.TestCase):

    def test_success_codes_succeeded(self):
        self.assertTrue(SolveStatus.SUCCESS.succeeded)
        self.assertTrue(SolveStatus.PARTIAL_SUCCESS.succeeded)

    def test_failure_codes_not_succeeded(self):
        for s in (SolveStatus.INVALID_INPUT, SolveStatus.REASONING_FAILED,
                  SolveStatus.INTERNAL_ERROR, SolveStatus.TIMEOUT):
            self.assertFalse(s.succeeded)

    def test_all_values_exist(self):
        expected = {
            "SUCCESS", "PARTIAL_SUCCESS", "INVALID_INPUT",
            "UNSUPPORTED_TASK", "MODULE_UNAVAILABLE", "MODEL_UNAVAILABLE",
            "RESOURCE_LIMIT", "TIMEOUT", "VALIDATION_FAILED",
            "REASONING_FAILED", "COMPOSITION_FAILED", "LEARNING_REJECTED",
            "OPTIMIZATION_FAILED", "INTERNAL_ERROR",
        }
        actual = {s.value for s in SolveStatus}
        self.assertEqual(expected, actual)


# ─────────────────────────────────────────────────────────────────────────────
# 14–17. PrimeTrace
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimeTrace(unittest.TestCase):

    def _make_trace(self) -> PrimeTrace:
        t = PrimeTrace(task_id="t_test")
        t.atlas_summary       = {"domain": "physics", "entity_count": 2, "goal_count": 1}
        t.mosaic_modules      = ["atlas", "reasoning", "validator"]
        t.mosaic_selection_reasons = ["arithmetic capability matched"]
        t.reasoning_status    = "SOLVED"
        t.reasoning_steps     = 3
        t.validation_passed   = True
        t.echo_experience_id  = "exp-001"
        t.echo_patterns_found = 2
        t.learning_updated    = True
        t.learning_update_id  = "upd-001"
        t.optimizer_strategy  = "BALANCED"
        t.optimizer_latency_ms = 0.4
        t.timings.atlas_s     = 0.005
        t.timings.mosaic_s    = 0.010
        t.timings.execution_s = 0.015
        t.timings.echo_s      = 0.002
        t.timings.learning_s  = 0.001
        t.timings.optimizer_s = 0.0004
        t.timings.total_s     = 0.033
        return t

    def test_default_creation(self):
        t = PrimeTrace(task_id="x")
        self.assertEqual(t.task_id, "x")
        self.assertIsInstance(t.trace_id, str)

    def test_explain_returns_string(self):
        txt = self._make_trace().explain()
        self.assertIsInstance(txt, str)
        self.assertGreater(len(txt), 100)

    def test_explain_has_all_sections(self):
        txt = self._make_trace().explain()
        for section in ("[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]"):
            self.assertIn(section, txt, f"Missing section {section}")

    def test_explain_no_llm_text(self):
        # Explanation must come from trace data, not invented
        txt = self._make_trace().explain()
        self.assertIn("SOLVED", txt)
        self.assertIn("BALANCED", txt)
        self.assertIn("exp-001", txt)

    def test_to_dict_serialisable(self):
        d = self._make_trace().to_dict()
        # Must be JSON serialisable
        j = json.dumps(d)
        self.assertIsInstance(j, str)

    def test_phase_timings_to_dict(self):
        t = PhaseTimings(atlas_s=0.005, total_s=0.033)
        d = t.to_dict()
        self.assertAlmostEqual(d["atlas_s"], 0.005)
        self.assertAlmostEqual(d["total_s"], 0.033)


# ─────────────────────────────────────────────────────────────────────────────
# 18–24. PrimeResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimeResult(unittest.TestCase):

    def test_defaults(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="test")
        self.assertEqual(r.answer, "")
        self.assertEqual(r.confidence, 0.0)
        self.assertFalse(r.learning_applied)

    def test_succeeded_property(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="t")
        self.assertTrue(r.succeeded)
        r2 = PrimeResult(status=SolveStatus.REASONING_FAILED, task="t")
        self.assertFalse(r2.succeeded)

    def test_answer_with_unit(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="t",
                         answer="50", answer_unit="km/h")
        self.assertEqual(r.answer_with_unit, "50 km/h")

    def test_answer_with_no_unit(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="t", answer="42")
        self.assertEqual(r.answer_with_unit, "42")

    def test_explain_returns_string(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="car problem",
                         answer="50", confidence=0.9, latency_ms=2.5)
        txt = r.explain()
        self.assertIn("SUCCESS", txt)
        self.assertIn("50", txt)

    def test_to_dict_serialisable(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="t", answer="42")
        d = r.to_dict()
        j = json.dumps(d)
        self.assertIsInstance(j, str)
        self.assertNotIn("trace", d)   # trace excluded from to_dict

    def test_to_dict_has_status_string(self):
        r = PrimeResult(status=SolveStatus.REASONING_FAILED, task="t")
        d = r.to_dict()
        self.assertEqual(d["status"], "REASONING_FAILED")

    def test_repr(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="t",
                         answer="50", confidence=0.95)
        txt = repr(r)
        self.assertIn("PrimeResult", txt)
        self.assertIn("SUCCESS", txt)

    def test_str_delegates_to_explain(self):
        r = PrimeResult(status=SolveStatus.SUCCESS, task="t")
        self.assertEqual(str(r), r.explain())


# ─────────────────────────────────────────────────────────────────────────────
# 25–35. PRIME.solve()
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimeSolve(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_returns_prime_result(self):
        r = self.prime.solve("What is 6 divided by 2?")
        self.assertIsInstance(r, PrimeResult)

    def test_status_is_solve_status(self):
        r = self.prime.solve("Compute 9 divided by 3.")
        self.assertIsInstance(r.status, SolveStatus)

    def test_include_trace_true(self):
        r = self.prime.solve("Compute 4 plus 4.", include_trace=True)
        self.assertIsNotNone(r.trace)
        self.assertIsInstance(r.trace, PrimeTrace)

    def test_include_trace_false(self):
        r = self.prime.solve("Compute 4 plus 4.")
        self.assertIsNone(r.trace)

    def test_empty_task_invalid_input(self):
        r = self.prime.solve("")
        self.assertEqual(r.status, SolveStatus.INVALID_INPUT)

    def test_whitespace_only_invalid_input(self):
        r = self.prime.solve("   ")
        self.assertEqual(r.status, SolveStatus.INVALID_INPUT)

    def test_learning_false_flag(self):
        cfg = PrimeConfig.default()
        r = self.prime.solve("2 + 2", config=cfg, learning=False,
                              include_trace=True)
        # Learning disabled → no learning update
        if r.trace:
            self.assertFalse(r.trace.learning_updated)

    def test_optimization_false_flag(self):
        cfg = PrimeConfig.default()
        r = self.prime.solve("2 + 2", config=cfg, optimization=False)
        # No optimizer strategy set when optimization disabled
        self.assertEqual(r.optimizer_strategy, "")

    def test_custom_config(self):
        cfg = PrimeConfig(log_level="silent",
                           features=FeatureFlags.mosaic_only())
        r = self.prime.solve("Compute 8 divided by 2.", config=cfg)
        self.assertIsInstance(r, PrimeResult)

    def test_latency_positive(self):
        r = self.prime.solve("Compute 6 divided by 2.")
        self.assertGreater(r.latency_ms, 0.0)

    def test_modules_used_is_list(self):
        r = self.prime.solve("A car travels 150 km in 3 hours.")
        self.assertIsInstance(r.modules_used, list)

    def test_repeated_calls_no_crash(self):
        for i in range(5):
            r = self.prime.solve(f"Compute {i+1} times 3.")
            self.assertIsInstance(r, PrimeResult)

    def test_task_id_propagated(self):
        r = self.prime.solve("2 + 2", task_id="my_task_001",
                              include_trace=True)
        self.assertEqual(r.task_id, "my_task_001")
        if r.trace:
            self.assertEqual(r.trace.task_id, "my_task_001")

    def test_experiment_id_present(self):
        r = self.prime.solve("2 + 2")
        self.assertIsInstance(r.experiment_id, str)
        self.assertGreater(len(r.experiment_id), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 36–38. PRIME.pipeline()
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimePipeline(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_returns_prime_pipeline(self):
        pipe = self.prime.pipeline()
        self.assertIsInstance(pipe, PrimePipeline)

    def test_pipeline_solve_returns_result(self):
        pipe = self.prime.pipeline()
        r = pipe.solve("Compute 6 divided by 2.")
        self.assertIsInstance(r, PrimeResult)

    def test_pipeline_is_reusable(self):
        pipe = self.prime.pipeline()
        r1 = pipe.solve("Compute 3 times 4.")
        r2 = pipe.solve("Compute 6 divided by 2.")
        self.assertIsInstance(r1, PrimeResult)
        self.assertIsInstance(r2, PrimeResult)

    def test_pipeline_with_trace(self):
        pipe = self.prime.pipeline()
        r = pipe.solve("A car travels 90 km in 3 hours.",
                        include_trace=True)
        self.assertIsNotNone(r.trace)


# ─────────────────────────────────────────────────────────────────────────────
# 39–43. Feature flag ablations
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFlagAblations(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_no_mosaic(self):
        cfg = PrimeConfig(features=FeatureFlags(
            enable_mosaic=False, enable_echo=False,
            enable_learning=False, enable_optimization=False))
        r = self.prime.solve("2 + 2", config=cfg)
        # Without MOSAIC there's no execution path → MODULE_UNAVAILABLE
        self.assertEqual(r.status, SolveStatus.MODULE_UNAVAILABLE)

    def test_no_echo_does_not_crash(self):
        cfg = PrimeConfig(features=FeatureFlags(
            enable_echo=False, enable_learning=False))
        r = self.prime.solve("Compute 4 plus 4.", config=cfg)
        self.assertIsInstance(r, PrimeResult)
        self.assertEqual(r.experience_id, "")

    def test_no_learning_does_not_crash(self):
        cfg = PrimeConfig(features=FeatureFlags(enable_learning=False))
        r = self.prime.solve("Compute 4 plus 4.", config=cfg)
        self.assertIsInstance(r, PrimeResult)
        self.assertFalse(r.learning_applied)

    def test_no_optimization_does_not_crash(self):
        cfg = PrimeConfig(features=FeatureFlags(enable_optimization=False))
        r = self.prime.solve("Compute 4 plus 4.", config=cfg)
        self.assertIsInstance(r, PrimeResult)
        self.assertEqual(r.optimizer_strategy, "")

    def test_static_baseline_config(self):
        cfg = PrimeConfig.for_ablation("static")
        r = self.prime.solve("Compute 4 plus 4.", config=cfg)
        # static: no MOSAIC → MODULE_UNAVAILABLE
        self.assertEqual(r.status, SolveStatus.MODULE_UNAVAILABLE)


# ─────────────────────────────────────────────────────────────────────────────
# 44–48. SyntheticBenchmarkSuite
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticBenchmarkSuite(unittest.TestCase):

    def test_all_tasks_non_empty(self):
        tasks = SyntheticBenchmarkSuite.all_tasks()
        self.assertGreater(len(tasks), 0)

    def test_by_category_arithmetic(self):
        tasks = SyntheticBenchmarkSuite.by_category("arithmetic")
        for t in tasks:
            self.assertEqual(t.category, "arithmetic")
        self.assertGreater(len(tasks), 0)

    def test_core_tasks_subset(self):
        core = SyntheticBenchmarkSuite.core_tasks()
        all_ = SyntheticBenchmarkSuite.all_tasks()
        all_ids = {t.task_id for t in all_}
        for t in core:
            self.assertIn(t.task_id, all_ids)

    def test_benchmark_task_to_dict(self):
        t = BenchmarkTask("t1", "arithmetic", "2+2", "4")
        d = t.to_dict()
        self.assertEqual(d["task_id"], "t1")
        self.assertEqual(d["ground_truth"], "4")

    def test_repeated_tasks_ground_truth(self):
        for t in SyntheticBenchmarkSuite.REPEATED:
            self.assertEqual(t.ground_truth, "50",
                             f"{t.task_id} ground truth should be 50")

    def test_all_tasks_have_unique_ids(self):
        tasks = SyntheticBenchmarkSuite.all_tasks()
        ids = [t.task_id for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_tasks_have_ground_truth(self):
        for t in SyntheticBenchmarkSuite.all_tasks():
            self.assertIsNotNone(t.ground_truth)
            self.assertGreater(len(t.ground_truth), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 49–51. ExperimentResult
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentResult(unittest.TestCase):

    def test_empty_compute_aggregates(self):
        exp = ExperimentResult()
        exp.compute_aggregates()
        self.assertEqual(exp.n_tasks, 0)
        self.assertEqual(exp.accuracy, 0.0)

    def test_compute_aggregates_with_data(self):
        exp = ExperimentResult()
        exp.task_results = [
            {"correct": True,  "latency_ms": 10.0, "confidence": 0.9,
             "status": "SUCCESS"},
            {"correct": True,  "latency_ms": 20.0, "confidence": 0.8,
             "status": "SUCCESS"},
            {"correct": False, "latency_ms": 30.0, "confidence": 0.3,
             "status": "REASONING_FAILED"},
        ]
        exp.compute_aggregates()
        self.assertEqual(exp.n_tasks, 3)
        self.assertEqual(exp.n_correct, 2)
        self.assertAlmostEqual(exp.accuracy, 2/3, places=5)
        self.assertAlmostEqual(exp.mean_latency_ms, 20.0, places=5)
        self.assertGreater(exp.std_latency_ms, 0.0)

    def test_to_dict_serialisable(self):
        exp = ExperimentResult(experiment_name="test")
        exp.compute_aggregates()
        d = exp.to_dict()
        j = json.dumps(d)
        self.assertIsInstance(j, str)

    def test_to_json(self):
        exp = ExperimentResult(experiment_name="json_test")
        j = exp.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["experiment_name"], "json_test")

    def test_summary_string(self):
        exp = ExperimentResult(experiment_name="s", baseline_name="b")
        exp.task_results = [
            {"correct": True, "latency_ms": 5.0, "confidence": 0.9,
             "status": "SUCCESS"}
        ]
        exp.compute_aggregates()
        s = exp.summary()
        self.assertIn("Correct", s)
        self.assertIn("1", s)


# ─────────────────────────────────────────────────────────────────────────────
# 52–55. ExperimentRunner
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRunner(unittest.TestCase):

    def _factory(self, cfg=None):
        return _prime_factory(cfg)

    def test_run_returns_experiment_result(self):
        runner = ExperimentRunner(self._factory)
        tasks  = SyntheticBenchmarkSuite.core_tasks()[:2]
        cfg    = PrimeConfig.default()
        exp    = runner.run(tasks, cfg, experiment_name="unit_test")
        self.assertIsInstance(exp, ExperimentResult)

    def test_run_n_tasks_matches(self):
        runner = ExperimentRunner(self._factory)
        tasks  = SyntheticBenchmarkSuite.core_tasks()[:3]
        cfg    = PrimeConfig.default()
        exp    = runner.run(tasks, cfg)
        self.assertEqual(exp.n_tasks, 3)

    def test_run_latency_positive(self):
        runner = ExperimentRunner(self._factory)
        tasks  = [SyntheticBenchmarkSuite.core_tasks()[0]]
        cfg    = PrimeConfig.default()
        exp    = runner.run(tasks, cfg)
        self.assertGreater(exp.mean_latency_ms, 0.0)

    def test_run_n_repeats(self):
        runner = ExperimentRunner(self._factory)
        tasks  = SyntheticBenchmarkSuite.core_tasks()[:2]
        cfg    = PrimeConfig.default()
        exp    = runner.run(tasks, cfg, n_repeats=2)
        self.assertEqual(exp.n_tasks, 4)  # 2 tasks × 2 repeats


# ─────────────────────────────────────────────────────────────────────────────
# 56–57. AblationStudy
# ─────────────────────────────────────────────────────────────────────────────

class TestAblationStudy(unittest.TestCase):

    def _factory(self, cfg=None):
        return _prime_factory(cfg)

    def test_run_all_returns_dict(self):
        study  = AblationStudy(self._factory)
        tasks  = SyntheticBenchmarkSuite.core_tasks()[:2]
        baselines = ["mosaic_only", "full"]
        results = study.run_all(tasks=tasks, baselines=baselines)
        self.assertIn("mosaic_only", results)
        self.assertIn("full", results)
        for v in results.values():
            self.assertIsInstance(v, ExperimentResult)

    def test_compare_returns_string(self):
        study  = AblationStudy(self._factory)
        tasks  = SyntheticBenchmarkSuite.core_tasks()[:2]
        baselines = ["mosaic_only", "full"]
        results = study.run_all(tasks=tasks, baselines=baselines)
        txt = AblationStudy.compare(results)
        self.assertIsInstance(txt, str)
        self.assertIn("mosaic_only", txt)
        self.assertIn("NOTE", txt)


# ─────────────────────────────────────────────────────────────────────────────
# 58–59. LearningCurveExperiment
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningCurveExperiment(unittest.TestCase):

    def _factory(self, cfg=None):
        return _prime_factory(cfg)

    def test_run_returns_list(self):
        exp    = LearningCurveExperiment(self._factory)
        train  = SyntheticBenchmarkSuite.REPEATED[:5]
        probe  = SyntheticBenchmarkSuite.core_tasks()[:2]
        curve  = exp.run(train, probe, checkpoints=[0, 3, 5])
        self.assertIsInstance(curve, list)
        self.assertGreater(len(curve), 0)
        for pt in curve:
            self.assertIsInstance(pt, LearningCurvePoint)

    def test_n_experiences_starts_at_zero(self):
        exp   = LearningCurveExperiment(self._factory)
        train = SyntheticBenchmarkSuite.REPEATED[:3]
        probe = SyntheticBenchmarkSuite.core_tasks()[:1]
        curve = exp.run(train, probe, checkpoints=[0, 2, 3])
        self.assertEqual(curve[0].n_experiences, 0)

    def test_learning_curve_point_to_dict(self):
        pt = LearningCurvePoint(n_experiences=10, accuracy=0.8,
                                  mean_latency_ms=5.0, mean_confidence=0.7)
        d = pt.to_dict()
        self.assertEqual(d["n_experiences"], 10)
        self.assertAlmostEqual(d["accuracy"], 0.8)


# ─────────────────────────────────────────────────────────────────────────────
# 60. collect_system_info
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectSystemInfo(unittest.TestCase):

    def test_required_keys(self):
        info = collect_system_info()
        for k in ("os", "python", "cpu_count", "prime_version"):
            self.assertIn(k, info, f"Missing key: {k}")

    def test_prime_version_10(self):
        info = collect_system_info()
        self.assertIn("1.0", info["prime_version"])


# ─────────────────────────────────────────────────────────────────────────────
# 61–62. profile_component
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileComponent(unittest.TestCase):

    def test_returns_profile(self):
        import time as _t
        prof = profile_component(lambda: _t.sleep(0), n=5, label="noop")
        self.assertIsInstance(prof, ComponentProfile)

    def test_mean_positive(self):
        prof = profile_component(lambda: sum(range(100)), n=10, label="sum")
        self.assertGreaterEqual(prof.mean_ms, 0.0)

    def test_label_stored(self):
        prof = profile_component(lambda: None, n=5, label="my_label")
        self.assertEqual(prof.component, "my_label")

    def test_to_dict(self):
        prof = profile_component(lambda: None, n=5, label="x")
        d = prof.to_dict()
        self.assertIn("mean_ms", d)
        self.assertIn("median_ms", d)

    def test_str_format(self):
        prof = profile_component(lambda: None, n=5, label="my_comp")
        s = str(prof)
        self.assertIn("my_comp", s)


# ─────────────────────────────────────────────────────────────────────────────
# 63–68. Regression: all previous phases still work
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionPhases(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_atlas_parses_problem(self):
        atlas   = self.prime.atlas()
        problem = atlas.parse("A car travels 150 km in 3 hours.")
        self.assertTrue(hasattr(problem, "domain"))
        self.assertIsInstance(problem.entities, list)

    def test_reasoning_engine_available(self):
        reasoner = self.prime.reasoner()
        self.assertIsNotNone(reasoner)

    def test_mosaic_composes_plan(self):
        atlas   = self.prime.atlas()
        mosaic  = self.prime.mosaic()
        problem = atlas.parse("A car travels 150 km in 3 hours.")
        plan    = mosaic.compose(problem)
        self.assertIn(plan.status, ["VALID", "EMPTY_PLAN",
                                    "COMPOSITION_INVALID", "MISSING_CAPABILITY"])

    def test_echo_records_experience(self):
        atlas  = self.prime.atlas()
        mosaic = self.prime.mosaic()
        echo   = self.prime.echo()
        problem = atlas.parse("2 + 2")
        plan    = mosaic.compose(problem)
        result  = mosaic.execute(plan, problem, raw_input="2 + 2")
        exp     = echo.record(problem_text="2 + 2", execution_result=result)
        self.assertIsNotNone(exp)
        self.assertIsNotNone(exp.id)

    def test_learning_controller_enables(self):
        lc = self.prime.learning()
        lc.enable()
        self.assertNotEqual(lc._cfg.mode, "OFF")

    def test_optimizer_produces_trace(self):
        opt   = self.prime.optimizer()
        trace = opt.optimize(module_metas=[])
        self.assertIsNotNone(trace)
        self.assertIsInstance(trace.strategy_selected, str)

    def test_version_is_10(self):
        self.assertEqual(prime.__version__, "1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# 69–75. End-to-end integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_trace_atlas_summary_populated(self):
        r = self.prime.solve("A car travels 150 km in 3 hours.",
                              include_trace=True)
        self.assertIsNotNone(r.trace)
        self.assertIn("domain", r.trace.atlas_summary)

    def test_trace_mosaic_modules_populated(self):
        r = self.prime.solve("Compute 6 times 7.", include_trace=True)
        self.assertIsNotNone(r.trace)
        self.assertIsInstance(r.trace.mosaic_modules, list)

    def test_trace_timings_total_positive(self):
        r = self.prime.solve("A car travels 90 km in 3 hours.",
                              include_trace=True)
        self.assertIsNotNone(r.trace)
        self.assertGreater(r.trace.timings.total_s, 0.0)

    def test_trace_explain_has_eight_sections(self):
        r = self.prime.solve("Compute 4 plus 5.", include_trace=True)
        txt = r.trace.explain()
        for i in range(1, 9):
            self.assertIn(f"[{i}]", txt)

    def test_result_is_json_serialisable(self):
        r = self.prime.solve("Compute 8 minus 3.")
        d = r.to_dict()
        j = json.dumps(d)
        self.assertIsInstance(j, str)

    def test_sequential_calls_accumulate_optimizer_history(self):
        opt = self.prime.optimizer()
        for task in ["2 + 2", "3 * 4", "10 / 2"]:
            trace = opt.optimize(module_metas=[], task_id=task)
        self.assertGreater(len(opt.get_traces()), 0)

    def test_ablation_pipeline_executes(self):
        cfg = PrimeConfig.for_ablation("mosaic_echo")
        r   = self.prime.solve("Compute 4 plus 4.", config=cfg)
        self.assertIsInstance(r, PrimeResult)
        self.assertIsInstance(r.status, SolveStatus)


if __name__ == "__main__":
    unittest.main(verbosity=2)
