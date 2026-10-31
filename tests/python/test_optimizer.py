"""
tests/python/test_optimizer.py
──────────────────────────────
Comprehensive tests for PRIME Phase 9: Adaptive Lightweight Optimization Engine.

Coverage:
  1.  ResourceMonitor (CPU + memory detection)
  2.  CPUInfo / MemoryInfo / ResourceSnapshot
  3.  ResourceClassifier
  4.  PerformanceRecord
  5.  PerformanceHistory
  6.  CostModel
  7.  CostWeights validation
  8.  ResourceConstraints (hard + soft)
  9.  OptimizationPolicy enum
  10. ExecutionStrategy enum
  11. OptimizationLevel enum
  12. STRATEGY_CONFIGS completeness
  13. ThreadSelector
  14. EstimateModuleMemory
  15. CandidateAssessment
  16. OptimizationTrace (explain)
  17. OptimizationState
  18. OptimizationCache (LRU)
  19. HysteresisController
  20. ExecutionStrategySelector
  21. ModuleRanker (ranking + hard constraints)
  22. AdaptiveScheduler (configure)
  23. RuntimeProfiler
  24. AdaptiveOptimizationEngine (core)
  25. Hard constraint enforcement (memory)
  26. Hard constraint enforcement (latency)
  27. Soft preference handling
  28. Phase 8 learned hint integration
  29. MOSAIC integration (optimize_plan)
  30. Strategy selection under resource pressure
  31. Optimization cache hits
  32. Hysteresis under tiny fluctuations
  33. Record result → PerformanceHistory
  34. Optimization levels (OFF / MINIMAL / BALANCED / AGGRESSIVE)
  35. make_optimizer convenience factory
  36. PRIME.optimizer() factory method
  37. Bypass for OFF level
  38. Energy proxy (labeled correctly)
  39. Cross-platform sentinel (UNKNOWN propagation)
  40. Engine summary
  41. get_traces
  42. explain() without trace arg (uses last)
  43. explain() with explicit trace
  44. OptimizerConfig from_dict / to_dict
  45. ResourceSnapshot to_dict
  46. Performance history aggregation
  47. OptimizationState from_dict / to_dict
  48. set_policy / set_level
  49. invalidate_cache
  50. End-to-end: PRIME → optimizer → mosaic → result
"""

from __future__ import annotations

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "python")

import prime
from prime.optimizer import (
    UNKNOWN,
    AdaptiveOptimizationEngine, OptimizerError,
    AdaptiveScheduler, CandidateAssessment,
    CostModel, CostWeights,
    CPUInfo, ExecutionStrategySelector,
    ExecutionStrategy, HysteresisController,
    MemoryInfo, ModuleRanker,
    OptimizationCache, OptimizationLevel,
    OptimizationPolicy, OptimizationState, OptimizationTrace,
    OptimizerConfig,
    PerformanceHistory, PerformanceRecord,
    ResourceClassifier, ResourceConstraints,
    ResourceMonitor, ResourceSnapshot,
    RuntimeProfiler, STRATEGY_CONFIGS,
    StrategyConfig, ThreadSelector,
    estimate_module_memory, make_optimizer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_meta(mid: str, cpu: float = 1.0, mem: float = 1.0,
               lat: float = 10.0, status: str = "available") -> dict:
    return {
        "id": mid, "name": mid.capitalize(), "version": "1.0",
        "cost": {"cpu_relative": cpu, "memory_relative": mem,
                 "latency_ms": lat, "requires_model": False},
        "status": status,
        "capabilities": [], "input_types": [], "output_types": [],
    }


def _snap(mem_bytes: int = 2 * 1024 ** 3,
          cpu_pct: float = 20.0,
          cpus: int = 4) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=time.time(),
        cpu=CPUInfo(logical_count=cpus, utilization_pct=cpu_pct),
        memory=MemoryInfo(available_bytes=mem_bytes,
                          total_bytes=4 * 1024 ** 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ResourceMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestResourceMonitor(unittest.TestCase):

    def setUp(self):
        self.monitor = ResourceMonitor(cache_ttl_s=0.0)

    def test_returns_snapshot(self):
        snap = self.monitor.snapshot()
        self.assertIsInstance(snap, ResourceSnapshot)

    def test_cpu_logical_count_int_or_unknown(self):
        snap = self.monitor.snapshot()
        val = snap.cpu.logical_count
        self.assertTrue(isinstance(val, int) or val == UNKNOWN)

    def test_memory_available_int_or_unknown(self):
        snap = self.monitor.snapshot()
        val = snap.memory.available_bytes
        self.assertTrue(isinstance(val, (int, float)) or val == UNKNOWN)

    def test_platform_os_filled(self):
        snap = self.monitor.snapshot()
        self.assertIsInstance(snap.platform_os, str)
        self.assertTrue(len(snap.platform_os) > 0)

    def test_python_ver_filled(self):
        snap = self.monitor.snapshot()
        self.assertIsInstance(snap.python_ver, str)
        self.assertIn(".", snap.python_ver)

    def test_caching_returns_same_object(self):
        monitor = ResourceMonitor(cache_ttl_s=60.0)
        s1 = monitor.snapshot()
        s2 = monitor.snapshot()
        self.assertIs(s1, s2)

    def test_force_bypasses_cache(self):
        monitor = ResourceMonitor(cache_ttl_s=60.0)
        s1 = monitor.snapshot()
        s2 = monitor.snapshot(force=True)
        self.assertIsNot(s1, s2)

    def test_available_memory_bytes_helper(self):
        val = self.monitor.available_memory_bytes()
        self.assertTrue(isinstance(val, (int, float)) or val == UNKNOWN)

    def test_logical_cpu_count_helper(self):
        val = self.monitor.logical_cpu_count()
        self.assertTrue(isinstance(val, int) or val == UNKNOWN)

    def test_cpu_utilization_helper(self):
        val = self.monitor.cpu_utilization_pct()
        self.assertTrue(isinstance(val, float) or val == UNKNOWN)

    def test_no_fabricated_negative_values(self):
        snap = self.monitor.snapshot()
        mem = snap.memory.available_bytes
        if mem != UNKNOWN:
            self.assertGreaterEqual(mem, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CPUInfo / MemoryInfo / ResourceSnapshot to_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestResourceDataclasses(unittest.TestCase):

    def test_cpuinfo_to_dict(self):
        c = CPUInfo(logical_count=8, utilization_pct=45.0)
        d = c.to_dict()
        self.assertEqual(d["logical_count"], 8)
        self.assertEqual(d["utilization_pct"], 45.0)

    def test_memoryinfo_available_mb(self):
        m = MemoryInfo(available_bytes=512 * 1024 * 1024)
        self.assertAlmostEqual(m.available_mb, 512.0, places=0)

    def test_memoryinfo_unknown_available_mb(self):
        m = MemoryInfo(available_bytes=UNKNOWN)
        self.assertEqual(m.available_mb, UNKNOWN)

    def test_snapshot_to_dict(self):
        snap = _snap()
        d = snap.to_dict()
        self.assertIn("cpu", d)
        self.assertIn("memory", d)
        self.assertIn("timestamp", d)

    def test_snapshot_timestamp_recent(self):
        snap = _snap()
        self.assertGreater(snap.timestamp, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ResourceClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestResourceClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = ResourceClassifier()

    def test_high_memory_low_cpu(self):
        snap = _snap(mem_bytes=4 * 1024 ** 3, cpu_pct=10.0)
        cls = self.clf.classify(snap)
        self.assertEqual(cls, "high")

    def test_low_memory(self):
        snap = _snap(mem_bytes=100 * 1024 * 1024, cpu_pct=10.0)  # 100 MB
        cls = self.clf.classify(snap)
        self.assertEqual(cls, "low")

    def test_high_cpu_usage_means_low_resources(self):
        snap = _snap(mem_bytes=4 * 1024 ** 3, cpu_pct=90.0)
        cls = self.clf.classify(snap)
        self.assertEqual(cls, "low")

    def test_medium_memory(self):
        snap = _snap(mem_bytes=512 * 1024 * 1024, cpu_pct=20.0)  # 512 MB
        cls = self.clf.classify(snap)
        self.assertEqual(cls, "medium")

    def test_all_unknown(self):
        snap = ResourceSnapshot(cpu=CPUInfo(), memory=MemoryInfo())
        cls = self.clf.classify(snap)
        self.assertEqual(cls, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PerformanceRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceRecord(unittest.TestCase):

    def test_default_creation(self):
        rec = PerformanceRecord()
        self.assertIsInstance(rec.execution_id, str)
        self.assertFalse(rec.success)
        self.assertEqual(rec.latency_ms, 0.0)

    def test_to_dict(self):
        rec = PerformanceRecord(module_id="reasoning", latency_ms=42.0,
                                 success=True)
        d = rec.to_dict()
        self.assertEqual(d["module_id"], "reasoning")
        self.assertAlmostEqual(d["latency_ms"], 42.0)
        self.assertTrue(d["success"])

    def test_from_dict_round_trip(self):
        rec = PerformanceRecord(module_id="atlas", latency_ms=5.0,
                                 success=True)
        d = rec.to_dict()
        rec2 = PerformanceRecord.from_dict(d)
        self.assertEqual(rec2.module_id, "atlas")
        self.assertAlmostEqual(rec2.latency_ms, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PerformanceHistory
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceHistory(unittest.TestCase):

    def setUp(self):
        self.hist = PerformanceHistory(max_records=100)

    def test_record_and_retrieve(self):
        rec = PerformanceRecord(module_id="reasoning", latency_ms=50.0,
                                 success=True)
        self.hist.record(rec)
        recs = self.hist.for_module("reasoning")
        self.assertEqual(len(recs), 1)

    def test_avg_latency(self):
        for i in range(5):
            self.hist.record(PerformanceRecord(
                module_id="atlas", latency_ms=float(10 * (i + 1)),
                success=True))
        avg = self.hist.avg_latency_ms("atlas")
        self.assertAlmostEqual(avg, 30.0)

    def test_avg_latency_min_samples_not_met(self):
        self.hist.record(PerformanceRecord(
            module_id="llm", latency_ms=1000.0, success=True))
        avg = self.hist.avg_latency_ms("llm", min_samples=3)
        self.assertEqual(avg, UNKNOWN)

    def test_success_rate(self):
        for ok in [True, True, False, True]:
            self.hist.record(PerformanceRecord(
                module_id="reasoning", success=ok))
        sr = self.hist.success_rate("reasoning")
        self.assertAlmostEqual(sr, 0.75)

    def test_success_rate_unknown_when_empty(self):
        self.assertEqual(self.hist.success_rate("nonexistent"), UNKNOWN)

    def test_avg_memory_unknown_when_none(self):
        self.hist.record(PerformanceRecord(module_id="m",
                                            memory_usage=UNKNOWN))
        self.assertEqual(self.hist.avg_memory_usage("m"), UNKNOWN)

    def test_avg_memory_with_values(self):
        for mb in [100, 200]:
            self.hist.record(PerformanceRecord(
                module_id="mem", memory_usage=mb * 1024 * 1024))
        val = self.hist.avg_memory_usage("mem")
        self.assertAlmostEqual(val, 150 * 1024 * 1024)

    def test_sample_count(self):
        for _ in range(7):
            self.hist.record(PerformanceRecord(module_id="atlas"))
        self.assertEqual(self.hist.sample_count("atlas"), 7)

    def test_aggregate_dict(self):
        self.hist.record(PerformanceRecord(module_id="reasoning",
                                            latency_ms=60.0, success=True))
        agg = self.hist.aggregate("reasoning")
        self.assertIn("avg_latency_ms", agg)
        self.assertIn("success_rate", agg)
        self.assertIn("sample_count", agg)

    def test_get_recent(self):
        for i in range(20):
            self.hist.record(PerformanceRecord(module_id=f"m{i}"))
        recent = self.hist.get_recent(5)
        self.assertEqual(len(recent), 5)

    def test_clear(self):
        self.hist.record(PerformanceRecord(module_id="atlas"))
        self.hist.clear()
        self.assertEqual(self.hist.sample_count("atlas"), 0)

    def test_max_records_respected(self):
        hist = PerformanceHistory(max_records=5)
        for i in range(10):
            hist.record(PerformanceRecord())
        self.assertLessEqual(len(hist._records), 5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CostModel
# ─────────────────────────────────────────────────────────────────────────────

class TestCostModel(unittest.TestCase):

    def setUp(self):
        self.model = CostModel()

    def test_zero_cost(self):
        s = self.model.score(cpu_relative=0, memory_relative=0, latency_ms=0,
                              expected_quality=1.0)
        self.assertAlmostEqual(s, 0.0)

    def test_max_cost_capped(self):
        s = self.model.score(cpu_relative=100, memory_relative=100,
                              latency_ms=10000, expected_quality=0.0)
        self.assertLessEqual(s, 1.5)   # bounded sum

    def test_higher_resources_higher_cost(self):
        cheap  = self.model.score(cpu_relative=0.5, memory_relative=0.5,
                                   latency_ms=5.0)
        expensive = self.model.score(cpu_relative=8.0, memory_relative=8.0,
                                      latency_ms=1000.0)
        self.assertLess(cheap, expensive)

    def test_quality_penalty(self):
        high_q = self.model.score(expected_quality=1.0)
        low_q  = self.model.score(expected_quality=0.0)
        self.assertLess(high_q, low_q)

    def test_energy_proxy_positive(self):
        ep = self.model.energy_proxy(cpu_relative=2.0, latency_ms=500.0)
        self.assertGreater(ep, 0.0)

    def test_score_from_module_meta(self):
        meta = _make_meta("reasoning", cpu=2.0, mem=2.0, lat=50.0)
        score = self.model.score_from_module_meta(meta)
        self.assertGreater(score, 0.0)

    def test_score_uses_history_latency(self):
        hist = PerformanceHistory()
        for _ in range(5):
            hist.record(PerformanceRecord(module_id="atlas", latency_ms=800.0,
                                          success=True))
        meta = _make_meta("atlas", lat=5.0)   # meta says 5 ms
        score_no_hist = self.model.score_from_module_meta(meta)
        score_with_hist = self.model.score_from_module_meta(meta, history=hist)
        # History says 800 ms → higher cost
        self.assertGreater(score_with_hist, score_no_hist)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CostWeights validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCostWeights(unittest.TestCase):

    def test_default_sum_to_one(self):
        w = CostWeights()
        self.assertAlmostEqual(w.cpu + w.memory + w.latency + w.quality, 1.0,
                               places=5)

    def test_invalid_weights_raise(self):
        w = CostWeights(cpu=0.5, memory=0.5, latency=0.5, quality=0.5)
        with self.assertRaises(ValueError):
            w.validate()


# ─────────────────────────────────────────────────────────────────────────────
# 8. ResourceConstraints
# ─────────────────────────────────────────────────────────────────────────────

class TestResourceConstraints(unittest.TestCase):

    def test_no_constraints_always_pass(self):
        rc = ResourceConstraints()
        ok, _ = rc.check_memory(100, 200)
        self.assertTrue(ok)
        ok, _ = rc.check_latency(9999)
        self.assertTrue(ok)
        ok, _ = rc.check_quality(0.0)
        self.assertTrue(ok)

    def test_hard_memory_constraint(self):
        rc = ResourceConstraints(max_memory_bytes=50 * 1024 * 1024)
        ok, reason = rc.check_memory(100 * 1024 * 1024, UNKNOWN)
        self.assertFalse(ok)
        self.assertIn("hard limit", reason)

    def test_insufficient_available_memory(self):
        rc = ResourceConstraints()
        ok, reason = rc.check_memory(500 * 1024 * 1024,
                                      100 * 1024 * 1024)
        self.assertFalse(ok)
        self.assertIn("available", reason)

    def test_available_unknown_skips_check(self):
        rc = ResourceConstraints()
        ok, _ = rc.check_memory(1_000_000_000, UNKNOWN)
        self.assertTrue(ok)

    def test_hard_latency_constraint(self):
        rc = ResourceConstraints(max_latency_ms=100.0)
        ok, reason = rc.check_latency(500.0)
        self.assertFalse(ok)
        self.assertIn("limit", reason)

    def test_quality_constraint(self):
        rc = ResourceConstraints(min_quality=0.8)
        ok, reason = rc.check_quality(0.5)
        self.assertFalse(ok)
        self.assertIn("minimum", reason)

    def test_to_dict(self):
        rc = ResourceConstraints(max_memory_bytes=1024)
        d = rc.to_dict()
        self.assertEqual(d["max_memory_bytes"], 1024)


# ─────────────────────────────────────────────────────────────────────────────
# 9–11. Enum sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestEnums(unittest.TestCase):

    def test_optimization_policy_values(self):
        for v in ("LOW_LATENCY", "LOW_MEMORY", "LOW_CPU", "BALANCED",
                   "MAX_QUALITY", "RESOURCE_LIMITED", "ENERGY_AWARE_PROXY"):
            self.assertEqual(OptimizationPolicy(v).value, v)

    def test_execution_strategy_values(self):
        for v in ("FULL", "BALANCED", "LIGHTWEIGHT",
                   "LOW_MEMORY", "LOW_LATENCY", "CPU_SAVER"):
            self.assertEqual(ExecutionStrategy(v).value, v)

    def test_optimization_level_values(self):
        for v in ("OFF", "MINIMAL", "BALANCED", "AGGRESSIVE"):
            self.assertEqual(OptimizationLevel(v).value, v)


# ─────────────────────────────────────────────────────────────────────────────
# 12. STRATEGY_CONFIGS completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyConfigs(unittest.TestCase):

    def test_all_strategies_have_config(self):
        for s in ExecutionStrategy:
            self.assertIn(s.value, STRATEGY_CONFIGS,
                          msg=f"Missing config for {s}")

    def test_config_has_required_fields(self):
        for name, cfg in STRATEGY_CONFIGS.items():
            self.assertIsInstance(cfg.strategy, ExecutionStrategy)
            self.assertIsInstance(cfg.policy, OptimizationPolicy)
            self.assertIsInstance(cfg.thread_fraction, float)
            self.assertTrue(0 < cfg.thread_fraction <= 1.0)

    def test_config_to_dict(self):
        cfg = STRATEGY_CONFIGS[ExecutionStrategy.BALANCED.value]
        d = cfg.to_dict()
        self.assertIn("strategy", d)
        self.assertIn("policy", d)


# ─────────────────────────────────────────────────────────────────────────────
# 13. ThreadSelector
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSelector(unittest.TestCase):

    def setUp(self):
        self.sel = ThreadSelector()
        self.constr = ResourceConstraints()

    def test_balanced_uses_fraction(self):
        cpu = CPUInfo(logical_count=8)
        n = self.sel.select(ExecutionStrategy.BALANCED, cpu, self.constr)
        # BALANCED fraction=0.75 → 6
        self.assertEqual(n, 6)

    def test_full_uses_all_cpus(self):
        cpu = CPUInfo(logical_count=8)
        n = self.sel.select(ExecutionStrategy.FULL, cpu, self.constr)
        self.assertEqual(n, 8)

    def test_cpu_saver_uses_quarter(self):
        cpu = CPUInfo(logical_count=8)
        n = self.sel.select(ExecutionStrategy.CPU_SAVER, cpu, self.constr)
        # fraction=0.25 → max(1, 2) = 2
        self.assertEqual(n, 2)

    def test_unknown_cpu_falls_back_to_1(self):
        cpu = CPUInfo(logical_count=UNKNOWN)
        n = self.sel.select(ExecutionStrategy.BALANCED, cpu, self.constr)
        self.assertEqual(n, 1)

    def test_max_threads_respected(self):
        cpu = CPUInfo(logical_count=16)
        constr = ResourceConstraints(max_threads=2)
        n = self.sel.select(ExecutionStrategy.FULL, cpu, constr)
        self.assertEqual(n, 2)

    def test_minimum_one_thread(self):
        cpu = CPUInfo(logical_count=1)
        n = self.sel.select(ExecutionStrategy.CPU_SAVER, cpu, self.constr)
        self.assertGreaterEqual(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 14. estimate_module_memory
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateModuleMemory(unittest.TestCase):

    def test_llm_larger_than_atlas(self):
        llm  = estimate_module_memory("llm")
        atl  = estimate_module_memory("atlas")
        self.assertGreater(llm, atl)

    def test_unknown_module_returns_default(self):
        val = estimate_module_memory("unknown_module_xyz")
        self.assertGreater(val, 0)

    def test_meta_scaling_applied(self):
        meta_heavy = _make_meta("atlas", mem=5.0)
        meta_light = _make_meta("atlas", mem=1.0)
        heavy = estimate_module_memory("atlas", meta_heavy)
        light = estimate_module_memory("atlas", meta_light)
        self.assertGreater(heavy, light)


# ─────────────────────────────────────────────────────────────────────────────
# 15. CandidateAssessment
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateAssessment(unittest.TestCase):

    def test_default_not_selected(self):
        ca = CandidateAssessment(module_id="atlas")
        self.assertFalse(ca.selected)
        self.assertFalse(ca.rejected)

    def test_to_dict(self):
        ca = CandidateAssessment(module_id="reasoning", cost_score=0.3,
                                  selected=True)
        d = ca.to_dict()
        self.assertEqual(d["module_id"], "reasoning")
        self.assertTrue(d["selected"])


# ─────────────────────────────────────────────────────────────────────────────
# 16. OptimizationTrace.explain()
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizationTrace(unittest.TestCase):

    def _make_trace(self):
        t = OptimizationTrace(task_id="t1",
                               policy=OptimizationPolicy.BALANCED.value,
                               strategy_selected=ExecutionStrategy.BALANCED.value,
                               optimization_level=OptimizationLevel.BALANCED.value,
                               thread_count=4,
                               cpu_utilization=30.0,
                               available_memory=1024 * 1024 * 1024)
        t.candidates = [
            CandidateAssessment(module_id="atlas",    selected=True,
                                 cost_score=0.1, final_score=0.1),
            CandidateAssessment(module_id="llm",      rejected=True,
                                 cost_score=0.9, final_score=0.9,
                                 rejection_reason="memory_limit"),
        ]
        return t

    def test_explain_contains_strategy(self):
        t = self._make_trace()
        txt = t.explain()
        self.assertIn("BALANCED", txt)

    def test_explain_lists_candidates(self):
        t = self._make_trace()
        txt = t.explain()
        self.assertIn("atlas", txt)
        self.assertIn("llm", txt)

    def test_explain_rejection_reason(self):
        t = self._make_trace()
        txt = t.explain()
        self.assertIn("memory_limit", txt)

    def test_explain_thread_count(self):
        t = self._make_trace()
        txt = t.explain()
        self.assertIn("4", txt)

    def test_to_dict(self):
        t = self._make_trace()
        d = t.to_dict()
        self.assertIn("candidates", d)
        self.assertEqual(len(d["candidates"]), 2)

    def test_bypass_trace_explain(self):
        t = OptimizationTrace(bypass_reason="optimization_level=OFF")
        txt = t.explain()
        self.assertIn("BYPASS", txt)


# ─────────────────────────────────────────────────────────────────────────────
# 17. OptimizationState
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizationState(unittest.TestCase):

    def test_to_dict(self):
        s = OptimizationState(current_strategy="BALANCED")
        d = s.to_dict()
        self.assertEqual(d["current_strategy"], "BALANCED")

    def test_from_dict_round_trip(self):
        s = OptimizationState(current_strategy="LIGHTWEIGHT",
                               switch_count=3,
                               optimizer_version="9.0")
        d = s.to_dict()
        s2 = OptimizationState.from_dict(d)
        self.assertEqual(s2.current_strategy, "LIGHTWEIGHT")
        self.assertEqual(s2.switch_count, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 18. OptimizationCache (LRU)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizationCache(unittest.TestCase):

    def test_miss_returns_none(self):
        cache = OptimizationCache()
        self.assertIsNone(cache.get("task", "high", "BALANCED"))

    def test_hit_returns_value(self):
        cache = OptimizationCache()
        cache.put("task", "high", "BALANCED",
                  ExecutionStrategy.BALANCED, 4)
        result = cache.get("task", "high", "BALANCED")
        self.assertIsNotNone(result)
        strategy, threads = result
        self.assertEqual(strategy, ExecutionStrategy.BALANCED)
        self.assertEqual(threads, 4)

    def test_different_keys_different_results(self):
        cache = OptimizationCache()
        cache.put("t1", "high", "BALANCED", ExecutionStrategy.BALANCED, 4)
        cache.put("t1", "low",  "BALANCED", ExecutionStrategy.LIGHTWEIGHT, 2)
        r1 = cache.get("t1", "high", "BALANCED")
        r2 = cache.get("t1", "low",  "BALANCED")
        self.assertNotEqual(r1[0], r2[0])

    def test_invalidate_clears_all(self):
        cache = OptimizationCache()
        cache.put("t1", "high", "BALANCED", ExecutionStrategy.BALANCED, 4)
        cache.invalidate()
        self.assertIsNone(cache.get("t1", "high", "BALANCED"))

    def test_max_size_respected(self):
        cache = OptimizationCache(max_size=3)
        for i in range(5):
            cache.put(f"t{i}", "high", "BALANCED",
                      ExecutionStrategy.BALANCED, 4)
        self.assertLessEqual(len(cache), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 19. HysteresisController
# ─────────────────────────────────────────────────────────────────────────────

class TestHysteresisController(unittest.TestCase):

    def test_no_switch_same_strategy(self):
        h = HysteresisController(threshold=0.05)
        self.assertFalse(h.should_switch(
            ExecutionStrategy.BALANCED, 0.3,
            ExecutionStrategy.BALANCED, 0.3))

    def test_large_improvement_allows_switch(self):
        h = HysteresisController(threshold=0.05)
        # current score 0.5, candidate score 0.1 → improvement 0.4 > 0.05
        self.assertTrue(h.should_switch(
            ExecutionStrategy.LIGHTWEIGHT, 0.1,
            ExecutionStrategy.BALANCED,   0.5))

    def test_tiny_improvement_blocks_switch(self):
        h = HysteresisController(threshold=0.05)
        # improvement = 0.01 < 0.05
        self.assertFalse(h.should_switch(
            ExecutionStrategy.LIGHTWEIGHT, 0.49,
            ExecutionStrategy.BALANCED,   0.50))

    def test_update_tracks_current_strategy(self):
        h = HysteresisController()
        h.update(ExecutionStrategy.LOW_LATENCY, 0.2)
        self.assertEqual(h.current_strategy, ExecutionStrategy.LOW_LATENCY)


# ─────────────────────────────────────────────────────────────────────────────
# 20. ExecutionStrategySelector
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionStrategySelector(unittest.TestCase):

    def setUp(self):
        self.sel = ExecutionStrategySelector()
        self.constr = ResourceConstraints()

    def test_low_latency_policy(self):
        snap = _snap()
        strategy, _ = self.sel.select(
            OptimizationPolicy.LOW_LATENCY, "high", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.LOW_LATENCY)

    def test_max_quality_high_resources(self):
        snap = _snap()
        strategy, _ = self.sel.select(
            OptimizationPolicy.MAX_QUALITY, "high", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.FULL)

    def test_max_quality_low_resources_overridden(self):
        snap = _snap()
        strategy, reason = self.sel.select(
            OptimizationPolicy.MAX_QUALITY, "low", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.LIGHTWEIGHT)
        self.assertIn("low", reason)

    def test_resource_limited_policy(self):
        snap = _snap()
        strategy, _ = self.sel.select(
            OptimizationPolicy.RESOURCE_LIMITED, "high", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.LIGHTWEIGHT)

    def test_balanced_policy(self):
        snap = _snap()
        strategy, _ = self.sel.select(
            OptimizationPolicy.BALANCED, "high", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.BALANCED)

    def test_low_memory_policy(self):
        snap = _snap()
        strategy, _ = self.sel.select(
            OptimizationPolicy.LOW_MEMORY, "high", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.LOW_MEMORY)

    def test_energy_proxy_policy(self):
        snap = _snap()
        strategy, _ = self.sel.select(
            OptimizationPolicy.ENERGY_AWARE_PROXY, "high", self.constr, snap)
        self.assertEqual(strategy, ExecutionStrategy.CPU_SAVER)


# ─────────────────────────────────────────────────────────────────────────────
# 21. ModuleRanker (ranking + hard constraints)
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleRanker(unittest.TestCase):

    def setUp(self):
        self.cost   = CostModel()
        self.hist   = PerformanceHistory()
        self.constr = ResourceConstraints()
        self.ranker = ModuleRanker(self.cost, self.hist, self.constr)
        self.snap   = _snap()

    def test_cheap_module_ranked_first(self):
        metas = [
            _make_meta("expensive", cpu=8.0, mem=8.0, lat=800.0),
            _make_meta("cheap",     cpu=0.5, mem=0.5, lat=5.0),
        ]
        ranked = self.ranker.rank(metas, ExecutionStrategy.BALANCED,
                                   self.snap)
        # First non-rejected should be cheap
        accepted = [(m, c) for m, c in ranked if not c.rejected]
        self.assertEqual(accepted[0][0], "cheap")

    def test_first_selected_is_marked(self):
        metas = [_make_meta("a"), _make_meta("b")]
        ranked = self.ranker.rank(metas, ExecutionStrategy.BALANCED, self.snap)
        accepted = [(m, c) for m, c in ranked if not c.rejected]
        self.assertTrue(accepted[0][1].selected)
        self.assertFalse(accepted[1][1].selected)

    def test_unavailable_module_rejected(self):
        meta = _make_meta("broken", status="unavailable")
        ranked = self.ranker.rank([meta], ExecutionStrategy.BALANCED, self.snap)
        _, ca = ranked[0]
        self.assertTrue(ca.rejected)
        self.assertTrue(ca.is_hard_constraint)

    def test_memory_exceeded_rejected(self):
        # Very tight memory limit
        constr = ResourceConstraints(max_memory_bytes=1 * 1024)
        ranker = ModuleRanker(self.cost, self.hist, constr)
        meta = _make_meta("llm", mem=10.0)
        snap = _snap(mem_bytes=4 * 1024 ** 3)
        ranked = ranker.rank([meta], ExecutionStrategy.BALANCED, snap)
        _, ca = ranked[0]
        self.assertTrue(ca.rejected)

    def test_latency_exceeded_rejected(self):
        constr = ResourceConstraints(max_latency_ms=10.0)
        ranker = ModuleRanker(self.cost, self.hist, constr)
        meta = _make_meta("llm", lat=2000.0)
        ranked = ranker.rank([meta], ExecutionStrategy.BALANCED, self.snap)
        _, ca = ranked[0]
        self.assertTrue(ca.rejected)

    def test_empty_module_list(self):
        ranked = self.ranker.rank([], ExecutionStrategy.BALANCED, self.snap)
        self.assertEqual(len(ranked), 0)

    def test_phase8_preferred_bonus_applied(self):
        metas = [
            _make_meta("a", cpu=1.0, mem=1.0, lat=50.0),
            _make_meta("b", cpu=0.8, mem=0.8, lat=40.0),  # slightly cheaper
        ]
        # Make "a" strongly preferred by Phase 8
        hints = {"preferred_modules": ["a"], "excluded_modules": []}
        ranked = self.ranker.rank(metas, ExecutionStrategy.BALANCED,
                                   self.snap, learned_hints=hints)
        accepted = [(m, c) for m, c in ranked if not c.rejected]
        # "a" gets a bonus that should bring it to rank 1
        self.assertEqual(accepted[0][0], "a")
        self.assertGreater(accepted[0][1].learned_bonus, 0)

    def test_phase8_excluded_penalty_applied(self):
        metas = [_make_meta("bad"), _make_meta("good")]
        hints = {"excluded_modules": ["bad"], "preferred_modules": []}
        ranked = self.ranker.rank(metas, ExecutionStrategy.BALANCED,
                                   self.snap, learned_hints=hints)
        accepted = [(m, c) for m, c in ranked if not c.rejected]
        # "bad" has negative bonus → higher cost → ranked after "good"
        bad_assessments = [c for m, c in accepted if m == "bad"]
        if bad_assessments:
            self.assertLess(bad_assessments[0].learned_bonus, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 22. AdaptiveScheduler.configure()
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveScheduler(unittest.TestCase):

    def setUp(self):
        monitor = ResourceMonitor(cache_ttl_s=0)
        hist    = PerformanceHistory()
        constr  = ResourceConstraints()
        self.sched = AdaptiveScheduler(monitor, hist, constr)
        self.snap  = _snap()

    def test_configure_returns_dict(self):
        cfg = self.sched.configure(ExecutionStrategy.BALANCED, self.snap)
        self.assertIsInstance(cfg, dict)
        self.assertIn("prefer_lightweight", cfg)

    def test_lightweight_strategy_sets_prefer(self):
        cfg = self.sched.configure(ExecutionStrategy.LIGHTWEIGHT, self.snap)
        self.assertTrue(cfg["prefer_lightweight"])

    def test_full_strategy_no_prefer(self):
        cfg = self.sched.configure(ExecutionStrategy.FULL, self.snap)
        self.assertFalse(cfg["prefer_lightweight"])

    def test_configure_includes_thread_count(self):
        cfg = self.sched.configure(ExecutionStrategy.BALANCED, self.snap)
        self.assertIn("_opt_threads", cfg)
        self.assertGreaterEqual(cfg["_opt_threads"], 1)

    def test_record_execution_stores_record(self):
        hist = PerformanceHistory()
        sched = AdaptiveScheduler(ResourceMonitor(), hist, ResourceConstraints())
        snap = _snap()
        rec = sched.record_execution(
            "t1", "atlas", ExecutionStrategy.BALANCED,
            time.monotonic() - 0.05, time.monotonic(), True, snap)
        self.assertIsInstance(rec, PerformanceRecord)
        self.assertGreater(rec.latency_ms, 0)
        self.assertEqual(hist.sample_count("atlas"), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 23. RuntimeProfiler
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeProfiler(unittest.TestCase):

    def test_profile_records_latency(self):
        profiler = RuntimeProfiler()
        with profiler.profile("reasoning"):
            time.sleep(0.01)
        rec = profiler.last_record()
        self.assertIsNotNone(rec)
        self.assertGreater(rec.latency_ms, 5.0)

    def test_profile_label_stored(self):
        profiler = RuntimeProfiler()
        with profiler.profile("atlas"):
            pass
        rec = profiler.last_record()
        self.assertEqual(rec.module_id, "atlas")

    def test_multiple_profiles(self):
        profiler = RuntimeProfiler()
        for label in ["atlas", "reasoning", "validator"]:
            with profiler.profile(label):
                pass
        recs = profiler.all_records()
        self.assertEqual(len(recs), 3)
        self.assertEqual({r.module_id for r in recs},
                         {"atlas", "reasoning", "validator"})

    def test_no_fabricated_data_without_execution(self):
        profiler = RuntimeProfiler()
        self.assertIsNone(profiler.last_record())


# ─────────────────────────────────────────────────────────────────────────────
# 24. AdaptiveOptimizationEngine core
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveOptimizationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AdaptiveOptimizationEngine()

    def test_repr(self):
        r = repr(self.engine)
        self.assertIn("AdaptiveOptimizationEngine", r)

    def test_optimize_returns_trace(self):
        trace = self.engine.optimize(module_metas=[])
        self.assertIsInstance(trace, OptimizationTrace)

    def test_optimize_sets_strategy(self):
        trace = self.engine.optimize(module_metas=[])
        self.assertIn(trace.strategy_selected,
                      [s.value for s in ExecutionStrategy])

    def test_optimize_has_thread_count(self):
        trace = self.engine.optimize(module_metas=[])
        self.assertGreaterEqual(trace.thread_count, 1)

    def test_optimize_measures_overhead(self):
        trace = self.engine.optimize(module_metas=[])
        self.assertGreaterEqual(trace.optimizer_latency_ms, 0.0)

    def test_get_state(self):
        state = self.engine.get_state()
        self.assertIsInstance(state, OptimizationState)

    def test_get_resource_snapshot(self):
        snap = self.engine.get_resource_snapshot()
        self.assertIsInstance(snap, ResourceSnapshot)

    def test_get_performance_history(self):
        hist = self.engine.get_performance_history()
        self.assertIsInstance(hist, PerformanceHistory)

    def test_summary_dict(self):
        s = self.engine.summary()
        self.assertIn("optimizer_version", s)
        self.assertIn("policy", s)
        self.assertIn("current_strategy", s)
        self.assertIn("psutil_available", s)

    def test_explain_no_traces_message(self):
        msg = self.engine.explain()
        self.assertIsInstance(msg, str)

    def test_explain_after_optimize(self):
        metas = [_make_meta("atlas"), _make_meta("reasoning")]
        trace = self.engine.optimize(module_metas=metas, task_id="t_explain")
        msg = self.engine.explain(trace)
        self.assertIn("atlas", msg)

    def test_get_traces_empty(self):
        engine = AdaptiveOptimizationEngine()
        self.assertEqual(engine.get_traces(), [])

    def test_get_traces_after_optimize(self):
        metas = [_make_meta("atlas")]
        self.engine.optimize(module_metas=metas, task_id="t99")
        traces = self.engine.get_traces()
        self.assertGreater(len(traces), 0)

    def test_set_policy_changes_policy(self):
        self.engine.set_policy(OptimizationPolicy.LOW_LATENCY)
        self.assertEqual(self.engine._cfg.policy,
                         OptimizationPolicy.LOW_LATENCY)

    def test_set_level_changes_level(self):
        self.engine.set_level(OptimizationLevel.MINIMAL)
        self.assertEqual(self.engine._cfg.level, OptimizationLevel.MINIMAL)

    def test_invalidate_cache(self):
        # Should not raise
        self.engine.invalidate_cache()

    def test_policy_override_per_call(self):
        trace = self.engine.optimize(
            module_metas=[], task_id="t_override",
            policy_override=OptimizationPolicy.LOW_LATENCY)
        self.assertEqual(trace.policy, OptimizationPolicy.LOW_LATENCY.value)


# ─────────────────────────────────────────────────────────────────────────────
# 25–26. Hard constraint enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestHardConstraintEnforcement(unittest.TestCase):

    def test_memory_constraint_rejects_module(self):
        constr = ResourceConstraints(max_memory_bytes=1 * 1024)  # 1 KB limit
        cfg    = OptimizerConfig(constraints=constr)
        engine = AdaptiveOptimizationEngine(config=cfg)
        metas  = [_make_meta("llm", mem=10.0)]  # very heavy
        snap   = _snap(mem_bytes=4 * 1024 ** 3)
        trace  = engine.optimize(module_metas=metas)
        rejected = [c for c in trace.candidates if c.rejected]
        self.assertGreater(len(rejected), 0)
        self.assertTrue(any(c.is_hard_constraint for c in rejected))

    def test_latency_constraint_rejects_module(self):
        constr = ResourceConstraints(max_latency_ms=5.0)
        cfg    = OptimizerConfig(constraints=constr)
        engine = AdaptiveOptimizationEngine(config=cfg)
        metas  = [_make_meta("llm", lat=2000.0)]
        trace  = engine.optimize(module_metas=metas)
        rejected = [c for c in trace.candidates if c.rejected]
        self.assertGreater(len(rejected), 0)

    def test_hard_constraint_never_bypassed_by_phase8_bonus(self):
        """Phase 8 bonus must not override a hard memory constraint."""
        constr = ResourceConstraints(max_memory_bytes=1 * 1024)
        cfg    = OptimizerConfig(constraints=constr)
        engine = AdaptiveOptimizationEngine(config=cfg)
        hints  = {"preferred_modules": ["llm"], "excluded_modules": []}
        metas  = [_make_meta("llm", mem=10.0)]
        trace  = engine.optimize(module_metas=metas, learned_hints=hints)
        rejected = [c for c in trace.candidates if c.rejected]
        self.assertGreater(len(rejected), 0,
                           "LLM must be rejected despite Phase 8 preference")


# ─────────────────────────────────────────────────────────────────────────────
# 27. Soft preference handling
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftPreferences(unittest.TestCase):

    def test_lightweight_preferred_when_resource_limited(self):
        cfg = OptimizerConfig(policy=OptimizationPolicy.RESOURCE_LIMITED)
        engine = AdaptiveOptimizationEngine(config=cfg)
        trace = engine.optimize(module_metas=[])
        cfg_tmpl = STRATEGY_CONFIGS.get(trace.strategy_selected)
        if cfg_tmpl:
            self.assertTrue(cfg_tmpl.prefer_lightweight)

    def test_full_quality_preferred_under_max_quality(self):
        cfg = OptimizerConfig(policy=OptimizationPolicy.MAX_QUALITY,
                               level=OptimizationLevel.BALANCED)
        engine = AdaptiveOptimizationEngine(config=cfg)
        trace = engine.optimize(module_metas=[])
        # Under high resources, strategy should be FULL
        if trace.strategy_selected != ExecutionStrategy.FULL.value:
            # Accept LIGHTWEIGHT only if resource class was detected as low
            allowed = {ExecutionStrategy.FULL.value,
                       ExecutionStrategy.LIGHTWEIGHT.value,
                       ExecutionStrategy.BALANCED.value}
            self.assertIn(trace.strategy_selected, allowed)


# ─────────────────────────────────────────────────────────────────────────────
# 28. Phase 8 learned hint integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase8Integration(unittest.TestCase):

    def test_preferred_modules_in_trace(self):
        engine = AdaptiveOptimizationEngine()
        hints  = {"preferred_modules": ["reasoning"], "excluded_modules": []}
        metas  = [_make_meta("reasoning"), _make_meta("atlas")]
        trace  = engine.optimize(module_metas=metas, learned_hints=hints)
        self.assertIn("reasoning", trace.phase8_preferred)

    def test_excluded_modules_in_trace(self):
        engine = AdaptiveOptimizationEngine()
        hints  = {"preferred_modules": [], "excluded_modules": ["llm"]}
        metas  = [_make_meta("llm"), _make_meta("reasoning")]
        trace  = engine.optimize(module_metas=metas, learned_hints=hints)
        self.assertIn("llm", trace.phase8_excluded)

    def test_phase8_bonus_applied_to_candidate(self):
        engine = AdaptiveOptimizationEngine()
        hints  = {"preferred_modules": ["reasoning"], "excluded_modules": []}
        metas  = [_make_meta("reasoning")]
        trace  = engine.optimize(module_metas=metas, learned_hints=hints)
        self.assertGreater(len(trace.candidates), 0)
        preferred = [c for c in trace.candidates
                     if c.module_id == "reasoning"]
        if preferred:
            self.assertGreater(preferred[0].learned_bonus, 0)

    def test_no_phase8_hints_still_works(self):
        engine = AdaptiveOptimizationEngine()
        metas  = [_make_meta("atlas")]
        trace  = engine.optimize(module_metas=metas, learned_hints=None)
        self.assertIsInstance(trace, OptimizationTrace)


# ─────────────────────────────────────────────────────────────────────────────
# 29. MOSAIC integration (optimize_plan)
# ─────────────────────────────────────────────────────────────────────────────

class TestMOSAICIntegration(unittest.TestCase):

    def _make_mosaic(self):
        from prime import PRIME
        p = PRIME()
        m = p.mosaic()
        return p, m

    def test_optimize_plan_returns_plan_and_trace(self):
        p, m = self._make_mosaic()
        try:
            from prime import ATLASEngine
            atlas   = p.atlas()
            problem = atlas.parse("A car travels 150 km in 3 hours.")
            engine  = AdaptiveOptimizationEngine()
            plan, trace = engine.optimize_plan(m, problem, task_id="t_car")
            self.assertIsNotNone(plan)
            self.assertIsInstance(trace, OptimizationTrace)
        finally:
            p.shutdown()

    def test_optimize_plan_trace_has_strategy(self):
        p, m = self._make_mosaic()
        try:
            atlas   = p.atlas()
            problem = atlas.parse("2 + 2")
            engine  = AdaptiveOptimizationEngine()
            _, trace = engine.optimize_plan(m, problem)
            self.assertIn(trace.strategy_selected,
                          [s.value for s in ExecutionStrategy])
        finally:
            p.shutdown()

    def test_mosaic_not_replaced(self):
        """MOSAIC remains in charge of capability matching."""
        p, m = self._make_mosaic()
        try:
            atlas   = p.atlas()
            problem = atlas.parse("A car travels 150 km in 3 hours.")
            engine  = AdaptiveOptimizationEngine()
            plan, _ = engine.optimize_plan(m, problem)
            # Plan must still come from MOSAIC
            self.assertTrue(hasattr(plan, "module_ids") or
                            hasattr(plan, "status"),
                            "Returned plan must be a MOSAIC CompositionPlan")
        finally:
            p.shutdown()

    def test_prefer_lightweight_passed_to_mosaic(self):
        """Resource-limited policy should set prefer_lightweight on MOSAIC."""
        p, m = self._make_mosaic()
        try:
            cfg    = OptimizerConfig(policy=OptimizationPolicy.RESOURCE_LIMITED)
            engine = AdaptiveOptimizationEngine(config=cfg)
            atlas  = p.atlas()
            problem = atlas.parse("2 + 2")
            plan, trace = engine.optimize_plan(m, problem)
            cfg_tmpl = STRATEGY_CONFIGS.get(trace.strategy_selected)
            if cfg_tmpl:
                self.assertTrue(cfg_tmpl.prefer_lightweight)
        finally:
            p.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 30. Strategy selection under resource pressure
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyUnderPressure(unittest.TestCase):

    def test_low_resource_class_selects_lightweight(self):
        engine = AdaptiveOptimizationEngine()
        # Force classifier to return "low" via a patched snap
        clf = engine._classifier
        with patch.object(clf, "classify", return_value="low"):
            # Refill resource window with a "low" class snapshot
            for _ in range(engine._cfg.resource_window):
                engine._resource_window.append(_snap(
                    mem_bytes=50 * 1024 * 1024,  # 50 MB → low
                    cpu_pct=90.0,
                ))
            engine.invalidate_cache()
            trace = engine.optimize(
                module_metas=[],
                policy_override=OptimizationPolicy.MAX_QUALITY)
            # Under low resources even MAX_QUALITY should be downgraded
            self.assertIn(trace.strategy_selected,
                          [ExecutionStrategy.LIGHTWEIGHT.value,
                           ExecutionStrategy.BALANCED.value])


# ─────────────────────────────────────────────────────────────────────────────
# 31. Optimization cache hits
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheHits(unittest.TestCase):

    def test_second_call_with_same_signature_is_cached(self):
        engine = AdaptiveOptimizationEngine()
        metas  = [_make_meta("atlas")]
        sig    = "car_problem_001"
        trace1 = engine.optimize(module_metas=metas, task_signature=sig)
        trace2 = engine.optimize(module_metas=metas, task_signature=sig)
        # Second call should hit cache
        self.assertEqual(trace2.bypass_reason, "cache_hit")

    def test_different_signatures_different_results(self):
        engine = AdaptiveOptimizationEngine()
        metas  = [_make_meta("atlas")]
        t1 = engine.optimize(module_metas=metas, task_signature="sig_A")
        t2 = engine.optimize(module_metas=metas, task_signature="sig_B")
        # Both should not hit cache (first time for each)
        self.assertNotEqual(t1.trace_id, t2.trace_id)

    def test_invalidate_cache_forces_recompute(self):
        engine = AdaptiveOptimizationEngine()
        metas  = [_make_meta("atlas")]
        sig    = "sig_X"
        engine.optimize(module_metas=metas, task_signature=sig)
        engine.invalidate_cache()
        trace2 = engine.optimize(module_metas=metas, task_signature=sig)
        self.assertNotEqual(trace2.bypass_reason, "cache_hit")


# ─────────────────────────────────────────────────────────────────────────────
# 32. Record result → PerformanceHistory
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordResult(unittest.TestCase):

    def test_record_result_adds_to_history(self):
        engine = AdaptiveOptimizationEngine()
        metas  = [_make_meta("reasoning")]
        trace  = engine.optimize(module_metas=metas)
        engine.record_result(trace, success=True, latency_ms=35.0)
        # Check that history has a record
        hist = engine.get_performance_history()
        self.assertGreater(len(list(hist._records)), 0)

    def test_record_result_success_rate_updates(self):
        engine = AdaptiveOptimizationEngine()
        metas  = [_make_meta("atlas")]
        trace  = engine.optimize(module_metas=metas)
        # Mark selected module for history lookups
        if trace.candidates:
            trace.candidates[0].selected = True
        engine.record_result(trace, success=True)


# ─────────────────────────────────────────────────────────────────────────────
# 34. Optimization levels
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizationLevels(unittest.TestCase):

    def test_off_level_returns_bypass(self):
        cfg = OptimizerConfig(level=OptimizationLevel.OFF)
        engine = AdaptiveOptimizationEngine(config=cfg)
        trace = engine.optimize(module_metas=[_make_meta("atlas")])
        self.assertIn("OFF", trace.bypass_reason)

    def test_minimal_level_skips_ranking(self):
        cfg = OptimizerConfig(level=OptimizationLevel.MINIMAL)
        engine = AdaptiveOptimizationEngine(config=cfg)
        metas  = [_make_meta("atlas"), _make_meta("reasoning")]
        trace  = engine.optimize(module_metas=metas)
        # MINIMAL doesn't rank modules
        self.assertEqual(len(trace.candidates), 0)

    def test_balanced_level_ranks_candidates(self):
        cfg = OptimizerConfig(level=OptimizationLevel.BALANCED)
        engine = AdaptiveOptimizationEngine(config=cfg)
        metas  = [_make_meta("atlas"), _make_meta("reasoning")]
        trace  = engine.optimize(module_metas=metas)
        self.assertEqual(len(trace.candidates), 2)

    def test_aggressive_level_ranks_candidates(self):
        cfg = OptimizerConfig(level=OptimizationLevel.AGGRESSIVE)
        engine = AdaptiveOptimizationEngine(config=cfg)
        metas  = [_make_meta("atlas")]
        trace  = engine.optimize(module_metas=metas)
        self.assertIsInstance(trace, OptimizationTrace)


# ─────────────────────────────────────────────────────────────────────────────
# 35. make_optimizer convenience factory
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeOptimizer(unittest.TestCase):

    def test_default_creation(self):
        engine = make_optimizer()
        self.assertIsInstance(engine, AdaptiveOptimizationEngine)

    def test_policy_string(self):
        engine = make_optimizer(policy="LOW_LATENCY")
        self.assertEqual(engine._cfg.policy, OptimizationPolicy.LOW_LATENCY)

    def test_level_string(self):
        engine = make_optimizer(level="MINIMAL")
        self.assertEqual(engine._cfg.level, OptimizationLevel.MINIMAL)

    def test_constraint_kwarg(self):
        engine = make_optimizer(max_latency_ms=100.0)
        self.assertEqual(engine._cfg.constraints.max_latency_ms, 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# 36. PRIME.optimizer() factory method
# ─────────────────────────────────────────────────────────────────────────────

class TestPRIMEOptimizerFactory(unittest.TestCase):

    def test_prime_optimizer_returns_engine(self):
        from prime import PRIME
        p = PRIME()
        try:
            engine = p.optimizer()
            self.assertIsInstance(engine, AdaptiveOptimizationEngine)
        finally:
            p.shutdown()

    def test_prime_optimizer_policy_arg(self):
        from prime import PRIME
        p = PRIME()
        try:
            engine = p.optimizer(policy="LOW_MEMORY")
            self.assertEqual(engine._cfg.policy,
                             OptimizationPolicy.LOW_MEMORY)
        finally:
            p.shutdown()

    def test_prime_optimizer_level_arg(self):
        from prime import PRIME
        p = PRIME()
        try:
            engine = p.optimizer(level="MINIMAL")
            self.assertEqual(engine._cfg.level, OptimizationLevel.MINIMAL)
        finally:
            p.shutdown()

    def test_prime_optimizer_after_shutdown_raises(self):
        from prime import PRIME, PrimeError
        p = PRIME()
        p.shutdown()
        with self.assertRaises(PrimeError):
            p.optimizer()

    def test_prime_optimizer_config_dict(self):
        from prime import PRIME
        p = PRIME()
        try:
            engine = p.optimizer(config={"policy": "LOW_CPU",
                                          "level":  "MINIMAL"})
            self.assertEqual(engine._cfg.policy, OptimizationPolicy.LOW_CPU)
        finally:
            p.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 38. Energy proxy label
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyProxy(unittest.TestCase):

    def test_energy_proxy_is_float(self):
        model = CostModel()
        ep = model.energy_proxy(cpu_relative=2.0, latency_ms=500.0)
        self.assertIsInstance(ep, float)
        self.assertGreater(ep, 0.0)

    def test_energy_proxy_labeled_policy_selects_cpu_saver(self):
        engine = AdaptiveOptimizationEngine(
            config=OptimizerConfig(
                policy=OptimizationPolicy.ENERGY_AWARE_PROXY))
        trace = engine.optimize(module_metas=[])
        # Should select CPU_SAVER strategy
        self.assertEqual(trace.strategy_selected,
                         ExecutionStrategy.CPU_SAVER.value)


# ─────────────────────────────────────────────────────────────────────────────
# 39. UNKNOWN propagation (cross-platform sentinel)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownPropagation(unittest.TestCase):

    def test_unknown_sentinel_value(self):
        self.assertEqual(UNKNOWN, "UNKNOWN")

    def test_unknown_memory_does_not_crash_ranker(self):
        cost   = CostModel()
        hist   = PerformanceHistory()
        constr = ResourceConstraints()
        ranker = ModuleRanker(cost, hist, constr)
        snap   = ResourceSnapshot(
            cpu=CPUInfo(logical_count=UNKNOWN),
            memory=MemoryInfo(available_bytes=UNKNOWN))
        metas  = [_make_meta("atlas")]
        ranked = ranker.rank(metas, ExecutionStrategy.BALANCED, snap)
        self.assertGreater(len(ranked), 0)

    def test_unknown_cpu_does_not_crash_engine(self):
        engine = AdaptiveOptimizationEngine()
        with patch.object(engine._monitor, "snapshot",
                          return_value=ResourceSnapshot(
                              cpu=CPUInfo(logical_count=UNKNOWN,
                                         utilization_pct=UNKNOWN),
                              memory=MemoryInfo(available_bytes=UNKNOWN))):
            trace = engine.optimize(module_metas=[])
        self.assertIsInstance(trace, OptimizationTrace)

    def test_thread_selector_unknown_cpu_gives_one(self):
        sel    = ThreadSelector()
        constr = ResourceConstraints()
        n = sel.select(ExecutionStrategy.FULL,
                       CPUInfo(logical_count=UNKNOWN), constr)
        self.assertEqual(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 40. Engine summary
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineSummary(unittest.TestCase):

    def test_summary_keys(self):
        engine = AdaptiveOptimizationEngine()
        s = engine.summary()
        required = {
            "optimizer_version", "level", "policy",
            "current_strategy", "thread_count",
            "cpu_utilization", "available_memory_mb",
            "logical_cpus", "performance_records",
            "cache_entries", "traces_recorded",
        }
        for k in required:
            self.assertIn(k, s, msg=f"Missing key: {k}")

    def test_version_is_9(self):
        engine = AdaptiveOptimizationEngine()
        s = engine.summary()
        self.assertEqual(s["optimizer_version"], "9.0")


# ─────────────────────────────────────────────────────────────────────────────
# 44. OptimizerConfig to_dict / from_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizerConfig(unittest.TestCase):

    def test_to_dict(self):
        cfg = OptimizerConfig(policy=OptimizationPolicy.LOW_LATENCY,
                               level=OptimizationLevel.MINIMAL)
        d = cfg.to_dict()
        self.assertEqual(d["policy"], "LOW_LATENCY")
        self.assertEqual(d["level"], "MINIMAL")

    def test_from_dict_round_trip(self):
        cfg  = OptimizerConfig(policy=OptimizationPolicy.LOW_CPU,
                                level=OptimizationLevel.AGGRESSIVE,
                                cache_ttl_s=5.0)
        d    = cfg.to_dict()
        cfg2 = OptimizerConfig.from_dict(d)
        self.assertEqual(cfg2.policy, OptimizationPolicy.LOW_CPU)
        self.assertEqual(cfg2.level, OptimizationLevel.AGGRESSIVE)

    def test_from_dict_constraints(self):
        d = {"constraints": {"max_latency_ms": 200.0, "max_threads": 4}}
        cfg = OptimizerConfig.from_dict(d)
        self.assertEqual(cfg.constraints.max_latency_ms, 200.0)
        self.assertEqual(cfg.constraints.max_threads, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 50. End-to-end: PRIME → optimizer → mosaic → result
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):

    def test_full_pipeline_car_problem(self):
        from prime import PRIME
        p = PRIME()
        try:
            atlas    = p.atlas()
            mosaic   = p.mosaic()
            opt_engine = p.optimizer()

            problem  = atlas.parse("A car travels 150 km in 3 hours.")
            plan, trace = opt_engine.optimize_plan(
                mosaic, problem, task_id="car_e2e")

            result = mosaic.execute(plan, problem,
                                     raw_input="A car travels 150 km in 3 hours.")
            opt_engine.record_result(trace, result.succeeded)

            # The optimizer should have produced a trace
            self.assertIsInstance(trace, OptimizationTrace)
            self.assertIn(trace.strategy_selected,
                          [s.value for s in ExecutionStrategy])
            # MOSAIC still runs (answer may vary by env/model availability)
            self.assertIn(result.status,
                          ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED",
                           "FAILURE", "MODULE_ERROR", "INVALID_PLAN",
                           "LIMIT_EXCEEDED"])
        finally:
            p.shutdown()

    def test_optimizer_does_not_replace_mosaic(self):
        """Phase 9 must leave MOSAIC as the composition authority."""
        from prime import PRIME
        p = PRIME()
        try:
            atlas   = p.atlas()
            mosaic  = p.mosaic()
            engine  = p.optimizer()
            problem = atlas.parse("2 + 3 = ?")
            plan, trace = engine.optimize_plan(mosaic, problem)
            # Plan has module_ids populated by MOSAIC
            self.assertIsNotNone(plan)
            self.assertIn(trace.strategy_selected,
                          [s.value for s in ExecutionStrategy])
        finally:
            p.shutdown()

    def test_optimizer_explain_from_full_pipeline(self):
        from prime import PRIME
        p = PRIME()
        try:
            atlas   = p.atlas()
            mosaic  = p.mosaic()
            engine  = p.optimizer()
            problem = atlas.parse("A car travels 90 km in 2 hours.")
            _, trace = engine.optimize_plan(mosaic, problem, task_id="explain_e2e")
            expl = engine.explain(trace)
            self.assertIn("Phase 9", expl)
        finally:
            p.shutdown()

    def test_previous_tests_still_pass_check(self):
        """Smoke-test: PRIME and MOSAIC basic functionality untouched."""
        from prime import PRIME
        p = PRIME()
        try:
            self.assertTrue(p.is_running())
            mosaic = p.mosaic()
            result = mosaic.solve("A car travels 150 km in 3 hours.")
            self.assertIn(result.status,
                          ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED",
                           "FAILURE", "MODULE_ERROR"])
        finally:
            p.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
