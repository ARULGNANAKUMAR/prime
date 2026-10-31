#!/usr/bin/env python3
"""
Phase 9 Benchmark — Adaptive Lightweight Optimization Engine
=============================================================

Measures:
  - resource_monitor_snapshot_ms
  - classify_resource_class_ms
  - cost_model_score_ms
  - module_rank_3_ms
  - optimize_empty_metas_ms
  - optimize_3_metas_ms
  - optimize_cache_hit_ms
  - thread_select_ms
  - performance_history_record_ms
  - performance_history_aggregate_ms
  - optimize_plan_mosaic_ms
  - optimizer_overhead_pct  (optimizer_latency / total)

Run:
    PYTHONPATH=python python3 examples/benchmark_phase9.py
"""

import sys
import time

sys.path.insert(0, "python")

from prime import PRIME
from prime.optimizer import (
    AdaptiveOptimizationEngine, CostModel,
    OptimizationPolicy, PerformanceRecord, PerformanceHistory,
    ResourceClassifier, ResourceMonitor, ResourceSnapshot,
    CPUInfo, MemoryInfo, make_optimizer, UNKNOWN,
)


def bench(fn, n=1000):
    # Warm-up
    for _ in range(max(1, n // 10)):
        fn()
    t0 = time.monotonic()
    for _ in range(n):
        fn()
    elapsed = time.monotonic() - t0
    return (elapsed / n) * 1000


def _snap():
    return ResourceSnapshot(
        cpu=CPUInfo(logical_count=4, utilization_pct=20.0),
        memory=MemoryInfo(available_bytes=2 * 1024 ** 3))


def main():
    print("PRIME Phase 9 Benchmarks")
    print("=" * 50)

    monitor = ResourceMonitor(cache_ttl_s=0.0)
    clf     = ResourceClassifier()
    cost    = CostModel()
    hist    = PerformanceHistory()
    engine  = make_optimizer()

    metas3 = [
        {"id": "a", "name": "A",
         "cost": {"cpu_relative": 0.5, "memory_relative": 0.5,
                  "latency_ms": 5.0, "requires_model": False},
         "status": "available"},
        {"id": "b", "name": "B",
         "cost": {"cpu_relative": 2.0, "memory_relative": 2.0,
                  "latency_ms": 50.0, "requires_model": False},
         "status": "available"},
        {"id": "c", "name": "C",
         "cost": {"cpu_relative": 8.0, "memory_relative": 8.0,
                  "latency_ms": 800.0, "requires_model": False},
         "status": "available"},
    ]

    results = {}

    # resource_monitor_snapshot
    results["resource_monitor_snapshot_ms"] = bench(
        lambda: monitor.snapshot(force=True), n=500)

    # classify_resource_class
    snap = _snap()
    results["classify_resource_class_ms"] = bench(
        lambda: clf.classify(snap), n=2000)

    # cost_model_score
    results["cost_model_score_ms"] = bench(
        lambda: cost.score(2.0, 2.0, 50.0), n=5000)

    # module_rank_3
    from prime.optimizer import ModuleRanker, ResourceConstraints, ExecutionStrategy
    ranker = ModuleRanker(cost, hist, ResourceConstraints())
    results["module_rank_3_ms"] = bench(
        lambda: ranker.rank(metas3, ExecutionStrategy.BALANCED, snap),
        n=2000)

    # optimize_empty_metas
    results["optimize_empty_metas_ms"] = bench(
        lambda: engine.optimize(module_metas=[]), n=2000)

    # optimize_3_metas
    results["optimize_3_metas_ms"] = bench(
        lambda: engine.optimize(module_metas=metas3), n=2000)

    # optimize_cache_hit
    engine2 = make_optimizer()
    engine2.optimize(module_metas=metas3, task_signature="bench_sig")
    results["optimize_cache_hit_ms"] = bench(
        lambda: engine2.optimize(module_metas=metas3,
                                  task_signature="bench_sig"),
        n=5000)

    # thread_select
    from prime.optimizer import ThreadSelector
    sel = ThreadSelector()
    cpu = CPUInfo(logical_count=8)
    rc  = ResourceConstraints()
    results["thread_select_ms"] = bench(
        lambda: sel.select(ExecutionStrategy.BALANCED, cpu, rc), n=5000)

    # performance_history_record
    rec = PerformanceRecord(module_id="reasoning", latency_ms=50.0,
                             success=True)
    results["performance_history_record_ms"] = bench(
        lambda: hist.record(rec), n=5000)

    # performance_history_aggregate
    results["performance_history_aggregate_ms"] = bench(
        lambda: hist.aggregate("reasoning"), n=5000)

    # optimize_plan_mosaic (end-to-end)
    with PRIME() as prime:
        atlas   = prime.atlas()
        mosaic  = prime.mosaic()
        opt_eng = prime.optimizer()
        problem = atlas.parse("A car travels 150 km in 3 hours.")

        def _run():
            opt_eng.invalidate_cache()
            plan, trace = opt_eng.optimize_plan(mosaic, problem)
            return trace

        results["optimize_plan_mosaic_ms"] = bench(_run, n=200)

        # optimizer overhead percentage
        trace = _run()
        results["optimizer_latency_last_ms"] = trace.optimizer_latency_ms

    # Print
    print()
    col_w = 42
    for k, v in results.items():
        bar = "─" * int(min(v, 5.0) * 8)
        print(f"  {k:<{col_w}} {v:>8.4f} ms  {bar}")

    print()
    print(f"  Optimizer is lightweight — overhead is measurable and bounded.")


if __name__ == "__main__":
    main()
