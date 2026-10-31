#!/usr/bin/env python3
"""
Phase 9 Demo — Adaptive Lightweight Optimization Engine
========================================================

Shows the full Phase 9 pipeline:

    ATLAS → MOSAIC Capability Matching
          → Phase 8 Learned Preferences (simulated)
          → Phase 9 Resource / Performance Analysis
          → Execution Strategy
          → Runtime
          → Performance Monitoring → Feedback → Optimization State

Run:
    PYTHONPATH=python python3 examples/phase9_demo.py
"""

import sys
import time

sys.path.insert(0, "python")

from prime import PRIME
from prime.optimizer import (
    AdaptiveOptimizationEngine, OptimizerConfig,
    OptimizationPolicy, OptimizationLevel, ExecutionStrategy,
    ResourceConstraints, make_optimizer, UNKNOWN,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main() -> None:
    print("PRIME Phase 9: Adaptive Lightweight Optimization Engine")
    print("Version: 0.9.0.dev1")

    # ── 1. Basic resource inspection ────────────────────────────────────────
    section("1. Resource Monitor — Runtime Detection")
    engine = make_optimizer()
    snap   = engine.get_resource_snapshot(force=True)
    cpu    = snap.cpu
    mem    = snap.memory

    print(f"  Platform:         {snap.platform_os}")
    print(f"  Python:           {snap.python_ver}")
    print(f"  Logical CPUs:     {cpu.logical_count}")
    print(f"  CPU utilization:  "
          f"{cpu.utilization_pct:.1f}%"
          if cpu.utilization_pct != UNKNOWN else
          f"  CPU utilization:  {UNKNOWN}")
    if mem.available_bytes != UNKNOWN:
        print(f"  Available RAM:    {mem.available_bytes/1e9:.2f} GB")
    else:
        print(f"  Available RAM:    {UNKNOWN}")
    if mem.process_rss != UNKNOWN:
        print(f"  Process RSS:      {mem.process_rss/1e6:.1f} MB")

    # ── 2. Resource classification ──────────────────────────────────────────
    section("2. Resource Classification")
    from prime.optimizer import ResourceClassifier
    clf = ResourceClassifier()
    cls = clf.classify(snap)
    print(f"  Resource class:   {cls}")
    print(f"  (high / medium / low / unknown)")

    # ── 3. Optimization policies and strategies ─────────────────────────────
    section("3. Policy → Strategy Mapping")
    policies = list(OptimizationPolicy)
    for policy in policies:
        eng = make_optimizer(policy=policy.value)
        trace = eng.optimize(module_metas=[])
        print(f"  {policy.value:<22s} → {trace.strategy_selected}")

    # ── 4. Module ranking with resource costs ───────────────────────────────
    section("4. Module Ranking — Resource-Aware Selection")
    metas = [
        {"id": "heavy_module", "name": "Heavy Module",
         "cost": {"cpu_relative": 8.0, "memory_relative": 8.0,
                  "latency_ms": 800.0, "requires_model": False},
         "status": "available"},
        {"id": "light_module", "name": "Light Module",
         "cost": {"cpu_relative": 0.5, "memory_relative": 0.5,
                  "latency_ms": 10.0, "requires_model": False},
         "status": "available"},
        {"id": "medium_module", "name": "Medium Module",
         "cost": {"cpu_relative": 2.0, "memory_relative": 2.0,
                  "latency_ms": 50.0, "requires_model": False},
         "status": "available"},
    ]

    for strategy in [ExecutionStrategy.FULL,
                      ExecutionStrategy.BALANCED,
                      ExecutionStrategy.LIGHTWEIGHT]:
        eng   = make_optimizer()
        trace = eng.optimize(module_metas=metas,
                             policy_override=OptimizationPolicy.BALANCED)
        # Force strategy via config
        from prime.optimizer import STRATEGY_CONFIGS
        ranked = eng._ranker.rank(metas, strategy, snap)
        selected = [m for m, c in ranked if c.selected]
        print(f"  Strategy {strategy.value:<12s}: selects → "
              f"{selected[0] if selected else 'none'}")

    # ── 5. Phase 8 integration ──────────────────────────────────────────────
    section("5. Phase 8 Learned Hints Integration")
    learned_hints = {
        "preferred_modules": ["light_module"],
        "excluded_modules":  ["heavy_module"],
    }
    eng   = make_optimizer()
    trace = eng.optimize(module_metas=metas, task_id="demo_task",
                          learned_hints=learned_hints)
    print(f"  Phase 8 preferred: {trace.phase8_preferred}")
    print(f"  Phase 8 excluded:  {trace.phase8_excluded}")
    selected = [c for c in trace.candidates if c.selected]
    if selected:
        c = selected[0]
        print(f"  Selected module:   {c.module_id} "
              f"(cost={c.cost_score:.3f}, bonus={c.learned_bonus:+.3f})")

    # ── 6. Hard constraint enforcement ──────────────────────────────────────
    section("6. Hard Constraint Enforcement")
    constrained_engine = make_optimizer(
        max_latency_ms=20.0,   # Only allow < 20 ms
        max_memory_bytes=50 * 1024 * 1024,  # 50 MB max
    )
    trace = constrained_engine.optimize(module_metas=metas)
    accepted = [c for c in trace.candidates if not c.rejected]
    rejected = [c for c in trace.candidates if c.rejected]
    print(f"  Max latency:      20 ms  |  Max memory: 50 MB")
    print(f"  Accepted:         {[c.module_id for c in accepted]}")
    print(f"  Hard rejected:    {[(c.module_id, c.rejection_reason[:30])
                                    for c in rejected]}")

    # ── 7. Optimization levels ──────────────────────────────────────────────
    section("7. Optimization Levels")
    for level in OptimizationLevel:
        eng   = make_optimizer(level=level.value)
        trace = eng.optimize(module_metas=metas)
        note  = f"(bypass: {trace.bypass_reason})" if trace.bypass_reason else ""
        print(f"  {level.value:<12s}: strategy={trace.strategy_selected} {note}")

    # ── 8. Thread selection ─────────────────────────────────────────────────
    section("8. Resource-Aware Thread Selection")
    for strategy in ExecutionStrategy:
        eng   = make_optimizer()
        trace = eng.optimize(module_metas=[], task_id="thread_demo",
                             policy_override=OptimizationPolicy.BALANCED)
        from prime.optimizer import STRATEGY_CONFIGS, ThreadSelector
        from prime.optimizer import ResourceConstraints as RC
        sel = ThreadSelector()
        n   = sel.select(strategy, snap.cpu, RC())
        print(f"  {strategy.value:<14s}: {n} thread(s)"
              f"  (fraction={STRATEGY_CONFIGS[strategy.value].thread_fraction})")

    # ── 9. Performance history ──────────────────────────────────────────────
    section("9. Performance History (Not a Learning Engine)")
    eng = make_optimizer()
    # Simulate a few executions
    from prime.optimizer import PerformanceRecord
    for i in range(5):
        eng._history.record(PerformanceRecord(
            module_id="reasoning",
            latency_ms=float(40 + i * 5),
            success=(i < 4),
        ))
    agg = eng._history.aggregate("reasoning")
    print(f"  Module:           reasoning")
    print(f"  Samples:          {agg['sample_count']}")
    print(f"  Avg latency:      {agg['avg_latency_ms']:.1f} ms")
    print(f"  Success rate:     {agg['success_rate']:.2f}")

    # ── 10. Optimization trace / explainability ──────────────────────────────
    section("10. Optimization Trace — Explainability")
    eng   = make_optimizer()
    trace = eng.optimize(module_metas=metas, task_id="explain_demo",
                          learned_hints=learned_hints)
    print(eng.explain(trace))

    # ── 11. Optimization cache ──────────────────────────────────────────────
    section("11. Optimization Cache")
    eng = make_optimizer()
    sig = "demo_task_signature"
    t1  = time.monotonic()
    eng.optimize(module_metas=metas, task_signature=sig)
    t1_ms = (time.monotonic() - t1) * 1000

    t2   = time.monotonic()
    trace2 = eng.optimize(module_metas=metas, task_signature=sig)
    t2_ms  = (time.monotonic() - t2) * 1000
    print(f"  First  call: {t1_ms:.3f} ms")
    print(f"  Second call: {t2_ms:.3f} ms  (cache_hit={trace2.bypass_reason!r})")
    print(f"  Cache entries: {len(eng._cache)}")

    # ── 12. Energy proxy ────────────────────────────────────────────────────
    section("12. Energy-Aware Proxy (Clearly Labeled)")
    from prime.optimizer import CostModel
    cost = CostModel()
    ep   = cost.energy_proxy(cpu_relative=2.0, latency_ms=500.0)
    print(f"  Energy proxy score: {ep:.4f}")
    print(f"  NOTE: This is a dimensionless proxy based on CPU × time.")
    print(f"        Not actual power consumption measurement.")

    # ── 13. Full PRIME integration ──────────────────────────────────────────
    section("13. Full Integration: PRIME → Optimizer → MOSAIC")
    with PRIME() as prime:
        atlas    = prime.atlas()
        mosaic   = prime.mosaic()
        opt_eng  = prime.optimizer(policy="BALANCED", level="BALANCED")

        problem  = atlas.parse("A car travels 150 km in 3 hours.")
        plan, trace = opt_eng.optimize_plan(mosaic, problem,
                                             task_id="car_p9")
        result   = mosaic.execute(plan, problem,
                                   raw_input="A car travels 150 km in 3 hours.")
        opt_eng.record_result(trace, result.succeeded)

        print(f"  Task:             'A car travels 150 km in 3 hours.'")
        print(f"  Strategy chosen:  {trace.strategy_selected}")
        print(f"  Thread count:     {trace.thread_count}")
        print(f"  Optimizer time:   {trace.optimizer_latency_ms:.3f} ms")
        print(f"  MOSAIC status:    {result.status}")
        print(f"  Ph8 preferred:    {trace.phase8_preferred or '(none)'}")

        s = opt_eng.summary()
        print(f"\n  Optimizer Summary:")
        print(f"    Version:     {s['optimizer_version']}")
        print(f"    Policy:      {s['policy']}")
        print(f"    Level:       {s['level']}")
        print(f"    Strategy:    {s['current_strategy']}")
        print(f"    CPU util:    {s['cpu_utilization']}")
        print(f"    RAM avail:   "
              + (f"{s['available_memory_mb']:.0f} MB"
                 if s['available_memory_mb'] != UNKNOWN else UNKNOWN))
        print(f"    psutil:      {s['psutil_available']}")

    print("\n✓ Phase 9 demo complete.")


if __name__ == "__main__":
    main()
