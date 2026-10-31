#!/usr/bin/env python3
"""
PRIME Phase 10 — Full Integration Demo
=======================================

Demonstrates the complete unified API:

    prime.solve(task)          → PrimeResult
    prime.pipeline()           → PrimePipeline (reusable)
    PrimeConfig / FeatureFlags → unified configuration + ablation
    PrimeTrace                 → full explainability from real data
    AblationStudy              → baseline comparison
    LearningCurveExperiment    → adaptation measurement
    profile_component()        → per-component overhead

Run:
    PYTHONPATH=python python3 examples/phase10_demo.py
"""

import json
import sys
import time

sys.path.insert(0, "python")

from prime import PRIME, PrimeConfig, FeatureFlags, SolveStatus
from prime import (
    AblationStudy, LearningCurveExperiment,
    SyntheticBenchmarkSuite, ExperimentRunner,
    collect_system_info, profile_component,
)


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def main() -> None:
    print("PRIME v1.0.0 — Full Integration & Research Validation")
    print(f"System: {collect_system_info()['os']} / "
          f"Python {collect_system_info()['python']} / "
          f"{collect_system_info().get('cpu_count','?')} CPUs")

    # ── 1. Simplest possible API ────────────────────────────────────────────
    section("1. Simplest API — prime.solve()")
    with PRIME() as prime:
        result = prime.solve("A car travels 150 km in 3 hours.")
        print(f"  Task:     'A car travels 150 km in 3 hours.'")
        print(f"  Status:   {result.status.value}")
        print(f"  Answer:   {result.answer!r}")
        print(f"  Latency:  {result.latency_ms:.1f} ms")
        print(f"  Modules:  {result.modules_used}")
        print(f"  Strategy: {result.optimizer_strategy}")

    # ── 2. Full trace + explainability ─────────────────────────────────────
    section("2. Full Trace — Every Stage Explained")
    with PRIME() as prime:
        result = prime.solve(
            "A rectangle has width 8 m and height 5 m. What is its area?",
            include_trace=True)
        if result.trace:
            print(result.trace.explain())

    # ── 3. Feature flags / ablation baselines ──────────────────────────────
    section("3. Feature Flags & Ablation Baselines")
    baselines = ["static", "mosaic_only", "mosaic_echo",
                  "mosaic_echo_learning", "full"]
    for b in baselines:
        cfg = PrimeConfig.for_ablation(b)
        with PRIME() as prime:
            r = prime.solve("Compute 6 divided by 2.", config=cfg)
        on_flags = [k for k, v in cfg.features.to_dict().items() if v]
        print(f"  {b:<22s}: {r.status.value:<22s} | {on_flags}")

    # ── 4. Reusable pipeline ────────────────────────────────────────────────
    section("4. Reusable Pipeline (amortised subsystem init)")
    with PRIME() as prime:
        pipe = prime.pipeline()
        tasks = [
            "Compute 3 times 4.",
            "A train travels at 80 km/h for 2.5 hours. How far?",
            "What is 15 percent of 200?",
        ]
        for task in tasks:
            t0 = time.monotonic()
            r  = pipe.solve(task)
            ms = (time.monotonic() - t0) * 1000
            print(f"  [{r.status.value:<7s}] {task[:48]:<50s} {ms:.1f} ms")

    # ── 5. Ablation study ───────────────────────────────────────────────────
    section("5. Ablation Study — Baseline Comparison")
    tasks = SyntheticBenchmarkSuite.core_tasks()[:3]

    def prime_factory(cfg=None):
        p = PRIME()
        _orig = p.solve
        def _wrapped(task, **kw):
            return _orig(task, config=cfg or PrimeConfig.default(), **kw)
        p.solve = _wrapped
        return p

    study   = AblationStudy(prime_factory)
    results = study.run_all(
        tasks     = tasks,
        baselines = ["mosaic_only", "full"],
    )
    print(AblationStudy.compare(results))

    # ── 6. Learning curve experiment ────────────────────────────────────────
    section("6. Learning Curve — Adaptation Over Experiences")
    train_tasks = SyntheticBenchmarkSuite.REPEATED[:8]
    probe_tasks = SyntheticBenchmarkSuite.core_tasks()[:3]

    exp   = LearningCurveExperiment(prime_factory)
    curve = exp.run(
        training_tasks = train_tasks,
        probe_tasks    = probe_tasks,
        checkpoints    = [0, 4, 8],
    )
    print(f"  {'Experiences':>12s}  {'Accuracy':>9s}  {'Mean Latency':>13s}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*13}")
    for pt in curve:
        print(f"  {pt.n_experiences:>12d}  {pt.accuracy*100:>8.1f}%  "
              f"{pt.mean_latency_ms:>12.1f} ms")
    print()
    print("  NOTE: Accuracy depends on MOSAIC's reasoning capability")
    print("        in this environment (no local LLM). Results reflect")
    print("        symbolic reasoning only.")

    # ── 7. Component profiling ──────────────────────────────────────────────
    section("7. Component Performance Profile")
    with PRIME() as prime:
        atlas   = prime.atlas()
        mosaic  = prime.mosaic()
        opt_eng = prime.optimizer()

        profiles = [
            profile_component(
                lambda: atlas.parse("A car travels 150 km in 3 hours."),
                n=200, label="ATLAS.parse"),
            profile_component(
                lambda: mosaic.compose(atlas.parse("Compute 4 times 3.")),
                n=100, label="MOSAIC.compose"),
            profile_component(
                lambda: opt_eng.optimize([]),
                n=500, label="Optimizer.optimize"),
        ]

    print(f"  {'Component':<24s} {'mean':>8s}  {'median':>8s}  {'std':>8s}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}")
    for p in profiles:
        print(f"  {p.component:<24s} {p.mean_ms:>7.3f}ms  "
              f"{p.median_ms:>7.3f}ms  {p.std_ms:>7.3f}ms")

    # ── 8. Result serialisation ─────────────────────────────────────────────
    section("8. JSON-Serialisable Result")
    with PRIME() as prime:
        r  = prime.solve("What is 15 percent of 200?")
        d  = r.to_dict()
        js = json.dumps(d, indent=2)
    print(js[:600])
    print("  ...")
    print(f"\n  Full result is {len(js)} bytes of JSON.")

    print("\n✓ Phase 10 demo complete.")


if __name__ == "__main__":
    main()
