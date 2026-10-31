#!/usr/bin/env python3
"""
benchmark_phase8.py — PRIME Phase 8: Self-Learning Performance Benchmarks

Measures:
  - learn_from() latency
  - strategy score update latency
  - adapter update latency
  - replay latency
  - rollback latency
  - get_mosaic_hints() latency
  - full end-to-end pipeline (MOSAIC → ECHO → learn)

Run:
    cd PRIME
    PYTHONPATH=python python3 examples/benchmark_phase8.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from prime import PRIME
from prime.echo import Experience, ExperienceOutcome
from prime.learning import (
    LearningConfig, LearningController, LearningMode,
    StrategyBiasAdapter, ExperienceReplayBuffer, ReplayEntry,
    make_context_key,
)


def bench(label: str, fn, reps: int = 1000) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {label:<50} {ms:>8.4f} ms  ({reps} reps)")
    return ms


def make_exp(outcome=ExperienceOutcome.SUCCESS.value) -> Experience:
    return Experience(
        task_domain="physics", task_category="arithmetic",
        modules_used=["reasoning", "validator"],
        outcome=outcome, validated=(outcome == ExperienceOutcome.SUCCESS.value),
        result_confidence=0.95, task_text="benchmark task",
        result_status="SUCCESS" if outcome == ExperienceOutcome.SUCCESS.value else "FAILURE",
    )


def main():
    print("PRIME Phase 8 — Self-Learning Benchmarks")
    print("=" * 65)

    prime = PRIME()
    cfg   = LearningConfig(
        mode=LearningMode.STRATEGY_LEARNING.value,
        min_evidence=1, min_confidence=0.0, learning_rate=0.10,
    )
    lc  = LearningController(config=cfg)
    lc.enable()

    # Pre-populate
    N = 50
    print(f"\n  Pre-populating with {N} experiences…")
    exp = make_exp()
    for _ in range(N):
        lc._strategy.learn(exp)
    ctx = make_context_key(exp)
    print(f"  Strategy entries: {len(lc._state.list_stats())}")
    print(f"  Adapter params:   {len(lc._state.adapter._biases)}")
    print()

    print("Latency (lower is better)")
    print("-" * 65)

    # 1. learn_from (full eligibility + strategy + knowledge path)
    fresh_exp = make_exp()
    bench("learn_from() — full pipeline",
          lambda: lc.learn_from(fresh_exp), reps=500)

    # 2. Strategy learner (inner update)
    bench("StrategyLearner.learn() — direct",
          lambda: lc._strategy.learn(make_exp()), reps=2000)

    # 3. Adapter update
    adapter = StrategyBiasAdapter()
    bench("StrategyBiasAdapter.update()",
          lambda: adapter.update("ctx", "r", 1.0, 0.10), reps=10_000)

    # 4. Adapter get_bias
    bench("StrategyBiasAdapter.get_bias()",
          lambda: adapter.get_bias("ctx", "r"), reps=50_000)

    # 5. Replay buffer append
    buf   = ExperienceReplayBuffer()
    entry = ReplayEntry("e1", ctx, "reasoning", "SUCCESS", 1.0, 0.9, signature="")
    bench("ReplayBuffer.append()",
          lambda: buf.append(entry), reps=10_000)

    # 6. Replay sample
    for i in range(100):
        buf.append(ReplayEntry(f"e{i}", ctx, "r", "SUCCESS", 1.0, 0.9))
    bench("ReplayBuffer.sample(n=8, recent)",
          lambda: buf.sample(n=8, strategy="recent"), reps=5_000)

    # 7. Replay update
    bench("LearningController.replay(n=8)",
          lambda: lc.replay(n=8), reps=1000)

    # 8. MOSAIC hints
    bench("get_mosaic_hints()",
          lambda: lc.get_mosaic_hints(ctx), reps=5_000)

    # 9. Strategy stats lookup
    bench("get_strategy_stats(context_key=ctx)",
          lambda: lc.get_strategy_stats(context_key=ctx), reps=10_000)

    # 10. LearningState snapshot
    bench("LearningState.snapshot()",
          lambda: lc._state.snapshot(), reps=2000)

    # 11. Rollback
    versions = lc.list_versions()
    if versions:
        vid = versions[0].version_id
        bench("LearningController.rollback(v0)",
              lambda: lc.rollback(vid), reps=200)

    # 12. get_trace
    bench("get_trace(n=50)",
          lambda: lc.get_trace(n=50), reps=5_000)

    # 13. End-to-end MOSAIC → learn
    mosaic = prime.mosaic()
    echo   = prime.echo()
    def e2e():
        result = mosaic.solve("A car travels 150 km in 3 hours.")
        exp2   = echo.record(problem_text="A car travels 150 km in 3 hours.",
                             execution_result=result)
        lc.learn_from(exp2)

    bench("End-to-end: MOSAIC.solve() → echo.record() → learn_from()",
          e2e, reps=20)

    # Summary
    print()
    print("Learning state size")
    print("-" * 65)
    s = lc.summary()
    print(f"  {'total_strategy_entries':<45} {s['total_strategy_entries']}")
    print(f"  {'adapter_parameters':<45} {s['adapter_parameters']}")
    print(f"  {'replay_buffer_size':<45} {s['replay_buffer_size']}")
    print(f"  {'versions_available':<45} {s['versions_available']}")
    m = s["metrics"]
    print(f"  {'updates_committed':<45} {m['total_updates_committed']}")
    print(f"  {'avg_delta':<45} {m['avg_delta']:.4f}")

    echo.close()
    prime.shutdown()
    print()
    print("=" * 65)
    print("  Benchmark complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
