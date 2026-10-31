#!/usr/bin/env python3
"""
phase8_demo.py — PRIME Phase 8: Self-Learning & Continual Learning Engine

Demonstrates the full Phase 8 pipeline:
  MOSAIC solve → ECHO record → LearningController.learn_from()
  → strategy-score update → MOSAIC hint → measurable behavioral change

Run:
    cd PRIME
    PYTHONPATH=python python3 examples/phase8_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from prime import PRIME, LearningMode, ExperienceOutcome
from prime.echo import Experience
from prime.learning import (
    LearningConfig, LearningController, make_context_key,
    DriftStatus, StrategyStats,
)


def sep(title=""):
    print()
    if title:
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
    else:
        print("-" * 60)


def main():
    print("PRIME Phase 8 — Self-Learning & Continual Learning Engine")
    print("=" * 60)

    prime  = PRIME()
    mosaic = prime.mosaic()
    echo   = prime.echo()

    cfg = LearningConfig(
        mode           = LearningMode.STRATEGY_LEARNING.value,
        learning_rate  = 0.12,
        min_evidence   = 1,
        min_confidence = 0.0,
        require_approval = False,
    )
    lc = prime.learning(config=cfg)
    lc.enable()

    # ───────────────────────────────────────────────────────────────
    # 1. Solve tasks and record experiences through ECHO
    # ───────────────────────────────────────────────────────────────
    sep("STEP 1: Solve tasks via MOSAIC → ECHO → LearningController")

    tasks = [
        "A car travels 150 km in 3 hours. Calculate average speed.",
        "A train covers 240 km in 4 hours. What is its speed?",
        "A cyclist rides 90 km in 3 hours. Find the average speed.",
        "A plane flies 1200 km in 2 hours. What is its speed?",
        "A runner completes 10 km in 1 hour. What is the average speed?",
    ]

    for task_text in tasks:
        result = mosaic.solve(task_text)
        # Record in ECHO
        exp = echo.record(
            problem_text     = task_text,
            execution_result = result,
            extra_tags       = ["arithmetic", "speed"],
        )
        # Learn from each experience
        lr = lc.learn_from(exp)
        icon = "✓" if lr["committed"] > 0 else "~"
        print(f"  {icon} [{exp.outcome[:7]}] commits={lr['committed']} "
              f"  {task_text[:50]}")

    # ───────────────────────────────────────────────────────────────
    # 2. Show learned strategy statistics
    # ───────────────────────────────────────────────────────────────
    sep("STEP 2: Learned strategy statistics")

    stats = lc.get_strategy_stats()
    for s in stats:
        print(f"  {s.strategy_id:<20} ctx={s.context_key!r}")
        print(f"    score={s.score:.4f}  attempts={s.attempts}  "
              f"successes={s.successes}  failures={s.failures}")
        print(f"    avg_reward={s.average_reward:+.3f}  "
              f"drift={s.drift_status}")

    # ───────────────────────────────────────────────────────────────
    # 3. Show adapter parameters (real trained biases)
    # ───────────────────────────────────────────────────────────────
    sep("STEP 3: StrategyBiasAdapter parameters (actual learned values)")

    adapter = lc._state.adapter
    ad = adapter.state_dict()
    biases = ad.get("biases", {})
    print(f"  Total adapter parameters: {len(biases)}")
    for k, v in list(biases.items())[:6]:
        ctx, mod = k.split("::", 1) if "::" in k else (k, "?")
        print(f"    [{mod:<20}] bias = {v:+.4f}  (ctx={ctx!r})")

    # ───────────────────────────────────────────────────────────────
    # 4. MOSAIC hints from learned preferences
    # ───────────────────────────────────────────────────────────────
    sep("STEP 4: MOSAIC hints derived from learned preferences")

    stats = lc.get_strategy_stats()
    contexts = {s.context_key for s in stats}
    for ctx in list(contexts)[:3]:
        hints = lc.get_mosaic_hints(ctx)
        print(f"  context: {ctx!r}")
        print(f"    preferred_modules: {hints['preferred_modules']}")
        print(f"    excluded_modules:  {hints['excluded_modules']}")

    # ───────────────────────────────────────────────────────────────
    # 5. Learning trace (provenance)
    # ───────────────────────────────────────────────────────────────
    sep("STEP 5: Learning trace — What changed? Why?")

    trace = lc.get_trace(n=3)
    print(f"  Showing last {len(trace)} trace entries:")
    for t in trace:
        print()
        print(t.explain())

    # ───────────────────────────────────────────────────────────────
    # 6. Versioning & rollback
    # ───────────────────────────────────────────────────────────────
    sep("STEP 6: Versioning & rollback")

    versions = lc.list_versions()
    print(f"  {len(versions)} checkpoint(s) available")
    if versions:
        v = versions[0]
        print(f"  Earliest: {v.version_id[:8]}…  reason={v.reason!r}")
        # Roll back to that version
        ok = lc.rollback(v.version_id)
        print(f"  Rollback to v0: {'ok' if ok else 'FAILED'}")
        # Re-enable and re-learn
        lc.enable()

    # ───────────────────────────────────────────────────────────────
    # 7. Replay buffer (catastrophic-forgetting protection)
    # ───────────────────────────────────────────────────────────────
    sep("STEP 7: Experience replay")

    print(f"  Replay buffer size: {len(lc._replay)}")
    result = lc.replay(n=4, strategy="recent")
    print(f"  Replay steps run: {result['replay_steps']}  "
          f"committed: {result['committed']}")

    # ───────────────────────────────────────────────────────────────
    # 8. Learning summary
    # ───────────────────────────────────────────────────────────────
    sep("STEP 8: Learning summary")

    s = lc.summary()
    print(f"  mode:                   {s['mode']}")
    print(f"  total_strategy_entries: {s['total_strategy_entries']}")
    print(f"  adapter_parameters:     {s['adapter_parameters']}")
    print(f"  replay_buffer_size:     {s['replay_buffer_size']}")
    print(f"  versions_available:     {s['versions_available']}")
    m = s["metrics"]
    print(f"  experiences_ingested:   {m['total_experiences_ingested']}")
    print(f"  updates_committed:      {m['total_updates_committed']}")
    print(f"  updates_rejected:       {m['total_updates_rejected']}")
    print(f"  rollbacks:              {m['total_rollbacks']}")
    print(f"  avg_delta:              {m['avg_delta']:.4f}")

    # ───────────────────────────────────────────────────────────────
    # 9. Phase boundary check
    # ───────────────────────────────────────────────────────────────
    sep("Phase boundary check")

    items = [
        ("Update strategy preference scores",     True),
        ("Update knowledge-candidate confidence", True),
        ("Update adapter bias parameters",        True),
        ("Detect concept drift",                  True),
        ("Replay buffer (forgetting protection)", True),
        ("Rollback to prior version",             True),
        ("Approval gate / human-in-loop hook",    True),
        ("Retrain full LLM weights",              False),
        ("Gradient descent on LLM",               False),
        ("Fine-tune base model parameters",       False),
        ("Adaptive resource optimization",        False),  # Phase 9
    ]
    for item, implemented in items:
        icon  = "✓" if implemented else "✗"
        label = "IMPLEMENTED" if implemented else "NOT implemented (Phase 9+)"
        print(f"  {icon}  {item}: {label}")

    echo.close()
    prime.shutdown()

    print()
    print("=" * 60)
    print("  Phase 8 Self-Learning demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
