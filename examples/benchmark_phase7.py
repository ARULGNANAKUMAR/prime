#!/usr/bin/env python3
"""
benchmark_phase7.py — PRIME Phase 7: ECHO Performance Benchmarks

Measures:
  - experience record latency
  - observation extraction latency
  - pattern detection over N experiences
  - generalization over N patterns
  - knowledge extraction latency
  - similarity search latency
  - query latency
  - full analyze() pipeline latency

Run:
    cd PRIME
    PYTHONPATH=python python3 examples/benchmark_phase7.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from prime import PRIME, ECHOEngine, ExperienceOutcome
from prime.echo import (
    Experience, Observation, Pattern,
    ExperienceStore, PatternDetector, GeneralizationEngine,
    PrincipleExtractor, ExperienceSimilarity, ConfidenceEstimator,
)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench(label: str, fn, reps: int = 1000) -> float:
    fn()   # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = time.perf_counter() - t0
    ms_per = elapsed / reps * 1000
    print(f"  {label:<42} {ms_per:>8.4f} ms  ({reps} reps)")
    return ms_per


def make_exp(domain="physics", category="arithmetic",
             outcome=ExperienceOutcome.SUCCESS.value) -> Experience:
    return Experience(
        task_text     = "A car travels 150 km in 3 hours.",
        task_domain   = domain,
        task_category = category,
        modules_used  = ["reasoning", "validator"],
        result_answer = "50",
        result_answer_unit = "km/h",
        result_confidence  = 0.95,
        result_status = "SUCCESS",
        reasoning_status = "SOLVED",
        outcome       = outcome,
        validated     = True,
        tags          = ["arithmetic"],
    )


def main() -> None:
    print("PRIME Phase 7 — ECHO Benchmarks")
    print("=" * 60)

    prime = PRIME()
    echo  = ECHOEngine()
    store = echo._store
    sim   = ExperienceSimilarity()

    # ---------------------------------------------------------------
    # Pre-populate the store
    # ---------------------------------------------------------------
    N = 100
    print(f"\n  Populating store with {N} experiences…")
    exps = []
    for _ in range(N):
        e = make_exp()
        store.add_experience(e)
        exps.append(e)
    obs_list = []
    for e in exps:
        for o in echo._obs_ext.extract(e):
            store.add_observation(o)
            obs_list.append(o)
    print(f"  {len(obs_list)} observations stored.\n")

    print("Latency (lower is better)")
    print("-" * 60)

    # 1. Record (collect + store)
    sample_exp = make_exp()
    bench("experience record (collect + store)",
          lambda: echo.record(problem_text="A car travels 150 km in 3 hours."))

    # 2. Observation extraction
    bench("observation extraction (1 experience)",
          lambda: echo._obs_ext.extract(sample_exp))

    # 3. Pattern detection (100 observations)
    all_obs  = store.list_observations()
    all_pats = store.list_patterns()
    bench("pattern detection (100 obs)",
          lambda: echo._pat_det.detect(all_obs, all_pats),
          reps=500)

    # 4. Generalization
    pats = echo.detect_patterns()
    all_stored_exps = store.list_experiences()
    bench("generalization (over detected patterns)",
          lambda: echo._gen_eng.generalize(pats, all_stored_exps),
          reps=500)

    # 5. Principle extraction
    gens = echo.generalize(pats)
    bench("principle extraction",
          lambda: echo._pri_ext.extract_principles(pats, gens),
          reps=500)

    # 6. Confidence estimation
    bench("confidence estimation",
          lambda: ConfidenceEstimator.estimate(50, 45, 5, validated_count=40),
          reps=10_000)

    # 7. Similarity (one vs 100)
    target = exps[0]
    bench("similarity search (1 vs 100 experiences)",
          lambda: sim.find_similar(target, exps, top_k=5),
          reps=500)

    # 8. Full analyze() pipeline on a fresh experience
    bench("full analyze() pipeline",
          lambda: echo.analyze(echo.record(problem_text="speed task")),
          reps=200)

    # 9. Query: get_experiences
    bench("query: get_experiences() (all)",
          lambda: echo.get_experiences(),
          reps=1000)

    # 10. Query: get_patterns
    bench("query: get_patterns()",
          lambda: echo.get_patterns(),
          reps=1000)

    # 11. Query: get_knowledge
    echo.extract_knowledge()
    bench("query: get_knowledge()",
          lambda: echo.get_knowledge(),
          reps=1000)

    # 12. Query: find_similar via engine
    bench("query: find_similar() via engine",
          lambda: echo.find_similar(target, top_k=5),
          reps=500)

    # ---------------------------------------------------------------
    # Memory footprint
    # ---------------------------------------------------------------
    print()
    print("Memory / scale")
    print("-" * 60)
    s = echo.summary()
    print(f"  {'Total experiences stored':<42} {s['total_experiences']}")
    print(f"  {'Total observations':<42} {s['total_observations']}")
    print(f"  {'Total patterns':<42} {s['total_patterns']}")
    print(f"  {'Total knowledge candidates':<42} {s['total_knowledge_candidates']}")

    echo.close()
    prime.shutdown()

    print()
    print("=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
