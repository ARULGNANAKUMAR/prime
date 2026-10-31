"""
benchmark_phase4.py — Performance benchmarks for PRIME Phase 4: ATLAS.

Measures:
  - Parsing latency (simple / medium / complex inputs)
  - Entity extraction throughput
  - JSON serialization / deserialization latency
  - Validation latency
  - Round-trip latency
  - Memory usage (approximate)

Run:
    PYTHONPATH=python python3 examples/benchmark_phase4.py
"""

import time
import tracemalloc
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from prime import ATLASEngine

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def timer(fn, n=1):
    """Return (elapsed_ms, last_result)."""
    start = time.perf_counter()
    result = None
    for _ in range(n):
        result = fn()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, result


def bench(label, fn, n=1):
    elapsed, result = timer(fn, n)
    per = elapsed / n
    if n > 1:
        print(f"  {label:<45} {elapsed:>8.2f} ms total  {per:>8.3f} ms/call  ({n} calls)")
    else:
        print(f"  {label:<45} {elapsed:>8.2f} ms")
    return result, elapsed


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------

SIMPLE   = "A car travels 120 km in 3 hours."
MEDIUM   = (
    "If a train travels 100 km in 2 hours, what is its average speed? "
    "Assume no air resistance and neglect friction. "
    "The mass of the train is 50000 kg."
)
COMPLEX  = (
    "A manufacturing machine operates at a temperature of 150 degrees C. "
    "The maximum allowed temperature is 200 degrees C. "
    "The machine processes 500 kg of material per hour at a cost of 7000 dollars. "
    "Revenue per hour is 10000 dollars. "
    "If cost > revenue, then issue a warning. "
    "Calculate the profit per hour and determine if operations are sustainable. "
    "The machine has been running for 8 hours today. "
    "Total output = rate * time. "
    "Assume constant throughput and no downtime."
)

DOMAINS = {
    "physics":     "An object has mass 10 kg and acceleration 2 m/s^2. Find the force.",
    "mathematics": "Solve x^2 + 5x + 6 = 0 by factoring the polynomial.",
    "programming": "Sort an array of integers in ascending order using a fast algorithm.",
    "business":    "Revenue is 10000 and cost is 7000. Calculate the profit margin.",
    "engineering": "A beam must support a load of 500 N. Stress must not exceed 200 Pa.",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    atlas = ATLASEngine()
    print("=" * 65)
    print("PRIME Phase 4 — ATLAS Benchmarks")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Parsing latency
    # ------------------------------------------------------------------
    print("\n1. Parsing latency\n")

    p_simple,  t_simple  = bench("parse(simple)",  lambda: atlas.parse(SIMPLE),  n=1)[0:2]  # type: ignore
    p_medium,  t_medium  = bench("parse(medium)",  lambda: atlas.parse(MEDIUM),  n=1)[0:2]  # type: ignore
    p_complex, t_complex = bench("parse(complex)", lambda: atlas.parse(COMPLEX), n=1)[0:2]  # type: ignore

    # Warm up then measure 100 runs
    bench("parse(simple)  × 100",  lambda: atlas.parse(SIMPLE),  n=100)
    bench("parse(medium)  × 100",  lambda: atlas.parse(MEDIUM),  n=100)
    bench("parse(complex) × 100",  lambda: atlas.parse(COMPLEX), n=100)

    # ------------------------------------------------------------------
    # 2. Domain classification
    # ------------------------------------------------------------------
    print("\n2. Domain classification latency\n")
    for domain, text in DOMAINS.items():
        bench(f"parse({domain})", lambda t=text: atlas.parse(t))

    # ------------------------------------------------------------------
    # 3. Entity extraction throughput
    # ------------------------------------------------------------------
    print("\n3. Entity extraction throughput\n")

    N_PARSE = 1000
    elapsed_ms, _ = timer(lambda: atlas.parse(MEDIUM), n=N_PARSE)
    p_sample = atlas.parse(MEDIUM)
    entities_per_parse = len(p_sample.entities)
    total_entities = entities_per_parse * N_PARSE
    entities_per_sec = total_entities / (elapsed_ms / 1000.0)
    print(f"  {'Entities/sec (from ' + str(N_PARSE) + ' parses)':<45} {entities_per_sec:>10,.0f}")
    print(f"  {'Entities per parse (medium)':<45} {entities_per_parse:>10}")

    # ------------------------------------------------------------------
    # 4. JSON serialization
    # ------------------------------------------------------------------
    print("\n4. JSON serialization latency\n")

    p = atlas.parse(COMPLEX)
    bench("to_json(complex problem)",        lambda: atlas.to_json(p))
    bench("to_json(complex problem) × 1000", lambda: atlas.to_json(p), n=1000)

    j = atlas.to_json(p)
    bench("from_json(complex problem)",        lambda: atlas.from_json(j))
    bench("from_json(complex problem) × 1000", lambda: atlas.from_json(j), n=1000)

    print(f"\n  {'JSON output size (complex)':<45} {len(j):>8} bytes")

    # ------------------------------------------------------------------
    # 5. Validation latency
    # ------------------------------------------------------------------
    print("\n5. Validation latency\n")
    bench("validate(complex problem)",        lambda: atlas.validate(p))
    bench("validate(complex problem) × 1000", lambda: atlas.validate(p), n=1000)

    # ------------------------------------------------------------------
    # 6. Round-trip latency
    # ------------------------------------------------------------------
    print("\n6. Round-trip latency (parse → to_json → from_json → validate)\n")

    def round_trip():
        prob = atlas.parse(MEDIUM)
        j2   = atlas.to_json(prob)
        prob2 = atlas.from_json(j2)
        atlas.validate(prob2)
        return prob2

    bench("round_trip(medium)",        round_trip)
    bench("round_trip(medium) × 100",  round_trip, n=100)

    # ------------------------------------------------------------------
    # 7. Entity/relationship counts on complex input
    # ------------------------------------------------------------------
    print("\n7. Representation counts (complex input)\n")
    pc = atlas.parse(COMPLEX)
    print(f"  {'entities':<30} {len(pc.entities)}")
    print(f"  {'relationships':<30} {len(pc.relationships)}")
    print(f"  {'constraints':<30} {len(pc.constraints)}")
    print(f"  {'goals':<30} {len(pc.goals)}")
    print(f"  {'operations':<30} {len(pc.operations)}")
    print(f"  {'expressions':<30} {len(pc.expressions)}")
    print(f"  {'assumptions':<30} {len(pc.assumptions)}")
    print(f"  {'logic_clauses':<30} {len(pc.to_dict().get('logic_clauses', []))}")
    print(f"  {'states':<30} {len(pc.states)}")
    print(f"  {'graph nodes':<30} {pc.graph['node_count']}")
    print(f"  {'graph edges':<30} {pc.graph['edge_count']}")

    # ------------------------------------------------------------------
    # 8. Memory usage (tracemalloc — Python heap only)
    # ------------------------------------------------------------------
    print("\n8. Memory usage (Python heap via tracemalloc)\n")
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    problems = [atlas.parse(MEDIUM) for _ in range(100)]
    jsons    = [atlas.to_json(p) for p in problems]

    snap2   = tracemalloc.take_snapshot()
    stats   = snap2.compare_to(snap1, 'lineno')
    py_delta = sum(s.size_diff for s in stats)

    print(f"  {'Python heap delta (100 parses + to_json)':<45} {py_delta / 1024:>8.1f} KB")
    print(f"  Note: C++ heap is not tracked by tracemalloc.")
    tracemalloc.stop()

    del problems, jsons

    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
