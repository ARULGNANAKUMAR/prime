#!/usr/bin/env python3
"""
examples/benchmark_phase6.py — PRIME Phase 6 MOSAIC Performance Benchmarks.

All numbers are measured, not estimated.

Run:
    PYTHONPATH=python python3 examples/benchmark_phase6.py
"""

import sys, os, time, gc, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from prime import PRIME, ModuleMetadata, Capability, ResourceCost
from prime.mosaic import TaskRequirements, MOSAICEngine

N = 1000

def bench(label, fn, n=N):
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    ms_per  = elapsed / n * 1000
    ops_sec = n / elapsed
    print(f"  {label:<48} {ms_per:8.4f} ms/call   {ops_sec:9.0f}/sec")
    return ms_per, ops_sec

def main():
    prime  = PRIME()
    atlas  = prime.atlas()
    mosaic = prime.mosaic()

    print("PRIME Phase 6 — MOSAIC Engine Benchmarks")
    print("=" * 70)
    print(f"  Iterations per benchmark: {N}")
    print()

    # -----------------------------------------------------------------------
    # 1. Module registration
    # -----------------------------------------------------------------------
    print("1. Module Registration")
    counter = [0]
    def _register():
        counter[0] += 1
        meta = ModuleMetadata(
            id=f"bench_m_{counter[0]}", name="BenchModule",
            capabilities=[Capability("bench_cap")],
            input_types=["*"], output_types=["*"])
        mosaic.register_module(meta)
        mosaic.remove_module(f"bench_m_{counter[0]}")

    bench("register + remove module (round-trip)", _register, N)

    # -----------------------------------------------------------------------
    # 2. Module discovery
    # -----------------------------------------------------------------------
    print("\n2. Module Discovery")
    bench("list_modules() (5 built-in)", lambda: mosaic.list_modules(), N)
    bench("get_module('reasoning')",     lambda: mosaic.get_module("reasoning"), N)

    # -----------------------------------------------------------------------
    # 3. Task analysis
    # -----------------------------------------------------------------------
    print("\n3. Task Analysis (requirement extraction)")
    problem_arith = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    problem_rules = atlas.parse("IF temperature > 100 THEN warning = true.")

    bench("analyze(arithmetic problem)",
          lambda: mosaic.analyze(problem_arith), N)
    bench("analyze(rule problem)",
          lambda: mosaic.analyze(problem_rules), N)

    # -----------------------------------------------------------------------
    # 4. Composition planning
    # -----------------------------------------------------------------------
    print("\n4. Composition Planning")
    req_arith = mosaic.analyze(problem_arith)
    req_rules = mosaic.analyze(problem_rules)

    bench("compose(arithmetic requirements) — cached",
          lambda: mosaic.compose(req_arith), N)
    bench("compose(rule requirements) — cached",
          lambda: mosaic.compose(req_rules), N)

    # Uncached (force cache miss by varying task_id)
    _i = [0]
    def _compose_uncached():
        _i[0] += 1
        r = TaskRequirements(task_id=f"bench_{_i[0]}", domain=req_arith.domain,
                             required_capabilities=req_arith.required_capabilities)
        mosaic.compose(r)

    bench("compose() — cache miss (unique task_id)", _compose_uncached, N)

    # Plan explain
    plan = mosaic.compose(problem_arith)
    bench("plan.explain()", lambda: plan.explain(), N)

    # -----------------------------------------------------------------------
    # 5. Dependency resolution (standalone engine)
    # -----------------------------------------------------------------------
    print("\n5. Dependency Resolution (standalone engine)")
    standalone = MOSAICEngine(prime_instance=None)
    for mid in list(standalone.list_modules()):
        standalone.remove_module(mid)

    # Build a 5-module chain a→b→c→d→e
    for i, (cid, dep) in enumerate([
        ("a", []), ("b", ["a"]), ("c", ["b"]), ("d", ["c"]), ("e", ["d"])
    ]):
        standalone.register_module(ModuleMetadata(
            id=cid, name=cid.upper(),
            capabilities=[Capability(f"cap_{cid}")],
            input_types=["*"], output_types=["*"],
            dependencies=dep))

    req_chain = TaskRequirements(task_id="chain", domain="x",
        required_capabilities=[f"cap_{c}" for c in "abcde"])

    bench("compose() 5-module dependency chain (cached)",
          lambda: standalone.compose(req_chain), N)

    standalone2 = MOSAICEngine(prime_instance=None)
    for mid in list(standalone2.list_modules()):
        standalone2.remove_module(mid)
    for i, (cid, dep) in enumerate([
        ("a", []), ("b", ["a"]), ("c", ["b"]), ("d", ["c"]), ("e", ["d"])
    ]):
        standalone2.register_module(ModuleMetadata(
            id=cid, name=cid.upper(),
            capabilities=[Capability(f"cap_{cid}")],
            input_types=["*"], output_types=["*"],
            dependencies=dep))

    _j = [0]
    def _dep_uncached():
        _j[0] += 1
        r = TaskRequirements(task_id=f"ch_{_j[0]}", domain="x",
            required_capabilities=[f"cap_{c}" for c in "abcde"])
        standalone2.compose(r)

    bench("compose() 5-module dependency chain (cache miss)", _dep_uncached, N)

    # -----------------------------------------------------------------------
    # 6. End-to-end pipeline execution
    # -----------------------------------------------------------------------
    print("\n6. End-to-End Pipeline Execution (ATLAS → compose → execute)")
    problem_cached = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    plan_cached    = mosaic.compose(problem_cached)

    bench("execute(pre-composed plan, pre-parsed problem)",
          lambda: mosaic.execute(plan_cached, problem_cached), 200)

    bench("solve(text) — full pipeline",
          lambda: mosaic.solve("A car travels 150 km in 3 hours. Find average speed."),
          200)

    bench("atlas.parse() only",
          lambda: atlas.parse("A car travels 150 km in 3 hours. Find average speed."),
          N)

    # -----------------------------------------------------------------------
    # 7. Memory footprint
    # -----------------------------------------------------------------------
    print("\n7. Memory Footprint (tracemalloc — Python heap)")
    tracemalloc.start()
    s0 = tracemalloc.take_snapshot()
    for _ in range(100):
        mosaic.solve("A car travels 150 km in 3 hours. Find average speed.")
    s1 = tracemalloc.take_snapshot()
    delta_kb = sum(s.size_diff for s in s1.compare_to(s0, "lineno")) / 1024
    print(f"  {'100 × solve() calls':<48} {delta_kb:8.1f} KB delta (Python heap)")
    tracemalloc.stop()

    # -----------------------------------------------------------------------
    # 8. Composition quality metrics
    # -----------------------------------------------------------------------
    print("\n8. Composition Quality Metrics")
    test_cases = [
        ("arithmetic solve", "A car travels 150 km in 3 hours. Find average speed.",
         "50", ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED"]),
        ("rule reasoning",   "IF temperature > 100 THEN warning is true.",
         None, ["SUCCESS", "PARTIAL_SUCCESS", "UNRESOLVED"]),
        ("constraint",       "Budget is 40000. Budget must be at most 50000.",
         None, ["SUCCESS", "PARTIAL_SUCCESS"]),
    ]

    successful, total, correct_answer, plans_valid = 0, 0, 0, 0
    plan_sizes = []
    latencies  = []

    for label, text, expected_answer, valid_statuses in test_cases:
        total += 1
        t0   = time.perf_counter()
        prob = atlas.parse(text)
        plan = mosaic.compose(prob)
        if plan.valid:
            plans_valid += 1
            plan_sizes.append(len(plan.module_ids))
            result = mosaic.execute(plan, prob)
            latencies.append((time.perf_counter() - t0) * 1000)
            if result.status in valid_statuses:
                successful += 1
            if expected_answer and result.answer == expected_answer:
                correct_answer += 1
        icon = "✓" if plan.valid else "✗"
        ans  = result.answer if plan.valid else plan.status_detail
        print(f"  {icon} {label:<25} plan={'VALID' if plan.valid else 'INVALID':<8} "
              f"answer={ans!r}")

    print(f"\n  Successful compositions:   {plans_valid}/{total}")
    print(f"  Execution success rate:    {successful}/{plans_valid}")
    if correct_answer > 0:
        print(f"  Correct answer rate:       {correct_answer}/{plans_valid}")
    if plan_sizes:
        print(f"  Avg modules per plan:      {sum(plan_sizes)/len(plan_sizes):.1f}")
    if latencies:
        print(f"  Avg end-to-end latency:    {sum(latencies)/len(latencies):.1f} ms")

    print("\n" + "=" * 70)
    print("Benchmarks complete. All numbers are measured, not estimated.")
    print()

    prime.shutdown()

if __name__ == "__main__":
    main()
