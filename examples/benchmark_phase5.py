#!/usr/bin/env python3
"""
examples/benchmark_phase5.py — PRIME Phase 5 Performance Benchmarks.

Measures actual performance — no invented numbers.

Run:
    PYTHONPATH=python python3 examples/benchmark_phase5.py
"""

import sys, os, time, gc, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from prime import PRIME, make_rule, make_condition, make_conclusion

N = 1000   # iterations for throughput benchmarks

def bench(label, fn, iterations=N):
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / iterations * 1000
    throughput   = iterations / elapsed
    print(f"  {label:<42} {per_call_ms:8.4f} ms/call   {throughput:9.0f}/sec")
    return per_call_ms, throughput

def main():
    prime    = PRIME()
    atlas    = prime.atlas()
    reasoner = prime.reasoner()

    print("PRIME Phase 5 — Reasoning Engine Benchmarks")
    print("=" * 65)
    print(f"  Iterations per benchmark: {N}")
    print()

    # -----------------------------------------------------------------------
    # 1. Safe Expression Evaluator
    # -----------------------------------------------------------------------
    print("1. Expression Evaluator")
    bench("eval: constant (42)",
          lambda: reasoner.eval_expression("42"))
    bench("eval: simple arithmetic (150 / 3)",
          lambda: reasoner.eval_expression("150 / 3"))
    bench("eval: with variables (x * y)",
          lambda: reasoner.eval_expression("x * y", {"x": 6.0, "y": 7.0}))
    bench("eval: complex expr ((a+b)*c-d/2)",
          lambda: reasoner.eval_expression("(a+b)*c-d/2",
                                           {"a": 3, "b": 7, "c": 4, "d": 20}))
    bench("eval: with functions (sqrt + abs)",
          lambda: reasoner.eval_expression("sqrt(abs(-16)) + min(3, 1, 2)"))

    # -----------------------------------------------------------------------
    # 2. Rule Reasoning
    # -----------------------------------------------------------------------
    print("\n2. Rule Reasoning (rules/sec)")
    simple_rule = [make_rule("r1",
        conditions=[make_condition("temperature", ">", "100")],
        conclusions=[make_conclusion("warning", True)])]

    bench("rule: 1 rule, condition met",
          lambda: reasoner.solve_facts(
              facts={"temperature": 120}, rules=simple_rule))

    bench("rule: 1 rule, condition NOT met",
          lambda: reasoner.solve_facts(
              facts={"temperature": 80}, rules=simple_rule))

    # 10-rule chain
    chain_rules = [
        make_rule(f"r{i}",
            conditions=[make_condition(f"f{i}", "=", "1")],
            conclusions=[make_conclusion(f"f{i+1}", 1)])
        for i in range(10)
    ]
    bench("rule: 10-step forward chain",
          lambda: reasoner.solve_facts(
              facts={"f0": 1}, rules=chain_rules, goals=["f10"]))

    # 5 rules, compound conditions
    compound_rules = [
        make_rule(f"r{i}",
            conditions=[make_condition(f"x{i}", ">", "0"),
                        make_condition(f"y{i}", ">", "0")],
            conclusions=[make_conclusion(f"z{i}", 1)])
        for i in range(5)
    ]
    compound_facts = {f"x{i}": i+1 for i in range(5)}
    compound_facts.update({f"y{i}": i+1 for i in range(5)})
    bench("rule: 5 rules compound conditions",
          lambda: reasoner.solve_facts(
              facts=dict(compound_facts), rules=compound_rules))

    # -----------------------------------------------------------------------
    # 3. Symbolic Expression Evaluation
    # -----------------------------------------------------------------------
    print("\n3. Symbolic Evaluation (expressions/sec)")
    bench("symbolic: divide (distance/time)",
          lambda: reasoner.solve_facts(
              facts={"distance": 150, "time": 3},
              expressions=[{"id": "e1", "op": "divide", "output_id": "speed",
                            "operand_ids": ["distance", "time"], "text": ""}],
              goals=["speed"]))

    bench("symbolic: 3-step chain (A→B→C→D)",
          lambda: reasoner.solve_facts(
              facts={"A": 10},
              expressions=[
                  {"id":"e1","op":"add",     "output_id":"B","operand_ids":["A","5"],"text":""},
                  {"id":"e2","op":"multiply","output_id":"C","operand_ids":["B","2"],"text":""},
                  {"id":"e3","op":"subtract","output_id":"D","operand_ids":["C","3"],"text":""},
              ],
              goals=["D"]))

    bench("symbolic: 5-expression chain",
          lambda: reasoner.solve_facts(
              facts={"v0": 10},
              expressions=[
                  {"id":f"e{i}","op":"add","output_id":f"v{i+1}",
                   "operand_ids":[f"v{i}","1"],"text":""} for i in range(5)
              ],
              goals=["v5"]))

    # -----------------------------------------------------------------------
    # 4. Constraint Checking
    # -----------------------------------------------------------------------
    print("\n4. Constraint Checking (constraint checks/sec)")
    constraints = [
        {"id": f"c{i}", "left": f"var{i}", "op": "<=", "right": "100", "confidence": 1.0}
        for i in range(5)
    ]
    facts_for_constraints = {f"var{i}": float(i * 10) for i in range(5)}

    bench("constraint: 5 constraints (all satisfied)",
          lambda: reasoner.solve_facts(
              facts=dict(facts_for_constraints),
              constraints=constraints))

    bench("constraint: check_constraint() API",
          lambda: reasoner.check_constraint("x", "<=", "100", {"x": 50.0}))

    # -----------------------------------------------------------------------
    # 5. Hybrid reasoning (rules + symbolic)
    # -----------------------------------------------------------------------
    print("\n5. Hybrid Mode (rules + symbolic)")
    hybrid_rules = [
        make_rule("classify",
            conditions=[make_condition("speed", ">", "30")],
            conclusions=[make_conclusion("fast", True)])
    ]
    bench("hybrid: symbolic → rule",
          lambda: reasoner.solve_facts(
              facts={"distance": 150, "time": 3},
              expressions=[{"id":"e1","op":"divide","output_id":"speed",
                            "operand_ids":["distance","time"],"text":""}],
              rules=hybrid_rules,
              goals=["fast"],
              mode="hybrid"))

    # -----------------------------------------------------------------------
    # 6. ATLAS → Reasoning end-to-end
    # -----------------------------------------------------------------------
    print("\n6. ATLAS → Reasoning (end-to-end)")
    # Pre-parse to isolate reasoning latency
    problem = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    bench("solve(pre-parsed problem)",
          lambda: reasoner.solve(problem))

    bench("atlas.parse() + solve() end-to-end",
          lambda: reasoner.solve(
              atlas.parse("A car travels 150 km in 3 hours. Find average speed.")),
          iterations=200)

    # -----------------------------------------------------------------------
    # 7. Memory usage
    # -----------------------------------------------------------------------
    print("\n7. Memory Usage (tracemalloc — Python heap only)")
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()

    for _ in range(100):
        reasoner.solve_facts(
            facts={"x": 10},
            expressions=[{"id":"e1","op":"multiply","output_id":"y",
                          "operand_ids":["x","3"],"text":""}],
            goals=["y"])

    snap1 = tracemalloc.take_snapshot()
    stats = snap1.compare_to(snap0, 'lineno')
    total_kb = sum(s.size_diff for s in stats) / 1024
    print(f"  {'100 hybrid solve_facts calls':<42} {total_kb:8.1f} KB delta (Python heap)")
    tracemalloc.stop()

    # -----------------------------------------------------------------------
    # 8. Reasoning quality metrics
    # -----------------------------------------------------------------------
    print("\n8. Reasoning Quality Metrics (controlled test suite)")
    test_cases = [
        ("arithmetic",  lambda: reasoner.solve_facts(
            facts={"d":150,"t":3},
            expressions=[{"id":"e1","op":"divide","output_id":"s","operand_ids":["d","t"],"text":""}],
            goals=["s"]), "SOLVED"),
        ("rule_fires",   lambda: reasoner.solve_facts(
            facts={"temp":120},
            rules=[make_rule("r1",[make_condition("temp",">","100")],
                             [make_conclusion("warn",True)])],
            goals=["warn"]), "SOLVED"),
        ("unresolved",  lambda: reasoner.solve_facts(
            facts={},
            expressions=[{"id":"e1","op":"add","output_id":"r","operand_ids":["x","5"],"text":""}],
            goals=["r"]), "UNRESOLVED"),
        ("constraint_ok",lambda: reasoner.solve_facts(
            facts={"b":40000},
            constraints=[{"id":"c1","left":"b","op":"<=","right":"50000","confidence":1.0}]),
            "SOLVED"),
    ]

    correct = 0
    total   = len(test_cases)
    for name, fn, expected_status in test_cases:
        r = fn()
        ok = (r.status == expected_status) or (
            expected_status == "SOLVED" and r.status in ("SOLVED","PARTIALLY_SOLVED"))
        icon = "✓" if ok else "✗"
        if ok: correct += 1
        print(f"  {icon} {name:<20} expected={expected_status:<25} got={r.status}")
    print(f"  Correct: {correct}/{total} ({100*correct//total}%)")

    print("\n" + "=" * 65)
    print("Benchmarks complete. All numbers are measured, not estimated.")
    print()

    prime.shutdown()

if __name__ == "__main__":
    main()
