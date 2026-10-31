"""
phase4_demo.py — Demonstration of PRIME Phase 4: ATLAS engine.

Shows:
  - Text parsing across multiple domains
  - Structured input
  - Validation
  - JSON round-trip
  - Integration with the Phase 3 Memory Engine

Run:
    PYTHONPATH=python python3 examples/phase4_demo.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from prime import PRIME, ATLASEngine, UniversalProblemProto


def separator(title=""):
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    else:
        print()


def show_problem(problem, title=""):
    if title:
        print(f"\n  Problem: {title!r}")
    print(f"    ID              : {problem.problem_id}")
    print(f"    Domain          : {problem.domain}  (confidence={problem.domain_confidence:.2f})")
    print(f"    Entities        : {len(problem.entities)}")
    print(f"    Relationships   : {len(problem.relationships)}")
    print(f"    Constraints     : {len(problem.constraints)}")
    print(f"    Goals           : {len(problem.goals)}")
    print(f"    Operations      : {len(problem.operations)}")
    print(f"    Expressions     : {len(problem.expressions)}")
    print(f"    Assumptions     : {len(problem.assumptions)}")
    print(f"    Graph nodes/edges: {problem.graph['node_count']} / {problem.graph['edge_count']}")

    if problem.entities:
        print(f"\n    Entities:")
        for e in problem.entities[:6]:
            qty = ""
            if e.get("quantity"):
                q = e["quantity"]
                qty = f"  [{q['value']} {q['unit']}  dim={q['dimension']}]"
            print(f"      [{e['type']:10}] {e['name']!r:20}{qty}")

    if problem.goals:
        print(f"\n    Goals:")
        for g in problem.goals[:3]:
            desc = g['description'][:60]
            print(f"      {desc!r}")

    if problem.expressions:
        print(f"\n    Expressions:")
        for ex in problem.expressions[:3]:
            print(f"      {ex['text']!r}  op={ex['op']!r}")

    if problem.assumptions:
        print(f"\n    Assumptions:")
        for a in problem.assumptions[:3]:
            print(f"      {a['description']!r}")


def main():
    print("=" * 60)
    print("PRIME Phase 4 — ATLAS: Universal Problem Representation")
    print("=" * 60)
    print()
    print("ATLAS converts problems into structured representations.")
    print("ATLAS does NOT solve problems — reasoning belongs to Phase 5.")

    with PRIME() as prime:
        atlas = prime.atlas()

        # ------------------------------------------------------------------
        separator("1. Physics: motion problem")
        # ------------------------------------------------------------------
        p_physics = atlas.parse(
            "If a train travels 100 km in 2 hours, what is its average speed? "
            "Assume no air resistance."
        )
        show_problem(p_physics, "Train speed problem")

        # ------------------------------------------------------------------
        separator("2. Mathematics: equation")
        # ------------------------------------------------------------------
        p_math = atlas.parse(
            "Solve x^2 + 5x + 6 = 0 by factoring. Find all values of x."
        )
        show_problem(p_math, "Quadratic equation")

        # ------------------------------------------------------------------
        separator("3. Programming: algorithm")
        # ------------------------------------------------------------------
        p_prog = atlas.parse(
            "Sort an array of integers in ascending order. "
            "Use a fast sorting algorithm. Return the sorted array."
        )
        show_problem(p_prog, "Sorting problem")

        # ------------------------------------------------------------------
        separator("4. Business: profit calculation")
        # ------------------------------------------------------------------
        p_biz = atlas.parse(
            "Revenue is 10000 and cost is 7000. "
            "profit = revenue - cost. "
            "Calculate the profit and determine if the business is profitable."
        )
        show_problem(p_biz, "Profit calculation")

        # ------------------------------------------------------------------
        separator("5. Engineering: constraint problem")
        # ------------------------------------------------------------------
        p_eng = atlas.parse(
            "A machine operates at a temperature of 150 C. "
            "The maximum allowed temperature is temperature < 200. "
            "If temperature > 180, then issue a safety warning. "
            "Assume constant operating conditions."
        )
        show_problem(p_eng, "Temperature constraint")

        # ------------------------------------------------------------------
        separator("6. Validation")
        # ------------------------------------------------------------------
        r = atlas.validate(p_physics)
        print(f"\n  Validation result: valid={r.valid}")
        if r.errors:
            print(f"  Errors:   {r.errors}")
        if r.warnings:
            print(f"  Warnings: {r.warnings}")

        # ------------------------------------------------------------------
        separator("7. JSON round-trip")
        # ------------------------------------------------------------------
        json_str = atlas.to_json(p_physics)
        print(f"\n  Serialized to {len(json_str)} bytes of JSON")

        p_restored = atlas.from_json(json_str)
        print(f"  Restored:  id={p_restored.problem_id}  domain={p_restored.domain}")
        print(f"  Entities match: {len(p_restored.entities) == len(p_physics.entities)}")
        print(f"  Goals    match: {len(p_restored.goals)    == len(p_physics.goals)}")

        # ------------------------------------------------------------------
        separator("8. Structured input (pre-known entities)")
        # ------------------------------------------------------------------
        proto = UniversalProblemProto(
            original_input="Budget optimization problem.",
            domain="business",
            entities=[{
                "id":         "e1",
                "type":       "quantity",
                "name":       "budget",
                "quantity":   {"value": 50000.0, "unit": "USD", "dimension": "currency"},
                "confidence": 1.0,
                "attributes": [],
                "metadata":   {},
            }],
            constraints=[{
                "id":         "c1",
                "left":       "budget",
                "op":         "<=",
                "right":      "50000",
                "type":       "financial",
                "confidence": 1.0,
                "metadata":   {},
            }],
            goals=[{
                "id":            "g1",
                "description":   "maximize profit within budget",
                "target":        "profit",
                "desired_state": "profit > 0",
                "priority":      0,
                "confidence":    0.9,
                "metadata":      {},
            }],
        )
        p_structured = atlas.parse_structured(proto)
        show_problem(p_structured, "Structured business problem")

        # ------------------------------------------------------------------
        separator("9. Memory Engine integration (Phase 3 + Phase 4)")
        # ------------------------------------------------------------------
        mem = prime.memory()

        problems_to_store = [p_physics, p_math, p_prog, p_biz, p_eng]
        stored_ids = []
        for prob in problems_to_store:
            mid = mem.store(
                atlas.to_json(prob),
                memory_type="long_term",
                importance=0.7,
                metadata={
                    "domain":     prob.domain,
                    "problem_id": prob.problem_id,
                    "entities":   str(len(prob.entities)),
                },
            )
            stored_ids.append(mid)

        print(f"\n  Stored {len(stored_ids)} problem representations in Long-Term Memory")

        # Search for physics problems
        results = mem.search("train travels speed", limit=5)
        print(f"\n  Memory search 'train travels speed': {len(results)} results")
        for item in results[:2]:
            domain = item.metadata.get("domain", "?")
            pid    = item.metadata.get("problem_id", "?")
            print(f"    domain={domain}  id={pid}  importance={item.importance:.2f}")

        # Reconstruct from memory
        if results:
            p_from_mem = atlas.from_json(results[0].content)
            print(f"\n  Reconstructed from memory: domain={p_from_mem.domain}  "
                  f"entities={len(p_from_mem.entities)}")

        print(f"\n  Total items in Long-Term Memory: {mem.count('long_term')}")

        # ------------------------------------------------------------------
        separator("10. Boundary verification — ATLAS does NOT solve")
        # ------------------------------------------------------------------
        p_train = atlas.parse(
            "If a train travels 100 km in 2 hours, what is its average speed?"
        )
        print(f"\n  Input : 'If a train travels 100 km in 2 hours, what is its average speed?'")
        print(f"  Domain: {p_train.domain}")
        print(f"\n  ATLAS represents:")
        for e in p_train.entities:
            if e.get("quantity"):
                q = e["quantity"]
                print(f"    Entity: {e['name']}  = {q['value']} {q['unit']}")
        for g in p_train.goals:
            print(f"    Goal  : {g['description'][:60]}")
        print(f"\n  ATLAS does NOT output '50 km/h' — that belongs to Phase 5 (Reasoning Engine).")

    # ------------------------------------------------------------------
    separator()
    print("Phase 4 demo complete.")
    print()
    print("What ATLAS provides to Phase 5:")
    print("  problem.entities       — structured quantities and objects")
    print("  problem.relationships  — connections between entities")
    print("  problem.constraints    — logical and mathematical bounds")
    print("  problem.goals          — what needs to be determined")
    print("  problem.operations     — required computational steps")
    print("  problem.expressions    — formula structure")
    print("  problem.states         — initial problem state")
    print("  problem.graph          — connected problem graph")


if __name__ == "__main__":
    main()
