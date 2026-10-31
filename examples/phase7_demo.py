#!/usr/bin/env python3
"""
phase7_demo.py — PRIME Phase 7: ECHO Experience-to-Intelligence Engine

Demonstrates the full ECHO lifecycle:
  record → observe → detect patterns → generalise → extract knowledge → query

Run:
    cd PRIME
    PYTHONPATH=python python3 examples/phase7_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from prime import (
    PRIME,
    ExperienceOutcome,
    KnowledgeType,
)


def separator(title: str = "") -> None:
    print()
    if title:
        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)


def main() -> None:
    print("PRIME Phase 7 — ECHO Experience-to-Intelligence Engine")
    print("=" * 60)

    prime = PRIME()
    mosaic = prime.mosaic()
    echo   = prime.echo()

    # ---------------------------------------------------------------
    # 1. Solve several arithmetic tasks via MOSAIC and record them
    # ---------------------------------------------------------------
    separator("STEP 1: Solve tasks via MOSAIC and record experiences")

    arithmetic_tasks = [
        "A car travels 150 km in 3 hours. Calculate average speed.",
        "A train covers 240 km in 4 hours. What is its speed?",
        "A cyclist rides 90 km in 3 hours. Find the average speed.",
        "A plane flies 1200 km in 2 hours. What is its speed?",
        "A runner completes 10 km in 1 hour. What is the average speed?",
        "A boat travels 60 km in 3 hours. Calculate the speed.",
    ]

    experiences = []
    for task_text in arithmetic_tasks:
        try:
            result = mosaic.solve(task_text)
            exp = echo.record(
                problem_text     = task_text,
                execution_result = result,
                extra_tags       = ["arithmetic", "speed", "demo"],
            )
            experiences.append(exp)
            status_icon = "✓" if result.succeeded else "✗"
            print(f"  {status_icon} [{exp.outcome[:7]}] {task_text[:55]}")
            print(f"      answer={result.answer_with_unit!r}  "
                  f"confidence={result.confidence:.2f}")
        except Exception as e:
            print(f"  ! Error: {e}")

    print(f"\n  Recorded {len(experiences)} experience(s).")

    # ---------------------------------------------------------------
    # 2. Observe each experience
    # ---------------------------------------------------------------
    separator("STEP 2: Extract observations from each experience")

    all_obs = []
    for exp in experiences:
        obs_list = echo.observe(exp)
        all_obs.extend(obs_list)

    print(f"  Extracted {len(all_obs)} observation(s).")
    if all_obs:
        o = all_obs[0]
        print(f"  Sample observation:")
        print(f"    domain={o.task_domain!r}  category={o.task_category!r}")
        print(f"    primary_module={o.primary_module!r}")
        print(f"    outcome={o.outcome!r}  validated={o.validated}")

    # ---------------------------------------------------------------
    # 3. Detect patterns
    # ---------------------------------------------------------------
    separator("STEP 3: Detect patterns across observations")

    patterns = echo.detect_patterns()
    print(f"  Found {len(patterns)} pattern(s).")
    for pat in patterns:
        icon = "⚡" if pat.is_contradicted else "✓"
        print(f"  {icon} {pat.description}")
        print(f"     obs={pat.observation_count}  "
              f"succ={pat.success_count}  fail={pat.failure_count}  "
              f"confidence={pat.confidence:.2f}")

    # ---------------------------------------------------------------
    # 4. Generalise
    # ---------------------------------------------------------------
    separator("STEP 4: Generalise from patterns")

    gens = echo.generalize(patterns)
    print(f"  Created {len(gens)} generalisation(s).")
    for gen in gens:
        print(f"  ► {gen.statement}")
        if gen.limitations:
            print(f"    Limitations: {gen.limitations[0]}")

    # ---------------------------------------------------------------
    # 5. Extract knowledge candidates
    # ---------------------------------------------------------------
    separator("STEP 5: Extract knowledge candidates")

    knowledge = echo.extract_knowledge()
    print(f"  Extracted {len(knowledge)} knowledge candidate(s).")
    for kc in knowledge:
        print(f"  [{kc.knowledge_type}] conf={kc.confidence:.2f} "
              f"status={kc.validation_status}")
        print(f"    {kc.statement[:100]}…" if len(kc.statement) > 100
              else f"    {kc.statement}")

    # ---------------------------------------------------------------
    # 6. Query API
    # ---------------------------------------------------------------
    separator("STEP 6: Query the experience store")

    print("  Recent experiences (last 3):")
    for exp in echo.get_recent_experiences(n=3):
        print(f"    [{exp.outcome[:7]}] {exp.task_text[:55]}")

    print("\n  Patterns for domain 'physics':")
    for pat in echo.get_patterns(domain="physics"):
        print(f"    {pat.description[:70]}")

    print("\n  Principles:")
    for kc in echo.get_principles():
        print(f"    [{kc.validation_status}] {kc.statement[:80]}…")

    print("\n  Contradictions:", len(echo.get_contradictions()))

    # ---------------------------------------------------------------
    # 7. Similarity search
    # ---------------------------------------------------------------
    separator("STEP 7: Similarity search")

    if len(experiences) >= 2:
        target = experiences[0]
        similar = echo.find_similar(target, top_k=3)
        print(f"  Most similar to: {target.task_text[:50]}")
        for exp, score in similar:
            print(f"    score={score:.2f}  {exp.task_text[:55]}")

    # ---------------------------------------------------------------
    # 8. EchoTrace — provenance
    # ---------------------------------------------------------------
    separator("STEP 8: EchoTrace — full provenance for knowledge")

    all_kc = echo.get_knowledge()
    if all_kc:
        kc = all_kc[0]
        trace = echo.trace(kc.id)
        print(f"  Provenance for: {kc.statement[:60]}…")
        if trace:
            print(f"    confidence:        {trace['confidence']:.2f}")
            print(f"    validation_status: {trace['validation_status']}")
            print(f"    supporting patterns:    "
                  f"{len(trace['evidence']['patterns'])}")
            print(f"    supporting experiences: "
                  f"{len(trace['evidence']['experiences'])}")

    # ---------------------------------------------------------------
    # 9. Summary
    # ---------------------------------------------------------------
    separator("ECHO Summary")

    s = echo.summary()
    print(f"  Total experiences:          {s['total_experiences']}")
    print(f"  Total observations:         {s['total_observations']}")
    print(f"  Total patterns:             {s['total_patterns']}")
    print(f"  Total generalisations:      {s['total_generalizations']}")
    print(f"  Total knowledge candidates: {s['total_knowledge_candidates']}")
    print(f"  Contradicted patterns:      {s['contradicted_patterns']}")
    print(f"  High-confidence patterns:   {s['high_confidence_patterns']}")
    print(f"  Outcome distribution:       {s['outcome_distribution']}")
    print(f"  Knowledge by type:          {s['knowledge_by_type']}")

    # ---------------------------------------------------------------
    # 10. Phase boundary check
    # ---------------------------------------------------------------
    separator("Phase boundary check (ECHO MUST NOT do these)")

    boundary_items = [
        ("Update LLM weights",         False),
        ("Fine-tune models",            False),
        ("Train neural networks",       False),
        ("Perform gradient descent",    False),
        ("Modify model parameters",     False),
        ("Implement continual learning",False),
        ("Record experiences",          True),
        ("Extract observations",        True),
        ("Detect patterns",             True),
        ("Create knowledge candidates", True),
        ("Query experience store",      True),
    ]
    for item, implemented in boundary_items:
        icon = "✓" if implemented else "✗"
        label = "IMPLEMENTED" if implemented else "NOT implemented (Phase 8+)"
        print(f"  {icon} {item}: {label}")

    echo.close()
    prime.shutdown()

    print()
    print("=" * 60)
    print("  Phase 7 ECHO demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
