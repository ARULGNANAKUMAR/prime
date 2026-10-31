#!/usr/bin/env python3
"""
examples/phase5_demo.py — PRIME Phase 5 Reasoning Engine Demo.

Demonstrates the full pipeline:
  User Input → ATLAS → UniversalProblem → ReasoningEngine → ReasoningResult

Run:
    PYTHONPATH=python python3 examples/phase5_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from prime import (
    PRIME, make_rule, make_condition, make_compound_condition, make_conclusion,
    MockLLMProvider,
)

def separator(title=""):
    w = 60
    if title:
        print(f"\n{'=' * w}")
        print(f"  {title}")
        print(f"{'=' * w}")
    else:
        print("-" * w)

def main():
    prime    = PRIME()
    atlas    = prime.atlas()
    reasoner = prime.reasoner()

    # -----------------------------------------------------------------------
    # Demo 1: Arithmetic — ATLAS + Reasoning end-to-end
    # -----------------------------------------------------------------------
    separator("Demo 1: Arithmetic Reasoning (ATLAS → Engine)")
    problem = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    print(f"Input:  {problem.original_input!r}")
    print(f"Domain: {problem.domain}")
    print(f"Entities: {[(e['name'], e.get('quantity')) for e in problem.entities]}")

    result = reasoner.solve(problem)
    print(f"\nResult:     {result.answer} {result.answer_unit}")
    print(f"Status:     {result.status}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Steps:      {result.trace.total_steps}")
    print(result.explain())

    # -----------------------------------------------------------------------
    # Demo 2: Rule-Based Reasoning
    # -----------------------------------------------------------------------
    separator("Demo 2: Rule-Based Reasoning")
    result2 = reasoner.solve_facts(
        facts={"temperature": 120},
        rules=[
            make_rule("high_temp_warning",
                conditions=[make_condition("temperature", ">", "100")],
                conclusions=[make_conclusion("warning", True), make_conclusion("alert_level", "HIGH")],
                name="High Temperature Warning",
                priority=10),
            make_rule("critical_temp",
                conditions=[make_condition("temperature", ">", "200")],
                conclusions=[make_conclusion("shutdown", True)],
                name="Critical Temperature Shutdown",
                priority=20),
        ],
        goals=["warning", "alert_level"],
    )
    print(f"Input:          temperature = 120")
    print(f"Rule 1:         IF temperature > 100 THEN warning = true, alert_level = HIGH")
    print(f"Rule 2:         IF temperature > 200 THEN shutdown = true")
    print(f"\nStatus:         {result2.status}")
    print(f"Derived facts:  {result2.derived_facts}")
    print(f"Rules fired:    {result2.trace.rules_fired}")
    print(result2.explain())

    # -----------------------------------------------------------------------
    # Demo 3: Multi-Step Symbolic Reasoning
    # -----------------------------------------------------------------------
    separator("Demo 3: Multi-Step Symbolic Reasoning")
    result3 = reasoner.solve_facts(
        facts={"A": 10},
        expressions=[
            {"id": "e1", "op": "add",      "output_id": "B", "operand_ids": ["A", "5"],  "text": "B = A + 5"},
            {"id": "e2", "op": "multiply", "output_id": "C", "operand_ids": ["B", "2"],  "text": "C = B * 2"},
        ],
        goals=["C"],
    )
    print("Input:  A = 10")
    print("Step 1: B = A + 5 = 15")
    print("Step 2: C = B * 2 = 30")
    print(f"\nStatus: {result3.status}")
    print(f"C =     {result3.derived_facts.get('C')}")
    print("\nTrace:")
    for step in result3.trace:
        if step.type in ("FACT_INIT", "EXPRESSION_EVAL", "GOAL_CHECK"):
            print(f"  [{step.step_id}] {step.type:<20} {step.description}")

    # -----------------------------------------------------------------------
    # Demo 4: Constraint Checking
    # -----------------------------------------------------------------------
    separator("Demo 4: Constraint Reasoning")
    result4 = reasoner.solve_facts(
        facts={"budget": 40000, "timeline_weeks": 12},
        constraints=[
            {"id": "c1", "left": "budget",         "op": "<=", "right": "50000", "confidence": 1.0},
            {"id": "c2", "left": "timeline_weeks",  "op": "<=", "right": "16",    "confidence": 1.0},
        ],
    )
    print("Facts:  budget=40000, timeline=12 weeks")
    print("C1:     budget <= 50000")
    print("C2:     timeline <= 16 weeks")
    print(f"\nStatus: {result4.status}")
    for cr in result4.constraint_results:
        icon = "✓" if cr.satisfied else "✗"
        print(f"  {icon} {cr.description}")

    # -----------------------------------------------------------------------
    # Demo 5: Compound Logical Conditions
    # -----------------------------------------------------------------------
    separator("Demo 5: Compound Logical Reasoning (AND / OR / NOT)")
    result5 = reasoner.solve_facts(
        facts={"alarm": "true", "door": "true", "person_present": "false"},
        rules=[
            make_rule("security_alert",
                conditions=[
                    make_compound_condition("AND",
                        make_condition("alarm", "=", "true"),
                        make_condition("door",  "=", "true"))
                ],
                conclusions=[make_conclusion("security_breach", True)],
                name="Security Alert"),
            make_rule("all_clear",
                conditions=[
                    make_compound_condition("NOT",
                        make_condition("alarm", "=", "true"))
                ],
                conclusions=[make_conclusion("all_clear", True)],
                name="All Clear"),
        ],
        goals=["security_breach"],
    )
    print("Facts:  alarm=true, door=true, person_present=false")
    print("Rule 1: IF alarm=true AND door=true THEN security_breach=true")
    print(f"\nStatus: {result5.status}")
    print(f"Derived: {result5.derived_facts}")

    # -----------------------------------------------------------------------
    # Demo 6: Deductive Reasoning
    # -----------------------------------------------------------------------
    separator("Demo 6: Deductive Reasoning")
    result6 = reasoner.solve_facts(
        facts={"machine_temp": 120},
        rules=[
            make_rule("inspection_rule",
                conditions=[make_condition("machine_temp", ">", "100")],
                conclusions=[make_conclusion("requires_inspection", True)],
                name="Machines above 100°C require inspection"),
        ],
        goals=["requires_inspection"],
    )
    print("Fact 1: All machines above 100°C require inspection (rule)")
    print("Fact 2: machine_temp = 120°C")
    print(f"\nConclusion:  requires_inspection = {result6.derived_facts.get('requires_inspection')}")
    print(f"Status:      {result6.status}")

    # -----------------------------------------------------------------------
    # Demo 7: UNRESOLVED — correct refusal to fabricate
    # -----------------------------------------------------------------------
    separator("Demo 7: UNRESOLVED — Refusing to Fabricate")
    result7 = reasoner.solve_facts(
        facts={},  # x is unknown
        expressions=[
            {"id": "e1", "op": "add", "output_id": "result",
             "operand_ids": ["x", "10"], "text": "result = x + 10"},
        ],
        goals=["result"],
    )
    print("Input:  x = unknown (unbound)")
    print("Goal:   result = x + 10")
    print(f"\nStatus: {result7.status}  (correctly refuses to invent a value)")
    print(f"Result: {result7.derived_facts.get('result', '(none)')}")

    # -----------------------------------------------------------------------
    # Demo 8: Standalone expression evaluator
    # -----------------------------------------------------------------------
    separator("Demo 8: Safe Expression Evaluator")
    expressions = [
        ("150 / 3",           {}),
        ("(a + b) * c",       {"a": 3, "b": 7, "c": 4}),
        ("sqrt(9) + abs(-4)", {}),
        ("2 ^ 10",            {}),
        ("min(5, 3, 8, 1)",   {}),
    ]
    for expr, bindings in expressions:
        val = reasoner.eval_expression(expr, bindings)
        print(f"  {expr:<30} {'with ' + str(bindings) if bindings else '':<20} → {val}")

    # -----------------------------------------------------------------------
    # Demo 9: LLM Provider Interface (mock)
    # -----------------------------------------------------------------------
    separator("Demo 9: LLM-Assisted Reasoning (MockLLMProvider)")
    llm_provider = MockLLMProvider(
        rules=[
            make_rule("llm_rule",
                conditions=[make_condition("risk_score", ">", "70")],
                conclusions=[make_conclusion("high_risk", True)])
        ]
    )
    llm_reasoner = prime.reasoner(llm_provider=llm_provider)
    result9 = llm_reasoner.solve_facts(
        facts={"risk_score": 85},
        goals=["high_risk"],
        mode="llm_assisted",
    )
    print("LLM proposed rule: IF risk_score > 70 THEN high_risk = true")
    print(f"Input:   risk_score = 85")
    print(f"Status:  {result9.status}")
    print(f"Derived: {result9.derived_facts}")
    print(f"Note:    LLM proposal validated by deterministic rule engine")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    separator("Summary")
    print("Phase 5 Reasoning Engine demonstrated:")
    print("  ✓ ATLAS → UniversalProblem → ReasoningEngine pipeline")
    print("  ✓ Symbolic arithmetic reasoning")
    print("  ✓ Rule-based forward chaining")
    print("  ✓ Multi-step chained reasoning with trace")
    print("  ✓ Constraint satisfaction checking")
    print("  ✓ Compound conditions (AND / OR / NOT)")
    print("  ✓ Deductive reasoning with provenance")
    print("  ✓ Correct UNRESOLVED status (no fabrication)")
    print("  ✓ Safe expression evaluator (no eval())")
    print("  ✓ LLM provider interface with validation")
    print()

    prime.shutdown()

if __name__ == "__main__":
    main()
