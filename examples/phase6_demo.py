#!/usr/bin/env python3
"""
examples/phase6_demo.py — PRIME Phase 6 MOSAIC Demo.

Demonstrates:
  - Dynamic composition pipeline
  - Module registry management
  - Task analysis
  - Resource-aware selection
  - Custom module registration
  - Fallback handling
  - Composition explanation

Run:
    PYTHONPATH=python python3 examples/phase6_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from prime import (
    PRIME, MOSAICEngine, ModuleMetadata, Capability, ResourceCost,
    make_rule, make_condition, make_conclusion,
)
from prime.mosaic import CompositionTraceStep

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def main():
    prime  = PRIME()
    atlas  = prime.atlas()
    mosaic = prime.mosaic()

    # -----------------------------------------------------------------------
    # Demo 1: Full pipeline — ATLAS → MOSAIC → Result
    # -----------------------------------------------------------------------
    section("Demo 1: End-to-End Arithmetic Pipeline")
    problem = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    plan    = mosaic.compose(problem)

    print(f"Input:         {problem.original_input!r}")
    print(f"Domain:        {problem.domain}")
    print(f"Plan status:   {plan.status}")
    print(f"Modules:       {' → '.join(plan.module_ids)}")
    print(f"Est. cost:     {plan.estimated_cost:.1f}")
    print()
    print(plan.explain())

    result = mosaic.execute(plan, problem)
    print(f"\nExecution:     {result.status}")
    print(f"Answer:        {result.answer_with_unit}")
    print(f"Confidence:    {result.confidence:.2f}")
    print()
    print(result.explain())

    # -----------------------------------------------------------------------
    # Demo 2: Task analysis — what capabilities are needed?
    # -----------------------------------------------------------------------
    section("Demo 2: Task Analysis (Requirement Extraction)")
    texts = [
        "A car travels 150 km in 3 hours. Find average speed.",
        "Budget is 40000. It must not exceed 50000.",
        "IF temperature > 100 THEN start cooling fan.",
    ]
    for text in texts:
        prob = atlas.parse(text)
        req  = mosaic.analyze(prob)
        print(f"Input:    {text!r}")
        print(f"Required: {req.required_capabilities}")
        print(f"Optional: {req.optional_capabilities}")
        print(f"Flags:    arithmetic={req.has_arithmetic} "
              f"rules={req.has_rules} constraints={req.has_constraints}")
        print()

    # -----------------------------------------------------------------------
    # Demo 3: Module registry inspection
    # -----------------------------------------------------------------------
    section("Demo 3: Registered Modules")
    for mid in mosaic.list_modules():
        m = mosaic.get_module(mid)
        caps = [c["name"] for c in m["capabilities"]]
        cost = m["cost"]
        print(f"  {mid:<15} caps={caps}")
        print(f"  {'':15} cost: cpu={cost['cpu_relative']:.1f} "
              f"mem={cost['memory_relative']:.1f} "
              f"model={cost['requires_model']}")

    # -----------------------------------------------------------------------
    # Demo 4: Resource-aware selection
    # -----------------------------------------------------------------------
    section("Demo 4: Resource-Aware Module Selection")

    # Build a standalone engine with two modules for the same capability
    standalone = MOSAICEngine(prime_instance=prime)
    for mid in list(standalone.list_modules()):
        standalone.remove_module(mid)

    standalone.register_module(ModuleMetadata(
        id="light_calc", name="LightCalculator",
        capabilities=[Capability("arithmetic", "arithmetic")],
        input_types=["*"], output_types=["reasoning_result"],
        cost=ResourceCost(cpu_relative=0.5, memory_relative=0.5),
        priority=5))

    standalone.register_module(ModuleMetadata(
        id="heavy_calc", name="HeavyCalculator",
        capabilities=[Capability("arithmetic", "arithmetic")],
        input_types=["*"], output_types=["reasoning_result"],
        cost=ResourceCost(cpu_relative=15.0, memory_relative=12.0),
        priority=5))

    from prime.mosaic import TaskRequirements
    req = TaskRequirements(task_id="ra", domain="math",
                           required_capabilities=["arithmetic"])

    plan_lw = standalone.compose(req, prefer_lightweight=True)
    plan_hv = standalone.compose(req, prefer_lightweight=False)

    print(f"prefer_lightweight=True  → selected: {plan_lw.module_ids}")
    print(f"prefer_lightweight=False → selected: {plan_hv.module_ids}")
    print()

    # -----------------------------------------------------------------------
    # Demo 5: Custom module registration
    # -----------------------------------------------------------------------
    section("Demo 5: Dynamic Custom Module Registration")
    call_log = []

    def my_calculator(ctx):
        """A custom arithmetic module."""
        call_log.append("my_calculator called")
        problem = ctx.get("atlas_problem")
        if problem:
            from prime.reasoning import ReasoningResult
            # Delegate to the reasoning engine
            reasoner = prime.reasoner()
            rr = reasoner.solve(problem)
            ctx["reasoning_result"] = rr
        return ctx

    ok = mosaic.register_module(
        ModuleMetadata(
            id="my_calc", name="MyCustomCalculator",
            capabilities=[Capability("arithmetic", "arithmetic"),
                          Capability("reasoning")],
            input_types=["*"], output_types=["reasoning_result"],
            priority=150,  # very high priority
        ),
        executor=my_calculator)

    print(f"Registered my_calc: {ok}")
    prob2   = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    plan2   = mosaic.compose(prob2, preferred_modules=["my_calc"],
                              excluded_modules=["reasoning"])
    print(f"Plan with my_calc: {plan2.module_ids}")
    if "my_calc" in plan2.module_ids:
        result2 = mosaic.execute(plan2, prob2)
        print(f"Result:  {result2.status} — answer: {result2.answer}")
        print(f"my_calculator called: {len(call_log)} times")
    mosaic.remove_module("my_calc")

    # -----------------------------------------------------------------------
    # Demo 6: Fallback handling
    # -----------------------------------------------------------------------
    section("Demo 6: Fallback Module Handling")

    calls = []

    def failing_module(ctx):
        calls.append("primary")
        raise RuntimeError("Primary module failed (intentional)")

    def fallback_module(ctx):
        calls.append("fallback")
        # Delegate to real reasoning
        problem = ctx.get("atlas_problem")
        if problem:
            from prime.reasoning import ReasoningResult
            reasoner = prime.reasoner()
            ctx["reasoning_result"] = reasoner.solve(problem)
        return ctx

    mosaic.register_module(
        ModuleMetadata(id="primary_demo", name="PrimaryDemo",
                       capabilities=[Capability("arithmetic", "arithmetic"),
                                     Capability("reasoning")],
                       input_types=["*"], output_types=["reasoning_result"],
                       priority=200),
        executor=failing_module)
    mosaic.register_module(
        ModuleMetadata(id="fallback_demo", name="FallbackDemo",
                       capabilities=[Capability("arithmetic", "arithmetic"),
                                     Capability("reasoning")],
                       input_types=["*"], output_types=["reasoning_result"],
                       priority=1),
        executor=fallback_module,
        fallback_for="primary_demo")

    prob3  = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    plan3  = mosaic.compose(prob3, preferred_modules=["primary_demo"],
                             excluded_modules=["reasoning"])
    if "primary_demo" in plan3.module_ids:
        result3 = mosaic.execute(plan3, prob3, allow_fallback=True)
        print(f"Primary failed → Fallback ran: {calls}")
        print(f"Status: {result3.status}")
        for s in result3.trace_steps:
            fb = " [FALLBACK]" if s.fallback_used else ""
            print(f"  [{s.step_index+1}] {s.module_id}{fb} → {s.status}")
    else:
        print("(primary_demo not in plan — skipping fallback demo)")

    mosaic.remove_module("primary_demo")
    mosaic.remove_module("fallback_demo")

    # -----------------------------------------------------------------------
    # Demo 7: LLM module availability
    # -----------------------------------------------------------------------
    section("Demo 7: LLM Module Availability")
    llm_status = mosaic.get_module("llm")
    print(f"LLM module status: {llm_status['status'] if llm_status else 'not registered'}")

    prob4 = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    plan_nollm = mosaic.compose(prob4, allow_llm=False)
    plan_llm   = mosaic.compose(prob4, allow_llm=True)
    print(f"allow_llm=False → modules: {plan_nollm.module_ids}")
    print(f"allow_llm=True  → modules: {plan_llm.module_ids}")
    print(f"LLM selected when allowed: {'llm' in plan_llm.module_ids}")

    # -----------------------------------------------------------------------
    # Demo 8: Cross-domain composition
    # -----------------------------------------------------------------------
    section("Demo 8: Cross-Domain Composition")
    domains = [
        ("Mathematics",  "A car travels 150 km in 3 hours. Find average speed."),
        ("Physics",      "Force is 50 N and mass is 10 kg. Find acceleration."),
        ("Business",     "Revenue is 1000 and costs are 600. Find profit."),
        ("Engineering",  "Budget is 40000. Budget must be at most 50000."),
    ]
    for domain, text in domains:
        prob   = atlas.parse(text)
        plan   = mosaic.compose(prob)
        result = mosaic.execute(plan, prob) if plan.valid else None
        status = result.status if result else plan.status
        answer = result.answer if result and result.answer else "—"
        print(f"  {domain:<15} → {' → '.join(plan.module_ids) if plan.valid else plan.status:<40} answer={answer!r}")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    section("Summary")
    print("MOSAIC Phase 6 demonstrated:")
    print("  ✓ End-to-end pipeline: ATLAS → compose → execute")
    print("  ✓ Task analysis: requirement extraction from ATLAS problem")
    print("  ✓ Module registry: registration, lookup, status management")
    print("  ✓ Resource-aware selection: lightweight vs heavyweight")
    print("  ✓ Custom module registration with executor callback")
    print("  ✓ Fallback module support on primary failure")
    print("  ✓ LLM module availability policy (allow_llm flag)")
    print("  ✓ Cross-domain composition: math, physics, business, engineering")
    print("  ✓ Explainable plans: selection reasons from actual metadata")
    print("  ✓ Bounded, inspectable execution (no autonomous agent loop)")
    print()

    prime.shutdown()

if __name__ == "__main__":
    main()
