"""
tests/python/test_mosaic.py — PRIME Phase 6 MOSAIC Tests.

Covers:
  - ModuleRegistry (registration, duplicate, lookup, removal, capability search)
  - CapabilityRegistry
  - TaskAnalyzer (requirement extraction)
  - CompositionPlanner (matching, dependency, input/output, cycle detection)
  - Execution engine (sequential, context passing, failure, fallback, limits)
  - All 8 required test cases from the spec
  - Cross-domain composition
  - Determinism
  - Resource-aware selection
  - LLM module availability
  - Custom module registration
  - Composition trace
  - Python API (MOSAICEngine, CompositionPlan, ExecutionResult)
  - PRIME integration (ATLAS + MOSAIC end-to-end)
  - Backward compatibility with Phases 1–5
"""

import sys, os, time, uuid
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import _prime_mosaic
from prime import (
    PRIME, MOSAICEngine, MOSAICError, CompositionPlan, ExecutionResult,
    TaskRequirements, ModuleMetadata, Capability, ResourceCost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cap(name: str, category: str = "general",
        inp=None, out=None) -> dict:
    return {"name": name, "category": category,
            "input_types": inp or [], "output_types": out or [],
            "description": name, "confidence": 1.0}


def module_dict(mid: str, name: str, caps: list,
                inp=None, out=None, deps=None,
                cpu=1.0, mem=1.0, latency=0.0, model=False,
                priority=0, status="available",
                is_fallback=False, fallback_for="") -> dict:
    return {
        "id": mid, "name": name, "version": "1.0",
        "capabilities": caps,
        "input_types":  inp  or ["*"],
        "output_types": out  or ["*"],
        "dependencies": deps or [],
        "cost": {"cpu_relative": cpu, "memory_relative": mem,
                 "latency_ms": latency, "requires_model": model},
        "priority": priority, "deterministic": True,
        "status": status,
        "is_fallback": is_fallback, "fallback_for": fallback_for,
    }


def make_engine(cfg=None) -> MOSAICEngine:
    """Create a bare MOSAICEngine (no built-in modules)."""
    e = MOSAICEngine(prime_instance=None, config=cfg or {})
    # Remove built-in modules so tests start clean
    for mid in list(e.list_modules()):
        e.remove_module(mid)
    return e


# ===========================================================================
# 1. C extension basics
# ===========================================================================
class TestCExtension(unittest.TestCase):
    def test_version_string(self):
        self.assertIsInstance(_prime_mosaic.VERSION, str)
        self.assertTrue(_prime_mosaic.VERSION.startswith("0.6"))

    def test_create_engine(self):
        cap = _prime_mosaic.create_engine({})
        self.assertIsNotNone(cap)

    def test_error_type(self):
        self.assertTrue(issubclass(_prime_mosaic.MOSAICError, Exception))

    def test_register_and_list(self):
        cap_obj = _prime_mosaic.create_engine({})
        m = module_dict("m1", "M1", [cap("arithmetic", "arithmetic")])
        ok = _prime_mosaic.register_module(cap_obj, m)
        self.assertTrue(ok)
        ids = _prime_mosaic.list_modules(cap_obj)
        self.assertIn("m1", ids)

    def test_duplicate_rejected(self):
        cap_obj = _prime_mosaic.create_engine({})
        m = module_dict("dup", "Dup", [cap("x")])
        _prime_mosaic.register_module(cap_obj, m)
        ok2 = _prime_mosaic.register_module(cap_obj, m)
        self.assertFalse(ok2)

    def test_remove_module(self):
        cap_obj = _prime_mosaic.create_engine({})
        _prime_mosaic.register_module(cap_obj, module_dict("r1", "R1", [cap("x")]))
        self.assertTrue(_prime_mosaic.remove_module(cap_obj, "r1"))
        self.assertFalse("r1" in _prime_mosaic.list_modules(cap_obj))

    def test_get_module_returns_none_for_missing(self):
        cap_obj = _prime_mosaic.create_engine({})
        self.assertIsNone(_prime_mosaic.get_module(cap_obj, "does_not_exist"))

    def test_get_module_returns_dict(self):
        cap_obj = _prime_mosaic.create_engine({})
        _prime_mosaic.register_module(cap_obj, module_dict("m1", "M1", [cap("x")]))
        d = _prime_mosaic.get_module(cap_obj, "m1")
        self.assertIsNotNone(d)
        self.assertEqual(d["id"], "m1")
        self.assertEqual(d["name"], "M1")

    def test_set_module_status(self):
        cap_obj = _prime_mosaic.create_engine({})
        _prime_mosaic.register_module(cap_obj, module_dict("m1", "M1", [cap("x")]))
        ok = _prime_mosaic.set_module_status(cap_obj, "m1", "unavailable")
        self.assertTrue(ok)
        d = _prime_mosaic.get_module(cap_obj, "m1")
        self.assertEqual(d["status"], "unavailable")


# ===========================================================================
# 2. Module Registry via Python API
# ===========================================================================
class TestModuleRegistry(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_register_module(self):
        meta = ModuleMetadata(
            id="calc", name="Calculator",
            capabilities=[Capability("arithmetic", "arithmetic")],
            input_types=["*"], output_types=["*"])
        ok = self.engine.register_module(meta)
        self.assertTrue(ok)
        self.assertIn("calc", self.engine.list_modules())

    def test_duplicate_registration_rejected(self):
        meta = ModuleMetadata(id="dup", name="Dup",
                              capabilities=[Capability("x")])
        self.engine.register_module(meta)
        ok2 = self.engine.register_module(meta)
        self.assertFalse(ok2)

    def test_list_modules_sorted(self):
        for mid in ["z_mod", "a_mod", "m_mod"]:
            self.engine.register_module(
                ModuleMetadata(id=mid, name=mid,
                               capabilities=[Capability("x")]))
        ids = self.engine.list_modules()
        self.assertEqual(ids, sorted(ids))

    def test_remove_module(self):
        self.engine.register_module(
            ModuleMetadata(id="rm1", name="RM1",
                           capabilities=[Capability("y")]))
        ok = self.engine.remove_module("rm1")
        self.assertTrue(ok)
        self.assertNotIn("rm1", self.engine.list_modules())

    def test_remove_nonexistent_returns_false(self):
        self.assertFalse(self.engine.remove_module("ghost"))

    def test_get_module_metadata(self):
        self.engine.register_module(
            ModuleMetadata(id="g1", name="G1",
                           capabilities=[Capability("z")],
                           priority=5))
        d = self.engine.get_module("g1")
        self.assertIsNotNone(d)
        self.assertEqual(d["priority"], 5)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.engine.get_module("missing"))

    def test_capability_search_via_compose(self):
        """Modules supporting a capability are discoverable through compose."""
        self.engine.register_module(
            ModuleMetadata(id="arith", name="Arithmetic",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"]))
        req = TaskRequirements(task_id="t1", domain="math",
                               required_capabilities=["arithmetic"])
        plan = self.engine.compose(req)
        self.assertIn("arith", plan.module_ids)

    def test_unavailable_module_not_selected(self):
        self.engine.register_module(
            ModuleMetadata(id="ua", name="Unavailable",
                           capabilities=[Capability("special")],
                           input_types=["*"], output_types=["*"],
                           status="unavailable"))
        req = TaskRequirements(task_id="t1", domain="x",
                               required_capabilities=["special"])
        plan = self.engine.compose(req)
        # Should fail — no available module
        self.assertNotEqual(plan.status, "VALID")

    def test_module_status_update(self):
        self.engine.register_module(
            ModuleMetadata(id="toggled", name="Toggled",
                           capabilities=[Capability("togglecap")],
                           input_types=["*"], output_types=["*"]))
        # Mark unavailable
        self.engine.update_module_status("toggled", "unavailable")
        d = self.engine.get_module("toggled")
        self.assertEqual(d["status"], "unavailable")
        # Mark available again
        self.engine.update_module_status("toggled", "available")
        d = self.engine.get_module("toggled")
        self.assertEqual(d["status"], "available")


# ===========================================================================
# 3. Task Analyzer
# ===========================================================================
class TestTaskAnalyzer(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def _analyze(self, **kwargs):
        defaults = {
            "task_id": "t1", "domain": "physics",
            "has_expressions": False, "has_rules": False,
            "has_constraints": False, "has_goals": False,
            "has_entities": True, "has_logic_clauses": False,
            "has_relationships": False, "entity_count": 2,
            "has_memory_context": False,
            "operation_types": [], "goal_targets": [],
        }
        defaults.update(kwargs)
        raw = _prime_mosaic.analyze_task(self.engine._capsule, defaults)
        return TaskRequirements(**raw)

    def test_arithmetic_with_expressions(self):
        req = self._analyze(has_expressions=True)
        self.assertIn("arithmetic", req.required_capabilities)
        self.assertIn("symbolic_reasoning", req.required_capabilities)

    def test_rule_reasoning_with_rules(self):
        req = self._analyze(has_rules=True)
        self.assertIn("rule_reasoning", req.required_capabilities)
        self.assertTrue(req.has_rules)

    def test_constraint_with_constraints(self):
        req = self._analyze(has_constraints=True)
        self.assertIn("constraint_checking", req.required_capabilities)
        self.assertTrue(req.has_constraints)

    def test_memory_context_adds_retrieval(self):
        req = self._analyze(has_memory_context=True)
        self.assertIn("memory_retrieval", req.required_capabilities)
        self.assertTrue(req.needs_memory)

    def test_atlas_representation_always_required(self):
        req = self._analyze()
        self.assertIn("atlas_representation", req.required_capabilities)

    def test_reasoning_always_required(self):
        req = self._analyze()
        self.assertIn("reasoning", req.required_capabilities)

    def test_no_duplicates_in_capabilities(self):
        req = self._analyze(has_expressions=True, has_rules=True)
        seen = set()
        for c in req.required_capabilities:
            self.assertNotIn(c, seen, f"Duplicate capability: {c}")
            seen.add(c)

    def test_arithmetic_ops_trigger_arithmetic(self):
        req = self._analyze(operation_types=["divide"])
        self.assertIn("arithmetic", req.required_capabilities)

    def test_validation_in_optional_when_goals(self):
        req = self._analyze(has_goals=True, goal_targets=["speed"])
        self.assertIn("validation", req.optional_capabilities)


# ===========================================================================
# 4. Composition Planner
# ===========================================================================
class TestCompositionPlanner(unittest.TestCase):
    def _engine_with(self, *mods) -> MOSAICEngine:
        e = make_engine()
        for m in mods:
            e.register_module(m if isinstance(m, ModuleMetadata)
                              else ModuleMetadata(**m))
        return e

    def test_simple_capability_match(self):
        e = self._engine_with(
            ModuleMetadata(id="a", name="A",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"]))
        req = TaskRequirements(task_id="t", domain="math",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req)
        self.assertTrue(plan.valid)
        self.assertIn("a", plan.module_ids)

    def test_missing_capability_returns_error(self):
        e = make_engine()
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["nonexistent_cap"])
        plan = e.compose(req)
        self.assertFalse(plan.valid)
        self.assertIn("MISSING", plan.status)

    def test_dependency_ordering(self):
        e = self._engine_with(
            ModuleMetadata(id="b", name="B",
                           capabilities=[Capability("cap_b")],
                           input_types=["cap_a_output"], output_types=["cap_b_output"],
                           dependencies=["a"]),
            ModuleMetadata(id="a", name="A",
                           capabilities=[Capability("cap_a")],
                           input_types=["*"], output_types=["cap_a_output", "cap_b_output"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["cap_a", "cap_b"])
        plan = e.compose(req)
        self.assertTrue(plan.valid)
        # 'a' must come before 'b'
        idx_a = plan.module_ids.index("a")
        idx_b = plan.module_ids.index("b")
        self.assertLess(idx_a, idx_b)

    def test_cycle_detection(self):
        """A → B → A must be rejected as COMPOSITION_CYCLE."""
        e = self._engine_with(
            ModuleMetadata(id="a", name="A",
                           capabilities=[Capability("cap_a")],
                           input_types=["*"], output_types=["*"],
                           dependencies=["b"]),
            ModuleMetadata(id="b", name="B",
                           capabilities=[Capability("cap_b")],
                           input_types=["*"], output_types=["*"],
                           dependencies=["a"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["cap_a", "cap_b"])
        plan = e.compose(req)
        self.assertEqual(plan.status, "COMPOSITION_CYCLE")

    def test_input_mismatch_rejected(self):
        """Module requires input type 'foo_type' but pipeline only provides raw_text."""
        e = self._engine_with(
            ModuleMetadata(id="m", name="M",
                           capabilities=[Capability("special")],
                           input_types=["foo_type"],  # pipeline never produces this
                           output_types=["*"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["special"])
        plan = e.compose(req)
        self.assertEqual(plan.status, "INPUT_MISMATCH")

    def test_priority_determines_selection(self):
        """Higher priority wins when two modules satisfy same capability."""
        e = self._engine_with(
            ModuleMetadata(id="low_prio", name="LowPrio",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           priority=1),
            ModuleMetadata(id="high_prio", name="HighPrio",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           priority=10))
        req = TaskRequirements(task_id="t", domain="math",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req)
        self.assertIn("high_prio", plan.module_ids)
        self.assertNotIn("low_prio", plan.module_ids)

    def test_max_modules_limit(self):
        """Plan fails with RESOURCE_EXCEEDED if max_modules exceeded."""
        e = make_engine()
        for i in range(6):
            e.register_module(ModuleMetadata(
                id=f"m{i}", name=f"M{i}",
                capabilities=[Capability(f"cap{i}")],
                input_types=["*"], output_types=["*"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=[f"cap{i}" for i in range(6)])
        plan = e.compose(req, max_modules=3)
        self.assertNotEqual(plan.status, "VALID")

    def test_excluded_modules_not_selected(self):
        e = self._engine_with(
            ModuleMetadata(id="excluded", name="Excluded",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           priority=99),
            ModuleMetadata(id="allowed", name="Allowed",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           priority=1))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req, excluded_modules=["excluded"])
        self.assertNotIn("excluded", plan.module_ids)
        self.assertIn("allowed", plan.module_ids)

    def test_preferred_modules_selected(self):
        e = self._engine_with(
            ModuleMetadata(id="preferred", name="Preferred",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           priority=1),
            ModuleMetadata(id="other", name="Other",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           priority=10))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req, preferred_modules=["preferred"])
        self.assertIn("preferred", plan.module_ids)

    def test_llm_not_selected_without_allow_llm(self):
        e = self._engine_with(
            ModuleMetadata(id="llm_mod", name="LLM",
                           capabilities=[Capability("text_generation", "language")],
                           input_types=["*"], output_types=["text_output"],
                           cost=ResourceCost(requires_model=True)))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["text_generation"])
        # allow_llm=False by default → MISSING_CAPABILITY
        plan = e.compose(req, allow_llm=False)
        self.assertNotEqual(plan.status, "VALID")

    def test_llm_selected_when_allow_llm(self):
        e = self._engine_with(
            ModuleMetadata(id="llm_mod", name="LLM",
                           capabilities=[Capability("text_generation", "language")],
                           input_types=["*"], output_types=["text_output"],
                           cost=ResourceCost(requires_model=True)))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["text_generation"])
        plan = e.compose(req, allow_llm=True)
        self.assertTrue(plan.valid)
        self.assertIn("llm_mod", plan.module_ids)

    def test_lightweight_preferred(self):
        """Lightweight module beats heavy when prefer_lightweight=True."""
        e = self._engine_with(
            ModuleMetadata(id="light", name="Light",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           cost=ResourceCost(cpu_relative=0.5, memory_relative=0.5),
                           priority=5),
            ModuleMetadata(id="heavy", name="Heavy",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["reasoning_result"],
                           cost=ResourceCost(cpu_relative=10.0, memory_relative=10.0),
                           priority=5))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req, prefer_lightweight=True)
        self.assertIn("light", plan.module_ids)
        self.assertNotIn("heavy", plan.module_ids)

    def test_plan_explain_returns_string(self):
        e = self._engine_with(
            ModuleMetadata(id="a", name="A",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["*"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req)
        explanation = plan.explain()
        self.assertIsInstance(explanation, str)
        self.assertIn("MOSAIC", explanation)

    def test_selection_reasons_populated(self):
        e = self._engine_with(
            ModuleMetadata(id="a", name="A",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["*"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req)
        if plan.valid:
            self.assertTrue(len(plan.selection_reasons) > 0)

    def test_estimated_cost_non_negative(self):
        e = self._engine_with(
            ModuleMetadata(id="a", name="A",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["*"]))
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        plan = e.compose(req)
        self.assertGreaterEqual(plan.estimated_cost, 0.0)


# ===========================================================================
# 5. Composition Plan cache and determinism
# ===========================================================================
class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.register_module(ModuleMetadata(
            id="a", name="A",
            capabilities=[Capability("arithmetic", "arithmetic"),
                          Capability("reasoning")],
            input_types=["*"], output_types=["reasoning_result"]))

    def test_same_input_same_plan(self):
        req = TaskRequirements(task_id="t1", domain="math",
                               required_capabilities=["arithmetic"])
        plan1 = self.engine.compose(req)
        plan2 = self.engine.compose(req)
        self.assertEqual(plan1.module_ids, plan2.module_ids)
        self.assertEqual(plan1.status, plan2.status)

    def test_different_tasks_may_differ(self):
        req1 = TaskRequirements(task_id="t1", domain="math",
                                required_capabilities=["arithmetic"])
        req2 = TaskRequirements(task_id="t2", domain="knowledge",
                                required_capabilities=["reasoning"])
        plan1 = self.engine.compose(req1)
        plan2 = self.engine.compose(req2)
        # Both valid, same module (it has both caps) — plans are equal here
        self.assertEqual(plan1.status, plan2.status)


# ===========================================================================
# 6. Execution Engine
# ===========================================================================
class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        self.prime = PRIME()
        self.engine = self.prime.mosaic()

    def tearDown(self):
        self.prime.shutdown()

    def test_execute_valid_plan(self):
        atlas    = self.prime.atlas()
        problem  = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
        plan     = self.engine.compose(problem)
        self.assertTrue(plan.valid)
        result   = self.engine.execute(plan, problem)
        self.assertIn(result.status, ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED"])

    def test_execute_invalid_plan_returns_error(self):
        # Manually build an invalid plan
        from prime.mosaic import CompositionPlan as CP
        raw = {"task_id": "bad", "domain": "x", "status": "MISSING_CAPABILITY",
               "status_detail": "test", "module_ids": [], "selection_reasons": [],
               "estimated_cost": 0.0, "valid": False, "requirements": {}}
        bad_plan = CP(raw, self.engine)
        result = self.engine.execute(bad_plan)
        self.assertEqual(result.status, "INVALID_PLAN")

    def test_context_passes_through_pipeline(self):
        """atlas_problem set in initial context reaches reasoning adapter."""
        atlas   = self.prime.atlas()
        problem = atlas.parse("Find average speed: 150 km in 3 hours.")
        plan    = self.engine.compose(problem)
        result  = self.engine.execute(plan, problem)
        # If reasoning ran, it should produce an answer
        if result.status == "SUCCESS":
            self.assertIsNotNone(result.answer)

    def test_unknown_module_in_plan_skipped(self):
        """If a module_id has no adapter, the step is skipped (not crashed)."""
        atlas   = self.prime.atlas()
        problem = atlas.parse("A = 10.")
        plan    = self.engine.compose(problem)
        # Inject a ghost module ID into the plan
        plan.module_ids.append("ghost_module_xyz")
        result  = self.engine.execute(plan, problem)
        skipped = [s for s in result.trace_steps if s.status == "skipped"]
        self.assertTrue(any(s.module_id == "ghost_module_xyz" for s in skipped))

    def test_trace_steps_populated(self):
        atlas   = self.prime.atlas()
        problem = atlas.parse("A car travels 150 km in 3 hours.")
        plan    = self.engine.compose(problem)
        result  = self.engine.execute(plan, problem)
        self.assertGreater(len(result.trace_steps), 0)

    def test_trace_step_has_required_fields(self):
        atlas   = self.prime.atlas()
        problem = atlas.parse("A car travels 150 km in 3 hours.")
        plan    = self.engine.compose(problem)
        result  = self.engine.execute(plan, problem)
        for step in result.trace_steps:
            self.assertIsNotNone(step.step_index)
            self.assertIsNotNone(step.module_id)
            self.assertIsNotNone(step.status)

    def test_solve_convenience(self):
        result = self.engine.solve("A car travels 150 km in 3 hours. Find average speed.")
        self.assertIn(result.status, ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED"])
        self.assertEqual(result.answer, "50")


# ===========================================================================
# 7. Required Test Cases
# ===========================================================================
class TestRequiredCases(unittest.TestCase):
    def setUp(self):
        self.prime  = PRIME()
        self.atlas  = self.prime.atlas()
        self.mosaic = self.prime.mosaic()

    def tearDown(self):
        self.prime.shutdown()

    # -----------------------------------------------------------------------
    # Test Case 1: Simple arithmetic — ATLAS + Symbolic + Validator pipeline
    # -----------------------------------------------------------------------
    def test_case_1_arithmetic(self):
        """Input: 150 km in 3 hours → pipeline: ATLAS → Reasoning → Validator → 50."""
        problem = self.atlas.parse(
            "A car travels 150 km in 3 hours. Find average speed.")
        plan = self.mosaic.compose(problem)

        self.assertTrue(plan.valid, f"Plan invalid: {plan.status} — {plan.status_detail}")
        self.assertIn("reasoning", plan.module_ids)

        result = self.mosaic.execute(plan, problem)
        self.assertIn(result.status, ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED"])
        self.assertEqual(result.answer, "50")

    # -----------------------------------------------------------------------
    # Test Case 2: Memory + Reasoning + Validation
    # -----------------------------------------------------------------------
    def test_case_2_memory_reasoning(self):
        """MOSAIC selects memory module when memory context is requested."""
        mem = self.prime.memory()
        mem.store("distance=150", memory_type="long_term", importance=0.9)
        mem.store("time=3",       memory_type="long_term", importance=0.9)

        req = TaskRequirements(
            task_id="mem_test", domain="physics",
            required_capabilities=["atlas_representation", "memory_retrieval",
                                   "reasoning"],
            needs_memory=True)
        plan = self.mosaic.compose(req)
        self.assertTrue(plan.valid)
        self.assertIn("memory", plan.module_ids)
        self.assertIn("reasoning", plan.module_ids)

        # memory must come before reasoning
        idx_mem = plan.module_ids.index("memory")
        idx_rea = plan.module_ids.index("reasoning")
        self.assertLess(idx_mem, idx_rea)

    # -----------------------------------------------------------------------
    # Test Case 3: LLM module — available vs unavailable
    # -----------------------------------------------------------------------
    def test_case_3_llm_unavailable_not_selected(self):
        """Without allow_llm, LLM module must not appear in the plan."""
        req = TaskRequirements(task_id="llm_test", domain="general",
                               required_capabilities=["atlas_representation",
                                                      "reasoning"])
        plan = self.mosaic.compose(req, allow_llm=False)
        self.assertNotIn("llm", plan.module_ids)

    def test_case_3_llm_optional_capability(self):
        """LLM is discoverable but not selected when a lighter alternative exists."""
        plan = self.mosaic.compose(
            self.atlas.parse("A car travels 150 km in 3 hours."),
            allow_llm=False)
        self.assertNotIn("llm", plan.module_ids)
        self.assertIn("reasoning", plan.module_ids)

    # -----------------------------------------------------------------------
    # Test Case 4: Custom module registration
    # -----------------------------------------------------------------------
    def test_case_4_custom_module(self):
        """Register a MockCalculator and verify MOSAIC discovers and uses it."""
        results = []

        def mock_calc_executor(ctx):
            results.append("called")
            # Produce a fake reasoning result
            from prime.reasoning import ReasoningResult
            ctx["reasoning_result"] = ReasoningResult({
                "status": "SOLVED", "answer": "999", "answer_unit": "",
                "answer_numeric": 999.0, "answer_is_numeric": True,
                "confidence": 1.0, "error_message": "", "mode_used": "mock",
                "trace": {"total_steps": 1, "rules_fired": 0,
                          "expressions_evaluated": 1, "constraints_checked": 0,
                          "steps": []},
                "goal_results": [], "constraint_results": [], "derived_facts": {},
            })
            return ctx

        meta = ModuleMetadata(
            id="mock_calc", name="MockCalculator",
            capabilities=[Capability("arithmetic", "arithmetic"),
                          Capability("reasoning")],
            input_types=["*"], output_types=["reasoning_result"],
            priority=200)
        ok = self.mosaic.register_module(meta, executor=mock_calc_executor)
        self.assertTrue(ok)
        self.assertIn("mock_calc", self.mosaic.list_modules())

        # Now compose with mock_calc preferred
        req = TaskRequirements(task_id="t4", domain="math",
                               required_capabilities=["arithmetic"])
        plan = self.mosaic.compose(req, preferred_modules=["mock_calc"],
                                   excluded_modules=["reasoning"])
        self.assertIn("mock_calc", plan.module_ids)

        # Execute — must call mock executor
        problem = self.atlas.parse("Calculate 6 * 7.")
        plan2   = self.mosaic.compose(problem, preferred_modules=["mock_calc"],
                                      excluded_modules=["reasoning"])
        if "mock_calc" in plan2.module_ids:
            result = self.mosaic.execute(plan2, problem)
            self.assertTrue(len(results) > 0)

    # -----------------------------------------------------------------------
    # Test Case 5: Incompatible module — COMPOSITION_INVALID / INPUT_MISMATCH
    # -----------------------------------------------------------------------
    def test_case_5_incompatible_module(self):
        """Register a module requiring an input type never produced → rejected."""
        engine = make_engine()
        engine.register_module(ModuleMetadata(
            id="incompatible", name="Incompatible",
            capabilities=[Capability("special_op")],
            input_types=["completely_unknown_type"],  # never in pipeline
            output_types=["*"]))
        req = TaskRequirements(task_id="t5", domain="x",
                               required_capabilities=["special_op"])
        plan = engine.compose(req)
        self.assertNotEqual(plan.status, "VALID")
        self.assertIn(plan.status, ["INPUT_MISMATCH", "COMPOSITION_INVALID",
                                     "MISSING_CAPABILITY"])

    # -----------------------------------------------------------------------
    # Test Case 6: Circular dependency → COMPOSITION_CYCLE
    # -----------------------------------------------------------------------
    def test_case_6_circular_dependency(self):
        """A → B → A must yield COMPOSITION_CYCLE."""
        engine = make_engine()
        engine.register_module(ModuleMetadata(
            id="mod_a", name="ModA",
            capabilities=[Capability("cap_a")],
            input_types=["*"], output_types=["*"],
            dependencies=["mod_b"]))
        engine.register_module(ModuleMetadata(
            id="mod_b", name="ModB",
            capabilities=[Capability("cap_b")],
            input_types=["*"], output_types=["*"],
            dependencies=["mod_a"]))
        req = TaskRequirements(task_id="t6", domain="x",
                               required_capabilities=["cap_a", "cap_b"])
        plan = engine.compose(req)
        self.assertEqual(plan.status, "COMPOSITION_CYCLE")

    # -----------------------------------------------------------------------
    # Test Case 7: Resource-aware selection
    # -----------------------------------------------------------------------
    def test_case_7_lightweight_preferred(self):
        """LightModule beats HeavyModule when prefer_lightweight=True."""
        engine = make_engine()
        engine.register_module(ModuleMetadata(
            id="light_mod", name="LightModule",
            capabilities=[Capability("arithmetic", "arithmetic")],
            input_types=["*"], output_types=["reasoning_result"],
            cost=ResourceCost(cpu_relative=0.5, memory_relative=0.5),
            priority=5))
        engine.register_module(ModuleMetadata(
            id="heavy_mod", name="HeavyModule",
            capabilities=[Capability("arithmetic", "arithmetic")],
            input_types=["*"], output_types=["reasoning_result"],
            cost=ResourceCost(cpu_relative=20.0, memory_relative=20.0),
            priority=5))
        req = TaskRequirements(task_id="t7", domain="math",
                               required_capabilities=["arithmetic"])
        plan = engine.compose(req, prefer_lightweight=True)
        self.assertTrue(plan.valid)
        self.assertIn("light_mod", plan.module_ids)
        self.assertNotIn("heavy_mod", plan.module_ids)

    def test_case_7_heavy_selected_when_only_option(self):
        """When only heavy module available, it must still be selected."""
        engine = make_engine()
        engine.register_module(ModuleMetadata(
            id="heavy_only", name="HeavyOnly",
            capabilities=[Capability("arithmetic", "arithmetic")],
            input_types=["*"], output_types=["reasoning_result"],
            cost=ResourceCost(cpu_relative=20.0, memory_relative=20.0)))
        req = TaskRequirements(task_id="t7b", domain="math",
                               required_capabilities=["arithmetic"])
        plan = engine.compose(req, prefer_lightweight=True)
        self.assertTrue(plan.valid)
        self.assertIn("heavy_only", plan.module_ids)

    # -----------------------------------------------------------------------
    # Test Case 8: Fallback module
    # -----------------------------------------------------------------------
    def test_case_8_fallback_used_on_primary_failure(self):
        """PrimaryModule fails → FallbackModule succeeds."""
        calls = []

        def primary_fail(ctx):
            calls.append("primary")
            raise RuntimeError("Primary failed intentionally")

        def fallback_ok(ctx):
            calls.append("fallback")
            ctx["fallback_ran"] = True
            return ctx

        engine = MOSAICEngine(prime_instance=self.prime)
        # Remove default modules to avoid interference
        for mid in ["atlas", "reasoning", "memory", "validator", "llm"]:
            engine.remove_module(mid)

        engine.register_module(
            ModuleMetadata(id="primary_m", name="Primary",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["*"], priority=10),
            executor=primary_fail)
        engine.register_module(
            ModuleMetadata(id="fallback_m", name="Fallback",
                           capabilities=[Capability("arithmetic", "arithmetic")],
                           input_types=["*"], output_types=["*"], priority=1),
            executor=fallback_ok,
            fallback_for="primary_m")

        # Manually set up a plan with primary_m and rely on fallback
        from prime.mosaic import CompositionPlan as CP
        raw_plan = {
            "task_id": "t8", "domain": "x", "status": "VALID", "valid": True,
            "status_detail": "", "module_ids": ["primary_m"],
            "selection_reasons": ["test"], "estimated_cost": 1.0,
            "requirements": {"required_capabilities": ["arithmetic"],
                             "optional_capabilities": [], "domain": "x"},
        }
        plan = CP(raw_plan, engine)
        result = engine.execute(plan, allow_fallback=True)

        self.assertIn("primary", calls)
        self.assertIn("fallback", calls)
        self.assertIn(result.status,
                      ["SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED"])


# ===========================================================================
# 8. Cross-domain testing
# ===========================================================================
class TestCrossDomain(unittest.TestCase):
    def setUp(self):
        self.prime  = PRIME()
        self.atlas  = self.prime.atlas()
        self.mosaic = self.prime.mosaic()

    def tearDown(self):
        self.prime.shutdown()

    def _compose_and_check(self, text: str, domain_hint: str = ""):
        problem = self.atlas.parse(text)
        plan    = self.mosaic.compose(problem)
        self.assertIn(plan.status, ["VALID", "MISSING_CAPABILITY"],
                      f"Unexpected status for domain {domain_hint}: {plan.status}")
        return plan

    def test_mathematics(self):
        plan = self._compose_and_check(
            "A car travels 150 km in 3 hours. Find average speed.", "math")
        if plan.valid:
            self.assertIn("reasoning", plan.module_ids)

    def test_physics(self):
        plan = self._compose_and_check(
            "Force is 50 N and mass is 10 kg. Find acceleration.", "physics")
        self.assertIn(plan.status, ["VALID", "MISSING_CAPABILITY"])

    def test_business(self):
        plan = self._compose_and_check(
            "Revenue is 1000 and costs are 600. Find profit.", "business")
        if plan.valid:
            self.assertIn("reasoning", plan.module_ids)

    def test_engineering(self):
        plan = self._compose_and_check(
            "Budget is 40000. Budget must be at most 50000.", "engineering")
        self.assertIn(plan.status, ["VALID", "MISSING_CAPABILITY"])

    def test_composition_domain_independent(self):
        """The composition mechanism does not require a different pipeline per domain."""
        texts = [
            "A = 10, B = 20, find C = A + B.",
            "Revenue is 500. Find profit if costs are 300.",
            "Speed is distance divided by time. Distance=100, time=4.",
        ]
        for text in texts:
            problem = self.atlas.parse(text)
            plan    = self.mosaic.compose(problem)
            self.assertIn(plan.status, ["VALID", "MISSING_CAPABILITY",
                                        "EMPTY_PLAN"])


# ===========================================================================
# 9. Composition trace completeness
# ===========================================================================
class TestCompositionTrace(unittest.TestCase):
    def setUp(self):
        self.prime  = PRIME()
        self.mosaic = self.prime.mosaic()

    def tearDown(self):
        self.prime.shutdown()

    def test_trace_has_steps(self):
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem)
        result  = self.mosaic.execute(plan, problem)
        self.assertGreater(len(result.trace_steps), 0)

    def test_trace_step_indices_sequential(self):
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem)
        result  = self.mosaic.execute(plan, problem)
        for i, step in enumerate(result.trace_steps):
            self.assertEqual(step.step_index, i)

    def test_trace_records_module_names(self):
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem)
        result  = self.mosaic.execute(plan, problem)
        for step in result.trace_steps:
            self.assertIsNotNone(step.module_id)

    def test_trace_latency_non_negative(self):
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem)
        result  = self.mosaic.execute(plan, problem)
        for step in result.trace_steps:
            self.assertGreaterEqual(step.latency_ms, 0.0)


# ===========================================================================
# 10. Python API surface
# ===========================================================================
class TestPythonAPI(unittest.TestCase):
    def setUp(self):
        self.prime  = PRIME()
        self.mosaic = self.prime.mosaic()

    def tearDown(self):
        self.prime.shutdown()

    def test_mosaic_repr(self):
        self.assertIn("MOSAICEngine", repr(self.mosaic))

    def test_plan_repr(self):
        problem = self.prime.atlas().parse("A car travels 150 km.")
        plan    = self.mosaic.compose(problem)
        self.assertIn("CompositionPlan", repr(plan))

    def test_result_repr(self):
        result = self.mosaic.solve("A car travels 150 km in 3 hours.")
        self.assertIn("ExecutionResult", repr(result))

    def test_result_str_is_explanation(self):
        result = self.mosaic.solve("A car travels 150 km in 3 hours.")
        s = str(result)
        self.assertIn("MOSAIC Execution Result", s)

    def test_plan_str_is_explanation(self):
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem)
        s = str(plan)
        self.assertIn("MOSAIC", s)

    def test_execution_result_succeeded_property(self):
        # Use a task with an explicit goal that the reasoning engine solves.
        result = self.mosaic.solve(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        self.assertTrue(result.succeeded)

    def test_answer_with_unit(self):
        result = self.mosaic.solve(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        self.assertIsInstance(result.answer_with_unit, str)

    def test_validate_plan_returns_string(self):
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem)
        status  = self.mosaic.validate_plan(plan)
        self.assertIsInstance(status, str)

    def test_task_requirements_to_dict(self):
        req = TaskRequirements(task_id="t", domain="x",
                               required_capabilities=["arithmetic"])
        d = req.to_dict()
        self.assertEqual(d["task_id"], "t")
        self.assertIn("arithmetic", d["required_capabilities"])

    def test_module_metadata_to_dict(self):
        meta = ModuleMetadata(id="m", name="M",
                              capabilities=[Capability("x")])
        d = meta.to_dict()
        self.assertEqual(d["id"], "m")
        self.assertEqual(d["capabilities"][0]["name"], "x")

    def test_prime_mosaic_method(self):
        m = self.prime.mosaic()
        self.assertIsInstance(m, MOSAICEngine)


# ===========================================================================
# 11. Backward compatibility
# ===========================================================================
class TestBackwardCompatibility(unittest.TestCase):
    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_runtime_still_alive(self):
        self.assertIn("running", self.prime.status().lower())

    def test_atlas_works(self):
        atlas = self.prime.atlas()
        prob  = atlas.parse("A car travels 150 km in 3 hours.")
        self.assertEqual(prob.domain, "physics")

    def test_reasoner_works(self):
        from prime import make_rule, make_condition, make_conclusion
        r = self.prime.reasoner()
        result = r.solve_facts(
            facts={"temperature": 120},
            rules=[make_rule("r1",
                [make_condition("temperature", ">", "100")],
                [make_conclusion("warning", True)])],
            goals=["warning"])
        self.assertEqual(result.status, "SOLVED")

    def test_memory_works(self):
        mem = self.prime.memory()
        mid = mem.store("test content", memory_type="long_term", importance=0.8)
        self.assertIsNotNone(mid)

    def test_mosaic_does_not_break_reasoning(self):
        mosaic   = self.prime.mosaic()
        reasoner = self.prime.reasoner()
        result   = reasoner.solve_facts(
            facts={"x": 10, "y": 5},
            expressions=[{"id": "e1", "op": "add", "output_id": "z",
                          "operand_ids": ["x", "y"], "text": ""}],
            goals=["z"])
        self.assertEqual(result.status, "SOLVED")
        self.assertEqual(result.derived_facts.get("z"), 15.0)


# ===========================================================================
# 12. Determinism across multiple runs
# ===========================================================================
class TestDeterminismFull(unittest.TestCase):
    def setUp(self):
        self.prime  = PRIME()
        self.mosaic = self.prime.mosaic()
        self.atlas  = self.prime.atlas()

    def tearDown(self):
        self.prime.shutdown()

    def test_same_problem_same_plan(self):
        problem = self.atlas.parse("A car travels 150 km in 3 hours.")
        plans   = [self.mosaic.compose(problem) for _ in range(5)]
        ids     = [p.module_ids for p in plans]
        for other in ids[1:]:
            self.assertEqual(ids[0], other)

    def test_same_problem_same_execution_status(self):
        problem = self.atlas.parse("A car travels 150 km in 3 hours.")
        results = [self.mosaic.solve("A car travels 150 km in 3 hours.")
                   for _ in range(3)]
        for r in results:
            self.assertEqual(r.answer, results[0].answer)


# ===========================================================================
# 13. Execution limits
# ===========================================================================
class TestExecutionLimits(unittest.TestCase):
    def setUp(self):
        self.prime  = PRIME()
        self.mosaic = self.prime.mosaic()

    def tearDown(self):
        self.prime.shutdown()

    def test_max_modules_respected_in_compose(self):
        """compose() with max_modules=1 produces at most 1 module."""
        problem = self.prime.atlas().parse("A car travels 150 km in 3 hours.")
        plan    = self.mosaic.compose(problem, max_modules=1)
        if plan.valid:
            self.assertLessEqual(len(plan.module_ids), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
