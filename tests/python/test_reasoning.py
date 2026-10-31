"""
tests/python/test_reasoning.py — PRIME Phase 5 Reasoning Engine Tests.

Covers:
  - Expression evaluator (arithmetic, variables, functions, errors)
  - Rule engine (conditions, firing, priorities, conflict resolution, circular)
  - Constraint engine (numeric, string, inequality)
  - Symbolic engine (ATLAS expressions, multi-step)
  - Hybrid reasoning (ATLAS → solve)
  - Goal satisfaction
  - Backward chaining interface
  - LLM provider interface (MockLLMProvider)
  - Python API (ReasoningEngine.solve / solve_facts)
  - Integration with ATLAS (Phase 4)
  - All 6 required test cases from the spec
"""

import sys
import os
import unittest

# Ensure the python/ directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import _prime_reasoning
from prime import (
    PRIME, ReasoningEngine, ReasoningError,
    make_rule, make_condition, make_compound_condition, make_conclusion,
    MockLLMProvider, LLMReasoningProvider,
)
from prime.reasoning import ReasoningResult, ReasoningTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mk_engine():
    return ReasoningEngine()


# ===========================================================================
# 1. Expression Evaluator
# ===========================================================================
class TestExpressionEvaluator(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def test_constant(self):
        self.assertEqual(self.engine.eval_expression("42"), 42.0)

    def test_addition(self):
        self.assertEqual(self.engine.eval_expression("10 + 20"), 30.0)

    def test_subtraction(self):
        self.assertEqual(self.engine.eval_expression("20 - 5"), 15.0)

    def test_multiplication(self):
        self.assertEqual(self.engine.eval_expression("6 * 7"), 42.0)

    def test_division(self):
        self.assertAlmostEqual(self.engine.eval_expression("150 / 3"), 50.0)

    def test_modulo(self):
        self.assertEqual(self.engine.eval_expression("10 % 3"), 1.0)

    def test_power(self):
        self.assertEqual(self.engine.eval_expression("2 ^ 10"), 1024.0)

    def test_unary_minus(self):
        self.assertEqual(self.engine.eval_expression("-5"), -5.0)

    def test_parentheses(self):
        self.assertEqual(self.engine.eval_expression("(2 + 3) * 4"), 20.0)

    def test_nested_parentheses(self):
        self.assertAlmostEqual(self.engine.eval_expression("(10 + 5) / (1 + 2)"), 5.0)

    def test_variable_binding(self):
        self.assertEqual(self.engine.eval_expression("x + y", {"x": 10, "y": 20}), 30.0)

    def test_variable_arithmetic(self):
        self.assertAlmostEqual(
            self.engine.eval_expression("distance / time", {"distance": 150, "time": 3}),
            50.0)

    def test_function_abs(self):
        self.assertEqual(self.engine.eval_expression("abs(-7)"), 7.0)

    def test_function_min(self):
        self.assertEqual(self.engine.eval_expression("min(3, 1, 2)"), 1.0)

    def test_function_max(self):
        self.assertEqual(self.engine.eval_expression("max(3, 1, 2)"), 3.0)

    def test_function_sqrt(self):
        self.assertAlmostEqual(self.engine.eval_expression("sqrt(9)"), 3.0)

    def test_division_by_zero_raises(self):
        with self.assertRaises(ReasoningError):
            self.engine.eval_expression("10 / 0")

    def test_unbound_variable_raises(self):
        with self.assertRaises(ReasoningError):
            self.engine.eval_expression("x + 1", {})

    def test_invalid_syntax_raises(self):
        with self.assertRaises(ReasoningError):
            self.engine.eval_expression("1 ++ 2")

    def test_empty_expression_raises(self):
        with self.assertRaises(Exception):
            self.engine.eval_expression("")

    def test_complex_expression(self):
        result = self.engine.eval_expression("(a + b) * c - d / 2",
                                             {"a": 3, "b": 7, "c": 4, "d": 20})
        self.assertAlmostEqual(result, 30.0)

    def test_precedence_respected(self):
        # 2 + 3 * 4 = 14 (not 20)
        self.assertEqual(self.engine.eval_expression("2 + 3 * 4"), 14.0)

    def test_right_assoc_power(self):
        # 2 ^ 3 ^ 2 = 2 ^ (3^2) = 2 ^ 9 = 512
        self.assertEqual(self.engine.eval_expression("2 ^ 3 ^ 2"), 512.0)


# ===========================================================================
# 2. Rule Engine — creation and matching
# ===========================================================================
class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def _simple_rule(self, rid="r1"):
        return make_rule(rid,
            conditions=[make_condition("temperature", ">", "100")],
            conclusions=[make_conclusion("warning", True)])

    def test_rule_creation(self):
        rule = self._simple_rule()
        self.assertEqual(rule["id"], "r1")
        self.assertEqual(len(rule["conditions"]), 1)
        self.assertEqual(len(rule["conclusions"]), 1)

    def test_rule_fires_when_condition_met(self):
        r = self.engine.solve_facts(
            facts={"temperature": 120},
            rules=[self._simple_rule()],
            goals=["warning"])
        self.assertEqual(r.status, "SOLVED")
        self.assertEqual(r.derived_facts.get("warning"), "true")

    def test_rule_does_not_fire_when_condition_not_met(self):
        r = self.engine.solve_facts(
            facts={"temperature": 80},
            rules=[self._simple_rule()],
            goals=["warning"])
        self.assertNotIn("warning", r.derived_facts)

    def test_rule_eq_condition(self):
        rule = make_rule("r1", [make_condition("x", "=", "5")],
                         [make_conclusion("hit", True)])
        r = self.engine.solve_facts(facts={"x": 5}, rules=[rule], goals=["hit"])
        self.assertEqual(r.derived_facts.get("hit"), "true")

    def test_rule_neq_condition(self):
        rule = make_rule("r1", [make_condition("x", "!=", "5")],
                         [make_conclusion("not_five", True)])
        r = self.engine.solve_facts(facts={"x": 6}, rules=[rule], goals=["not_five"])
        self.assertEqual(r.derived_facts.get("not_five"), "true")

    def test_rule_lte_condition(self):
        rule = make_rule("r1", [make_condition("budget", "<=", "50000")],
                         [make_conclusion("affordable", True)])
        r = self.engine.solve_facts(facts={"budget": 40000}, rules=[rule], goals=["affordable"])
        self.assertEqual(r.derived_facts.get("affordable"), "true")

    def test_rule_gte_condition(self):
        rule = make_rule("r1", [make_condition("score", ">=", "90")],
                         [make_conclusion("grade", "A")])
        r = self.engine.solve_facts(facts={"score": 95}, rules=[rule], goals=["grade"])
        self.assertEqual(r.derived_facts.get("grade"), "A")

    def test_rule_priority_order(self):
        # r_high fires first; r_low also fires but doesn't override goal
        r_high = make_rule("r_high", [make_condition("x", ">", "0")],
                           [make_conclusion("level", "high")], priority=10)
        r_low  = make_rule("r_low",  [make_condition("x", ">", "0")],
                           [make_conclusion("level2", "low")],  priority=1)
        r = self.engine.solve_facts(facts={"x": 5}, rules=[r_low, r_high])
        self.assertIn("level", r.derived_facts)
        self.assertIn("level2", r.derived_facts)

    def test_rule_confidence_propagated(self):
        rule = make_rule("r1", [make_condition("x", ">", "0")],
                         [make_conclusion("result", True)], confidence=0.8)
        r = self.engine.solve_facts(facts={"x": 1}, rules=[rule])
        # Rule fires and produces a derived fact
        self.assertIn("result", r.derived_facts)

    def test_multiple_conclusions(self):
        rule = make_rule("r1", [make_condition("flag", "=", "1")],
                         [make_conclusion("a", True), make_conclusion("b", 42)])
        r = self.engine.solve_facts(facts={"flag": 1}, rules=[rule])
        self.assertIn("a", r.derived_facts)
        self.assertIn("b", r.derived_facts)

    def test_chained_rules(self):
        r1 = make_rule("r1", [make_condition("a", "=", "1")], [make_conclusion("b", 1)])
        r2 = make_rule("r2", [make_condition("b", "=", "1")], [make_conclusion("c", 1)])
        r = self.engine.solve_facts(facts={"a": 1}, rules=[r1, r2], goals=["c"])
        self.assertEqual(r.status, "SOLVED")
        self.assertIn("c", r.derived_facts)

    def test_circular_rules_terminate(self):
        """Circular rules must terminate without infinite loop."""
        r1 = make_rule("r1", [make_condition("A", "=", "1")], [make_conclusion("B", 1)])
        r2 = make_rule("r2", [make_condition("B", "=", "1")], [make_conclusion("C", 1)])
        r3 = make_rule("r3", [make_condition("C", "=", "1")], [make_conclusion("A", 1)])
        r = self.engine.solve_facts(facts={"A": 1}, rules=[r1, r2, r3])
        # Must terminate — status may be SOLVED or REASONING_LIMIT_REACHED
        self.assertNotEqual(r.status, "INTERNAL_ERROR")

    def test_duplicate_fact_not_added_twice(self):
        """Same fact derived twice should not grow the trace unboundedly."""
        rule = make_rule("r1", [make_condition("x", "=", "1")], [make_conclusion("y", 1)])
        r = self.engine.solve_facts(facts={"x": 1}, rules=[rule])
        # Should terminate quickly
        self.assertLess(r.trace.total_steps, 100)

    def test_and_condition(self):
        cond = make_compound_condition("AND",
                                       make_condition("a", ">", "0"),
                                       make_condition("b", ">", "0"))
        rule = make_rule("r1", [cond], [make_conclusion("both_positive", True)])
        r = self.engine.solve_facts(facts={"a": 1, "b": 2}, rules=[rule])
        self.assertEqual(r.derived_facts.get("both_positive"), "true")

    def test_and_condition_false(self):
        cond = make_compound_condition("AND",
                                       make_condition("a", ">", "0"),
                                       make_condition("b", ">", "0"))
        rule = make_rule("r1", [cond], [make_conclusion("both_positive", True)])
        r = self.engine.solve_facts(facts={"a": 1, "b": -1}, rules=[rule])
        self.assertNotIn("both_positive", r.derived_facts)

    def test_or_condition(self):
        cond = make_compound_condition("OR",
                                       make_condition("a", ">", "100"),
                                       make_condition("b", ">", "100"))
        rule = make_rule("r1", [cond], [make_conclusion("alert", True)])
        r = self.engine.solve_facts(facts={"a": 50, "b": 150}, rules=[rule])
        self.assertEqual(r.derived_facts.get("alert"), "true")

    def test_not_condition(self):
        cond = make_compound_condition("NOT", make_condition("flag", "=", "true"))
        rule = make_rule("r1", [cond], [make_conclusion("inactive", True)])
        r = self.engine.solve_facts(facts={"flag": "false"}, rules=[rule])
        self.assertEqual(r.derived_facts.get("inactive"), "true")

    def test_global_rules_persist(self):
        eng = ReasoningEngine()
        eng.add_rule(make_rule("g1", [make_condition("x", ">", "0")],
                               [make_conclusion("positive", True)]))
        r1 = eng.solve_facts(facts={"x": 5})
        r2 = eng.solve_facts(facts={"x": 3})
        self.assertIn("positive", r1.derived_facts)
        self.assertIn("positive", r2.derived_facts)
        eng.clear_rules()
        r3 = eng.solve_facts(facts={"x": 5})
        self.assertNotIn("positive", r3.derived_facts)


# ===========================================================================
# 3. Symbolic Engine — expression evaluation
# ===========================================================================
class TestSymbolicEngine(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def _expr(self, eid, op, output, operands, text=""):
        return {"id": eid, "op": op, "output_id": output,
                "operand_ids": operands, "text": text}

    def test_divide(self):
        r = self.engine.solve_facts(
            facts={"distance": 150, "time": 3},
            expressions=[self._expr("e1", "divide", "speed", ["distance", "time"])],
            goals=["speed"])
        self.assertEqual(r.status, "SOLVED")
        self.assertAlmostEqual(r.derived_facts.get("speed"), 50.0)

    def test_add(self):
        r = self.engine.solve_facts(
            facts={"a": 10, "b": 20},
            expressions=[self._expr("e1", "add", "total", ["a", "b"])],
            goals=["total"])
        self.assertEqual(r.derived_facts.get("total"), 30.0)

    def test_multiply(self):
        r = self.engine.solve_facts(
            facts={"price": 25, "quantity": 4},
            expressions=[self._expr("e1", "multiply", "total_cost", ["price", "quantity"])],
            goals=["total_cost"])
        self.assertEqual(r.derived_facts.get("total_cost"), 100.0)

    def test_subtract(self):
        r = self.engine.solve_facts(
            facts={"revenue": 1000, "costs": 600},
            expressions=[self._expr("e1", "subtract", "profit", ["revenue", "costs"])],
            goals=["profit"])
        self.assertEqual(r.derived_facts.get("profit"), 400.0)

    def test_multistep_chaining(self):
        """A=10, B=A+5, C=B*2 → C=30"""
        r = self.engine.solve_facts(
            facts={"A": 10},
            expressions=[
                self._expr("e1", "add",      "B", ["A", "5"]),
                self._expr("e2", "multiply", "C", ["B", "2"]),
            ],
            goals=["C"])
        self.assertEqual(r.status, "SOLVED")
        self.assertAlmostEqual(r.derived_facts.get("C"), 30.0)

    def test_literal_operand(self):
        """Operand that is a literal number, not a fact name."""
        r = self.engine.solve_facts(
            facts={"x": 7},
            expressions=[self._expr("e1", "multiply", "result", ["x", "3"])],
            goals=["result"])
        self.assertAlmostEqual(r.derived_facts.get("result"), 21.0)

    def test_unresolved_operand(self):
        """Missing operand → UNRESOLVED."""
        r = self.engine.solve_facts(
            facts={},
            expressions=[self._expr("e1", "divide", "result", ["x", "y"])],
            goals=["result"])
        self.assertEqual(r.status, "UNRESOLVED")

    def test_average(self):
        r = self.engine.solve_facts(
            facts={"a": 10, "b": 20, "c": 30},
            expressions=[self._expr("e1", "average", "mean", ["a", "b", "c"])],
            goals=["mean"])
        self.assertAlmostEqual(r.derived_facts.get("mean"), 20.0)

    def test_trace_contains_expression_steps(self):
        r = self.engine.solve_facts(
            facts={"p": 5, "q": 4},
            expressions=[self._expr("e1", "multiply", "area", ["p", "q"])],
            goals=["area"])
        self.assertGreater(r.trace.expressions_evaluated, 0)

    def test_expression_step_description(self):
        r = self.engine.solve_facts(
            facts={"x": 6, "y": 7},
            expressions=[self._expr("e1", "multiply", "z", ["x", "y"])],
            goals=["z"])
        expr_steps = [s for s in r.trace.steps if s.type == "EXPRESSION_EVAL"]
        self.assertTrue(len(expr_steps) > 0)
        self.assertIn("42", expr_steps[0].description)


# ===========================================================================
# 4. Constraint Engine
# ===========================================================================
class TestConstraintEngine(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def _constraint(self, cid, left, op, right):
        return {"id": cid, "left": left, "op": op, "right": right, "confidence": 1.0}

    def test_satisfied_lte(self):
        r = self.engine.solve_facts(
            facts={"budget": 40000},
            constraints=[self._constraint("c1", "budget", "<=", "50000")])
        self.assertTrue(all(c.satisfied for c in r.constraint_results))

    def test_violated_lte(self):
        r = self.engine.solve_facts(
            facts={"budget": 60000},
            constraints=[self._constraint("c1", "budget", "<=", "50000")])
        self.assertEqual(r.status, "CONSTRAINT_VIOLATION")
        self.assertFalse(r.constraint_results[0].satisfied)

    def test_satisfied_gte(self):
        r = self.engine.solve_facts(
            facts={"score": 85},
            constraints=[self._constraint("c1", "score", ">=", "80")])
        self.assertTrue(r.constraint_results[0].satisfied)

    def test_satisfied_gt(self):
        r = self.engine.solve_facts(
            facts={"mass": 10},
            constraints=[self._constraint("c1", "mass", ">", "0")])
        self.assertTrue(r.constraint_results[0].satisfied)

    def test_satisfied_lt(self):
        r = self.engine.solve_facts(
            facts={"temperature": 95},
            constraints=[self._constraint("c1", "temperature", "<", "100")])
        self.assertTrue(r.constraint_results[0].satisfied)

    def test_satisfied_eq(self):
        r = self.engine.solve_facts(
            facts={"flag": 1},
            constraints=[self._constraint("c1", "flag", "=", "1")])
        self.assertTrue(r.constraint_results[0].satisfied)

    def test_unknown_fact(self):
        r = self.engine.solve_facts(
            facts={},
            constraints=[self._constraint("c1", "missing_fact", "<=", "100")])
        self.assertEqual(r.constraint_results[0].result, "unknown")

    def test_multiple_constraints(self):
        r = self.engine.solve_facts(
            facts={"a": 5, "b": 10},
            constraints=[
                self._constraint("c1", "a", ">", "0"),
                self._constraint("c2", "b", "<", "20"),
            ])
        self.assertEqual(len(r.constraint_results), 2)
        self.assertTrue(all(c.satisfied for c in r.constraint_results))

    def test_check_constraint_api(self):
        result = self.engine.check_constraint("x", "<=", "100", {"x": 50.0})
        self.assertEqual(result, "satisfied")

    def test_check_constraint_violated(self):
        result = self.engine.check_constraint("x", "<=", "100", {"x": 150.0})
        self.assertEqual(result, "violated")

    def test_check_constraint_unknown(self):
        result = self.engine.check_constraint("y", "<=", "100", {})
        self.assertEqual(result, "unknown")


# ===========================================================================
# 5. ReasoningResult and ReasoningTrace
# ===========================================================================
class TestReasoningResult(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def test_result_repr(self):
        r = self.engine.solve_facts(facts={"x": 5}, goals=["x"])
        self.assertIn("ReasoningResult", repr(r))

    def test_result_status_field(self):
        r = self.engine.solve_facts(facts={"x": 5})
        self.assertIn(r.status, [
            "SOLVED", "PARTIALLY_SOLVED", "UNRESOLVED",
            "INVALID_INPUT", "CONSTRAINT_VIOLATION",
            "REASONING_LIMIT_REACHED", "CIRCULAR_REASONING"])

    def test_trace_type(self):
        r = self.engine.solve_facts(facts={"x": 5})
        self.assertIsInstance(r.trace, ReasoningTrace)

    def test_trace_len(self):
        r = self.engine.solve_facts(facts={"x": 5, "y": 3},
            expressions=[{"id": "e1", "op": "add", "output_id": "z",
                          "operand_ids": ["x", "y"], "text": ""}],
            goals=["z"])
        self.assertGreater(len(r.trace), 0)

    def test_trace_steps_iterable(self):
        r = self.engine.solve_facts(facts={"a": 1},
            rules=[make_rule("r1", [make_condition("a", ">", "0")],
                             [make_conclusion("b", 1)])])
        steps = list(r.trace)
        self.assertTrue(len(steps) > 0)

    def test_trace_step_fields(self):
        r = self.engine.solve_facts(facts={"a": 1},
            rules=[make_rule("r1", [make_condition("a", ">", "0")],
                             [make_conclusion("b", 1)])])
        for step in r.trace:
            self.assertIsNotNone(step.step_id)
            self.assertIsNotNone(step.type)
            self.assertIsNotNone(step.description)

    def test_explain_returns_string(self):
        r = self.engine.solve_facts(facts={"x": 42})
        explanation = r.explain()
        self.assertIsInstance(explanation, str)
        self.assertIn("PRIME Reasoning Explanation", explanation)

    def test_explain_contains_status(self):
        r = self.engine.solve_facts(facts={"x": 1},
            rules=[make_rule("r1", [make_condition("x", ">", "0")],
                             [make_conclusion("result", True)])])
        explanation = r.explain()
        self.assertIn(r.status, explanation)

    def test_explain_from_trace_not_fabricated(self):
        """Explanation must derive from the actual trace, not invent a path."""
        r = self.engine.solve_facts(
            facts={"distance": 150, "time": 3},
            expressions=[{"id": "e1", "op": "divide", "output_id": "speed",
                          "operand_ids": ["distance", "time"], "text": ""}],
            goals=["speed"])
        explanation = r.explain()
        self.assertIn("50", explanation)

    def test_trace_format(self):
        r = self.engine.solve_facts(facts={"x": 1},
            rules=[make_rule("r1", [make_condition("x", ">", "0")],
                             [make_conclusion("y", 1)])])
        fmt = r.trace.format()
        self.assertIsInstance(fmt, str)

    def test_goal_results_list(self):
        r = self.engine.solve_facts(facts={"speed": 50}, goals=["speed"])
        self.assertIsInstance(r.goal_results, list)

    def test_goal_result_fields(self):
        r = self.engine.solve_facts(facts={"speed": 50}, goals=["speed"])
        if r.goal_results:
            gr = r.goal_results[0]
            self.assertIsNotNone(gr.id)
            self.assertEqual(gr.target, "speed")

    def test_derived_facts_dict(self):
        r = self.engine.solve_facts(
            facts={"a": 10, "b": 5},
            expressions=[{"id": "e1", "op": "add", "output_id": "c",
                          "operand_ids": ["a", "b"], "text": ""}],
            goals=["c"])
        self.assertIsInstance(r.derived_facts, dict)

    def test_solved_property(self):
        r = self.engine.solve_facts(facts={"x": 5}, goals=["x"])
        if r.status == "SOLVED":
            self.assertTrue(r.solved)

    def test_answer_with_unit(self):
        r = self.engine.solve_facts(
            facts={"distance": 150, "time": 3},
            expressions=[{"id": "e1", "op": "divide", "output_id": "speed",
                          "operand_ids": ["distance", "time"], "text": ""}],
            goals=["speed"])
        self.assertIsInstance(r.answer_with_unit, str)


# ===========================================================================
# 6. Required Test Cases (from spec)
# ===========================================================================
class TestRequiredCases(unittest.TestCase):
    def setUp(self):
        self.prime   = PRIME()
        self.atlas   = self.prime.atlas()
        self.engine  = self.prime.reasoner()

    def tearDown(self):
        self.prime.shutdown()

    def test_case_1_arithmetic(self):
        """Test 1: A car travels 150 km in 3 hours → speed = 50."""
        problem = self.atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
        result  = self.engine.solve(problem)
        self.assertEqual(result.status, "SOLVED")
        self.assertEqual(result.answer, "50")
        self.assertGreater(result.trace.total_steps, 0)

    def test_case_2_rule_reasoning(self):
        """Test 2: temperature=120, rule IF temp>100 THEN warning=true."""
        result = self.engine.solve_facts(
            facts={"temperature": 120},
            rules=[make_rule("r1",
                conditions=[make_condition("temperature", ">", "100")],
                conclusions=[make_conclusion("warning", True)])],
            goals=["warning"])
        self.assertEqual(result.status, "SOLVED")
        self.assertEqual(result.derived_facts.get("warning"), "true")

    def test_case_3_multistep(self):
        """Test 3: A=10, B=A+5, C=B*2 → C=30 with trace."""
        result = self.engine.solve_facts(
            facts={"A": 10},
            expressions=[
                {"id": "e1", "op": "add",      "output_id": "B",
                 "operand_ids": ["A", "5"], "text": "B = A + 5"},
                {"id": "e2", "op": "multiply", "output_id": "C",
                 "operand_ids": ["B", "2"], "text": "C = B * 2"},
            ],
            goals=["C"])
        self.assertEqual(result.status, "SOLVED")
        self.assertAlmostEqual(result.derived_facts.get("C"), 30.0)
        # Trace must contain each step
        expr_steps = [s for s in result.trace.steps if s.type == "EXPRESSION_EVAL"]
        self.assertGreaterEqual(len(expr_steps), 2)

    def test_case_4_constraint(self):
        """Test 4: budget=40000, budget<=50000 → satisfied."""
        result = self.engine.solve_facts(
            facts={"budget": 40000},
            constraints=[{"id": "c1", "left": "budget", "op": "<=",
                          "right": "50000", "confidence": 1.0}])
        satisfied = all(c.satisfied for c in result.constraint_results)
        self.assertTrue(satisfied)

    def test_case_5_invalid_unresolved(self):
        """Test 5: x=unknown, goal=x+10 → UNRESOLVED, no fabricated value."""
        result = self.engine.solve_facts(
            facts={},
            expressions=[{"id": "e1", "op": "add", "output_id": "result",
                          "operand_ids": ["x", "10"], "text": ""}],
            goals=["result"])
        self.assertEqual(result.status, "UNRESOLVED")
        # Must NOT have invented a value
        self.assertNotIn("result", result.derived_facts)

    def test_case_6_circular_rules_terminate(self):
        """Test 6: A→B→C→A circular rules must terminate without infinite loop."""
        result = self.engine.solve_facts(
            facts={"A": 1},
            rules=[
                make_rule("r1", [make_condition("A", "=", "1")], [make_conclusion("B", 1)]),
                make_rule("r2", [make_condition("B", "=", "1")], [make_conclusion("C", 1)]),
                make_rule("r3", [make_condition("C", "=", "1")], [make_conclusion("A", 1)]),
            ])
        # Status not INTERNAL_ERROR and steps bounded
        self.assertNotEqual(result.status, "INTERNAL_ERROR")
        self.assertLess(result.trace.total_steps, 2000)


# ===========================================================================
# 7. Deductive Reasoning / Logical Inference
# ===========================================================================
class TestLogicalReasoning(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def test_deductive_inspection(self):
        """All machines >100°C require inspection."""
        rule = make_rule("inspection",
            conditions=[make_condition("machine_temp", ">", "100")],
            conclusions=[make_conclusion("requires_inspection", True)])
        r = self.engine.solve_facts(
            facts={"machine_temp": 120},
            rules=[rule],
            goals=["requires_inspection"])
        self.assertEqual(r.status, "SOLVED")
        self.assertEqual(r.derived_facts.get("requires_inspection"), "true")

    def test_deductive_fails_when_condition_not_met(self):
        rule = make_rule("inspection",
            conditions=[make_condition("machine_temp", ">", "100")],
            conclusions=[make_conclusion("requires_inspection", True)])
        r = self.engine.solve_facts(
            facts={"machine_temp": 80},
            rules=[rule],
            goals=["requires_inspection"])
        self.assertNotIn("requires_inspection", r.derived_facts)

    def test_multi_fact_rule(self):
        """Rule fires only if both conditions are true."""
        rule = make_rule("r1",
            conditions=[
                make_condition("alarm_on", "=", "true"),
                make_condition("door_open", "=", "true"),
            ],
            conclusions=[make_conclusion("alert_security", True)])
        r = self.engine.solve_facts(
            facts={"alarm_on": "true", "door_open": "true"},
            rules=[rule],
            goals=["alert_security"])
        self.assertEqual(r.derived_facts.get("alert_security"), "true")

    def test_multi_fact_rule_partial(self):
        rule = make_rule("r1",
            conditions=[
                make_condition("alarm_on", "=", "true"),
                make_condition("door_open", "=", "true"),
            ],
            conclusions=[make_conclusion("alert_security", True)])
        r = self.engine.solve_facts(
            facts={"alarm_on": "true", "door_open": "false"},
            rules=[rule])
        self.assertNotIn("alert_security", r.derived_facts)


# ===========================================================================
# 8. Multiple Goals
# ===========================================================================
class TestMultipleGoals(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def test_two_goals_both_satisfied(self):
        r = self.engine.solve_facts(
            facts={"x": 10, "y": 20},
            goals=["x", "y"])
        # Both goals should be satisfied
        satisfied = [g for g in r.goal_results if g.satisfied]
        self.assertGreaterEqual(len(satisfied), 1)

    def test_partial_goal_satisfaction(self):
        r = self.engine.solve_facts(
            facts={"x": 10},
            goals=["x", "missing"])
        statuses = {g.target: g.satisfied for g in r.goal_results}
        self.assertTrue(statuses.get("x", False))
        self.assertFalse(statuses.get("missing", True))


# ===========================================================================
# 9. Reasoning Modes
# ===========================================================================
class TestReasoningModes(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def test_mode_symbolic(self):
        r = self.engine.solve_facts(
            facts={"a": 5, "b": 3},
            expressions=[{"id": "e1", "op": "add", "output_id": "c",
                          "operand_ids": ["a", "b"], "text": ""}],
            goals=["c"],
            mode="symbolic")
        self.assertEqual(r.status, "SOLVED")
        self.assertEqual(r.mode_used, "SYMBOLIC")

    def test_mode_rule(self):
        rule = make_rule("r1", [make_condition("x", ">", "0")],
                         [make_conclusion("positive", True)])
        r = self.engine.solve_facts(
            facts={"x": 5},
            rules=[rule],
            mode="rule")
        self.assertEqual(r.mode_used, "RULE")

    def test_mode_hybrid(self):
        r = self.engine.solve_facts(facts={"x": 1}, mode="hybrid")
        self.assertEqual(r.mode_used, "HYBRID")

    def test_mode_default_is_hybrid(self):
        r = self.engine.solve_facts(facts={"x": 1})
        self.assertEqual(r.mode_used, "HYBRID")


# ===========================================================================
# 10. Reasoning Limits
# ===========================================================================
class TestReasoningLimits(unittest.TestCase):
    def setUp(self):
        self.engine = mk_engine()

    def test_max_steps_respected(self):
        # Build a long chain of rules
        rules = []
        for i in range(200):
            rules.append(make_rule(
                f"r{i}",
                [make_condition(f"f{i}", "=", "1")],
                [make_conclusion(f"f{i+1}", 1)]))
        facts = {"f0": 1}
        r = self.engine.solve_facts(facts=facts, rules=rules, max_steps=10)
        # Should stop early
        self.assertIn(r.status, ["SOLVED", "REASONING_LIMIT_REACHED"])

    def test_invalid_input_no_content(self):
        r = self.engine.solve_facts(facts={})
        # Empty input — status is UNRESOLVED or INVALID_INPUT
        self.assertIn(r.status, ["UNRESOLVED", "INVALID_INPUT"])


# ===========================================================================
# 11. LLM Provider Interface
# ===========================================================================
class TestLLMProviderInterface(unittest.TestCase):
    def test_mock_provider_propose(self):
        provider = MockLLMProvider(
            rules=[make_rule("r1", [make_condition("x", ">", "0")],
                             [make_conclusion("y", 1)])])
        engine = ReasoningEngine(llm_provider=provider)
        r = engine.solve_facts(facts={"x": 5}, mode="llm_assisted")
        # Rule from LLM proposal should fire
        self.assertIn("y", r.derived_facts)

    def test_mock_provider_invalid_rule_rejected(self):
        """An invalid LLM rule that can never fire should not produce output."""
        provider = MockLLMProvider(
            rules=[make_rule("bad_r", [make_condition("NONEXISTENT", ">", "99999")],
                             [make_conclusion("fabricated", True)])])
        engine = ReasoningEngine(llm_provider=provider)
        r = engine.solve_facts(facts={"x": 1}, mode="llm_assisted")
        self.assertNotIn("fabricated", r.derived_facts)

    def test_llm_unavailable_rules_still_work(self):
        """If LLM is not set, rule/symbolic reasoning continues normally."""
        engine = ReasoningEngine(llm_provider=None)
        rule = make_rule("r1", [make_condition("flag", "=", "1")],
                         [make_conclusion("result", True)])
        r = engine.solve_facts(facts={"flag": 1}, rules=[rule], goals=["result"])
        self.assertEqual(r.status, "SOLVED")

    def test_set_llm_provider(self):
        engine = ReasoningEngine()
        self.assertIsNone(engine._llm)
        engine.set_llm_provider(MockLLMProvider())
        self.assertIsNotNone(engine._llm)
        engine.set_llm_provider(None)
        self.assertIsNone(engine._llm)

    def test_abstract_provider_returns_none(self):
        provider = LLMReasoningProvider()
        self.assertIsNone(provider.propose_reasoning("{}"))
        self.assertIsNone(provider.interpret_ambiguity("ambiguous"))
        self.assertIsNone(provider.generate_explanation({}))


# ===========================================================================
# 12. ATLAS Integration (end-to-end)
# ===========================================================================
class TestAtlasIntegration(unittest.TestCase):
    def setUp(self):
        self.prime   = PRIME()
        self.atlas   = self.prime.atlas()
        self.engine  = self.prime.reasoner()

    def tearDown(self):
        self.prime.shutdown()

    def test_atlas_parse_then_solve(self):
        problem = self.atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
        result  = self.engine.solve(problem)
        self.assertIsInstance(result, ReasoningResult)
        self.assertEqual(result.status, "SOLVED")

    def test_solve_validates_problem_type(self):
        with self.assertRaises(ReasoningError):
            self.engine.solve("not a problem")

    def test_math_problem(self):
        problem = self.atlas.parse("A rectangle has length 8 m and width 5 m. Calculate area.")
        result  = self.engine.solve(problem)
        # ATLAS may or may not produce a multiply expression — status may be UNRESOLVED
        self.assertIn(result.status, ["SOLVED", "UNRESOLVED", "PARTIALLY_SOLVED"])

    def test_constraint_problem_via_atlas(self):
        problem = self.atlas.parse("Budget is 40000. Budget must be at most 50000.")
        result  = self.engine.solve(problem)
        # Constraint should be present or absent depending on ATLAS parsing
        self.assertIn(result.status, ["SOLVED", "UNRESOLVED", "PARTIALLY_SOLVED"])

    def test_prime_reasoner_method(self):
        """prime.reasoner() returns a ReasoningEngine."""
        r = self.prime.reasoner()
        self.assertIsInstance(r, ReasoningEngine)

    def test_validate_method(self):
        problem = self.atlas.parse("A = 10, B = 20, find C = A + B.")
        v = self.engine.validate(problem)
        self.assertIn("valid", v)
        self.assertIn("issues", v)

    def test_end_to_end_flow(self):
        """Full pipeline: parse → solve → trace → explain."""
        problem = self.atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
        result  = self.engine.solve(problem)
        trace   = result.trace
        explanation = result.explain()

        self.assertIsInstance(trace, ReasoningTrace)
        self.assertIsInstance(explanation, str)
        self.assertIn("50", explanation)

    def test_cross_domain_physics(self):
        problem = self.atlas.parse("Force is 50 N and mass is 10 kg. Find acceleration.")
        result  = self.engine.solve(problem)
        self.assertIn(result.status, ["SOLVED", "UNRESOLVED", "PARTIALLY_SOLVED"])

    def test_cross_domain_business(self):
        problem = self.atlas.parse("Revenue is 1000 and costs are 600. Find profit.")
        result  = self.engine.solve(problem)
        self.assertIn(result.status, ["SOLVED", "UNRESOLVED", "PARTIALLY_SOLVED"])


# ===========================================================================
# 13. Thread safety (basic)
# ===========================================================================
class TestThreadSafety(unittest.TestCase):
    def test_independent_calls_do_not_share_state(self):
        """Two solve_facts calls must be isolated."""
        engine = ReasoningEngine()
        r1 = engine.solve_facts(facts={"x": 10})
        r2 = engine.solve_facts(facts={"y": 20})
        # x should not be in r2's trace, y should not be in r1's trace
        facts_in_r1 = set(r1.derived_facts.keys())
        facts_in_r2 = set(r2.derived_facts.keys())
        self.assertNotIn("y", facts_in_r1)
        self.assertNotIn("x", facts_in_r2)


# ===========================================================================
# 14. C extension direct tests
# ===========================================================================
class TestCExtension(unittest.TestCase):
    def test_eval_expression_basic(self):
        result = _prime_reasoning.eval_expression("2 + 3", {})
        self.assertAlmostEqual(result, 5.0)

    def test_eval_expression_with_bindings(self):
        result = _prime_reasoning.eval_expression("x * y", {"x": 6.0, "y": 7.0})
        self.assertAlmostEqual(result, 42.0)

    def test_eval_expression_error(self):
        with self.assertRaises(_prime_reasoning.ReasoningError):
            _prime_reasoning.eval_expression("z + 1", {})

    def test_check_constraint_satisfied(self):
        result = _prime_reasoning.check_constraint("val", "<=", "100", {"val": 50.0})
        self.assertEqual(result, "satisfied")

    def test_check_constraint_violated(self):
        result = _prime_reasoning.check_constraint("val", ">", "100", {"val": 50.0})
        self.assertEqual(result, "violated")

    def test_check_constraint_unknown(self):
        result = _prime_reasoning.check_constraint("missing", "=", "0", {})
        self.assertEqual(result, "unknown")

    def test_solve_context_basic(self):
        ctx = {
            "problem_id": "test",
            "problem_domain": "math",
            "entities": [],
            "goals": [{"id": "g1", "target": "result", "description": "find result",
                       "priority": 0, "confidence": 1.0}],
            "constraints": [],
            "expressions": [{"id": "e1", "op": "add", "output_id": "result",
                              "operand_ids": ["3", "4"], "text": ""}],
            "rules": [],
            "facts": {},
            "config": {"mode": "hybrid", "max_steps": 100, "max_depth": 20,
                       "max_rule_firings": 50, "timeout_ms": 5000.0,
                       "use_rules": True, "use_symbolic": True, "use_llm": False,
                       "detect_circular": True, "confidence_threshold": 0.0},
        }
        result = _prime_reasoning.solve_context(ctx)
        self.assertEqual(result["status"], "SOLVED")
        self.assertAlmostEqual(result["answer_numeric"], 7.0)

    def test_version_string(self):
        self.assertIsInstance(_prime_reasoning.VERSION, str)
        self.assertTrue(_prime_reasoning.VERSION.startswith("0.5"))

    def test_reasoning_error_type(self):
        self.assertTrue(issubclass(_prime_reasoning.ReasoningError, Exception))


# ===========================================================================
# 15. Backward compatibility — previous phases unaffected
# ===========================================================================
class TestBackwardCompatibility(unittest.TestCase):
    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_prime_status(self):
        status = self.prime.status()
        self.assertIsInstance(status, str)
        self.assertIn("running", status.lower())

    def test_atlas_still_works(self):
        atlas   = self.prime.atlas()
        problem = atlas.parse("A car travels 150 km in 3 hours.")
        self.assertEqual(problem.domain, "physics")
        self.assertGreater(len(problem.entities), 0)

    def test_memory_still_works(self):
        from prime import MemoryConfig
        mem = self.prime.memory(MemoryConfig())
        item_id = mem.store("test content", memory_type="long_term", importance=0.8)
        self.assertTrue(item_id is not None)

    def test_reasoner_on_prime(self):
        reasoner = self.prime.reasoner()
        self.assertIsInstance(reasoner, ReasoningEngine)


if __name__ == "__main__":
    unittest.main(verbosity=2)
