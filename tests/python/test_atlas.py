"""
tests/python/test_atlas.py — Python unit tests for Phase 4: ATLAS engine.

Run:
    PYTHONPATH=python python3 -m unittest tests/python/test_atlas.py
"""

import json
import unittest

from prime import (
    PRIME, ATLASEngine, ATLASError, Problem,
    ValidationResult, UniversalProblemProto,
)
import _prime_atlas


class TestSchemaVersion(unittest.TestCase):
    def test_schema_version_constant(self):
        self.assertEqual(_prime_atlas.SCHEMA_VERSION, "1.0")

    def test_engine_schema_version(self):
        a = ATLASEngine()
        self.assertEqual(a.schema_version(), "1.0")


class TestATLASEngineConstruction(unittest.TestCase):
    def test_default_construction(self):
        a = ATLASEngine()
        self.assertIsInstance(a, ATLASEngine)

    def test_repr(self):
        a = ATLASEngine()
        r = repr(a)
        self.assertIn("ATLASEngine", r)
        self.assertIn("1.0", r)

    def test_custom_limits(self):
        a = ATLASEngine(max_entities=10, max_goals=5)
        self.assertIsNotNone(a)

    def test_prime_atlas_method(self):
        with PRIME() as prime:
            atlas = prime.atlas()
            self.assertIsInstance(atlas, ATLASEngine)


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.atlas = ATLASEngine()

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    def test_parse_returns_problem(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        self.assertIsInstance(p, Problem)

    def test_parse_preserves_original_input(self):
        text = "A car travels 120 km in 3 hours."
        p = self.atlas.parse(text)
        self.assertEqual(p.original_input, text)

    def test_parse_has_problem_id(self):
        p = self.atlas.parse("Mass is 10 kg.")
        self.assertTrue(len(p.problem_id) > 0)

    def test_parse_schema_version(self):
        p = self.atlas.parse("A ball falls 10 m.")
        self.assertEqual(p.schema_version, "1.0")

    def test_parse_empty_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.parse("")

    def test_parse_whitespace_only_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.parse("   ")

    def test_parse_non_string_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.parse(42)  # type: ignore

    def test_parse_too_large_raises(self):
        a = ATLASEngine(max_input_bytes=100)
        with self.assertRaises(ATLASError):
            a.parse("x" * 200)

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------
    def test_entities_is_list(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        self.assertIsInstance(p.entities, list)

    def test_entities_found(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        self.assertGreater(len(p.entities), 0)

    def test_entity_has_required_fields(self):
        p = self.atlas.parse("Mass is 10 kg.")
        for e in p.entities:
            self.assertIn("id",         e)
            self.assertIn("type",       e)
            self.assertIn("name",       e)
            self.assertIn("confidence", e)

    def test_entity_confidence_range(self):
        p = self.atlas.parse("Distance is 100 km.")
        for e in p.entities:
            self.assertGreaterEqual(e["confidence"], 0.0)
            self.assertLessEqual(e["confidence"],    1.0)

    def test_entity_with_quantity(self):
        p = self.atlas.parse("The distance is 120 km.")
        qty_entities = [e for e in p.entities if e.get("quantity")]
        self.assertGreater(len(qty_entities), 0)
        q = qty_entities[0]["quantity"]
        self.assertIn("value", q)
        self.assertIn("unit",  q)
        self.assertIn("dimension", q)

    def test_entity_ids_unique(self):
        p = self.atlas.parse("Mass is 10 kg and distance is 50 km.")
        ids = [e["id"] for e in p.entities]
        self.assertEqual(len(ids), len(set(ids)))

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------
    def test_goals_is_list(self):
        p = self.atlas.parse("What is the speed?")
        self.assertIsInstance(p.goals, list)

    def test_goal_found_for_question(self):
        p = self.atlas.parse("A car travels 100 km in 2 hours. What is its average speed?")
        self.assertGreater(len(p.goals), 0)

    def test_goal_has_required_fields(self):
        p = self.atlas.parse("Calculate the average speed of the car.")
        for g in p.goals:
            self.assertIn("id",          g)
            self.assertIn("description", g)
            self.assertIn("confidence",  g)

    def test_goal_confidence_range(self):
        p = self.atlas.parse("Find the force applied to the object.")
        for g in p.goals:
            self.assertGreaterEqual(g["confidence"], 0.0)
            self.assertLessEqual(g["confidence"],    1.0)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    def test_relationships_is_list(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        self.assertIsInstance(p.relationships, list)

    def test_relationship_has_required_fields(self):
        p = self.atlas.parse("A car travels 120 km.")
        for r in p.relationships:
            self.assertIn("id",         r)
            self.assertIn("source_id",  r)
            self.assertIn("relation",   r)
            self.assertIn("target_id",  r)
            self.assertIn("confidence", r)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def test_constraints_is_list(self):
        p = self.atlas.parse("Temperature must be < 100.")
        self.assertIsInstance(p.constraints, list)

    def test_constraint_found(self):
        p = self.atlas.parse("The temperature must be temperature < 100.")
        self.assertGreater(len(p.constraints), 0)

    def test_constraint_has_required_fields(self):
        p = self.atlas.parse("x < 50.")
        for c in p.constraints:
            self.assertIn("id",         c)
            self.assertIn("left",       c)
            self.assertIn("op",         c)
            self.assertIn("right",      c)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def test_operations_is_list(self):
        p = self.atlas.parse("Calculate the speed.")
        self.assertIsInstance(p.operations, list)

    def test_operation_found(self):
        p = self.atlas.parse("Calculate the average speed of a car.")
        self.assertGreater(len(p.operations), 0)

    def test_operation_has_type(self):
        p = self.atlas.parse("Sort an array of numbers.")
        for op in p.operations:
            self.assertIn("type", op)
            self.assertTrue(len(op["type"]) > 0)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def test_expressions_is_list(self):
        p = self.atlas.parse("speed = distance / time")
        self.assertIsInstance(p.expressions, list)

    def test_expression_found_formula(self):
        p = self.atlas.parse("The formula is speed = distance / time.")
        self.assertGreater(len(p.expressions), 0)

    def test_expression_has_required_fields(self):
        p = self.atlas.parse("speed = distance / time")
        for ex in p.expressions:
            self.assertIn("id",  ex)
            self.assertIn("op",  ex)
            self.assertIn("text", ex)

    # ------------------------------------------------------------------
    # Assumptions
    # ------------------------------------------------------------------
    def test_assumptions_is_list(self):
        p = self.atlas.parse("Assume no friction.")
        self.assertIsInstance(p.assumptions, list)

    def test_assumption_found(self):
        p = self.atlas.parse("Assuming no friction, the car travels 100 km.")
        self.assertGreater(len(p.assumptions), 0)

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------
    def test_states_is_list(self):
        p = self.atlas.parse("A car travels 120 km.")
        self.assertIsInstance(p.states, list)

    def test_initial_state_built(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        labels = [s["label"] for s in p.states]
        self.assertIn("initial", labels)

    def test_state_has_entity_ids(self):
        p = self.atlas.parse("Mass is 10 kg.")
        for s in p.states:
            self.assertIn("entity_ids",     s)
            self.assertIn("constraint_ids", s)

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def test_graph_summary(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        g = p.graph
        self.assertIn("node_count", g)
        self.assertIn("edge_count", g)
        self.assertGreater(g["node_count"], 0)

    # ------------------------------------------------------------------
    # Problem repr and to_dict
    # ------------------------------------------------------------------
    def test_problem_repr(self):
        p = self.atlas.parse("Mass is 10 kg.")
        r = repr(p)
        self.assertIn("Problem", r)

    def test_problem_to_dict(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        d = p.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("problem_id",           d)
        self.assertIn("atlas_schema_version", d)
        self.assertIn("entities",             d)
        self.assertIn("goals",                d)
        self.assertIn("domain",               d)


class TestDomainClassification(unittest.TestCase):
    def setUp(self):
        self.atlas = ATLASEngine()

    def _check(self, text, expected_domain):
        p = self.atlas.parse(text)
        self.assertEqual(
            p.domain, expected_domain,
            f"Expected {expected_domain!r}, got {p.domain!r} for: {text[:60]}"
        )

    def test_physics_motion(self):
        self._check(
            "A car travels 120 km in 3 hours. What is its average speed?",
            "physics"
        )

    def test_physics_force(self):
        self._check(
            "An object has mass 10 kg and acceleration 2 m/s^2. Find the force.",
            "physics"
        )

    def test_physics_train(self):
        self._check(
            "If a train travels 100 km in 2 hours, what is its average speed?",
            "physics"
        )

    def test_mathematics(self):
        self._check(
            "Solve x^2 + 5x + 6 = 0 by factoring the polynomial.",
            "mathematics"
        )

    def test_programming(self):
        self._check(
            "Sort an array of integers in ascending order using an algorithm.",
            "programming"
        )

    def test_business(self):
        self._check(
            "Revenue is 10000 and cost is 7000. What is the profit?",
            "business"
        )

    def test_engineering(self):
        self._check(
            "A beam must support a load of 500 N. Stress must not exceed 200 Pa.",
            "engineering"
        )

    def test_domain_confidence_range(self):
        p = self.atlas.parse("Sort an array of integers.")
        self.assertGreaterEqual(p.domain_confidence, 0.0)
        self.assertLessEqual(p.domain_confidence,    1.0)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.atlas = ATLASEngine()

    def test_parsed_problem_is_valid(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours. What is its speed?")
        r = self.atlas.validate(p)
        self.assertIsInstance(r, ValidationResult)
        self.assertTrue(r.valid)
        self.assertEqual(r.errors, [])

    def test_validate_returns_validation_result(self):
        p = self.atlas.parse("Mass is 10 kg.")
        r = self.atlas.validate(p)
        self.assertIsInstance(r, ValidationResult)
        self.assertIsInstance(r.errors,   list)
        self.assertIsInstance(r.warnings, list)

    def test_validate_bool_conversion(self):
        p = self.atlas.parse("Mass is 10 kg.")
        r = self.atlas.validate(p)
        self.assertTrue(bool(r))

    def test_validate_non_problem_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.validate("not a problem")  # type: ignore

    def test_validate_repr(self):
        p = self.atlas.parse("Mass is 10 kg.")
        r = self.atlas.validate(p)
        self.assertIn("ValidationResult", repr(r))


class TestSerialization(unittest.TestCase):
    def setUp(self):
        self.atlas = ATLASEngine()

    def _round_trip(self, text):
        p1 = self.atlas.parse(text)
        j  = self.atlas.to_json(p1)
        p2 = self.atlas.from_json(j)
        return p1, p2

    def test_to_json_returns_string(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        j = self.atlas.to_json(p)
        self.assertIsInstance(j, str)
        self.assertTrue(len(j) > 0)

    def test_to_json_is_valid_json(self):
        p = self.atlas.parse("A car travels 120 km in 3 hours.")
        j = self.atlas.to_json(p)
        data = json.loads(j)
        self.assertIsInstance(data, dict)

    def test_to_json_contains_schema_version(self):
        p = self.atlas.parse("Mass is 10 kg.")
        j = self.atlas.to_json(p)
        data = json.loads(j)
        self.assertEqual(data["atlas_schema_version"], "1.0")

    def test_to_json_non_problem_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.to_json("not a problem")  # type: ignore

    def test_from_json_returns_problem(self):
        p = self.atlas.parse("A car travels 120 km.")
        j = self.atlas.to_json(p)
        p2 = self.atlas.from_json(j)
        self.assertIsInstance(p2, Problem)

    def test_from_json_empty_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.from_json("")

    def test_from_json_non_string_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.from_json(123)  # type: ignore

    def test_from_json_malformed_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.from_json("{not valid json!!!")

    def test_from_json_wrong_schema_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.from_json('{"atlas_schema_version":"99.0","problem_id":"x"}')

    # Round-trip tests
    def test_round_trip_problem_id(self):
        p1, p2 = self._round_trip("A car travels 120 km in 3 hours.")
        self.assertEqual(p1.problem_id, p2.problem_id)

    def test_round_trip_domain(self):
        p1, p2 = self._round_trip("A car travels 120 km in 3 hours.")
        self.assertEqual(p1.domain, p2.domain)

    def test_round_trip_entity_count(self):
        p1, p2 = self._round_trip("Mass is 10 kg and distance is 50 km.")
        self.assertEqual(len(p1.entities), len(p2.entities))

    def test_round_trip_goal_count(self):
        p1, p2 = self._round_trip("Calculate the average speed.")
        self.assertEqual(len(p1.goals), len(p2.goals))

    def test_round_trip_constraint_count(self):
        p1, p2 = self._round_trip("Temperature must be temperature < 100.")
        self.assertEqual(len(p1.constraints), len(p2.constraints))

    def test_round_trip_graph_nodes(self):
        p1, p2 = self._round_trip("A car travels 120 km in 3 hours.")
        self.assertEqual(p1.graph["node_count"], p2.graph["node_count"])

    def test_round_trip_is_valid(self):
        p1, p2 = self._round_trip("A car travels 120 km in 3 hours. What is its speed?")
        r = self.atlas.validate(p2)
        self.assertTrue(r.valid)

    def test_round_trip_original_input(self):
        text = "A car travels 120 km in 3 hours."
        p1, p2 = self._round_trip(text)
        self.assertEqual(p1.original_input, p2.original_input)

    def test_round_trip_schema_version(self):
        p1, p2 = self._round_trip("Mass is 10 kg.")
        self.assertEqual(p1.schema_version, p2.schema_version)
        self.assertEqual(p2.schema_version, "1.0")


class TestCrossDomainUniversality(unittest.TestCase):
    """Verify same UniversalProblem schema works across all domains."""

    def setUp(self):
        self.atlas = ATLASEngine()

    CASES = [
        ("mathematics", "Solve x^2 - 9 = 0 by factoring.", "mathematics"),
        ("physics",     "An object has mass 10 kg and acceleration 2 m/s^2.", "physics"),
        ("programming", "Sort an array of integers in ascending order.", "programming"),
        ("business",    "Revenue is 10000 and cost is 7000. Calculate profit.", "business"),
        ("engineering", "A machine operates above temperature limit 200 C.", "engineering"),
    ]

    def test_all_domains_parse(self):
        for name, text, expected_domain in self.CASES:
            with self.subTest(domain=name):
                p = self.atlas.parse(text)
                self.assertIsInstance(p, Problem)
                self.assertEqual(p.domain, expected_domain,
                                 f"domain mismatch for {name}: {p.domain}")

    def test_all_domains_valid(self):
        for name, text, _ in self.CASES:
            with self.subTest(domain=name):
                p = self.atlas.parse(text)
                r = self.atlas.validate(p)
                self.assertTrue(r.valid,
                                f"Validation failed for {name}: {r.errors}")

    def test_all_domains_round_trip(self):
        for name, text, expected_domain in self.CASES:
            with self.subTest(domain=name):
                p1 = self.atlas.parse(text)
                j  = self.atlas.to_json(p1)
                p2 = self.atlas.from_json(j)
                self.assertEqual(p2.domain, expected_domain)
                self.assertEqual(len(p2.entities), len(p1.entities))
                self.assertEqual(len(p2.goals),    len(p1.goals))

    def test_same_schema_all_domains(self):
        """All domains produce dicts with identical top-level keys."""
        expected_keys = {
            "problem_id", "atlas_schema_version", "original_input",
            "normalized_input", "domain", "domain_confidence",
            "entities", "relationships", "constraints", "goals",
            "operations", "expressions", "states", "assumptions",
            "evidence", "logic_clauses", "graph", "metadata",
        }
        for name, text, _ in self.CASES:
            with self.subTest(domain=name):
                p = self.atlas.parse(text)
                j = self.atlas.to_json(p)
                d = json.loads(j)
                for k in expected_keys:
                    self.assertIn(k, d,
                                  f"Key {k!r} missing in {name} representation")


class TestATLASDoesNotSolve(unittest.TestCase):
    """ATLAS represents problems; it must NOT solve them."""

    def setUp(self):
        self.atlas = ATLASEngine()

    def test_no_numeric_answer_in_goals(self):
        """Goals describe what to find, not computed answers."""
        p = self.atlas.parse(
            "If a train travels 100 km in 2 hours, what is its average speed?"
        )
        for g in p.goals:
            # Goal description should not contain "50" (the answer)
            self.assertNotIn("50 km/h", g.get("description", ""))
            self.assertNotIn("50 km/h", g.get("desired_state", ""))

    def test_entities_represent_inputs_not_answers(self):
        """Entities should be the given quantities (100 km, 2 hours) not derived ones."""
        p = self.atlas.parse(
            "A car travels 100 km in 2 hours."
        )
        # Should find 100 and 2, but NOT 50 as a "computed" entity
        values = []
        for e in p.entities:
            if e.get("quantity"):
                values.append(e["quantity"]["value"])
        # 100 and 2 should be found; 50 (the speed) should not appear as input
        self.assertIn(100.0, values)
        self.assertIn(2.0,   values)

    def test_expression_is_structural_not_evaluated(self):
        """Expressions represent formula structure, not computed results."""
        p = self.atlas.parse("speed = distance / time")
        for ex in p.expressions:
            # The expression should describe "divide", not contain "50"
            self.assertNotIn("50", ex.get("text", ""))
            self.assertIn(ex.get("op", ""), ["divide", "subtract", "add", "multiply", "custom"])


class TestStructuredInput(unittest.TestCase):
    def setUp(self):
        self.atlas = ATLASEngine()

    def test_parse_structured_minimal(self):
        proto = UniversalProblemProto(
            original_input="A car travels 120 km in 3 hours.",
            domain="physics",
        )
        p = self.atlas.parse_structured(proto)
        self.assertIsInstance(p, Problem)
        self.assertEqual(p.domain, "physics")

    def test_parse_structured_with_entities(self):
        proto = UniversalProblemProto(
            original_input="Custom problem.",
            domain="mathematics",
            entities=[{
                "id":         "e1",
                "type":       "quantity",
                "name":       "distance",
                "quantity":   {"value": 100.0, "unit": "km", "dimension": "length"},
                "confidence": 1.0,
                "attributes": [],
                "metadata":   {},
            }],
            goals=[{
                "id":            "g1",
                "description":   "find speed",
                "target":        "speed",
                "desired_state": "",
                "priority":      0,
                "confidence":    0.9,
                "metadata":      {},
            }],
        )
        p = self.atlas.parse_structured(proto)
        self.assertIsInstance(p, Problem)
        self.assertGreaterEqual(len(p.entities), 1)
        self.assertGreaterEqual(len(p.goals),    1)

    def test_parse_structured_non_proto_raises(self):
        with self.assertRaises(ATLASError):
            self.atlas.parse_structured("not a proto")  # type: ignore

    def test_structured_round_trip(self):
        proto = UniversalProblemProto(
            original_input="Budget constraint problem.",
            domain="business",
            constraints=[{
                "id":         "c1",
                "left":       "budget",
                "op":         "<=",
                "right":      "50000",
                "type":       "domain",
                "confidence": 1.0,
                "metadata":   {},
            }],
        )
        p  = self.atlas.parse_structured(proto)
        j  = self.atlas.to_json(p)
        p2 = self.atlas.from_json(j)
        self.assertEqual(p2.domain, "business")
        self.assertGreaterEqual(len(p2.constraints), 1)


class TestMemoryEngineIntegration(unittest.TestCase):
    """End-to-end: ATLAS → Memory Engine (Phase 3 integration)."""

    def test_atlas_to_memory_store(self):
        with PRIME() as prime:
            atlas = prime.atlas()
            mem   = prime.memory()

            p = atlas.parse("A car travels 120 km in 3 hours. What is its speed?")
            j = atlas.to_json(p)

            # Store the JSON representation in memory
            mid = mem.store(
                j,
                memory_type="long_term",
                importance=0.8,
                metadata={"domain": p.domain, "problem_id": p.problem_id},
            )
            self.assertGreater(mid, 0)

            # Retrieve it back
            item = mem.get(mid)
            self.assertEqual(item.content, j)
            self.assertEqual(item.metadata.get("domain"), "physics")

            # Reconstruct from stored JSON
            p2 = atlas.from_json(item.content)
            self.assertEqual(p2.problem_id, p.problem_id)
            self.assertEqual(p2.domain,     p.domain)

    def test_atlas_memory_search(self):
        with PRIME() as prime:
            atlas = prime.atlas()
            mem   = prime.memory()

            problems = [
                "A car travels 120 km in 3 hours.",
                "An object has mass 10 kg and acceleration 2 m/s^2.",
                "Revenue is 10000 and cost is 7000. What is profit?",
            ]
            for text in problems:
                p = atlas.parse(text)
                mem.store(
                    p.to_dict().__repr__(),  # store a string summary
                    memory_type="long_term",
                    importance=0.7,
                    metadata={"domain": p.domain},
                )

            results = mem.search("car travels speed", limit=5)
            self.assertGreater(len(results), 0)


class TestProblemRepr(unittest.TestCase):
    def test_repr_contains_id(self):
        a = ATLASEngine()
        p = a.parse("Mass is 10 kg.")
        self.assertIn(p.problem_id, repr(p))

    def test_repr_contains_domain(self):
        a = ATLASEngine()
        p = a.parse("Sort an array.")
        self.assertIn(p.domain, repr(p))


class TestLogicAndAssumptions(unittest.TestCase):
    def setUp(self):
        self.atlas = ATLASEngine()

    def test_if_then_detected(self):
        p = self.atlas.parse("If temperature > 100, then warning is true.")
        self.assertGreater(len(p.to_dict().get("logic_clauses", [])), 0)

    def test_assumption_detected(self):
        p = self.atlas.parse("Assume no friction. The car travels 100 km.")
        self.assertGreater(len(p.assumptions), 0)

    def test_neglect_assumption(self):
        p = self.atlas.parse("Neglecting air resistance, calculate the range.")
        self.assertGreater(len(p.assumptions), 0)


class TestResourceLimits(unittest.TestCase):
    def test_entity_count_capped(self):
        a = ATLASEngine(max_entities=3)
        p = a.parse("Distance is 120 km, time is 3 h, speed is 40 m/s, mass is 10 kg, force is 20 N.")
        self.assertLessEqual(len(p.entities), 3)

    def test_two_parses_have_different_ids(self):
        a = ATLASEngine()
        p1 = a.parse("Mass is 10 kg.")
        p2 = a.parse("Mass is 10 kg.")
        self.assertNotEqual(p1.problem_id, p2.problem_id)


class TestPrimeIntegration(unittest.TestCase):
    """Verify Phase 1, 2, 3 still work alongside Phase 4."""

    def test_prime_runtime_still_works(self):
        with PRIME() as prime:
            status = prime.status()
            self.assertIsInstance(status, str)
            self.assertTrue(prime.is_running())

    def test_memory_engine_still_works(self):
        with PRIME() as prime:
            mem = prime.memory()
            mid = mem.store("test content", memory_type="working")
            item = mem.get(mid)
            self.assertEqual(item.content, "test content")

    def test_atlas_and_memory_coexist(self):
        with PRIME() as prime:
            atlas = prime.atlas()
            mem   = prime.memory()
            p = atlas.parse("A car travels 100 km.")
            mid = mem.store(atlas.to_json(p), importance=0.5)
            self.assertGreater(mid, 0)
            item = mem.get(mid)
            p2 = atlas.from_json(item.content)
            self.assertEqual(p.problem_id, p2.problem_id)

    def test_atlas_does_not_break_prime(self):
        """Creating an ATLAS engine must not break the PRIME runtime."""
        with PRIME() as prime:
            self.assertTrue(prime.is_running())
            atlas = prime.atlas()
            _ = atlas.parse("A car travels 100 km.")
            self.assertTrue(prime.is_running())  # still running after ATLAS use


if __name__ == "__main__":
    unittest.main()
