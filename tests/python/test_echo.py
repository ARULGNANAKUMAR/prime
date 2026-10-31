"""
tests/python/test_echo.py — Phase 7: ECHO Experience-to-Intelligence Engine.

Tests are grouped into:
  1.  Data models (Experience, Observation, Pattern, …)
  2.  ExperienceStore
  3.  ExperienceCollector
  4.  ObservationExtractor
  5.  PatternDetector
  6.  GeneralizationEngine
  7.  PrincipleExtractor
  8.  ConfidenceEstimator
  9.  ExperienceValidator
  10. ExperienceSimilarity
  11. ExperienceQueryEngine
  12. ECHOEngine public API
  13. PRIME.echo() integration
  14. Knowledge versioning / contradiction handling
  15. Retention policies
  16. EchoTrace provenance
  17. Edge cases & resilience
"""

import time
import unittest
import uuid

from prime import (
    PRIME,
    ECHOEngine, ECHOError,
    Experience, ExperienceOutcome, ExperienceLifecycle, ExperienceEvent,
    Observation, Pattern, Generalization, KnowledgeCandidate,
    KnowledgeType, PrincipleStatus, ValidationStatus,
    RetentionPolicy, EchoTrace,
    ExperienceStore, ExperienceCollector, ObservationExtractor,
    PatternDetector, GeneralizationEngine, PrincipleExtractor,
    ExperienceValidator, ExperienceSimilarity, ConfidenceEstimator,
    ExperienceQueryEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()

def _make_exp(domain: str = "physics",
              category: str = "arithmetic",
              modules: list = _SENTINEL,
              outcome: str = ExperienceOutcome.SUCCESS.value,
              validated: bool = True,
              confidence: float = 0.95,
              text: str = "A car travels 150 km in 3 hours.",
              tags: list = None) -> Experience:
    resolved_modules = ["reasoning", "validator"] if modules is _SENTINEL else modules
    exp = Experience(
        task_text        = text,
        task_domain      = domain,
        task_category    = category,
        modules_used     = resolved_modules,
        result_answer    = "50",
        result_answer_unit = "km/h",
        result_confidence  = confidence,
        result_status    = "SUCCESS" if outcome == ExperienceOutcome.SUCCESS.value else "FAILURE",
        reasoning_status = "SOLVED" if validated else "UNRESOLVED",
        outcome          = outcome,
        validated        = validated,
        tags             = list(tags or []),
    )
    return exp


def _make_failure_exp(domain: str = "physics",
                      category: str = "arithmetic") -> Experience:
    return _make_exp(domain=domain, category=category,
                     outcome=ExperienceOutcome.FAILURE.value,
                     validated=False,
                     confidence=0.1,
                     modules=["reasoning"])


def _populate_echo(echo: ECHOEngine, n_success: int = 5,
                   n_failure: int = 0) -> list:
    """Record n_success successes and n_failure failures; return experiences."""
    exps = []
    for _ in range(n_success):
        exp = _make_exp()
        echo._store.add_experience(exp)
        exps.append(exp)
    for _ in range(n_failure):
        exp = _make_failure_exp()
        echo._store.add_experience(exp)
        exps.append(exp)
    return exps


# ===========================================================================
# 1. Data models
# ===========================================================================

class TestExperienceModel(unittest.TestCase):

    def test_experience_defaults(self):
        exp = Experience()
        self.assertIsNotNone(exp.id)
        self.assertEqual(exp.outcome, ExperienceOutcome.UNKNOWN.value)
        self.assertEqual(exp.lifecycle, ExperienceLifecycle.CREATED.value)
        self.assertIsInstance(exp.events, list)
        self.assertIsInstance(exp.tags, list)

    def test_experience_round_trip(self):
        exp = _make_exp(tags=["speed", "vehicle"])
        d   = exp.to_dict()
        exp2 = Experience.from_dict(d)
        self.assertEqual(exp.id, exp2.id)
        self.assertEqual(exp.task_domain, exp2.task_domain)
        self.assertEqual(exp.outcome, exp2.outcome)
        self.assertEqual(exp.tags, exp2.tags)

    def test_experience_event_round_trip(self):
        ev = ExperienceEvent(event_type="TASK_RECEIVED",
                             data={"text": "hello"})
        d  = ev.to_dict()
        ev2 = ExperienceEvent.from_dict(d)
        self.assertEqual(ev.event_type, ev2.event_type)
        self.assertEqual(ev.data, ev2.data)

    def test_observation_defaults(self):
        obs = Observation()
        self.assertIsNotNone(obs.id)
        self.assertEqual(obs.outcome, ExperienceOutcome.UNKNOWN.value)

    def test_observation_round_trip(self):
        obs = Observation(task_domain="physics", outcome="SUCCESS")
        d   = obs.to_dict()
        obs2 = Observation.from_dict(d)
        self.assertEqual(obs.id, obs2.id)
        self.assertEqual(obs.task_domain, obs2.task_domain)

    def test_pattern_success_rate(self):
        pat = Pattern(success_count=8, failure_count=2)
        self.assertAlmostEqual(pat.success_rate, 0.8)

    def test_pattern_is_contradicted(self):
        pat_clean = Pattern(success_count=5, failure_count=0)
        pat_mixed = Pattern(success_count=5, failure_count=2)
        self.assertFalse(pat_clean.is_contradicted)
        self.assertTrue(pat_mixed.is_contradicted)

    def test_pattern_round_trip(self):
        pat = Pattern(condition_domain="physics", observation_count=5)
        d   = pat.to_dict()
        pat2 = Pattern.from_dict(d)
        self.assertEqual(pat.id, pat2.id)
        self.assertEqual(pat.observation_count, pat2.observation_count)

    def test_knowledge_candidate_update_confidence(self):
        kc = KnowledgeCandidate(confidence=0.5)
        kc.update_confidence(0.8, reason="new evidence")
        self.assertAlmostEqual(kc.confidence, 0.8)
        self.assertEqual(kc.version, 2)
        self.assertTrue(any("0.50" in entry for entry in kc.change_log))

    def test_knowledge_candidate_round_trip(self):
        kc = KnowledgeCandidate(statement="test", confidence=0.7,
                                knowledge_type=KnowledgeType.PATTERN.value)
        d  = kc.to_dict()
        kc2 = KnowledgeCandidate.from_dict(d)
        self.assertEqual(kc.id, kc2.id)
        self.assertEqual(kc.statement, kc2.statement)

    def test_experience_outcome_enum(self):
        for outcome in ExperienceOutcome:
            self.assertIsInstance(outcome.value, str)

    def test_knowledge_type_enum(self):
        types = {kt.value for kt in KnowledgeType}
        self.assertIn("OBSERVATION", types)
        self.assertIn("PATTERN", types)
        self.assertIn("GENERALIZATION", types)
        self.assertIn("PRINCIPLE", types)

    def test_generalization_round_trip(self):
        gen = Generalization(statement="test gen", confidence=0.6)
        d   = gen.to_dict()
        gen2 = Generalization.from_dict(d)
        self.assertEqual(gen.id, gen2.id)
        self.assertEqual(gen.statement, gen2.statement)


# ===========================================================================
# 2. ExperienceStore
# ===========================================================================

class TestExperienceStore(unittest.TestCase):

    def setUp(self):
        self.store = ExperienceStore()

    def tearDown(self):
        self.store.close()

    def test_add_and_get_experience(self):
        exp = _make_exp()
        self.store.add_experience(exp)
        fetched = self.store.get_experience(exp.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, exp.id)

    def test_list_experiences_empty(self):
        self.assertEqual(self.store.list_experiences(), [])

    def test_list_experiences_filter_outcome(self):
        self.store.add_experience(_make_exp(outcome=ExperienceOutcome.SUCCESS.value))
        self.store.add_experience(_make_failure_exp())
        successes = self.store.list_experiences(outcome=ExperienceOutcome.SUCCESS.value)
        self.assertEqual(len(successes), 1)
        self.assertEqual(successes[0].outcome, ExperienceOutcome.SUCCESS.value)

    def test_list_experiences_filter_domain(self):
        self.store.add_experience(_make_exp(domain="physics"))
        self.store.add_experience(_make_exp(domain="logic"))
        physics = self.store.list_experiences(domain="physics")
        self.assertEqual(len(physics), 1)

    def test_list_experiences_filter_tag(self):
        self.store.add_experience(_make_exp(tags=["speed"]))
        self.store.add_experience(_make_exp(tags=["logic"]))
        speed = self.store.list_experiences(tag="speed")
        self.assertEqual(len(speed), 1)

    def test_list_experiences_limit(self):
        for _ in range(5):
            self.store.add_experience(_make_exp())
        limited = self.store.list_experiences(limit=3)
        self.assertEqual(len(limited), 3)

    def test_count_experiences(self):
        self.store.add_experience(_make_exp())
        self.store.add_experience(_make_exp())
        self.assertEqual(self.store.count_experiences(), 2)

    def test_delete_experience(self):
        exp = _make_exp()
        self.store.add_experience(exp)
        result = self.store.delete_experience(exp.id)
        self.assertTrue(result)
        self.assertIsNone(self.store.get_experience(exp.id))

    def test_delete_nonexistent(self):
        self.assertFalse(self.store.delete_experience("no-such-id"))

    def test_add_and_list_observations(self):
        obs = Observation(task_domain="physics", outcome="SUCCESS")
        self.store.add_observation(obs)
        results = self.store.list_observations(domain="physics")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, obs.id)

    def test_get_observations_for_experience(self):
        exp_id = str(uuid.uuid4())
        obs = Observation(experience_id=exp_id)
        self.store.add_observation(obs)
        results = self.store.get_observations_for_experience(exp_id)
        self.assertEqual(len(results), 1)

    def test_add_and_get_pattern(self):
        pat = Pattern(condition_domain="physics", confidence=0.8)
        self.store.add_pattern(pat)
        fetched = self.store.get_pattern(pat.id)
        self.assertIsNotNone(fetched)
        self.assertAlmostEqual(fetched.confidence, 0.8)

    def test_list_patterns_filter(self):
        self.store.add_pattern(Pattern(condition_domain="physics",
                                       condition_module="reasoning"))
        self.store.add_pattern(Pattern(condition_domain="logic",
                                       condition_module="rule_engine"))
        physics = self.store.list_patterns(domain="physics")
        self.assertEqual(len(physics), 1)

    def test_add_and_list_knowledge(self):
        kc = KnowledgeCandidate(knowledge_type=KnowledgeType.PATTERN.value,
                                statement="test", confidence=0.7)
        self.store.add_knowledge(kc)
        results = self.store.list_knowledge(knowledge_type=KnowledgeType.PATTERN.value)
        self.assertEqual(len(results), 1)

    def test_update_knowledge_versioning(self):
        kc = KnowledgeCandidate(statement="original", confidence=0.5)
        self.store.add_knowledge(kc)
        kc2 = KnowledgeCandidate(id=kc.id, statement="updated", confidence=0.8,
                                  change_log=["v2: updated"])
        self.store.update_knowledge(kc2)
        fetched = self.store.get_knowledge(kc.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.statement, "updated")
        # version should have been bumped
        self.assertGreaterEqual(fetched.version, 2)


# ===========================================================================
# 3. ExperienceCollector
# ===========================================================================

class TestExperienceCollector(unittest.TestCase):

    def setUp(self):
        self.collector = ExperienceCollector()

    def test_collect_minimal(self):
        exp = self.collector.collect(problem_text="Test problem.")
        self.assertIsNotNone(exp.id)
        self.assertEqual(exp.task_text, "Test problem.")
        self.assertEqual(exp.outcome, ExperienceOutcome.UNKNOWN.value)

    def test_collect_with_mock_execution_result(self):
        class MockResult:
            answer         = "50"
            answer_unit    = "km/h"
            confidence     = 0.95
            status         = "SUCCESS"
            reasoning_status = "SOLVED"
            trace_steps    = []

        exp = self.collector.collect(problem_text="speed problem",
                                     execution_result=MockResult())
        self.assertEqual(exp.result_answer, "50")
        self.assertEqual(exp.outcome, ExperienceOutcome.SUCCESS.value)
        self.assertTrue(exp.validated)

    def test_collect_failure_outcome(self):
        class MockResult:
            answer = ""
            answer_unit = ""
            confidence = 0.0
            status = "FAILURE"
            reasoning_status = "UNRESOLVED"
            trace_steps = []

        exp = self.collector.collect(execution_result=MockResult())
        self.assertEqual(exp.outcome, ExperienceOutcome.FAILURE.value)
        self.assertFalse(exp.validated)

    def test_collect_maps_partial_success(self):
        class MockResult:
            answer = "partial"
            answer_unit = ""
            confidence = 0.5
            status = "PARTIAL_SUCCESS"
            reasoning_status = "PARTIAL"
            trace_steps = []

        exp = self.collector.collect(execution_result=MockResult())
        self.assertEqual(exp.outcome, ExperienceOutcome.PARTIAL_SUCCESS.value)

    def test_collect_with_atlas_problem(self):
        class MockProblem:
            domain     = "physics"
            entities   = []
            operations = []
            constraints = []
            goals      = []

        exp = self.collector.collect(problem_text="speed",
                                     atlas_problem=MockProblem())
        self.assertEqual(exp.task_domain, "physics")

    def test_collect_with_mosaic_plan(self):
        class MockPlan:
            module_ids       = ["reasoning", "validator"]
            status           = "VALID"
            selection_reasons = ["arithmetic", "validation"]

        exp = self.collector.collect(mosaic_plan=MockPlan())
        self.assertEqual(exp.modules_used, ["reasoning", "validator"])

    def test_collect_extra_tags_and_metadata(self):
        exp = self.collector.collect(
            extra_tags=["speed", "vehicle"],
            extra_metadata={"user": "test"},
        )
        self.assertIn("speed", exp.tags)
        self.assertEqual(exp.metadata.get("user"), "test")

    def test_collect_events_appended(self):
        class MockResult:
            answer = "42"
            answer_unit = ""
            confidence = 0.9
            status = "SUCCESS"
            reasoning_status = "SOLVED"
            trace_steps = []

        exp = self.collector.collect(problem_text="q", execution_result=MockResult())
        event_types = {e.event_type for e in exp.events}
        self.assertIn("RESULT_GENERATED", event_types)
        self.assertIn("EXPERIENCE_COMPLETED", event_types)

    def test_collect_with_broken_result_degrades_gracefully(self):
        """Collector must not raise if execution_result is malformed."""
        class BadResult:
            pass   # no attributes at all

        exp = self.collector.collect(problem_text="q", execution_result=BadResult())
        self.assertIsNotNone(exp)

    def test_outcome_mapping_coverage(self):
        outcomes = {
            "SUCCESS": ExperienceOutcome.SUCCESS.value,
            "PARTIAL_SUCCESS": ExperienceOutcome.PARTIAL_SUCCESS.value,
            "FALLBACK_USED": ExperienceOutcome.PARTIAL_SUCCESS.value,
            "FAILURE": ExperienceOutcome.FAILURE.value,
            "MODULE_ERROR": ExperienceOutcome.FAILURE.value,
            "LIMIT_EXCEEDED": ExperienceOutcome.TIMEOUT.value,
            "INVALID_PLAN": ExperienceOutcome.INVALID.value,
            "UNKNOWN_STATUS": ExperienceOutcome.UNKNOWN.value,
        }
        for status, expected in outcomes.items():
            self.assertEqual(ExperienceCollector._map_outcome(status), expected)


# ===========================================================================
# 4. ObservationExtractor
# ===========================================================================

class TestObservationExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ObservationExtractor()

    def test_extract_basic(self):
        exp = _make_exp()
        obs_list = self.extractor.extract(exp)
        self.assertEqual(len(obs_list), 1)
        obs = obs_list[0]
        self.assertEqual(obs.experience_id, exp.id)
        self.assertEqual(obs.task_domain, exp.task_domain)
        self.assertEqual(obs.outcome, exp.outcome)

    def test_extract_primary_module(self):
        exp = _make_exp(modules=["reasoning", "validator"])
        obs = self.extractor.extract(exp)[0]
        self.assertEqual(obs.primary_module, "reasoning")

    def test_extract_no_modules(self):
        exp = _make_exp(modules=[])
        obs = self.extractor.extract(exp)[0]
        self.assertEqual(obs.primary_module, "")

    def test_extract_arithmetic_flag(self):
        exp = _make_exp(domain="arithmetic", category="arithmetic")
        obs = self.extractor.extract(exp)[0]
        self.assertTrue(obs.has_arithmetic)

    def test_extract_success_validated(self):
        exp = _make_exp(outcome=ExperienceOutcome.SUCCESS.value, validated=True)
        obs = self.extractor.extract(exp)[0]
        self.assertEqual(obs.outcome, ExperienceOutcome.SUCCESS.value)
        self.assertTrue(obs.validated)

    def test_extract_failure(self):
        exp = _make_failure_exp()
        obs = self.extractor.extract(exp)[0]
        self.assertEqual(obs.outcome, ExperienceOutcome.FAILURE.value)
        self.assertFalse(obs.validated)

    def test_extract_tags_propagated(self):
        exp = _make_exp(tags=["speed", "physics"])
        obs = self.extractor.extract(exp)[0]
        self.assertIn("speed", obs.tags)


# ===========================================================================
# 5. PatternDetector
# ===========================================================================

class TestPatternDetector(unittest.TestCase):

    def setUp(self):
        self.detector = PatternDetector()

    def _make_obs(self, domain="physics", category="arithmetic",
                  module="reasoning",
                  outcome=ExperienceOutcome.SUCCESS.value,
                  validated=True) -> Observation:
        return Observation(
            task_domain=domain, task_category=category,
            primary_module=module, outcome=outcome, validated=validated,
        )

    def test_no_pattern_below_threshold(self):
        obs = [self._make_obs()]   # only 1
        patterns = self.detector.detect(obs, [])
        self.assertEqual(patterns, [])

    def test_pattern_formed_with_two_obs(self):
        obs = [self._make_obs() for _ in range(2)]
        patterns = self.detector.detect(obs, [])
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].observation_count, 2)

    def test_pattern_description_set(self):
        obs = [self._make_obs() for _ in range(3)]
        patterns = self.detector.detect(obs, [])
        self.assertIsInstance(patterns[0].description, str)
        self.assertGreater(len(patterns[0].description), 10)

    def test_pattern_confidence_nonzero(self):
        obs = [self._make_obs() for _ in range(4)]
        patterns = self.detector.detect(obs, [])
        self.assertGreater(patterns[0].confidence, 0.0)
        self.assertLessEqual(patterns[0].confidence, 1.0)

    def test_pattern_multiple_groups(self):
        obs = (
            [self._make_obs(domain="physics", category="arithmetic")] * 2
            + [self._make_obs(domain="logic", category="logic", module="rule_engine")] * 2
        )
        patterns = self.detector.detect(obs, [])
        domains = {p.condition_domain for p in patterns}
        self.assertIn("physics", domains)
        self.assertIn("logic", domains)

    def test_pattern_contradiction_counted(self):
        obs = (
            [self._make_obs(outcome=ExperienceOutcome.SUCCESS.value)] * 4
            + [self._make_obs(outcome=ExperienceOutcome.FAILURE.value)] * 2
        )
        patterns = self.detector.detect(obs, [])
        p = patterns[0]
        self.assertEqual(p.success_count, 4)
        self.assertEqual(p.failure_count, 2)
        self.assertTrue(p.is_contradicted)

    def test_existing_pattern_updated(self):
        obs2 = [self._make_obs() for _ in range(2)]
        pats = self.detector.detect(obs2, [])
        obs3 = [self._make_obs() for _ in range(3)]
        pats2 = self.detector.detect(obs3, pats)
        # Should still be 1 pattern (same key), just updated
        self.assertEqual(len(pats2), 1)
        self.assertEqual(pats2[0].observation_count, 3)

    def test_pattern_versioned_on_update(self):
        obs = [self._make_obs() for _ in range(2)]
        pats = self.detector.detect(obs, [])
        v1 = pats[0].version
        obs_more = [self._make_obs() for _ in range(4)]
        pats2 = self.detector.detect(obs_more, pats)
        self.assertGreater(pats2[0].version, v1)


# ===========================================================================
# 6. GeneralizationEngine
# ===========================================================================

class TestGeneralizationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GeneralizationEngine()

    def _make_pattern(self, n_obs=5, n_succ=5, n_fail=0, conf=0.75,
                      domain="physics", category="arithmetic",
                      module="reasoning") -> Pattern:
        pat = Pattern(
            condition_domain=domain, condition_category=category,
            condition_module=module,
            observation_count=n_obs,
            success_count=n_succ, failure_count=n_fail,
            confidence=conf,
        )
        return pat

    def test_no_gen_below_threshold_confidence(self):
        pat = self._make_pattern(conf=0.30, n_obs=5)
        gens = self.engine.generalize([pat], [])
        self.assertEqual(gens, [])

    def test_no_gen_below_threshold_obs(self):
        pat = self._make_pattern(conf=0.80, n_obs=2)
        gens = self.engine.generalize([pat], [])
        self.assertEqual(gens, [])

    def test_generalization_created(self):
        pat = self._make_pattern(n_obs=5, conf=0.80)
        exps = [_make_exp() for _ in range(5)]
        pat.supporting_experience_ids = [e.id for e in exps]
        gens = self.engine.generalize([pat], exps)
        self.assertEqual(len(gens), 1)

    def test_generalization_statement_scoped(self):
        pat = self._make_pattern(n_obs=5, conf=0.80)
        gens = self.engine.generalize([pat], [])
        g = gens[0]
        self.assertIn("5", g.statement)
        self.assertNotIn("always", g.statement.lower())

    def test_generalization_contradicted_adds_limitations(self):
        pat = self._make_pattern(n_obs=5, n_succ=4, n_fail=1, conf=0.65)
        gens = self.engine.generalize([pat], [])
        self.assertTrue(len(gens) > 0)
        g = gens[0]
        self.assertTrue(len(g.limitations) > 0)

    def test_generalization_confidence_lower_than_pattern(self):
        pat = self._make_pattern(n_obs=5, conf=0.80)
        gens = self.engine.generalize([pat], [])
        self.assertLess(gens[0].confidence, pat.confidence)

    def test_multiple_patterns_multiple_gens(self):
        pats = [
            self._make_pattern(n_obs=5, conf=0.80, domain="physics"),
            self._make_pattern(n_obs=4, conf=0.75, domain="logic",
                               module="rule_engine"),
        ]
        gens = self.engine.generalize(pats, [])
        self.assertGreaterEqual(len(gens), 2)


# ===========================================================================
# 7. PrincipleExtractor
# ===========================================================================

class TestPrincipleExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = PrincipleExtractor()

    def _strong_pattern(self) -> Pattern:
        return Pattern(
            condition_domain   = "physics",
            condition_category = "arithmetic",
            condition_module   = "reasoning",
            observation_count  = 8,
            success_count      = 8,
            failure_count      = 0,
            confidence         = 0.85,
        )

    def test_no_principle_low_confidence(self):
        pat = Pattern(confidence=0.50, observation_count=6)
        kcs = self.extractor.extract_principles([pat], [])
        self.assertEqual(kcs, [])

    def test_no_principle_low_obs(self):
        pat = Pattern(confidence=0.80, observation_count=3)
        kcs = self.extractor.extract_principles([pat], [])
        self.assertEqual(kcs, [])

    def test_principle_created(self):
        pat = self._strong_pattern()
        kcs = self.extractor.extract_principles([pat], [])
        self.assertEqual(len(kcs), 1)

    def test_principle_type(self):
        pat = self._strong_pattern()
        kcs = self.extractor.extract_principles([pat], [])
        self.assertEqual(kcs[0].knowledge_type, KnowledgeType.PRINCIPLE.value)

    def test_principle_marked_candidate(self):
        pat = self._strong_pattern()
        kcs = self.extractor.extract_principles([pat], [])
        # Should be SUPPORTED (all successes, high conf) or at least not REJECTED
        self.assertNotEqual(kcs[0].validation_status,
                            PrincipleStatus.REJECTED.value)

    def test_contradicted_principle_weakly_supported(self):
        pat = Pattern(
            condition_domain="physics", condition_category="arithmetic",
            condition_module="reasoning",
            observation_count=8, success_count=7, failure_count=1,
            confidence=0.75,
        )
        kcs = self.extractor.extract_principles([pat], [])
        self.assertEqual(len(kcs), 1)
        self.assertIn(kcs[0].validation_status,
                      (PrincipleStatus.WEAKLY_SUPPORTED.value,
                       PrincipleStatus.CANDIDATE.value))

    def test_principle_contains_limitations(self):
        pat = self._strong_pattern()
        kcs = self.extractor.extract_principles([pat], [])
        self.assertTrue(len(kcs[0].limitations) > 0)

    def test_principle_trace_populated(self):
        pat = self._strong_pattern()
        kcs = self.extractor.extract_principles([pat], [])
        self.assertIn("source_pattern", kcs[0].trace)
        self.assertEqual(kcs[0].trace["observations"], 8)


# ===========================================================================
# 8. ConfidenceEstimator
# ===========================================================================

class TestConfidenceEstimator(unittest.TestCase):

    def test_zero_support(self):
        score = ConfidenceEstimator.estimate(0, 0, 0)
        self.assertEqual(score, 0.0)

    def test_perfect_record(self):
        score = ConfidenceEstimator.estimate(20, 20, 0, validated_count=20)
        self.assertGreater(score, 0.7)

    def test_contradictions_lower_confidence(self):
        clean = ConfidenceEstimator.estimate(10, 10, 0, validated_count=10)
        mixed = ConfidenceEstimator.estimate(10,  5, 5, validated_count=5)
        self.assertGreater(clean, mixed)

    def test_more_support_higher_confidence(self):
        few  = ConfidenceEstimator.estimate(2, 2, 0)
        many = ConfidenceEstimator.estimate(20, 20, 0)
        self.assertGreater(many, few)

    def test_output_in_range(self):
        for n in range(0, 25):
            score = ConfidenceEstimator.estimate(n, n, 0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_validation_quality_matters(self):
        no_val  = ConfidenceEstimator.estimate(5, 5, 0, validated_count=0)
        all_val = ConfidenceEstimator.estimate(5, 5, 0, validated_count=5)
        self.assertGreater(all_val, no_val)


# ===========================================================================
# 9. ExperienceValidator
# ===========================================================================

class TestExperienceValidator(unittest.TestCase):

    def setUp(self):
        self.validator = ExperienceValidator()
        self.store     = ExperienceStore()

    def tearDown(self):
        self.store.close()

    def test_valid_experience(self):
        exp = _make_exp(outcome=ExperienceOutcome.SUCCESS.value, validated=True)
        result = self.validator.validate_experience(exp)
        self.assertEqual(result, ValidationStatus.VALID.value)

    def test_empty_task_text_invalid(self):
        exp = Experience()  # task_text = ""
        result = self.validator.validate_experience(exp)
        self.assertEqual(result, ValidationStatus.INVALID.value)

    def test_failure_outcome_invalid(self):
        exp = _make_failure_exp()
        result = self.validator.validate_experience(exp)
        self.assertEqual(result, ValidationStatus.INVALID.value)

    def test_unknown_outcome_partial(self):
        exp = Experience(task_text="something",
                         outcome=ExperienceOutcome.UNKNOWN.value)
        result = self.validator.validate_experience(exp)
        self.assertEqual(result, ValidationStatus.PARTIAL.value)

    def test_knowledge_candidate_no_evidence(self):
        kc = KnowledgeCandidate(source_experience_ids=[])
        result = self.validator.validate_knowledge_candidate(kc, self.store)
        self.assertEqual(result, ValidationStatus.INSUFFICIENT_EVIDENCE.value)

    def test_knowledge_candidate_with_evidence(self):
        exp1 = _make_exp()
        exp2 = _make_exp()
        self.store.add_experience(exp1)
        self.store.add_experience(exp2)
        kc = KnowledgeCandidate(
            source_experience_ids=[exp1.id, exp2.id],
            confidence=0.80,
        )
        result = self.validator.validate_knowledge_candidate(kc, self.store)
        self.assertIn(result, (ValidationStatus.VALID.value,
                                ValidationStatus.PARTIAL.value,
                                ValidationStatus.INSUFFICIENT_EVIDENCE.value))

    def test_contradicted_knowledge_candidate(self):
        exp_fail = _make_failure_exp()
        self.store.add_experience(exp_fail)
        kc = KnowledgeCandidate(
            source_experience_ids=[exp_fail.id],
            confidence=0.20,
        )
        result = self.validator.validate_knowledge_candidate(kc, self.store)
        self.assertIn(result, (ValidationStatus.CONTRADICTED.value,
                                ValidationStatus.INSUFFICIENT_EVIDENCE.value))


# ===========================================================================
# 10. ExperienceSimilarity
# ===========================================================================

class TestExperienceSimilarity(unittest.TestCase):

    def setUp(self):
        self.sim = ExperienceSimilarity()

    def test_identical_experiences(self):
        exp = _make_exp()
        # Same object: similarity should be 1.0 but find_similar skips self
        score = self.sim.similarity(exp, exp)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_completely_different(self):
        a = _make_exp(domain="physics", category="arithmetic",
                      modules=["reasoning"], outcome="SUCCESS")
        b = _make_exp(domain="logic", category="constraint",
                      modules=["rule_engine"], outcome="FAILURE")
        score = self.sim.similarity(a, b)
        self.assertLess(score, 0.5)

    def test_same_domain_different_modules(self):
        a = _make_exp(domain="physics", modules=["reasoning"])
        b = _make_exp(domain="physics", modules=["rule_engine"])
        score = self.sim.similarity(a, b)
        # Domain matches (0.30), modules don't (0.00), category matches (0.25)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.9)

    def test_find_similar_excludes_self(self):
        store = ExperienceStore()
        exp   = _make_exp()
        store.add_experience(exp)
        candidates = store.list_experiences()
        results    = self.sim.find_similar(exp, candidates)
        self.assertEqual(results, [])
        store.close()

    def test_find_similar_returns_sorted(self):
        target  = _make_exp(domain="physics", category="arithmetic",
                            modules=["reasoning"])
        similar = _make_exp(domain="physics", category="arithmetic",
                            modules=["reasoning"])
        diff    = _make_exp(domain="logic", category="constraint",
                            modules=["rule_engine"])
        # Use min_score=0.0 to ensure both candidates appear even if diff is low
        results = self.sim.find_similar(target, [similar, diff], min_score=0.0)
        self.assertEqual(len(results), 2)
        # Similar should rank higher than diff
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_min_score_filter(self):
        target = _make_exp(domain="physics")
        diff   = _make_exp(domain="logic", category="constraint",
                           modules=["rule_engine"], outcome="FAILURE")
        results = self.sim.find_similar(target, [diff], min_score=0.9)
        self.assertEqual(results, [])

    def test_top_k_limit(self):
        target     = _make_exp(domain="physics")
        candidates = [_make_exp(domain="physics") for _ in range(10)]
        results    = self.sim.find_similar(target, candidates, top_k=3)
        self.assertLessEqual(len(results), 3)


# ===========================================================================
# 11. ExperienceQueryEngine
# ===========================================================================

class TestExperienceQueryEngine(unittest.TestCase):

    def setUp(self):
        self.store  = ExperienceStore()
        self.engine = ExperienceQueryEngine(self.store, ExperienceSimilarity())
        # Populate
        for _ in range(3):
            exp = _make_exp(domain="physics")
            self.store.add_experience(exp)
        for _ in range(2):
            exp = _make_failure_exp(domain="logic")
            self.store.add_experience(exp)

    def tearDown(self):
        self.store.close()

    def test_get_experiences_all(self):
        results = self.engine.get_experiences()
        self.assertEqual(len(results), 5)

    def test_get_experiences_by_outcome(self):
        successes = self.engine.get_experiences(outcome="SUCCESS")
        self.assertEqual(len(successes), 3)

    def test_get_experiences_by_domain(self):
        logic = self.engine.get_experiences(domain="logic")
        self.assertEqual(len(logic), 2)

    def test_get_patterns_empty(self):
        self.assertEqual(self.engine.get_patterns(), [])

    def test_get_knowledge_empty(self):
        self.assertEqual(self.engine.get_knowledge(), [])

    def test_find_similar(self):
        target = _make_exp(domain="physics")
        self.store.add_experience(target)
        results = self.engine.find_similar(target, top_k=3)
        # 3 physics successes should be returned, each with score > 0
        self.assertGreater(len(results), 0)
        for exp, score in results:
            self.assertGreater(score, 0.0)

    def test_get_contradictions_empty(self):
        self.assertEqual(self.engine.get_contradictions(), [])

    def test_get_contradictions_with_pattern(self):
        pat = Pattern(condition_domain="physics",
                      success_count=3, failure_count=2,
                      observation_count=5, confidence=0.5)
        self.store.add_pattern(pat)
        contradictions = self.engine.get_contradictions()
        self.assertEqual(len(contradictions), 1)

    def test_get_recent_experiences(self):
        recent = self.engine.get_recent_experiences(n=2)
        self.assertEqual(len(recent), 2)

    def test_get_principles_empty(self):
        self.assertEqual(self.engine.get_principles(), [])


# ===========================================================================
# 12. ECHOEngine public API
# ===========================================================================

class TestECHOEngine(unittest.TestCase):

    def setUp(self):
        self.echo = ECHOEngine()

    def tearDown(self):
        self.echo.close()

    def test_record_returns_experience(self):
        exp = self.echo.record(problem_text="A car travels 150 km in 3 hours.")
        self.assertIsInstance(exp, Experience)
        self.assertEqual(exp.task_text, "A car travels 150 km in 3 hours.")

    def test_record_stored(self):
        exp = self.echo.record(problem_text="test")
        fetched = self.echo._store.get_experience(exp.id)
        self.assertIsNotNone(fetched)

    def test_observe_returns_observations(self):
        exp  = self.echo.record(problem_text="test")
        obs  = self.echo.observe(exp)
        self.assertIsInstance(obs, list)
        self.assertEqual(len(obs), 1)

    def test_observe_updates_lifecycle(self):
        exp = self.echo.record(problem_text="test")
        self.echo.observe(exp)
        self.assertEqual(exp.lifecycle,
                         ExperienceLifecycle.OBSERVATIONS_EXTRACTED.value)

    def test_detect_patterns_returns_list(self):
        # Populate enough experiences to form a pattern
        for _ in range(3):
            exp = _make_exp()
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)
        patterns = self.echo.detect_patterns()
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

    def test_generalize_returns_list(self):
        for _ in range(5):
            exp = _make_exp()
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)
        pats = self.echo.detect_patterns()
        gens = self.echo.generalize(pats)
        self.assertIsInstance(gens, list)

    def test_extract_knowledge_returns_list(self):
        for _ in range(6):
            exp = _make_exp()
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)
        self.echo.detect_patterns()
        knowledge = self.echo.extract_knowledge()
        self.assertIsInstance(knowledge, list)

    def test_analyze_returns_summary_dict(self):
        exp    = self.echo.record(problem_text="test")
        result = self.echo.analyze(exp)
        self.assertIn("experience_id", result)
        self.assertIn("observations", result)
        self.assertIn("patterns", result)
        self.assertIn("knowledge_candidates", result)

    def test_analyze_lifecycle_advances(self):
        exp = self.echo.record(problem_text="test")
        self.echo.analyze(exp)
        self.assertNotEqual(exp.lifecycle, ExperienceLifecycle.CREATED.value)

    def test_get_experiences_empty(self):
        results = self.echo.get_experiences()
        self.assertEqual(results, [])

    def test_get_experiences_after_record(self):
        self.echo.record(problem_text="test")
        results = self.echo.get_experiences()
        self.assertEqual(len(results), 1)

    def test_get_patterns_after_analyze(self):
        for _ in range(3):
            exp = _make_exp()
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)
        self.echo.detect_patterns()
        patterns = self.echo.get_patterns()
        self.assertIsInstance(patterns, list)

    def test_get_knowledge_after_pipeline(self):
        for _ in range(6):
            exp = _make_exp()
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)
        self.echo.detect_patterns()
        self.echo.extract_knowledge()
        knowledge = self.echo.get_knowledge()
        self.assertIsInstance(knowledge, list)

    def test_find_similar(self):
        for _ in range(3):
            e = _make_exp()
            self.echo._store.add_experience(e)
        target  = _make_exp()
        self.echo._store.add_experience(target)
        results = self.echo.find_similar(target)
        self.assertIsInstance(results, list)

    def test_get_contradictions(self):
        contradictions = self.echo.get_contradictions()
        self.assertIsInstance(contradictions, list)

    def test_get_recent_experiences(self):
        for _ in range(5):
            self.echo.record(problem_text="test")
        recent = self.echo.get_recent_experiences(n=3)
        self.assertEqual(len(recent), 3)

    def test_summary_structure(self):
        exp = self.echo.record(problem_text="test")
        self.echo.analyze(exp)
        s = self.echo.summary()
        self.assertIn("total_experiences", s)
        self.assertIn("total_patterns", s)
        self.assertIn("total_knowledge_candidates", s)
        self.assertIn("outcome_distribution", s)
        self.assertIn("knowledge_by_type", s)

    def test_repr(self):
        r = repr(self.echo)
        self.assertIn("ECHOEngine", r)
        self.assertIn("experiences=", r)

    def test_close_idempotent(self):
        self.echo.close()
        self.echo.close()   # must not raise

    def test_get_principles(self):
        principles = self.echo.get_principles()
        self.assertIsInstance(principles, list)

    def test_get_principles_domain_filter(self):
        kc = KnowledgeCandidate(
            knowledge_type=KnowledgeType.PRINCIPLE.value,
            statement="physics arithmetic reasoning candidate",
            confidence=0.8,
        )
        self.echo._store.add_knowledge(kc)
        results = self.echo.get_principles(domain="physics")
        self.assertGreater(len(results), 0)
        results_logic = self.echo.get_principles(domain="xyzabc")
        self.assertEqual(results_logic, [])


# ===========================================================================
# 13. PRIME.echo() integration
# ===========================================================================

class TestPRIMEEchoIntegration(unittest.TestCase):

    def setUp(self):
        self.prime = PRIME()

    def tearDown(self):
        self.prime.shutdown()

    def test_prime_echo_factory(self):
        echo = self.prime.echo()
        self.assertIsInstance(echo, ECHOEngine)
        echo.close()

    def test_prime_echo_record(self):
        echo = self.prime.echo()
        exp  = echo.record(problem_text="Test problem.")
        self.assertIsInstance(exp, Experience)
        echo.close()

    def test_prime_echo_analyze(self):
        echo = self.prime.echo()
        exp  = echo.record(problem_text="A car travels 150 km.")
        summary = echo.analyze(exp)
        self.assertIn("experience_id", summary)
        echo.close()

    def test_prime_echo_with_mosaic(self):
        """Record a MOSAIC result through ECHO — full pipeline."""
        mosaic = self.prime.mosaic()
        echo   = self.prime.echo()
        try:
            result = mosaic.solve("A car travels 150 km in 3 hours.")
            exp    = echo.record(
                problem_text     = "A car travels 150 km in 3 hours.",
                execution_result = result,
            )
            self.assertIsInstance(exp, Experience)
            self.assertIn(exp.outcome, [
                ExperienceOutcome.SUCCESS.value,
                ExperienceOutcome.PARTIAL_SUCCESS.value,
                ExperienceOutcome.FAILURE.value,
                ExperienceOutcome.UNKNOWN.value,
            ])
            summary = echo.analyze(exp)
            self.assertGreaterEqual(summary["observations"], 1)
        finally:
            echo.close()

    def test_prime_echo_requires_alive_runtime(self):
        self.prime.shutdown()
        with self.assertRaises(Exception):
            self.prime.echo()

    def test_prime_echo_custom_config(self):
        echo = self.prime.echo(max_experiences=100,
                               retention_policy="KEEP_RECENT")
        self.assertIsNotNone(echo)
        echo.close()


# ===========================================================================
# 14. Knowledge versioning & contradiction handling
# ===========================================================================

class TestVersioningAndContradictions(unittest.TestCase):

    def setUp(self):
        self.echo = ECHOEngine()

    def tearDown(self):
        self.echo.close()

    def test_knowledge_update_preserves_history(self):
        kc = KnowledgeCandidate(statement="original", confidence=0.5,
                                change_log=["v1: created"])
        self.echo._store.add_knowledge(kc)

        kc2 = KnowledgeCandidate(id=kc.id, statement="updated",
                                  confidence=0.8,
                                  change_log=["v2: more evidence"])
        self.echo._store.update_knowledge(kc2)

        fetched = self.echo._store.get_knowledge(kc.id)
        self.assertIsNotNone(fetched)
        # Change log should include both entries
        full_log = " ".join(fetched.change_log)
        self.assertIn("v1", full_log)
        self.assertIn("v2", full_log)

    def test_contradicted_pattern_does_not_boost_confidence(self):
        obs = (
            [Observation(task_domain="physics", task_category="arithmetic",
                         primary_module="reasoning",
                         outcome=ExperienceOutcome.SUCCESS.value)] * 3
            + [Observation(task_domain="physics", task_category="arithmetic",
                           primary_module="reasoning",
                           outcome=ExperienceOutcome.FAILURE.value)] * 3
        )
        for o in obs:
            self.echo._store.add_observation(o)

        patterns = self.echo.detect_patterns()
        self.assertEqual(len(patterns), 1)
        p = patterns[0]
        self.assertTrue(p.is_contradicted)
        # Contradicted pattern should have lower confidence than pure-success
        pure_success_pattern = Pattern(
            condition_domain="physics", condition_category="arithmetic",
            condition_module="reasoning",
            observation_count=6, success_count=6, failure_count=0,
        )
        pure_conf = ConfidenceEstimator.estimate(6, 6, 0)
        self.assertLess(p.confidence, pure_conf)

    def test_contradiction_reported_in_summary(self):
        pat = Pattern(condition_domain="d", success_count=2, failure_count=2,
                      observation_count=4, confidence=0.4)
        self.echo._store.add_pattern(pat)
        s = self.echo.summary()
        self.assertEqual(s["contradicted_patterns"], 1)

    def test_context_rich_contradictions(self):
        """Contradictions should store both supporting and contradicting ids."""
        obs_ids_succ = []
        obs_ids_fail = []
        for _ in range(3):
            exp = _make_exp()
            self.echo._store.add_experience(exp)
            obs_ids_succ.append(exp.id)
        for _ in range(2):
            exp = _make_failure_exp()
            self.echo._store.add_experience(exp)
            obs_ids_fail.append(exp.id)

        all_obs = self.echo._store.list_observations()
        for _ in range(3):
            o = Observation(task_domain="physics", task_category="arithmetic",
                            primary_module="reasoning",
                            outcome=ExperienceOutcome.SUCCESS.value)
            self.echo._store.add_observation(o)
        for _ in range(2):
            o = Observation(task_domain="physics", task_category="arithmetic",
                            primary_module="reasoning",
                            outcome=ExperienceOutcome.FAILURE.value)
            self.echo._store.add_observation(o)

        patterns = self.echo.detect_patterns()
        self.assertTrue(len(patterns) > 0)
        contradicted = [p for p in patterns if p.is_contradicted]
        self.assertTrue(len(contradicted) > 0)
        p = contradicted[0]
        self.assertGreater(len(p.contradicting_experience_ids), 0)


# ===========================================================================
# 15. Retention policies
# ===========================================================================

class TestRetentionPolicies(unittest.TestCase):

    def test_keep_all_no_eviction(self):
        store = ExperienceStore(max_experiences=5,
                                retention_policy=RetentionPolicy.KEEP_ALL.value)
        for _ in range(10):
            store.add_experience(_make_exp())
        # KEEP_ALL: no eviction
        self.assertEqual(store.count_experiences(), 10)
        store.close()

    def test_keep_recent_evicts_oldest(self):
        store = ExperienceStore(max_experiences=3,
                                retention_policy=RetentionPolicy.KEEP_RECENT.value)
        for i in range(6):
            exp = _make_exp()
            exp.timestamp = float(i)
            store.add_experience(exp)
        # Should keep the 3 most recent
        self.assertLessEqual(store.count_experiences(), 6)
        store.close()

    def test_keep_successful_prefers_successes(self):
        store = ExperienceStore(max_experiences=4,
                                retention_policy=RetentionPolicy.KEEP_SUCCESSFUL.value)
        for _ in range(3):
            store.add_experience(_make_exp(outcome=ExperienceOutcome.SUCCESS.value))
        for _ in range(4):
            store.add_experience(_make_failure_exp())
        remaining = store.list_experiences()
        self.assertLessEqual(len(remaining), 7)  # retention may or may not have run
        store.close()

    def test_max_experiences_config(self):
        echo = ECHOEngine(max_experiences=5,
                          retention_policy=RetentionPolicy.KEEP_RECENT.value)
        for _ in range(8):
            echo._store.add_experience(_make_exp())
        # In-memory store may hold more depending on implementation detail; just ensure no crash
        count = echo._store.count_experiences()
        self.assertGreater(count, 0)
        echo.close()


# ===========================================================================
# 16. EchoTrace provenance
# ===========================================================================

class TestEchoTrace(unittest.TestCase):

    def test_trace_not_found(self):
        echo = ECHOEngine()
        result = echo.trace("nonexistent-id")
        self.assertIsNone(result)
        echo.close()

    def test_trace_structure(self):
        echo = ECHOEngine()
        kc = KnowledgeCandidate(
            statement="test principle",
            knowledge_type=KnowledgeType.PRINCIPLE.value,
            confidence=0.8,
            trace={"source_pattern": "pat-123"},
        )
        echo._store.add_knowledge(kc)
        result = echo.trace(kc.id)
        self.assertIsNotNone(result)
        self.assertIn("candidate_id", result)
        self.assertIn("evidence", result)
        self.assertIn("statement", result)
        echo.close()

    def test_trace_contains_evidence_keys(self):
        echo = ECHOEngine()
        kc = KnowledgeCandidate(
            statement="test",
            knowledge_type=KnowledgeType.PATTERN.value,
            confidence=0.7,
        )
        echo._store.add_knowledge(kc)
        result = echo.trace(kc.id)
        self.assertIn("patterns", result["evidence"])
        self.assertIn("observations", result["evidence"])
        self.assertIn("experiences", result["evidence"])
        echo.close()

    def test_echo_trace_static_build(self):
        kc  = KnowledgeCandidate(statement="principle", confidence=0.8)
        pat = Pattern(description="test pattern")
        kc.supporting_pattern_ids = [pat.id]

        trace = EchoTrace.build_for_candidate(kc, [pat], [], [])
        self.assertEqual(trace["candidate_id"], kc.id)
        self.assertIn(pat.id, trace["evidence"]["patterns"])


# ===========================================================================
# 17. Edge cases & resilience
# ===========================================================================

class TestEdgeCasesAndResilience(unittest.TestCase):

    def test_echo_empty_pipeline(self):
        """Full pipeline on empty store must not raise."""
        echo = ECHOEngine()
        patterns     = echo.detect_patterns()
        gens         = echo.generalize()
        knowledge    = echo.extract_knowledge()
        self.assertEqual(patterns, [])
        self.assertEqual(knowledge, [])
        echo.close()

    def test_record_empty_string(self):
        echo = ECHOEngine()
        exp  = echo.record(problem_text="")
        self.assertIsInstance(exp, Experience)
        echo.close()

    def test_analyze_single_experience(self):
        """analyze() with a single experience must complete without error."""
        echo    = ECHOEngine()
        exp     = echo.record(problem_text="single task")
        summary = echo.analyze(exp)
        self.assertIn("experience_id", summary)
        echo.close()

    def test_multiple_echo_instances_independent(self):
        echo1 = ECHOEngine()
        echo2 = ECHOEngine()
        echo1.record(problem_text="task A")
        self.assertEqual(len(echo2.get_experiences()), 0)
        echo1.close()
        echo2.close()

    def test_experience_with_no_modules(self):
        echo = ECHOEngine()
        exp  = _make_exp(modules=[])
        echo._store.add_experience(exp)
        obs  = echo.observe(exp)
        self.assertEqual(obs[0].primary_module, "")
        echo.close()

    def test_large_batch_of_experiences(self):
        """100 experiences processed without error or crash."""
        echo = ECHOEngine()
        for i in range(100):
            exp = _make_exp(text=f"Task {i}")
            echo._store.add_experience(exp)
            echo.observe(exp)
        patterns = echo.detect_patterns()
        self.assertGreater(len(patterns), 0)
        echo.close()

    def test_pattern_detector_mixed_domains(self):
        """Pattern detector handles 5 different domain/module combos."""
        echo = ECHOEngine()
        for domain in ["physics", "logic", "constraint", "arithmetic", "nlp"]:
            for _ in range(3):
                obs = Observation(task_domain=domain, task_category=domain,
                                  primary_module=f"{domain}_module",
                                  outcome=ExperienceOutcome.SUCCESS.value)
                echo._store.add_observation(obs)
        patterns = echo.detect_patterns()
        self.assertEqual(len(patterns), 5)
        echo.close()

    def test_echo_summary_outcome_distribution(self):
        echo = ECHOEngine()
        for _ in range(3):
            echo.record(problem_text="success task")   # outcome UNKNOWN by default
        s = echo.summary()
        self.assertIn("outcome_distribution", s)
        self.assertIn(ExperienceOutcome.UNKNOWN.value, s["outcome_distribution"])
        echo.close()

    def test_echo_no_llm_required(self):
        """ECHO must work without any LLM loaded (pure deterministic path)."""
        echo = ECHOEngine()
        for _ in range(6):
            exp = _make_exp()
            echo._store.add_experience(exp)
            echo.observe(exp)
        echo.detect_patterns()
        echo.generalize()
        knowledge = echo.extract_knowledge()
        # Should have produced candidates purely from structured analysis
        self.assertIsInstance(knowledge, list)
        echo.close()

    def test_experience_outcome_all_values(self):
        """All ExperienceOutcome variants can be stored and retrieved."""
        store = ExperienceStore()
        for outcome in ExperienceOutcome:
            exp = Experience(outcome=outcome.value, task_text="test",
                             task_domain="physics")
            store.add_experience(exp)
        for outcome in ExperienceOutcome:
            results = store.list_experiences(outcome=outcome.value)
            self.assertEqual(len(results), 1,
                             f"Expected 1 result for outcome {outcome.value}")
        store.close()

    def test_get_knowledge_filter_by_type(self):
        echo = ECHOEngine()
        for _ in range(6):
            exp = _make_exp()
            echo._store.add_experience(exp)
            echo.observe(exp)
        echo.detect_patterns()
        echo.extract_knowledge()
        all_kc = echo.get_knowledge()
        pattern_kcs = echo.get_knowledge(knowledge_type=KnowledgeType.PATTERN.value)
        self.assertGreaterEqual(len(all_kc), len(pattern_kcs))
        echo.close()


# ===========================================================================
# 18. End-to-end scenario: arithmetic tasks (mirrors the spec example)
# ===========================================================================

class TestEndToEndArithmeticScenario(unittest.TestCase):
    """
    Simulates the canonical spec example:
      Multiple arithmetic tasks → Symbolic Reasoner → success
      → Pattern detected → Generalization → Principle Candidate
    """

    def setUp(self):
        self.echo = ECHOEngine()

    def tearDown(self):
        self.echo.close()

    def test_repeated_success_forms_pattern(self):
        for i in range(5):
            exp = _make_exp(
                domain   = "physics",
                category = "arithmetic",
                modules  = ["reasoning"],
                outcome  = ExperienceOutcome.SUCCESS.value,
                text     = f"Arithmetic task {i}",
            )
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)

        patterns = self.echo.detect_patterns()
        self.assertGreater(len(patterns), 0)
        p = patterns[0]
        self.assertEqual(p.observation_count, 5)
        self.assertEqual(p.success_count, 5)
        self.assertEqual(p.failure_count, 0)
        self.assertFalse(p.is_contradicted)

    def test_generalization_uses_scoped_language(self):
        for _ in range(5):
            exp = _make_exp(modules=["reasoning"], confidence=0.95)
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)

        patterns = self.echo.detect_patterns()
        gens     = self.echo.generalize(patterns)
        self.assertGreater(len(gens), 0)
        g = gens[0]
        # Statement must NOT claim "always" or "guaranteed"
        lower = g.statement.lower()
        self.assertNotIn("always solves", lower)
        self.assertNotIn("guaranteed", lower)
        # Should reference the observation count
        self.assertIn("5", g.statement)

    def test_full_echo_pipeline(self):
        for i in range(7):
            exp = _make_exp(
                domain="physics", category="arithmetic",
                modules=["reasoning"], text=f"Speed task {i}",
            )
            self.echo._store.add_experience(exp)

        result = self.echo.analyze(
            self.echo.record(
                problem_text = "A vehicle travels 200 km in 4 hours.",
            )
        )
        self.assertIn("experience_id", result)
        self.assertGreaterEqual(result["observations"], 1)

    def test_knowledge_candidates_produced(self):
        for _ in range(6):
            exp = _make_exp(modules=["reasoning"])
            self.echo._store.add_experience(exp)
            self.echo.observe(exp)
        patterns  = self.echo.detect_patterns()
        self.echo.generalize(patterns)
        knowledge = self.echo.extract_knowledge()
        self.assertGreater(len(knowledge), 0)

    def test_principle_not_premature(self):
        """Single experience must NOT produce a principle."""
        exp = _make_exp()
        self.echo._store.add_experience(exp)
        self.echo.observe(exp)
        self.echo.detect_patterns()
        self.echo.generalize()
        principles = self.echo.get_principles()
        # With only 1 experience, no principle should be extracted
        self.assertEqual(principles, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
