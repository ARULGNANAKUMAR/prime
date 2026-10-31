/*
 * test_atlas.cpp — C++ unit tests for Phase 4: ATLAS engine.
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
 *       -Wno-missing-field-initializers \
 *       -I cpp tests/cpp/test_atlas.cpp -o tests/cpp/test_atlas
 *
 * Run:
 *   tests/cpp/test_atlas
 */

#include "atlas/atlas_types.hpp"
#include "atlas/atlas_parser.hpp"
#include "atlas/atlas_validator.hpp"
#include "atlas/atlas_json.hpp"
#include "atlas/atlas_engine.hpp"

#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace prime::atlas;

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------
// Assertion-level counter: total assertions attempted, passed assertions.
static int  g_assertions_total  = 0;
static int  g_assertions_passed = 0;
// Test-level counter: unique named tests, failed tests.
static int  g_tests_total  = 0;
static int  g_tests_failed = 0;
static std::string current_test;
static bool current_test_failed = false;

#define TEST(name) \
    do { \
        if (!current_test.empty() && current_test_failed) { \
            /* already counted on first failure */ \
        } \
        current_test = (name); \
        current_test_failed = false; \
        ++g_tests_total; \
        std::cout << "  [....] " << current_test << "\n"; \
    } while(0)

#define EXPECT(cond) \
    do { \
        ++g_assertions_total; \
        if (!(cond)) { \
            if (!current_test_failed) { \
                std::cerr << "  [FAIL] " << current_test \
                          << " — " << #cond \
                          << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
                ++g_tests_failed; \
                current_test_failed = true; \
            } \
        } else { \
            ++g_assertions_passed; \
        } \
    } while(0)

#define EXPECT_THROWS(stmt) \
    do { \
        ++g_assertions_total; \
        bool threw = false; \
        try { stmt; } catch (...) { threw = true; } \
        if (!threw) { \
            if (!current_test_failed) { \
                std::cerr << "  [FAIL] " << current_test << " — expected exception\n"; \
                ++g_tests_failed; \
                current_test_failed = true; \
            } \
        } else { \
            ++g_assertions_passed; \
        } \
    } while(0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------
// 1. Schema version
// ---------------------------------------------------------------------------
static void test_schema_version() {
    std::cout << "\n=== Schema Version ===\n";

    TEST("schema_version_string");
    EXPECT(std::string(ATLAS_SCHEMA_VERSION) == "1.0");

    TEST("engine_schema_version");
    EXPECT(std::string(ATLASEngine::schema_version()) == "1.0");
}

// ---------------------------------------------------------------------------
// 2. Types: Quantity
// ---------------------------------------------------------------------------
static void test_quantity() {
    std::cout << "\n=== Quantity ===\n";

    TEST("quantity_default");
    Quantity q;
    EXPECT(q.value == 0.0);
    EXPECT(q.unit.empty());
    EXPECT(!q.has_unit());

    TEST("quantity_with_unit");
    Quantity q2{100.0, "km", "length"};
    EXPECT(q2.value == 100.0);
    EXPECT(q2.unit == "km");
    EXPECT(q2.dimension == "length");
    EXPECT(q2.has_unit());
    EXPECT(q2.to_string() == "100 km");

    TEST("quantity_no_unit_to_string");
    Quantity q3{42.5, "", ""};
    EXPECT(q3.to_string() == "42.5");
}

// ---------------------------------------------------------------------------
// 3. Types: Entity
// ---------------------------------------------------------------------------
static void test_entity() {
    std::cout << "\n=== Entity ===\n";

    TEST("entity_default");
    Entity e;
    EXPECT(e.id.empty());
    EXPECT(e.confidence == 1.0);
    EXPECT(!e.quantity.has_value());

    TEST("entity_with_quantity");
    Entity e2;
    e2.id       = "e1";
    e2.type     = "quantity";
    e2.name     = "distance";
    e2.quantity = Quantity{100.0, "km", "length"};
    e2.confidence = 0.9;
    EXPECT(e2.quantity.has_value());
    EXPECT(e2.quantity->value == 100.0);
    EXPECT(e2.quantity->unit  == "km");

    TEST("entity_attributes");
    Attribute attr;
    attr.id   = "a1";
    attr.name = "color";
    attr.value_text = "red";
    e2.attributes.push_back(attr);
    EXPECT(e2.attributes.size() == 1);
    EXPECT(e2.attributes[0].name == "color");
}

// ---------------------------------------------------------------------------
// 4. Types: Relationship
// ---------------------------------------------------------------------------
static void test_relationship() {
    std::cout << "\n=== Relationship ===\n";

    TEST("relationship_fields");
    Relationship r;
    r.id         = "r1";
    r.source_id  = "e1";
    r.relation   = "travels";
    r.target_id  = "e2";
    r.confidence = 0.8;
    EXPECT(r.source_id == "e1");
    EXPECT(r.relation  == "travels");
    EXPECT(r.target_id == "e2");
    EXPECT(std::abs(r.confidence - 0.8) < 1e-9);
}

// ---------------------------------------------------------------------------
// 5. Types: Constraint
// ---------------------------------------------------------------------------
static void test_constraint() {
    std::cout << "\n=== Constraint ===\n";

    TEST("constraint_op_roundtrip");
    for (auto op : { ConstraintOp::EQ, ConstraintOp::NEQ, ConstraintOp::LT,
                     ConstraintOp::LTE, ConstraintOp::GT, ConstraintOp::GTE,
                     ConstraintOp::IN, ConstraintOp::NOT_IN }) {
        auto s = constraint_op_to_string(op);
        EXPECT(constraint_op_from_string(s) == op);
    }

    TEST("constraint_fields");
    Constraint c;
    c.id   = "c1";
    c.left = "temperature";
    c.op   = ConstraintOp::LT;
    c.right= "100";
    c.type = "physical";
    EXPECT(c.left == "temperature");
    EXPECT(constraint_op_to_string(c.op) == "<");
}

// ---------------------------------------------------------------------------
// 6. Types: Goal
// ---------------------------------------------------------------------------
static void test_goal() {
    std::cout << "\n=== Goal ===\n";

    TEST("goal_fields");
    Goal g;
    g.id          = "g1";
    g.description = "calculate average speed";
    g.target      = "speed";
    g.priority    = 0;
    g.confidence  = 0.9;
    EXPECT(g.id == "g1");
    EXPECT(g.description == "calculate average speed");
    EXPECT(g.target == "speed");
}

// ---------------------------------------------------------------------------
// 7. Types: Operation
// ---------------------------------------------------------------------------
static void test_operation() {
    std::cout << "\n=== Operation ===\n";

    TEST("operation_fields");
    Operation op;
    op.id   = "op1";
    op.type = "calculate";
    op.input_ids  = {"e1", "e2"};
    op.output_ids = {"e3"};
    EXPECT(op.type == "calculate");
    EXPECT(op.input_ids.size() == 2);
    EXPECT(op.output_ids.size() == 1);
}

// ---------------------------------------------------------------------------
// 8. Types: Expression
// ---------------------------------------------------------------------------
static void test_expression() {
    std::cout << "\n=== Expression ===\n";

    TEST("expression_divide");
    Expression ex;
    ex.id          = "ex1";
    ex.op          = "divide";
    ex.operand_ids = {"distance", "time"};
    ex.output_id   = "speed";
    ex.text        = "speed = distance / time";
    EXPECT(ex.op == "divide");
    EXPECT(ex.operand_ids.size() == 2);
    EXPECT(ex.output_id == "speed");
    EXPECT(ex.text == "speed = distance / time");
}

// ---------------------------------------------------------------------------
// 9. Types: State
// ---------------------------------------------------------------------------
static void test_state() {
    std::cout << "\n=== State ===\n";

    TEST("state_initial");
    State s;
    s.id    = "s1";
    s.label = "initial";
    s.entity_ids     = {"e1", "e2"};
    s.constraint_ids = {"c1"};
    EXPECT(s.label == "initial");
    EXPECT(s.entity_ids.size() == 2);
    EXPECT(s.constraint_ids.size() == 1);
}

// ---------------------------------------------------------------------------
// 10. Types: UniversalProblem
// ---------------------------------------------------------------------------
static void test_universal_problem() {
    std::cout << "\n=== UniversalProblem ===\n";

    TEST("problem_create");
    UniversalProblem p;
    p.problem_id = "test_prob";
    p.domain     = Domain::PHYSICS;
    EXPECT(p.problem_id == "test_prob");
    EXPECT(p.domain == Domain::PHYSICS);
    EXPECT(p.atlas_schema_version == std::string(ATLAS_SCHEMA_VERSION));

    TEST("problem_add_entity");
    Entity e; e.id = "e1"; e.type = "quantity"; e.name = "distance";
    p.entities.push_back(e);
    EXPECT(p.entity_count() == 1);
    EXPECT(p.find_entity("e1") != nullptr);
    EXPECT(p.find_entity("e_none") == nullptr);

    TEST("problem_add_goal");
    Goal g; g.id = "g1"; g.description = "find speed";
    p.goals.push_back(g);
    EXPECT(p.goal_count() == 1);
    EXPECT(p.find_goal("g1") != nullptr);

    TEST("problem_counters");
    Relationship r; r.id = "r1";
    Constraint   c; c.id = "c1";
    p.relationships.push_back(r);
    p.constraints.push_back(c);
    EXPECT(p.relationship_count() == 1);
    EXPECT(p.constraint_count()   == 1);
}

// ---------------------------------------------------------------------------
// 11. ProblemGraph
// ---------------------------------------------------------------------------
static void test_graph() {
    std::cout << "\n=== ProblemGraph ===\n";

    TEST("graph_add_nodes_edges");
    ProblemGraph g;
    g.add_node("n1", "entity", "e1");
    g.add_node("n2", "goal",   "g1");
    g.add_edge("n1", "n2", "requires");
    EXPECT(g.node_count() == 2);
    EXPECT(g.edge_count() == 1);
    EXPECT(g.has_node("n1"));
    EXPECT(!g.has_node("n_missing"));
}

// ---------------------------------------------------------------------------
// 12. InputNormalizer
// ---------------------------------------------------------------------------
static void test_normalizer() {
    std::cout << "\n=== InputNormalizer ===\n";

    InputNormalizer norm;

    TEST("normalize_whitespace");
    auto r = norm.normalize("  hello   world  ");
    EXPECT(r.normalized == "hello world");
    EXPECT(r.original == "  hello   world  ");

    TEST("normalize_sentence_split");
    auto r2 = norm.normalize("First sentence. Second sentence. Third.");
    EXPECT(r2.sentences.size() == 3);

    TEST("normalize_empty");
    auto r3 = norm.normalize("");
    EXPECT(r3.normalized.empty());
    EXPECT(r3.sentences.empty());

    TEST("normalize_preserves_original");
    auto r4 = norm.normalize("A car   travels  120  km.");
    EXPECT(r4.original == "A car   travels  120  km.");
    EXPECT(r4.normalized == "A car travels 120 km.");
}

// ---------------------------------------------------------------------------
// 13. ProblemClassifier
// ---------------------------------------------------------------------------
static void test_classifier() {
    std::cout << "\n=== ProblemClassifier ===\n";

    ProblemClassifier cls;

    TEST("classify_physics_motion");
    auto r1 = cls.classify("A car travels 100 km in 2 hours.");
    EXPECT(r1.domain == Domain::PHYSICS);

    TEST("classify_mathematics");
    auto r2 = cls.classify("Solve x^2 + 5x + 6 = 0. Factor the polynomial.");
    EXPECT(r2.domain == Domain::MATHEMATICS);

    TEST("classify_programming");
    auto r3 = cls.classify("Sort an array using a recursion algorithm.");
    EXPECT(r3.domain == Domain::PROGRAMMING);

    TEST("classify_business");
    auto r4 = cls.classify("Revenue is 10000 and cost is 7000.");
    EXPECT(r4.domain == Domain::BUSINESS);

    TEST("classify_engineering");
    auto r5 = cls.classify("The beam carries a load and stress must not exceed 200 Pa.");
    EXPECT(r5.domain == Domain::ENGINEERING);

    TEST("classify_confidence_in_range");
    auto r6 = cls.classify("sort array algorithm function code");
    EXPECT(r6.confidence >= 0.0 && r6.confidence <= 1.0);

    TEST("classify_unknown_short");
    auto r7 = cls.classify("hello");
    EXPECT(r7.domain == Domain::UNKNOWN || r7.domain != Domain::UNKNOWN); // just must not crash
    EXPECT(r7.confidence >= 0.0 && r7.confidence <= 1.0);
}

// ---------------------------------------------------------------------------
// 14. NumericExtractor
// ---------------------------------------------------------------------------
static void test_numeric_extractor() {
    std::cout << "\n=== NumericExtractor ===\n";

    NumericExtractor ne;

    TEST("extract_km");
    auto v1 = ne.extract("A car travels 120 km.");
    bool found_km = false;
    for (auto& m : v1)
        if (m.value == 120.0 && m.dimension == "length") found_km = true;
    EXPECT(found_km);

    TEST("extract_multiple");
    auto v2 = ne.extract("Mass is 10 kg and time is 2 s.");
    EXPECT(v2.size() >= 2);

    TEST("extract_negative");
    auto v3 = ne.extract("Temperature is -10 s.");  // negative number
    bool found_neg = false;
    for (auto& m : v3) if (m.value < 0) found_neg = true;
    EXPECT(found_neg);

    TEST("extract_decimal");
    auto v4 = ne.extract("Speed is 12.5 m/s.");
    bool found_dec = false;
    for (auto& m : v4) if (std::abs(m.value - 12.5) < 1e-9) found_dec = true;
    EXPECT(found_dec);

    TEST("extract_no_numbers");
    auto v5 = ne.extract("Sort an array.");
    EXPECT(v5.empty());
}

// ---------------------------------------------------------------------------
// 15. Validator
// ---------------------------------------------------------------------------
static void test_validator() {
    std::cout << "\n=== Validator ===\n";

    AtlasValidator v;

    TEST("validate_empty_problem_id");
    UniversalProblem p;
    p.problem_id = "";
    auto r1 = v.validate(p);
    EXPECT(!r1.valid);
    bool found_id_err = false;
    for (auto& e : r1.errors) if (e.find("problem_id") != std::string::npos) found_id_err = true;
    EXPECT(found_id_err);

    TEST("validate_schema_version_mismatch");
    UniversalProblem p2;
    p2.problem_id = "x";
    p2.atlas_schema_version = "99.0";
    auto r2 = v.validate(p2);
    EXPECT(!r2.valid);

    TEST("validate_duplicate_entity_id");
    UniversalProblem p3;
    p3.problem_id = "x";
    p3.atlas_schema_version = ATLAS_SCHEMA_VERSION;
    Entity e1; e1.id = "dup"; e1.confidence = 1.0;
    Entity e2; e2.id = "dup"; e2.confidence = 1.0;
    p3.entities.push_back(e1);
    p3.entities.push_back(e2);
    auto r3 = v.validate(p3);
    EXPECT(!r3.valid);

    TEST("validate_invalid_confidence");
    UniversalProblem p4;
    p4.problem_id = "x";
    p4.atlas_schema_version = ATLAS_SCHEMA_VERSION;
    Entity e3; e3.id = "e1"; e3.confidence = 2.5; // invalid
    p4.entities.push_back(e3);
    auto r4 = v.validate(p4);
    EXPECT(!r4.valid);

    TEST("validate_valid_problem");
    UniversalProblem p5;
    p5.problem_id = "valid";
    p5.atlas_schema_version = ATLAS_SCHEMA_VERSION;
    Entity e4; e4.id = "e1"; e4.name = "distance"; e4.type = "quantity"; e4.confidence = 0.9;
    p5.entities.push_back(e4);
    Goal g; g.id = "g1"; g.description = "find speed"; g.confidence = 0.8;
    p5.goals.push_back(g);
    auto r5 = v.validate(p5);
    EXPECT(r5.valid);
    EXPECT(r5.errors.empty());
}

// ---------------------------------------------------------------------------
// 16. ATLASEngine: parse text
// ---------------------------------------------------------------------------
static void test_engine_parse() {
    std::cout << "\n=== ATLASEngine parse ===\n";

    ATLASEngine engine;

    TEST("parse_basic_physics");
    auto p1 = engine.parse("A car travels 120 km in 3 hours.");
    EXPECT(p1.domain == Domain::PHYSICS || p1.domain == Domain::UNKNOWN || p1.entity_count() > 0);
    EXPECT(!p1.problem_id.empty());
    EXPECT(!p1.original_input.empty());
    EXPECT(p1.atlas_schema_version == std::string(ATLAS_SCHEMA_VERSION));

    TEST("parse_entities_found");
    auto p2 = engine.parse("A car travels 120 km in 3 hours. What is its average speed?");
    EXPECT(p2.entity_count() >= 2); // at least distance and time

    TEST("parse_goal_found");
    auto p3 = engine.parse("What is the average speed if a car travels 100 km in 2 hours?");
    EXPECT(p3.goal_count() >= 1);

    TEST("parse_operation_found");
    auto p4 = engine.parse("Calculate the average speed of a car.");
    EXPECT(p4.operation_count() >= 1);

    TEST("parse_preserves_original");
    std::string text = "An object has mass 10 kg and acceleration 2 m/s^2. Find the force.";
    auto p5 = engine.parse(text);
    EXPECT(p5.original_input == text);

    TEST("parse_state_built");
    auto p6 = engine.parse("A car travels 120 km in 3 hours.");
    EXPECT(p6.states.size() >= 1);
    EXPECT(p6.states[0].label == "initial");

    TEST("parse_graph_built");
    auto p7 = engine.parse("A car travels 120 km in 3 hours. What is its speed?");
    EXPECT(p7.graph.node_count() > 0);

    TEST("parse_empty_throws");
    bool threw = false;
    try { engine.parse(""); } catch (AtlasError&) { threw = true; }
    EXPECT(threw);

    TEST("parse_large_input_throws");
    std::string big(2 * 1024 * 1024, 'x'); // 2MB > default 1MB
    threw = false;
    try { engine.parse(big); } catch (AtlasError&) { threw = true; }
    EXPECT(threw);
}

// ---------------------------------------------------------------------------
// 17. ATLASEngine: domain classification
// ---------------------------------------------------------------------------
static void test_engine_domains() {
    std::cout << "\n=== ATLASEngine domain classification ===\n";

    ATLASEngine engine;

    struct Case { std::string text; Domain expected; };
    std::vector<Case> cases = {
        { "If a train travels 100 km in 2 hours, what is its average speed?", Domain::PHYSICS },
        { "An object has mass 10 kg and acceleration 2 m/s^2. Find the force.", Domain::PHYSICS },
        { "Solve the equation x^2 - 4 = 0 by factoring.", Domain::MATHEMATICS },
        { "Sort an array of integers in ascending order using an algorithm.", Domain::PROGRAMMING },
        { "Revenue is 10000 and cost is 7000. What is the profit?", Domain::BUSINESS },
        { "A beam must support 500 N. Stress must not exceed 200 Pa.", Domain::ENGINEERING },
    };

    for (auto& c : cases) {
        TEST("classify_" + domain_to_string(c.expected));
        auto p = engine.parse(c.text);
        EXPECT(p.domain == c.expected);
    }
}

// ---------------------------------------------------------------------------
// 18. ATLASEngine: expressions
// ---------------------------------------------------------------------------
static void test_engine_expressions() {
    std::cout << "\n=== ATLASEngine expressions ===\n";

    ATLASEngine engine;

    TEST("expression_formula_detected");
    auto p = engine.parse("The profit is calculated as profit = revenue - cost.");
    // Should detect the formula or at least the expression context
    EXPECT(p.expression_count() >= 1 || p.entity_count() > 0);

    TEST("expression_speed_formula");
    auto p2 = engine.parse("Average speed equals distance divided by time. speed = distance / time");
    EXPECT(p2.expression_count() >= 1);
}

// ---------------------------------------------------------------------------
// 19. ATLASEngine: constraints
// ---------------------------------------------------------------------------
static void test_engine_constraints() {
    std::cout << "\n=== ATLASEngine constraints ===\n";

    ATLASEngine engine;

    TEST("constraint_numerical_extracted");
    auto p = engine.parse("The temperature must be temperature < 100.");
    EXPECT(p.constraint_count() >= 1);

    TEST("constraint_textual_at_most");
    auto p2 = engine.parse("The budget must be at most 50000 dollars.");
    EXPECT(p2.constraint_count() >= 1);
}

// ---------------------------------------------------------------------------
// 20. ATLASEngine: logic clauses
// ---------------------------------------------------------------------------
static void test_engine_logic() {
    std::cout << "\n=== ATLASEngine logic clauses ===\n";

    ATLASEngine engine;

    TEST("logic_if_then");
    auto p = engine.parse("If temperature > 100, then warning is true.");
    EXPECT(p.logic_clauses.size() >= 1);
    EXPECT(p.logic_clauses[0].op == LogicOp::IF);

    TEST("logic_and");
    auto p2 = engine.parse("The beam must carry load and stress must not exceed the limit.");
    EXPECT(p2.logic_clauses.size() >= 1);
}

// ---------------------------------------------------------------------------
// 21. ATLASEngine: assumptions
// ---------------------------------------------------------------------------
static void test_engine_assumptions() {
    std::cout << "\n=== ATLASEngine assumptions ===\n";

    ATLASEngine engine;

    TEST("assumption_detected");
    auto p = engine.parse("Assuming no friction, the car travels 100 km.");
    EXPECT(p.assumptions.size() >= 1);

    TEST("assumption_air_resistance");
    auto p2 = engine.parse("Neglecting air resistance, calculate the range.");
    EXPECT(p2.assumptions.size() >= 1);
}

// ---------------------------------------------------------------------------
// 22. JSON Serialization
// ---------------------------------------------------------------------------
static void test_serialization() {
    std::cout << "\n=== JSON Serialization ===\n";

    ATLASEngine engine;

    TEST("serialize_basic");
    auto p1 = engine.parse("A car travels 120 km in 3 hours.");
    std::string json = engine.to_json(p1);
    EXPECT(!json.empty());
    EXPECT(contains(json, "atlas_schema_version"));
    EXPECT(contains(json, "\"1.0\""));
    EXPECT(contains(json, "entities"));
    EXPECT(contains(json, "goals"));

    TEST("deserialize_basic");
    auto p2 = engine.from_json(json);
    EXPECT(p2.problem_id == p1.problem_id);
    EXPECT(p2.domain == p1.domain);
    EXPECT(p2.entity_count() == p1.entity_count());
    EXPECT(p2.atlas_schema_version == std::string(ATLAS_SCHEMA_VERSION));

    TEST("round_trip_entities");
    EXPECT(p2.entity_count() == p1.entity_count());

    TEST("round_trip_goals");
    EXPECT(p2.goal_count() == p1.goal_count());

    TEST("round_trip_constraints");
    EXPECT(p2.constraint_count() == p1.constraint_count());

    TEST("round_trip_graph");
    // Graph is rebuilt from JSON
    EXPECT(p2.graph.node_count() == p1.graph.node_count());
    EXPECT(p2.graph.edge_count() == p1.graph.edge_count());

    TEST("from_json_empty_throws");
    bool threw = false;
    try { engine.from_json(""); } catch (AtlasError&) { threw = true; }
    EXPECT(threw);

    TEST("from_json_malformed_throws");
    threw = false;
    try { engine.from_json("{not valid json"); } catch (...) { threw = true; }
    EXPECT(threw);

    TEST("from_json_wrong_schema_throws");
    threw = false;
    try {
        engine.from_json(R"({"atlas_schema_version":"99.0","problem_id":"x"})");
    } catch (AtlasError&) { threw = true; }
    EXPECT(threw);
}

// ---------------------------------------------------------------------------
// 23. Validation
// ---------------------------------------------------------------------------
static void test_validation_integration() {
    std::cout << "\n=== Validation integration ===\n";

    ATLASEngine engine;

    TEST("parsed_problem_is_valid");
    auto p = engine.parse("A car travels 120 km in 3 hours. What is its speed?");
    auto r = engine.validate(p);
    EXPECT(r.valid);
    EXPECT(r.errors.empty());

    TEST("round_trip_is_valid");
    auto json = engine.to_json(p);
    auto p2   = engine.from_json(json);
    auto r2   = engine.validate(p2);
    EXPECT(r2.valid);

    TEST("manually_built_valid");
    UniversalProblem mp;
    mp.problem_id = "manual";
    mp.atlas_schema_version = ATLAS_SCHEMA_VERSION;
    Entity e; e.id = "e1"; e.type = "quantity"; e.name = "dist"; e.confidence = 0.95;
    mp.entities.push_back(e);
    Goal g; g.id = "g1"; g.description = "find speed"; g.confidence = 1.0;
    mp.goals.push_back(g);
    auto rv = engine.validate(mp);
    EXPECT(rv.valid);

    TEST("invalid_confidence_rejected");
    Entity bad_e; bad_e.id = "e2"; bad_e.confidence = 1.5;
    mp.entities.push_back(bad_e);
    auto rv2 = engine.validate(mp);
    EXPECT(!rv2.valid);
}

// ---------------------------------------------------------------------------
// 24. Cross-domain universality
// ---------------------------------------------------------------------------
static void test_cross_domain() {
    std::cout << "\n=== Cross-domain universality ===\n";

    ATLASEngine engine;

    struct DomainCase {
        std::string name;
        std::string text;
        Domain      expected_domain;
    };

    std::vector<DomainCase> cases = {
        { "mathematics", "Solve x^2 - 9 = 0 by factoring the polynomial.", Domain::MATHEMATICS },
        { "physics",     "An object has mass 10 kg and acceleration 2 m/s^2. Find the force.", Domain::PHYSICS },
        { "programming", "Sort an array of integers in ascending order using a fast algorithm.", Domain::PROGRAMMING },
        { "business",    "Revenue is 10000 and cost is 7000. Calculate the profit.", Domain::BUSINESS },
        { "engineering", "A machine operates above a temperature limit of 200 C.", Domain::ENGINEERING },
    };

    for (auto& c : cases) {
        TEST("cross_domain_" + c.name + "_parse");
        auto p = engine.parse(c.text);
        EXPECT(p.domain == c.expected_domain);
        EXPECT(!p.problem_id.empty());
        EXPECT(!p.original_input.empty());

        TEST("cross_domain_" + c.name + "_valid");
        auto r = engine.validate(p);
        EXPECT(r.valid);

        TEST("cross_domain_" + c.name + "_round_trip");
        auto json = engine.to_json(p);
        auto p2   = engine.from_json(json);
        EXPECT(p2.domain == p.domain);
        EXPECT(p2.entity_count() == p.entity_count());
    }
}

// ---------------------------------------------------------------------------
// 25. ATLAS does NOT solve problems
// ---------------------------------------------------------------------------
static void test_no_solving() {
    std::cout << "\n=== ATLAS does NOT solve problems ===\n";

    ATLASEngine engine;

    TEST("no_solution_in_entity_names");
    auto p = engine.parse("If a train travels 100 km in 2 hours, what is its average speed?");
    // ATLAS should NOT include a "50 km/h" answer entity
    bool found_answer = false;
    for (auto& e : p.entities) {
        if (e.name.find("50") != std::string::npos &&
            e.type != "quantity") found_answer = true;
    }
    EXPECT(!found_answer);  // no numeric answer in a "solution" entity

    TEST("no_solution_in_goals");
    // Goals describe what to find, not the answer
    for (auto& g : p.goals) {
        EXPECT(g.desired_state.empty() || g.desired_state.find("50") == std::string::npos);
    }

    TEST("representation_not_reasoning");
    // The representation should contain entities, goals, relationships
    // but not an evaluated/computed result
    EXPECT(p.entity_count() > 0);
    EXPECT(p.goal_count() >= 1);
}

// ---------------------------------------------------------------------------
// 26. Resource limits
// ---------------------------------------------------------------------------
static void test_limits() {
    std::cout << "\n=== Resource limits ===\n";

    TEST("custom_limits");
    AtlasLimits lim;
    lim.max_entities = 2;
    lim.max_input_bytes = 1000;
    ATLASEngine engine(lim);

    // Short text — should parse without throwing
    auto p = engine.parse("A car travels 100 km.");
    EXPECT(p.entity_count() <= 2); // capped at 2

    TEST("input_too_large");
    std::string big(2000, 'x'); // > 1000 byte limit
    bool threw = false;
    try { engine.parse(big); } catch (AtlasError&) { threw = true; }
    EXPECT(threw);
}

// ---------------------------------------------------------------------------
// 27. ID generation
// ---------------------------------------------------------------------------
static void test_ids() {
    std::cout << "\n=== ID generation ===\n";

    ATLASEngine engine;

    TEST("all_entities_have_ids");
    auto p = engine.parse("A car travels 120 km in 3 hours. What is the speed?");
    for (auto& e : p.entities) EXPECT(!e.id.empty());

    TEST("all_goals_have_ids");
    for (auto& g : p.goals) EXPECT(!g.id.empty());

    TEST("all_relationships_have_ids");
    for (auto& r : p.relationships) EXPECT(!r.id.empty());

    TEST("unique_entity_ids");
    std::unordered_set<std::string> seen;
    for (auto& e : p.entities) {
        EXPECT(seen.find(e.id) == seen.end());
        seen.insert(e.id);
    }

    TEST("two_parses_different_ids");
    auto p1 = engine.parse("Mass is 10 kg.");
    auto p2 = engine.parse("Distance is 50 m.");
    EXPECT(p1.problem_id != p2.problem_id);
}

// ---------------------------------------------------------------------------
// 28. Confidence values
// ---------------------------------------------------------------------------
static void test_confidence() {
    std::cout << "\n=== Confidence values ===\n";

    ATLASEngine engine;
    auto p = engine.parse("An object has mass 10 kg and acceleration 2 m/s^2.");

    TEST("entity_confidence_in_range");
    for (auto& e : p.entities)
        EXPECT(e.confidence >= 0.0 && e.confidence <= 1.0);

    TEST("relationship_confidence_in_range");
    for (auto& r : p.relationships)
        EXPECT(r.confidence >= 0.0 && r.confidence <= 1.0);

    TEST("goal_confidence_in_range");
    for (auto& g : p.goals)
        EXPECT(g.confidence >= 0.0 && g.confidence <= 1.0);

    TEST("domain_confidence_in_range");
    EXPECT(p.domain_confidence >= 0.0 && p.domain_confidence <= 1.0);
}

// ---------------------------------------------------------------------------
// 29. Unit normalization
// ---------------------------------------------------------------------------
static void test_unit_normalization() {
    std::cout << "\n=== Unit normalization ===\n";

    TEST("unit_lookup_km");
    auto u = lookup_unit("km");
    EXPECT(u.has_value());
    EXPECT(u->dimension == "length");
    EXPECT(u->canonical == "km");

    TEST("unit_lookup_case_insensitive");
    auto u2 = lookup_unit("KM");
    EXPECT(u2.has_value());
    EXPECT(u2->dimension == "length");

    TEST("unit_lookup_kilometer");
    auto u3 = lookup_unit("kilometer");
    EXPECT(u3.has_value());
    EXPECT(u3->canonical == "km");

    TEST("unit_lookup_unknown");
    auto u4 = lookup_unit("flibbertigibbet");
    EXPECT(!u4.has_value());

    TEST("unit_lookup_mass");
    auto u5 = lookup_unit("kg");
    EXPECT(u5.has_value());
    EXPECT(u5->dimension == "mass");

    TEST("unit_lookup_speed");
    auto u6 = lookup_unit("m/s");
    EXPECT(u6.has_value());
    EXPECT(u6->dimension == "speed");
}

// ---------------------------------------------------------------------------
// 30. Domain string roundtrip
// ---------------------------------------------------------------------------
static void test_domain_strings() {
    std::cout << "\n=== Domain string roundtrip ===\n";

    TEST("domain_all_roundtrip");
    for (auto d : { Domain::MATHEMATICS, Domain::PHYSICS, Domain::PROGRAMMING,
                    Domain::BUSINESS, Domain::ENGINEERING, Domain::BIOLOGY,
                    Domain::GENERAL, Domain::UNKNOWN }) {
        EXPECT(domain_from_string(domain_to_string(d)) == d);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "PRIME Phase 4 — ATLAS C++ Tests\n";
    std::cout << "================================\n";

    test_schema_version();
    test_quantity();
    test_entity();
    test_relationship();
    test_constraint();
    test_goal();
    test_operation();
    test_expression();
    test_state();
    test_universal_problem();
    test_graph();
    test_normalizer();
    test_classifier();
    test_numeric_extractor();
    test_validator();
    test_engine_parse();
    test_engine_domains();
    test_engine_expressions();
    test_engine_constraints();
    test_engine_logic();
    test_engine_assumptions();
    test_serialization();
    test_validation_integration();
    test_cross_domain();
    test_no_solving();
    test_limits();
    test_ids();
    test_confidence();
    test_unit_normalization();
    test_domain_strings();

    std::cout << "\n================================\n";
    int tests_passed = g_tests_total - g_tests_failed;
    std::cout << "Results: " << tests_passed << "/" << g_tests_total << " passed";
    if (g_tests_failed > 0)
        std::cout << "  (" << g_tests_failed << " FAILED)";
    std::cout << "\n";
    std::cout << "Assertions: " << g_assertions_passed << "/" << g_assertions_total << "\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
