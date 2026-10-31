// tests/cpp/test_reasoning.cpp — C++ unit tests for PRIME Phase 5 Reasoning Engine.
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
//       -I cpp tests/cpp/test_reasoning.cpp -o tests/cpp/test_reasoning
// Run:
//   tests/cpp/test_reasoning

#include "reasoning/reasoning_types.hpp"
#include "reasoning/expression_evaluator.hpp"
#include "reasoning/rule_engine.hpp"
#include "reasoning/constraint_engine.hpp"
#include "reasoning/symbolic_engine.hpp"
#include "reasoning/reasoning_engine.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int  g_tests  = 0;
static int  g_passed = 0;
static int  g_failed = 0;
static std::string g_current_suite;

#define SUITE(name) do { g_current_suite = (name); \
    std::cout << "\n[" << g_current_suite << "]\n"; } while(0)

#define CHECK(expr) do { \
    ++g_tests; \
    if (!(expr)) { \
        ++g_failed; \
        std::cerr << "  FAIL  " << #expr << "  (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    } else { \
        ++g_passed; \
        std::cout << "  pass  " << #expr << "\n"; \
    } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) < (eps))
#define CHECK_THROWS(expr) do { \
    ++g_tests; \
    bool threw = false; \
    try { (void)(expr); } catch (...) { threw = true; } \
    if (!threw) { \
        ++g_failed; \
        std::cerr << "  FAIL  expected exception: " << #expr << "\n"; \
    } else { \
        ++g_passed; \
        std::cout << "  pass  throws: " << #expr << "\n"; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
using namespace prime::reasoning;

static Fact make_numeric_fact(const std::string& name, double value,
                               const std::string& unit = "") {
    Fact f;
    f.id         = name;
    f.name       = name;
    f.value      = FactValue{value};
    f.unit       = unit;
    f.confidence = 1.0;
    f.source     = "test";
    f.bound      = true;
    return f;
}

static Fact make_bool_fact(const std::string& name, bool value) {
    Fact f;
    f.id    = f.name = name;
    f.value = FactValue{value};
    f.bound = f.confidence = true;
    f.source = "test";
    return f;
}

static Fact make_string_fact(const std::string& name, const std::string& value) {
    Fact f;
    f.id    = f.name = name;
    f.value = FactValue{value};
    f.bound = true;
    f.confidence = 1.0;
    f.source = "test";
    return f;
}

static Rule make_simple_rule(const std::string& id,
                              const std::string& left, const std::string& op_s,
                              const std::string& right,
                              const std::string& conc_name, const std::string& conc_val,
                              int priority = 0) {
    Rule r;
    r.id       = id;
    r.name     = id;
    r.priority = priority;
    r.confidence = 1.0;

    Condition c;
    c.left  = left;
    c.right = right;
    if      (op_s == "=")  c.op = CondOp::EQ;
    else if (op_s == "!=") c.op = CondOp::NEQ;
    else if (op_s == "<")  c.op = CondOp::LT;
    else if (op_s == "<=") c.op = CondOp::LTE;
    else if (op_s == ">")  c.op = CondOp::GT;
    else if (op_s == ">=") c.op = CondOp::GTE;
    r.conditions.push_back(c);

    Conclusion conc;
    conc.name       = conc_name;
    conc.value_expr = conc_val;
    // Try numeric
    try {
        size_t idx;
        double d = std::stod(conc_val, &idx);
        if (idx == conc_val.size()) { conc.is_numeric = true; conc.numeric_value = d; }
    } catch (...) {}
    conc.confidence = 1.0;
    r.conclusions.push_back(conc);
    return r;
}

// ===========================================================================
// 1. ReasoningTypes
// ===========================================================================
void test_types() {
    SUITE("ReasoningTypes");

    // Status strings
    CHECK_EQ(status_to_string(ReasoningStatus::SOLVED),                  "SOLVED");
    CHECK_EQ(status_to_string(ReasoningStatus::UNRESOLVED),              "UNRESOLVED");
    CHECK_EQ(status_to_string(ReasoningStatus::INVALID_INPUT),           "INVALID_INPUT");
    CHECK_EQ(status_to_string(ReasoningStatus::CONSTRAINT_VIOLATION),    "CONSTRAINT_VIOLATION");
    CHECK_EQ(status_to_string(ReasoningStatus::REASONING_LIMIT_REACHED), "REASONING_LIMIT_REACHED");
    CHECK_EQ(status_to_string(ReasoningStatus::CIRCULAR_REASONING),      "CIRCULAR_REASONING");

    // Mode strings
    CHECK_EQ(mode_to_string(ReasoningMode::RULE),         "RULE");
    CHECK_EQ(mode_to_string(ReasoningMode::SYMBOLIC),     "SYMBOLIC");
    CHECK_EQ(mode_to_string(ReasoningMode::HYBRID),       "HYBRID");
    CHECK_EQ(mode_to_string(ReasoningMode::LLM_ASSISTED), "LLM_ASSISTED");

    // FactValue conversions
    FactValue fv_num{42.0};
    CHECK(fact_value_is_numeric(fv_num));
    CHECK_NEAR(fact_value_to_double(fv_num), 42.0, 1e-9);
    CHECK_EQ(fact_value_to_string(fv_num), "42");

    FactValue fv_bool{true};
    CHECK(!fact_value_is_numeric(fv_bool));
    CHECK_NEAR(fact_value_to_double(fv_bool), 1.0, 1e-9);
    CHECK_EQ(fact_value_to_string(fv_bool), "true");

    FactValue fv_str{std::string("hello")};
    CHECK(!fact_value_is_numeric(fv_str));
    CHECK_EQ(fact_value_to_string(fv_str), "hello");

    // StepType strings
    CHECK_EQ(step_type_to_string(StepType::FACT_INIT),        "FACT_INIT");
    CHECK_EQ(step_type_to_string(StepType::RULE_FIRE),        "RULE_FIRE");
    CHECK_EQ(step_type_to_string(StepType::EXPRESSION_EVAL),  "EXPRESSION_EVAL");
    CHECK_EQ(step_type_to_string(StepType::CONSTRAINT_CHECK), "CONSTRAINT_CHECK");
    CHECK_EQ(step_type_to_string(StepType::GOAL_CHECK),       "GOAL_CHECK");

    // ReasoningTrace
    ReasoningTrace trace;
    CHECK_EQ(trace.total_steps, 0);
    ReasoningStep s;
    s.type = StepType::FACT_INIT; s.success = true;
    trace.add_step(s);
    CHECK_EQ(trace.total_steps, 1);
    CHECK_EQ(trace.steps[0].step_id, 1);
}

// ===========================================================================
// 2. ExpressionEvaluator
// ===========================================================================
void test_expression_evaluator() {
    SUITE("ExpressionEvaluator");

    ExpressionEngine ee;

    // Constants
    CHECK_NEAR(ee.eval("42", {}), 42.0, 1e-9);
    CHECK_NEAR(ee.eval("3.14", {}), 3.14, 1e-4);

    // Basic arithmetic
    CHECK_NEAR(ee.eval("2 + 3", {}), 5.0, 1e-9);
    CHECK_NEAR(ee.eval("10 - 4", {}), 6.0, 1e-9);
    CHECK_NEAR(ee.eval("6 * 7", {}), 42.0, 1e-9);
    CHECK_NEAR(ee.eval("150 / 3", {}), 50.0, 1e-9);
    CHECK_NEAR(ee.eval("10 % 3", {}), 1.0, 1e-9);
    CHECK_NEAR(ee.eval("2 ^ 8", {}), 256.0, 1e-9);

    // Unary minus
    CHECK_NEAR(ee.eval("-5", {}), -5.0, 1e-9);
    CHECK_NEAR(ee.eval("-(3 + 2)", {}), -5.0, 1e-9);

    // Parentheses and precedence
    CHECK_NEAR(ee.eval("(2 + 3) * 4", {}), 20.0, 1e-9);
    CHECK_NEAR(ee.eval("2 + 3 * 4", {}), 14.0, 1e-9);   // precedence
    CHECK_NEAR(ee.eval("2 ^ 3 ^ 2", {}), 512.0, 1e-9);  // right-assoc

    // Variables
    ExprEvaluator::Bindings b = {{"x", 10.0}, {"y", 5.0}};
    CHECK_NEAR(ee.eval("x + y", b), 15.0, 1e-9);
    CHECK_NEAR(ee.eval("x * y", b), 50.0, 1e-9);
    CHECK_NEAR(ee.eval("x / y", b), 2.0, 1e-9);
    CHECK_NEAR(ee.eval("(x + y) * 2", b), 30.0, 1e-9);

    // Functions
    CHECK_NEAR(ee.eval("abs(-7)", {}), 7.0, 1e-9);
    CHECK_NEAR(ee.eval("min(3, 1, 2)", {}), 1.0, 1e-9);
    CHECK_NEAR(ee.eval("max(3, 1, 2)", {}), 3.0, 1e-9);
    CHECK_NEAR(ee.eval("sqrt(16)", {}), 4.0, 1e-9);

    // Variable collection
    auto tree = ee.compile("a + b * c");
    auto vars = ee.variables(tree);
    CHECK(vars.count("a") > 0);
    CHECK(vars.count("b") > 0);
    CHECK(vars.count("c") > 0);

    // Error cases
    CHECK_THROWS(ee.eval("10 / 0", {}));
    CHECK_THROWS(ee.eval("z + 1", {}));         // unbound variable
    CHECK_THROWS(ee.eval("sqrt(-1)", {}));       // sqrt of negative
    CHECK_THROWS(ee.eval("1 ++ 2", {}));         // syntax error
}

// ===========================================================================
// 3. RuleEngine — rule creation and condition evaluation
// ===========================================================================
void test_rule_engine() {
    SUITE("RuleEngine");

    RuleEngine engine;

    // Rule addition
    auto r1 = make_simple_rule("r1", "temperature", ">", "100", "warning", "true");
    CHECK(engine.add_rule(r1));
    CHECK_EQ(engine.rule_count(), static_cast<size_t>(1));

    // Duplicate rule ID rejected
    CHECK(!engine.add_rule(r1));
    CHECK_EQ(engine.rule_count(), static_cast<size_t>(1));

    // Forward chain: condition met
    std::unordered_map<std::string, Fact> facts;
    facts["temperature"] = make_numeric_fact("temperature", 120.0);

    ReasoningConfig cfg;
    ReasoningTrace trace;
    bool fired = engine.forward_chain(facts, trace, cfg);
    CHECK(fired);
    CHECK(facts.count("warning") > 0);
    CHECK_EQ(fact_value_to_string(facts["warning"].value), "true");
    CHECK(trace.rules_fired > 0);

    // Forward chain: condition NOT met
    RuleEngine engine2;
    engine2.add_rule(make_simple_rule("r1", "temperature", ">", "100", "warning", "true"));
    std::unordered_map<std::string, Fact> facts2;
    facts2["temperature"] = make_numeric_fact("temperature", 80.0);
    ReasoningTrace trace2;
    bool fired2 = engine2.forward_chain(facts2, trace2, cfg);
    CHECK(!fired2);
    CHECK(facts2.count("warning") == 0);
}

void test_rule_conditions() {
    SUITE("RuleConditions");

    ReasoningConfig cfg;

    auto check_fires = [&](const std::string& op, double fact_val, double cond_val) -> bool {
        RuleEngine eng;
        eng.add_rule(make_simple_rule("r", "x", op, std::to_string(cond_val), "fired", "true"));
        std::unordered_map<std::string, Fact> facts;
        facts["x"] = make_numeric_fact("x", fact_val);
        ReasoningTrace trace;
        return eng.forward_chain(facts, trace, cfg);
    };

    CHECK(check_fires("=",  5.0, 5.0));
    CHECK(!check_fires("=", 5.0, 6.0));
    CHECK(check_fires("!=", 5.0, 6.0));
    CHECK(!check_fires("!=",5.0, 5.0));
    CHECK(check_fires("<",  4.0, 5.0));
    CHECK(!check_fires("<", 5.0, 5.0));
    CHECK(check_fires("<=", 5.0, 5.0));
    CHECK(check_fires("<=", 4.0, 5.0));
    CHECK(!check_fires("<=",6.0, 5.0));
    CHECK(check_fires(">",  6.0, 5.0));
    CHECK(!check_fires(">", 5.0, 5.0));
    CHECK(check_fires(">=", 5.0, 5.0));
    CHECK(check_fires(">=", 6.0, 5.0));
    CHECK(!check_fires(">=",4.0, 5.0));
}

void test_rule_priorities() {
    SUITE("RulePriorities");

    RuleEngine engine;
    // High priority rule
    auto r_high = make_simple_rule("r_high", "x", ">", "0", "result", "high", 10);
    // Low priority rule
    auto r_low  = make_simple_rule("r_low",  "x", ">", "0", "result2", "low", 1);
    engine.add_rule(r_low);
    engine.add_rule(r_high);

    std::unordered_map<std::string, Fact> facts;
    facts["x"] = make_numeric_fact("x", 5.0);
    ReasoningConfig cfg;
    ReasoningTrace trace;
    engine.forward_chain(facts, trace, cfg);

    // Both should fire (different conclusion names)
    CHECK(facts.count("result")  > 0);
    CHECK(facts.count("result2") > 0);

    // High-priority rule should appear first in the RULE_FIRE steps
    std::string first_rule_id;
    for (auto& step : trace.steps) {
        if (step.type == StepType::RULE_FIRE) {
            first_rule_id = step.rule_id;
            break;
        }
    }
    CHECK_EQ(first_rule_id, "r_high");
}

void test_rule_chaining() {
    SUITE("RuleChaining");

    RuleEngine engine;
    // A=1 → B=1 → C=1
    engine.add_rule(make_simple_rule("r1", "A", "=", "1", "B", "1"));
    engine.add_rule(make_simple_rule("r2", "B", "=", "1", "C", "1"));

    std::unordered_map<std::string, Fact> facts;
    facts["A"] = make_numeric_fact("A", 1.0);

    ReasoningConfig cfg;
    ReasoningTrace trace;
    engine.forward_chain(facts, trace, cfg);

    CHECK(facts.count("B") > 0);
    CHECK(facts.count("C") > 0);
}

void test_circular_rule_detection() {
    SUITE("CircularRuleDetection");

    RuleEngine engine;
    engine.add_rule(make_simple_rule("r1", "A", "=", "1", "B", "1"));
    engine.add_rule(make_simple_rule("r2", "B", "=", "1", "C", "1"));
    engine.add_rule(make_simple_rule("r3", "C", "=", "1", "A", "1"));

    std::unordered_map<std::string, Fact> facts;
    facts["A"] = make_numeric_fact("A", 1.0);

    ReasoningConfig cfg;
    cfg.max_depth        = 50;
    cfg.max_rule_firings = 200;
    cfg.detect_circular  = true;

    ReasoningTrace trace;
    engine.forward_chain(facts, trace, cfg);
    // Must terminate — total steps bounded
    CHECK(trace.total_steps < 500);
    // B and C should be derived
    CHECK(facts.count("B") > 0);
    CHECK(facts.count("C") > 0);
}

void test_compound_conditions() {
    SUITE("CompoundConditions");

    ReasoningConfig cfg;

    // AND condition
    {
        Rule r;
        r.id = "r_and"; r.confidence = 1.0;
        Condition and_cond;
        and_cond.op = CondOp::AND;
        Condition c1; c1.left = "a"; c1.op = CondOp::GT; c1.right = "0";
        Condition c2; c2.left = "b"; c2.op = CondOp::GT; c2.right = "0";
        and_cond.sub_conditions = {c1, c2};
        r.conditions.push_back(and_cond);
        Conclusion conc; conc.name = "both"; conc.value_expr = "true"; r.conclusions.push_back(conc);

        RuleEngine eng; eng.add_rule(r);
        std::unordered_map<std::string, Fact> facts;
        facts["a"] = make_numeric_fact("a", 5.0);
        facts["b"] = make_numeric_fact("b", 3.0);
        ReasoningTrace trace;
        bool fired = eng.forward_chain(facts, trace, cfg);
        CHECK(fired);
        CHECK(facts.count("both") > 0);
    }

    // AND fails if one part is false
    {
        Rule r;
        r.id = "r_and2"; r.confidence = 1.0;
        Condition and_cond;
        and_cond.op = CondOp::AND;
        Condition c1; c1.left = "a"; c1.op = CondOp::GT; c1.right = "0";
        Condition c2; c2.left = "b"; c2.op = CondOp::GT; c2.right = "0";
        and_cond.sub_conditions = {c1, c2};
        r.conditions.push_back(and_cond);
        Conclusion conc; conc.name = "both2"; conc.value_expr = "true"; r.conclusions.push_back(conc);

        RuleEngine eng; eng.add_rule(r);
        std::unordered_map<std::string, Fact> facts;
        facts["a"] = make_numeric_fact("a", 5.0);
        facts["b"] = make_numeric_fact("b", -1.0);
        ReasoningTrace trace;
        bool fired = eng.forward_chain(facts, trace, cfg);
        CHECK(!fired);
        CHECK(facts.count("both2") == 0);
    }

    // OR condition
    {
        Rule r;
        r.id = "r_or"; r.confidence = 1.0;
        Condition or_cond;
        or_cond.op = CondOp::OR;
        Condition c1; c1.left = "a"; c1.op = CondOp::GT; c1.right = "100";
        Condition c2; c2.left = "b"; c2.op = CondOp::GT; c2.right = "100";
        or_cond.sub_conditions = {c1, c2};
        r.conditions.push_back(or_cond);
        Conclusion conc; conc.name = "alert"; conc.value_expr = "true"; r.conclusions.push_back(conc);

        RuleEngine eng; eng.add_rule(r);
        std::unordered_map<std::string, Fact> facts;
        facts["a"] = make_numeric_fact("a", 50.0);
        facts["b"] = make_numeric_fact("b", 150.0);
        ReasoningTrace trace;
        bool fired = eng.forward_chain(facts, trace, cfg);
        CHECK(fired);
        CHECK(facts.count("alert") > 0);
    }
}

// ===========================================================================
// 4. SymbolicEngine
// ===========================================================================
void test_symbolic_engine() {
    SUITE("SymbolicEngine");

    SymbolicEngine sym;

    // Divide
    {
        AtlasExpression e;
        e.id = "e1"; e.op = "divide"; e.output_id = "speed";
        e.operand_ids = {"distance", "time"};
        std::unordered_map<std::string, Fact> facts;
        facts["distance"] = make_numeric_fact("distance", 150.0, "km");
        facts["time"]     = make_numeric_fact("time", 3.0, "h");
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(ok);
        CHECK(err.empty());
        CHECK(facts.count("speed") > 0);
        CHECK_NEAR(fact_value_to_double(facts["speed"].value), 50.0, 1e-9);
        CHECK(trace.expressions_evaluated > 0);
    }

    // Add
    {
        AtlasExpression e;
        e.id = "e2"; e.op = "add"; e.output_id = "total";
        e.operand_ids = {"a", "b"};
        std::unordered_map<std::string, Fact> facts;
        facts["a"] = make_numeric_fact("a", 10.0);
        facts["b"] = make_numeric_fact("b", 20.0);
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(ok);
        CHECK_NEAR(fact_value_to_double(facts["total"].value), 30.0, 1e-9);
    }

    // Multiply
    {
        AtlasExpression e;
        e.id = "e3"; e.op = "multiply"; e.output_id = "product";
        e.operand_ids = {"x", "y"};
        std::unordered_map<std::string, Fact> facts;
        facts["x"] = make_numeric_fact("x", 6.0);
        facts["y"] = make_numeric_fact("y", 7.0);
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(ok);
        CHECK_NEAR(fact_value_to_double(facts["product"].value), 42.0, 1e-9);
    }

    // Literal operand (numeric string)
    {
        AtlasExpression e;
        e.id = "e4"; e.op = "add"; e.output_id = "B";
        e.operand_ids = {"A", "5"};   // "5" is a literal
        std::unordered_map<std::string, Fact> facts;
        facts["A"] = make_numeric_fact("A", 10.0);
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(ok);
        CHECK_NEAR(fact_value_to_double(facts["B"].value), 15.0, 1e-9);
    }

    // Multi-step: B=A+5, C=B*2
    {
        AtlasExpression e1; e1.id = "e1"; e1.op = "add";
        e1.output_id = "B"; e1.operand_ids = {"A", "5"};
        AtlasExpression e2; e2.id = "e2"; e2.op = "multiply";
        e2.output_id = "C"; e2.operand_ids = {"B", "2"};

        std::unordered_map<std::string, Fact> facts;
        facts["A"] = make_numeric_fact("A", 10.0);
        ReasoningTrace trace;
        std::string err;

        bool ok1 = sym.evaluate_expression(e1, facts, trace, err);
        CHECK(ok1);
        bool ok2 = sym.evaluate_expression(e2, facts, trace, err);
        CHECK(ok2);
        CHECK_NEAR(fact_value_to_double(facts["C"].value), 30.0, 1e-9);
    }

    // Average
    {
        AtlasExpression e;
        e.id = "e5"; e.op = "average"; e.output_id = "mean";
        e.operand_ids = {"a", "b", "c"};
        std::unordered_map<std::string, Fact> facts;
        facts["a"] = make_numeric_fact("a", 10.0);
        facts["b"] = make_numeric_fact("b", 20.0);
        facts["c"] = make_numeric_fact("c", 30.0);
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(ok);
        CHECK_NEAR(fact_value_to_double(facts["mean"].value), 20.0, 1e-9);
    }

    // Missing operand → fail
    {
        AtlasExpression e;
        e.id = "e6"; e.op = "divide"; e.output_id = "z";
        e.operand_ids = {"nonexistent_x", "nonexistent_y"};
        std::unordered_map<std::string, Fact> facts;
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(!ok);
        CHECK(!err.empty());
    }

    // Division by zero
    {
        AtlasExpression e;
        e.id = "e7"; e.op = "divide"; e.output_id = "z";
        e.operand_ids = {"x", "y"};
        std::unordered_map<std::string, Fact> facts;
        facts["x"] = make_numeric_fact("x", 10.0);
        facts["y"] = make_numeric_fact("y", 0.0);
        ReasoningTrace trace;
        std::string err;
        bool ok = sym.evaluate_expression(e, facts, trace, err);
        CHECK(!ok);
        CHECK(!err.empty());
    }

    // build_bindings
    {
        std::unordered_map<std::string, Fact> facts;
        facts["a"] = make_numeric_fact("a", 5.0);
        facts["b"] = make_numeric_fact("b", 3.0);
        auto bindings = SymbolicEngine::build_bindings(facts);
        CHECK(bindings.count("a") > 0);
        CHECK_NEAR(bindings["a"], 5.0, 1e-9);
    }
}

// ===========================================================================
// 5. ConstraintEngine
// ===========================================================================
void test_constraint_engine() {
    SUITE("ConstraintEngine");

    ConstraintEngine ce;

    auto make_c = [](const std::string& id, const std::string& left,
                     const std::string& op_s, const std::string& right) {
        return atlas_constraint_to_reasoning(id, left, op_s, right, 1.0);
    };

    std::unordered_map<std::string, Fact> facts;
    facts["budget"]      = make_numeric_fact("budget", 40000.0);
    facts["temperature"] = make_numeric_fact("temperature", 95.0);
    facts["score"]       = make_numeric_fact("score", 85.0);

    // budget <= 50000 → satisfied
    {
        auto c = make_c("c1", "budget", "<=", "50000");
        ReasoningTrace trace;
        auto r = ce.check(c, facts);
        CHECK_EQ(r.result, ConstraintResult::SATISFIED);
    }

    // budget <= 30000 → violated
    {
        auto c = make_c("c2", "budget", "<=", "30000");
        ReasoningTrace trace;
        auto r = ce.check(c, facts);
        CHECK_EQ(r.result, ConstraintResult::VIOLATED);
    }

    // temperature < 100 → satisfied
    {
        auto c = make_c("c3", "temperature", "<", "100");
        ReasoningTrace trace;
        auto r = ce.check(c, facts);
        CHECK_EQ(r.result, ConstraintResult::SATISFIED);
    }

    // score >= 80 → satisfied
    {
        auto c = make_c("c4", "score", ">=", "80");
        ReasoningTrace trace;
        auto r = ce.check(c, facts);
        CHECK_EQ(r.result, ConstraintResult::SATISFIED);
    }

    // score >= 90 → violated
    {
        auto c = make_c("c5", "score", ">=", "90");
        ReasoningTrace trace;
        auto r = ce.check(c, facts);
        CHECK_EQ(r.result, ConstraintResult::VIOLATED);
    }

    // missing fact → unknown
    {
        auto c = make_c("c6", "missing", "<=", "100");
        ReasoningTrace trace;
        auto r = ce.check(c, facts);
        CHECK_EQ(r.result, ConstraintResult::UNKNOWN);
    }

    // check_all appends to trace
    {
        std::vector<ReasoningConstraint> constraints = {
            make_c("c1", "budget", "<=", "50000"),
            make_c("c2", "score",  ">=", "80"),
        };
        ReasoningTrace trace;
        auto results = ce.check_all(constraints, facts, trace);
        CHECK_EQ(results.size(), static_cast<size_t>(2));
        CHECK_EQ(trace.constraints_checked, 2);
        CHECK(!ce.any_violated(results));
    }

    // any_violated
    {
        std::vector<ReasoningConstraint> constraints = {
            make_c("c1", "budget", "<=", "50000"),
            make_c("c2", "score",  ">=", "90"),  // violated
        };
        ReasoningTrace trace;
        auto results = ce.check_all(constraints, facts, trace);
        CHECK(ce.any_violated(results));
    }
}

// ===========================================================================
// 6. ReasoningEngine — full pipeline
// ===========================================================================
void test_reasoning_engine() {
    SUITE("ReasoningEngine");

    // Test 1: Arithmetic — distance/time → speed
    {
        ReasoningContext ctx;
        ctx.config.mode = ReasoningMode::HYBRID;

        Fact dist = make_numeric_fact("distance", 150.0, "km");
        Fact time_ = make_numeric_fact("time",    3.0,   "h");
        ctx.facts["distance"] = dist;
        ctx.facts["time"]     = time_;

        AtlasExpression e;
        e.id = "ex1"; e.op = "divide"; e.output_id = "speed";
        e.operand_ids = {"distance", "time"};
        ctx.expressions.push_back(e);

        Goal g; g.id = "g1"; g.target = "speed"; g.description = "Find speed";
        ctx.goals.push_back(g);

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::SOLVED);
        CHECK_NEAR(result.answer_numeric, 50.0, 1e-9);
        CHECK(result.answer_is_numeric);
        CHECK(result.trace.total_steps > 0);
        CHECK(result.trace.expressions_evaluated > 0);
    }

    // Test 2: Rule reasoning — temperature warning
    {
        ReasoningContext ctx;
        ctx.facts["temperature"] = make_numeric_fact("temperature", 120.0);

        Rule rule = make_simple_rule("r1", "temperature", ">", "100", "warning", "true");
        ctx.rules.push_back(rule);

        Goal g; g.id = "g1"; g.target = "warning";
        ctx.goals.push_back(g);

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::SOLVED);
        CHECK(result.trace.rules_fired > 0);
    }

    // Test 3: Multi-step A→B→C
    {
        ReasoningContext ctx;
        ctx.facts["A"] = make_numeric_fact("A", 10.0);

        AtlasExpression e1; e1.id="e1"; e1.op="add"; e1.output_id="B";
        e1.operand_ids = {"A", "5"};
        AtlasExpression e2; e2.id="e2"; e2.op="multiply"; e2.output_id="C";
        e2.operand_ids = {"B", "2"};
        ctx.expressions = {e1, e2};

        Goal g; g.id = "g1"; g.target = "C";
        ctx.goals.push_back(g);

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::SOLVED);
        CHECK_NEAR(result.answer_numeric, 30.0, 1e-9);
        CHECK_EQ(result.trace.expressions_evaluated, 2);
    }

    // Test 4: Constraint satisfaction
    {
        ReasoningContext ctx;
        ctx.facts["budget"] = make_numeric_fact("budget", 40000.0);
        ctx.constraints.push_back(
            atlas_constraint_to_reasoning("c1", "budget", "<=", "50000", 1.0));

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK(result.constraint_results.size() == 1);
        CHECK_EQ(result.constraint_results[0].result, ConstraintResult::SATISFIED);
        CHECK(result.status != ReasoningStatus::CONSTRAINT_VIOLATION);
    }

    // Test 5: Unresolved — missing operand
    {
        ReasoningContext ctx;
        // No facts — operand 'x' is unknown
        AtlasExpression e; e.id="e1"; e.op="add"; e.output_id="result";
        e.operand_ids = {"x", "10"};
        ctx.expressions.push_back(e);
        Goal g; g.id="g1"; g.target="result";
        ctx.goals.push_back(g);

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::UNRESOLVED);
        CHECK(result.derived_facts.count("result") == 0);
    }

    // Test 6: Circular rules — must terminate
    {
        ReasoningContext ctx;
        ctx.facts["A"] = make_numeric_fact("A", 1.0);
        ctx.rules.push_back(make_simple_rule("r1","A","=","1","B","1"));
        ctx.rules.push_back(make_simple_rule("r2","B","=","1","C","1"));
        ctx.rules.push_back(make_simple_rule("r3","C","=","1","A","1"));
        ctx.config.max_depth        = 50;
        ctx.config.max_rule_firings = 200;

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK(result.status != ReasoningStatus::INTERNAL_ERROR);
        CHECK(result.trace.total_steps < 1000);
    }

    // Test: Invalid input — no goals/rules/constraints/expressions
    {
        ReasoningContext ctx;
        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::INVALID_INPUT);
    }

    // Test: constraint violation detected
    {
        ReasoningContext ctx;
        ctx.facts["budget"] = make_numeric_fact("budget", 60000.0);
        ctx.constraints.push_back(
            atlas_constraint_to_reasoning("c1", "budget", "<=", "50000", 1.0));

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::CONSTRAINT_VIOLATION);
        CHECK(result.constraint_results[0].result == ConstraintResult::VIOLATED);
    }

    // Test: multiple goals — partial satisfaction
    {
        ReasoningContext ctx;
        ctx.facts["x"] = make_numeric_fact("x", 5.0);
        // 'y' is not in facts
        Goal g1; g1.id="g1"; g1.target="x";
        Goal g2; g2.id="g2"; g2.target="y";
        ctx.goals = {g1, g2};

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK(result.status == ReasoningStatus::PARTIALLY_SOLVED ||
              result.status == ReasoningStatus::UNRESOLVED);

        bool x_satisfied = false;
        for (auto& gr : result.goal_results)
            if (gr.target == "x" && gr.satisfied) x_satisfied = true;
        CHECK(x_satisfied);
    }
}

void test_reasoning_engine_limits() {
    SUITE("ReasoningEngineLimits");

    // max_rule_firings stops the engine
    {
        ReasoningContext ctx;
        // Build a chain of 200 rules
        for (int i = 0; i < 200; ++i) {
            ctx.rules.push_back(make_simple_rule(
                "r" + std::to_string(i),
                "f" + std::to_string(i), "=", "1",
                "f" + std::to_string(i+1), "1"));
        }
        ctx.facts["f0"] = make_numeric_fact("f0", 1.0);
        ctx.config.max_rule_firings = 10;

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        // Should stop before reaching f200
        CHECK(result.trace.rules_fired <= 15);
    }
}

// ===========================================================================
// 7. Deductive reasoning
// ===========================================================================
void test_deductive_reasoning() {
    SUITE("DeductiveReasoning");

    // "All machines above 100°C require inspection. Machine A = 120°C."
    ReasoningContext ctx;
    ctx.facts["machine_A_temp"] = make_numeric_fact("machine_A_temp", 120.0);
    ctx.rules.push_back(
        make_simple_rule("inspection_rule",
                         "machine_A_temp", ">", "100",
                         "machine_A_requires_inspection", "true"));
    Goal g; g.id = "g1"; g.target = "machine_A_requires_inspection";
    ctx.goals.push_back(g);

    ReasoningEngine engine;
    auto result = engine.solve(ctx);
    CHECK_EQ(result.status, ReasoningStatus::SOLVED);
    CHECK(result.derived_facts.count("machine_A_requires_inspection") > 0);

    // Verify trace provenance — a RULE_FIRE step must exist
    bool found_rule_step = false;
    for (auto& step : result.trace.steps)
        if (step.type == StepType::RULE_FIRE && step.success) found_rule_step = true;
    CHECK(found_rule_step);
}

// ===========================================================================
// 8. Goal satisfaction checks
// ===========================================================================
void test_goal_checking() {
    SUITE("GoalChecking");

    // Goal satisfied
    {
        ReasoningContext ctx;
        ctx.facts["result"] = make_numeric_fact("result", 42.0);
        Goal g; g.id="g1"; g.target="result";
        ctx.goals.push_back(g);

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::SOLVED);
        CHECK(!result.goal_results.empty());
        CHECK(result.goal_results[0].satisfied);
        CHECK_EQ(result.goal_results[0].derived_value, "42");
    }

    // Goal not found
    {
        ReasoningContext ctx;
        Goal g; g.id="g1"; g.target="nonexistent";
        ctx.goals.push_back(g);

        ReasoningEngine engine;
        auto result = engine.solve(ctx);
        CHECK_EQ(result.status, ReasoningStatus::UNRESOLVED);
        CHECK(!result.goal_results[0].satisfied);
    }
}

// ===========================================================================
// 9. Reasoning trace integrity
// ===========================================================================
void test_trace_integrity() {
    SUITE("TraceIntegrity");

    ReasoningContext ctx;
    ctx.facts["x"] = make_numeric_fact("x", 6.0);
    ctx.facts["y"] = make_numeric_fact("y", 7.0);

    AtlasExpression e; e.id="e1"; e.op="multiply"; e.output_id="z";
    e.operand_ids = {"x", "y"};
    ctx.expressions.push_back(e);
    Goal g; g.id="g1"; g.target="z";
    ctx.goals.push_back(g);

    ReasoningEngine engine;
    auto result = engine.solve(ctx);
    CHECK_EQ(result.status, ReasoningStatus::SOLVED);

    // Steps are ordered by step_id
    for (size_t i = 1; i < result.trace.steps.size(); ++i)
        CHECK(result.trace.steps[i].step_id > result.trace.steps[i-1].step_id);

    // At least one EXPRESSION_EVAL step with correct description
    bool found_expr = false;
    for (auto& step : result.trace.steps) {
        if (step.type == StepType::EXPRESSION_EVAL) {
            found_expr = true;
            CHECK(!step.description.empty());
        }
    }
    CHECK(found_expr);

    // Goal check step present
    bool found_goal = false;
    for (auto& step : result.trace.steps)
        if (step.type == StepType::GOAL_CHECK) found_goal = true;
    CHECK(found_goal);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    std::cout << "PRIME Phase 5 — C++ Reasoning Engine Tests\n";
    std::cout << "============================================\n";

    test_types();
    test_expression_evaluator();
    test_rule_engine();
    test_rule_conditions();
    test_rule_priorities();
    test_rule_chaining();
    test_circular_rule_detection();
    test_compound_conditions();
    test_symbolic_engine();
    test_constraint_engine();
    test_reasoning_engine();
    test_reasoning_engine_limits();
    test_deductive_reasoning();
    test_goal_checking();
    test_trace_integrity();

    std::cout << "\n============================================\n";
    std::cout << "Results: " << g_passed << "/" << g_tests << " passed";
    if (g_failed > 0)
        std::cout << "  (" << g_failed << " FAILED)";
    std::cout << "\n";

    return g_failed > 0 ? 1 : 0;
}
