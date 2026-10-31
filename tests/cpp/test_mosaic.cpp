// tests/cpp/test_mosaic.cpp — C++ unit tests for PRIME Phase 6 MOSAIC.
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -I cpp \
//       tests/cpp/test_mosaic.cpp -o tests/cpp/test_mosaic
// Run:
//   tests/cpp/test_mosaic

#include "mosaic/mosaic_types.hpp"
#include "mosaic/module_registry.hpp"
#include "mosaic/task_analyzer.hpp"
#include "mosaic/composition_planner.hpp"
#include "mosaic/mosaic_engine.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int g_tests  = 0;
static int g_passed = 0;
static int g_failed = 0;
static std::string g_suite;

#define SUITE(n) do { g_suite = (n); \
    std::cout << "\n[" << g_suite << "]\n"; } while(0)

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

#define CHECK_EQ(a,b) CHECK((a)==(b))
#define CHECK_NE(a,b) CHECK((a)!=(b))
#define CHECK_NEAR(a,b,e) CHECK(std::abs((double)(a)-(double)(b)) < (e))

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
using namespace prime::mosaic;

static Capability make_cap(const std::string& name,
                            CapabilityCategory cat = CapabilityCategory::GENERAL,
                            std::vector<std::string> inp = {},
                            std::vector<std::string> out = {}) {
    Capability c;
    c.name        = name;
    c.category    = cat;
    c.input_types  = std::move(inp);
    c.output_types = std::move(out);
    c.confidence  = 1.0;
    return c;
}

static ModuleMetadata make_module(
        const std::string& id,
        const std::string& name,
        std::vector<Capability> caps,
        std::vector<std::string> inp  = {"*"},
        std::vector<std::string> out  = {"*"},
        std::vector<std::string> deps = {},
        int  priority = 0,
        double cpu    = 1.0,
        double mem    = 1.0,
        bool model    = false,
        ModuleStatus status = ModuleStatus::AVAILABLE)
{
    ModuleMetadata m;
    m.id           = id;
    m.name         = name;
    m.capabilities = std::move(caps);
    m.input_types  = std::move(inp);
    m.output_types = std::move(out);
    m.dependencies = std::move(deps);
    m.priority     = priority;
    m.cost.cpu_relative    = cpu;
    m.cost.memory_relative = mem;
    m.cost.requires_model  = model;
    m.status       = status;
    return m;
}

// ===========================================================================
// 1. MOSAIC types
// ===========================================================================
void test_types() {
    SUITE("MOSAICTypes");

    CHECK_EQ(composition_status_to_string(CompositionStatus::VALID),             "VALID");
    CHECK_EQ(composition_status_to_string(CompositionStatus::COMPOSITION_CYCLE), "COMPOSITION_CYCLE");
    CHECK_EQ(composition_status_to_string(CompositionStatus::MISSING_CAPABILITY),"MISSING_CAPABILITY");
    CHECK_EQ(composition_status_to_string(CompositionStatus::INPUT_MISMATCH),    "INPUT_MISMATCH");
    CHECK_EQ(composition_status_to_string(CompositionStatus::EMPTY_PLAN),        "EMPTY_PLAN");

    CHECK_EQ(module_status_to_string(ModuleStatus::AVAILABLE),   "available");
    CHECK_EQ(module_status_to_string(ModuleStatus::UNAVAILABLE), "unavailable");
    CHECK_EQ(module_status_to_string(ModuleStatus::DEGRADED),    "degraded");

    CHECK_EQ(execution_status_to_string(ExecutionStatus::SUCCESS),       "SUCCESS");
    CHECK_EQ(execution_status_to_string(ExecutionStatus::FALLBACK_USED), "FALLBACK_USED");
    CHECK_EQ(execution_status_to_string(ExecutionStatus::INVALID_PLAN),  "INVALID_PLAN");

    CHECK_EQ(category_to_string(CapabilityCategory::ARITHMETIC),         "arithmetic");
    CHECK_EQ(category_to_string(CapabilityCategory::SYMBOLIC_REASONING), "symbolic_reasoning");
    CHECK_EQ(category_to_string(CapabilityCategory::VALIDATION),         "validation");

    // ModuleMetadata helpers
    auto m = make_module("m1", "M1", {make_cap("arithmetic")});
    CHECK(m.has_capability("arithmetic"));
    CHECK(!m.has_capability("nonexistent"));
    CHECK(m.accepts_input("*"));

    // ExecutionContext
    ExecutionContext ctx;
    ctx.set("x", ContextValue{42.0});
    CHECK(ctx.has("x"));
    CHECK_NEAR(ctx.get_double("x"), 42.0, 1e-9);
    CHECK_EQ(ctx.get_str("x"), "42");
    CHECK(!ctx.has("missing"));
    CHECK_EQ(ctx.get_str("missing", "dflt"), "dflt");
    CHECK_EQ(ctx.get_double("missing", 99.0), 99.0);

    ctx.set("flag", ContextValue{true});
    CHECK(ctx.get_bool("flag"));
    ctx.set("label", ContextValue{std::string("hello")});
    CHECK_EQ(ctx.get_str("label"), "hello");
}

// ===========================================================================
// 2. CapabilityRegistry
// ===========================================================================
void test_capability_registry() {
    SUITE("CapabilityRegistry");

    CapabilityRegistry reg;
    CHECK_EQ(reg.size(), static_cast<size_t>(0));

    Capability c1;
    c1.name     = "arithmetic";
    c1.category = CapabilityCategory::ARITHMETIC;
    CHECK(reg.register_capability(c1));
    CHECK_EQ(reg.size(), static_cast<size_t>(1));

    // Duplicate rejected
    CHECK(!reg.register_capability(c1));
    CHECK_EQ(reg.size(), static_cast<size_t>(1));

    // has / get
    CHECK(reg.has("arithmetic"));
    CHECK(!reg.has("unknown_cap"));
    auto* p = reg.get("arithmetic");
    CHECK(p != nullptr);
    CHECK_EQ(p->name, "arithmetic");
    CHECK(reg.get("missing") == nullptr);

    // list_names sorted
    Capability c2; c2.name = "validation"; c2.category = CapabilityCategory::VALIDATION;
    Capability c3; c3.name = "memory";     c3.category = CapabilityCategory::RETRIEVAL;
    reg.register_capability(c2);
    reg.register_capability(c3);
    auto names = reg.list_names();
    CHECK_EQ(names.size(), static_cast<size_t>(3));
    CHECK(std::is_sorted(names.begin(), names.end()));

    // find_by_category
    auto arith = reg.find_by_category(CapabilityCategory::ARITHMETIC);
    CHECK_EQ(arith.size(), static_cast<size_t>(1));
    CHECK_EQ(arith[0].name, "arithmetic");

    // ensure (idempotent)
    Capability c4; c4.name = "arithmetic"; c4.description = "updated";
    reg.ensure(c4);
    CHECK_EQ(reg.size(), static_cast<size_t>(3));
    CHECK_EQ(reg.get("arithmetic")->description, "updated");

    // clear
    reg.clear();
    CHECK_EQ(reg.size(), static_cast<size_t>(0));
}

// ===========================================================================
// 3. ModuleRegistry
// ===========================================================================
void test_module_registry() {
    SUITE("ModuleRegistry");

    ModuleRegistry reg;
    CHECK_EQ(reg.size(), static_cast<size_t>(0));

    auto m1 = make_module("calc", "Calculator",
                           {make_cap("arithmetic", CapabilityCategory::ARITHMETIC)});
    CHECK(reg.register_module(m1));
    CHECK_EQ(reg.size(), static_cast<size_t>(1));

    // Duplicate rejected
    CHECK(!reg.register_module(m1));
    CHECK_EQ(reg.size(), static_cast<size_t>(1));

    // has / get
    CHECK(reg.has("calc"));
    CHECK(!reg.has("ghost"));
    auto* p = reg.get("calc");
    CHECK(p != nullptr);
    CHECK_EQ(p->name, "Calculator");
    CHECK(reg.get("ghost") == nullptr);

    // find_capable
    auto caps = reg.find_capable("arithmetic");
    CHECK_EQ(caps.size(), static_cast<size_t>(1));
    CHECK_EQ(caps[0]->id, "calc");

    auto none = reg.find_capable("nonexistent");
    CHECK(none.empty());

    // list_ids sorted
    auto m2 = make_module("zmod", "Z", {make_cap("validation")});
    auto m3 = make_module("amod", "A", {make_cap("retrieval")});
    reg.register_module(m2);
    reg.register_module(m3);
    auto ids = reg.list_ids();
    CHECK_EQ(ids.size(), static_cast<size_t>(3));
    CHECK(std::is_sorted(ids.begin(), ids.end()));

    // remove
    CHECK(reg.remove("calc"));
    CHECK(!reg.has("calc"));
    CHECK(!reg.remove("calc")); // already removed
    CHECK_EQ(reg.size(), static_cast<size_t>(2));
    CHECK(reg.find_capable("arithmetic").empty());

    // set_status
    CHECK(reg.set_status("zmod", ModuleStatus::UNAVAILABLE));
    auto avail = reg.available_modules();
    // Only amod should be available
    CHECK_EQ(avail.size(), static_cast<size_t>(1));
    CHECK_EQ(avail[0]->id, "amod");
    CHECK(!reg.set_status("ghost", ModuleStatus::AVAILABLE));

    // Priority sorting in find_capable
    reg.register_module(make_module("high", "High",
        {make_cap("shared")}, {}, {}, {}, 10));
    reg.register_module(make_module("low", "Low",
        {make_cap("shared")}, {}, {}, {}, 1));
    auto shared = reg.find_capable("shared");
    CHECK_EQ(shared.size(), static_cast<size_t>(2));
    CHECK_EQ(shared[0]->id, "high"); // higher priority first

    // update_module
    auto m_upd = make_module("amod", "A-Updated", {make_cap("new_cap")});
    reg.update_module(m_upd);
    CHECK_EQ(reg.get("amod")->name, "A-Updated");
    CHECK(!reg.find_capable("retrieval").empty() == false); // old cap gone
    CHECK(!reg.find_capable("new_cap").empty());

    // clear
    reg.clear();
    CHECK_EQ(reg.size(), static_cast<size_t>(0));
}

// ===========================================================================
// 4. TaskAnalyzer
// ===========================================================================
void test_task_analyzer() {
    SUITE("TaskAnalyzer");

    TaskAnalyzer analyzer;

    // Basic: always requires atlas_representation and reasoning
    auto req = analyzer.analyze("t1", "physics",
        false, false, false, false, true, false, false,
        {}, {}, 2, false);
    CHECK(!req.required_capabilities.empty());
    bool has_atlas = false, has_reasoning = false;
    for (auto& c : req.required_capabilities) {
        if (c == "atlas_representation") has_atlas = true;
        if (c == "reasoning")            has_reasoning = true;
    }
    CHECK(has_atlas);
    CHECK(has_reasoning);

    // Expressions → arithmetic
    auto req2 = analyzer.analyze("t2", "math",
        true, false, false, false, true, false, false,
        {}, {}, 2, false);
    bool has_arith = false;
    for (auto& c : req2.required_capabilities)
        if (c == "arithmetic") has_arith = true;
    CHECK(has_arith);
    CHECK(req2.has_arithmetic);

    // Rules → rule_reasoning
    auto req3 = analyzer.analyze("t3", "engineering",
        false, true, false, false, true, false, false,
        {}, {}, 1, false);
    bool has_rule = false;
    for (auto& c : req3.required_capabilities)
        if (c == "rule_reasoning") has_rule = true;
    CHECK(has_rule);
    CHECK(req3.has_rules);

    // Constraints
    auto req4 = analyzer.analyze("t4", "business",
        false, false, true, false, true, false, false,
        {}, {}, 1, false);
    bool has_constr = false;
    for (auto& c : req4.required_capabilities)
        if (c == "constraint_checking") has_constr = true;
    CHECK(has_constr);
    CHECK(req4.has_constraints);

    // Memory context
    auto req5 = analyzer.analyze("t5", "general",
        false, false, false, false, false, false, false,
        {}, {}, 0, true);
    bool has_mem = false;
    for (auto& c : req5.required_capabilities)
        if (c == "memory_retrieval") has_mem = true;
    CHECK(has_mem);
    CHECK(req5.needs_memory);

    // Arithmetic operations
    auto req6 = analyzer.analyze("t6", "physics",
        false, false, false, false, true, false, false,
        {"divide", "find"}, {}, 2, false);
    bool has_arith2 = false;
    for (auto& c : req6.required_capabilities)
        if (c == "arithmetic") has_arith2 = true;
    CHECK(has_arith2);

    // No duplicates
    auto req7 = analyzer.analyze("t7", "math",
        true, true, true, true, true, true, true,
        {"divide"}, {"speed"}, 3, true);
    std::unordered_set<std::string> seen;
    for (auto& c : req7.required_capabilities) {
        CHECK(seen.insert(c).second); // must be unique
    }
    for (auto& c : req7.optional_capabilities) {
        CHECK(seen.insert(c).second);
    }
}

// ===========================================================================
// 5. CompositionPlanner
// ===========================================================================
void test_composition_planner() {
    SUITE("CompositionPlanner");

    CompositionPlanner planner;
    MOSAICConfig cfg;

    // --- Simple capability match ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("m1", "M1",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}));

        TaskRequirements req;
        req.task_id = "t1"; req.domain = "math";
        req.required_capabilities = {"arithmetic"};

        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::VALID);
        CHECK(!plan.module_ids.empty());
        CHECK_EQ(plan.module_ids[0], "m1");
        CHECK(!plan.selection_reasons.empty());
    }

    // --- Missing capability ---
    {
        ModuleRegistry reg;
        TaskRequirements req;
        req.task_id = "t2"; req.domain = "x";
        req.required_capabilities = {"nonexistent"};
        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::MISSING_CAPABILITY);
        CHECK(!plan.valid());
    }

    // --- Cycle detection ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("a", "A",
            {make_cap("cap_a")}, {"*"}, {"*"}, {"b"}));
        reg.register_module(make_module("b", "B",
            {make_cap("cap_b")}, {"*"}, {"*"}, {"a"}));

        TaskRequirements req;
        req.task_id = "t3"; req.domain = "x";
        req.required_capabilities = {"cap_a", "cap_b"};
        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::COMPOSITION_CYCLE);
    }

    // --- Dependency ordering ---
    {
        ModuleRegistry reg;
        // b depends on a → a must come first
        reg.register_module(make_module("a", "A",
            {make_cap("cap_a")},
            {"*"}, {"cap_a_output", "cap_b_output"}));
        reg.register_module(make_module("b", "B",
            {make_cap("cap_b")},
            {"cap_a_output"}, {"cap_b_output"}, {"a"}));

        TaskRequirements req;
        req.task_id = "t4"; req.domain = "x";
        req.required_capabilities = {"cap_a", "cap_b"};
        auto plan = planner.plan(req, reg, cfg);
        if (plan.status == CompositionStatus::VALID) {
            auto it_a = std::find(plan.module_ids.begin(), plan.module_ids.end(), "a");
            auto it_b = std::find(plan.module_ids.begin(), plan.module_ids.end(), "b");
            CHECK(it_a != plan.module_ids.end());
            CHECK(it_b != plan.module_ids.end());
            CHECK(it_a < it_b);
        }
    }

    // --- Priority selection ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("low_p", "LowPriority",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}, {}, 1));
        reg.register_module(make_module("high_p", "HighPriority",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}, {}, 10));

        TaskRequirements req;
        req.task_id = "t5"; req.domain = "math";
        req.required_capabilities = {"arithmetic"};
        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::VALID);
        CHECK_EQ(plan.module_ids[0], "high_p");
    }

    // --- Unavailable module not selected ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("unavail", "U",
            {make_cap("special")}, {"*"}, {"*"}, {}, 0,
            1.0, 1.0, false, ModuleStatus::UNAVAILABLE));
        TaskRequirements req;
        req.task_id = "t6"; req.domain = "x";
        req.required_capabilities = {"special"};
        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::MISSING_CAPABILITY);
    }

    // --- LLM policy ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("llm_m", "LLM",
            {make_cap("text_generation", CapabilityCategory::LANGUAGE,
                      {"*"}, {"text_output"})},
            {"*"}, {"text_output"}, {}, 0, 10.0, 8.0, true));

        TaskRequirements req;
        req.task_id = "t7"; req.domain = "lang";
        req.required_capabilities = {"text_generation"};

        // allow_llm = false → rejected
        MOSAICConfig cfg_nollm;
        cfg_nollm.allow_llm = false;
        auto plan_no = planner.plan(req, reg, cfg_nollm);
        CHECK_NE(plan_no.status, CompositionStatus::VALID);

        // allow_llm = true → selected
        MOSAICConfig cfg_llm;
        cfg_llm.allow_llm = true;
        auto plan_ok = planner.plan(req, reg, cfg_llm);
        CHECK_EQ(plan_ok.status, CompositionStatus::VALID);
        CHECK_EQ(plan_ok.module_ids[0], "llm_m");
    }

    // --- Input mismatch ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("m_bad", "Bad",
            {make_cap("special")},
            {"completely_unknown_type"}, {"*"}));
        TaskRequirements req;
        req.task_id = "t8"; req.domain = "x";
        req.required_capabilities = {"special"};
        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::INPUT_MISMATCH);
    }

    // --- Excluded modules ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("ex", "Excluded",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}, {}, 99));
        reg.register_module(make_module("ok", "OK",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}, {}, 1));

        TaskRequirements req;
        req.task_id = "t9"; req.domain = "math";
        req.required_capabilities = {"arithmetic"};
        MOSAICConfig cfg_ex;
        cfg_ex.excluded_modules = {"ex"};
        auto plan = planner.plan(req, reg, cfg_ex);
        CHECK_EQ(plan.status, CompositionStatus::VALID);
        bool has_ex = false, has_ok = false;
        for (auto& id : plan.module_ids) {
            if (id == "ex") has_ex = true;
            if (id == "ok") has_ok = true;
        }
        CHECK(!has_ex);
        CHECK(has_ok);
    }

    // --- Resource-aware (lightweight preferred) ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("light", "Light",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}, {}, 5, 0.5, 0.5));
        reg.register_module(make_module("heavy", "Heavy",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}, {}, 5, 20.0, 20.0));

        TaskRequirements req;
        req.task_id = "t10"; req.domain = "math";
        req.required_capabilities = {"arithmetic"};
        MOSAICConfig cfg_lw;
        cfg_lw.prefer_lightweight = true;
        auto plan = planner.plan(req, reg, cfg_lw);
        CHECK_EQ(plan.status, CompositionStatus::VALID);
        CHECK_EQ(plan.module_ids[0], "light");
    }

    // --- max_modules limit ---
    {
        ModuleRegistry reg;
        for (int i = 0; i < 6; ++i)
            reg.register_module(make_module("m" + std::to_string(i), "M" + std::to_string(i),
                {make_cap("cap" + std::to_string(i))}));

        TaskRequirements req;
        req.task_id = "tlimit"; req.domain = "x";
        for (int i = 0; i < 6; ++i)
            req.required_capabilities.push_back("cap" + std::to_string(i));

        MOSAICConfig cfg_limit;
        cfg_limit.max_modules = 3;
        auto plan = planner.plan(req, reg, cfg_limit);
        CHECK_NE(plan.status, CompositionStatus::VALID);
    }

    // --- Optional capabilities (best-effort) ---
    {
        ModuleRegistry reg;
        reg.register_module(make_module("r", "Required",
            {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                      {"*"}, {"reasoning_result"})},
            {"*"}, {"reasoning_result"}));
        // No module for "optional_cap" — should not fail

        TaskRequirements req;
        req.task_id = "topt"; req.domain = "math";
        req.required_capabilities = {"arithmetic"};
        req.optional_capabilities = {"optional_cap"};
        auto plan = planner.plan(req, reg, cfg);
        CHECK_EQ(plan.status, CompositionStatus::VALID);
    }
}

// ===========================================================================
// 6. MOSAICEngine — full orchestrator
// ===========================================================================
void test_mosaic_engine() {
    SUITE("MOSAICEngine");

    MOSAICEngine engine;

    // Register a module
    auto m = make_module("arith", "Arithmetic",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"});
    CHECK(engine.register_module(m));
    CHECK_EQ(engine.module_count(), static_cast<size_t>(1));

    // Duplicate rejected
    CHECK(!engine.register_module(m));

    // list_modules
    auto ids = engine.list_modules();
    CHECK_EQ(ids.size(), static_cast<size_t>(1));
    CHECK_EQ(ids[0], "arith");

    // get_module
    auto* p = engine.get_module("arith");
    CHECK(p != nullptr);
    CHECK_EQ(p->name, "Arithmetic");

    // set_module_status
    CHECK(engine.set_module_status("arith", ModuleStatus::UNAVAILABLE));
    CHECK_EQ(engine.get_module("arith")->status, ModuleStatus::UNAVAILABLE);
    CHECK(engine.set_module_status("arith", ModuleStatus::AVAILABLE));

    // remove_module
    CHECK(engine.remove_module("arith"));
    CHECK_EQ(engine.module_count(), static_cast<size_t>(0));
    CHECK(!engine.remove_module("arith")); // already gone

    // compose with missing capability
    TaskRequirements req;
    req.task_id = "t1"; req.domain = "x";
    req.required_capabilities = {"arithmetic"};
    auto plan = engine.compose(req);
    CHECK_NE(plan.status, CompositionStatus::VALID);

    // Register and compose successfully
    engine.register_module(make_module("a", "A",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"}));
    plan = engine.compose(req);
    CHECK_EQ(plan.status, CompositionStatus::VALID);
    CHECK(!plan.module_ids.empty());

    // explain_plan
    auto explanation = engine.explain_plan(plan);
    CHECK(!explanation.empty());
    CHECK(explanation.find("MOSAIC") != std::string::npos);

    // validate_plan
    auto vstatus = engine.validate_plan(plan);
    CHECK_EQ(vstatus, CompositionStatus::VALID);

    // Cache: same request returns same plan
    auto plan2 = engine.compose(req);
    CHECK_EQ(plan.module_ids, plan2.module_ids);

    // Config accessor
    auto& cfg = engine.config();
    cfg.max_modules = 20;
    CHECK_EQ(engine.config().max_modules, 20);
}

// ===========================================================================
// 7. Determinism
// ===========================================================================
void test_determinism() {
    SUITE("Determinism");

    MOSAICEngine engine;
    engine.register_module(make_module("a1", "A1",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"}, {}, 5));
    engine.register_module(make_module("a2", "A2",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"}, {}, 5));

    TaskRequirements req;
    req.task_id = "det"; req.domain = "math";
    req.required_capabilities = {"arithmetic"};

    std::vector<CompositionPlan> plans;
    for (int i = 0; i < 10; ++i)
        plans.push_back(engine.compose(req));

    for (auto& p : plans) {
        CHECK_EQ(p.module_ids, plans[0].module_ids);
        CHECK_EQ(p.status,     plans[0].status);
    }
}

// ===========================================================================
// 8. Resource-aware selection
// ===========================================================================
void test_resource_awareness() {
    SUITE("ResourceAwareness");

    MOSAICEngine engine;
    MOSAICConfig cfg;
    cfg.prefer_lightweight = true;

    engine.register_module(make_module("light", "LightCalc",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"}, {}, 5, 0.3, 0.3));
    engine.register_module(make_module("heavy", "HeavyCalc",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"}, {}, 5, 50.0, 50.0));

    TaskRequirements req;
    req.task_id = "ra"; req.domain = "math";
    req.required_capabilities = {"arithmetic"};

    engine.set_config(cfg);
    auto plan = engine.compose(req);
    CHECK_EQ(plan.status, CompositionStatus::VALID);
    CHECK(!plan.module_ids.empty());
    CHECK_EQ(plan.module_ids[0], "light");

    // Estimated cost is positive
    CHECK(plan.estimated_cost > 0.0);
}

// ===========================================================================
// 9. Cycle detection — required test 6
// ===========================================================================
void test_cycle_detection() {
    SUITE("CycleDetection");

    MOSAICEngine engine;
    engine.register_module(make_module("a", "A", {make_cap("cap_a")},
                                       {"*"}, {"*"}, {"b"}));
    engine.register_module(make_module("b", "B", {make_cap("cap_b")},
                                       {"*"}, {"*"}, {"a"}));

    TaskRequirements req;
    req.task_id = "cyc"; req.domain = "x";
    req.required_capabilities = {"cap_a", "cap_b"};

    auto plan = engine.compose(req);
    CHECK_EQ(plan.status, CompositionStatus::COMPOSITION_CYCLE);
    CHECK(!plan.valid());
}

// ===========================================================================
// 10. Cache management
// ===========================================================================
void test_cache() {
    SUITE("Cache");

    MOSAICEngine engine;
    engine.register_module(make_module("m", "M",
        {make_cap("arithmetic", CapabilityCategory::ARITHMETIC,
                  {"*"}, {"reasoning_result"})},
        {"*"}, {"reasoning_result"}));

    TaskRequirements req;
    req.task_id = "cache_t"; req.domain = "math";
    req.required_capabilities = {"arithmetic"};

    auto plan1 = engine.compose(req);
    auto plan2 = engine.compose(req); // should hit cache
    CHECK_EQ(plan1.module_ids, plan2.module_ids);

    // After removing module, cache is invalidated
    engine.remove_module("m");
    auto plan3 = engine.compose(req);
    CHECK_NE(plan3.status, CompositionStatus::VALID);

    // clear_cache
    engine.clear_cache();
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    std::cout << "PRIME Phase 6 — C++ MOSAIC Engine Tests\n";
    std::cout << "=========================================\n";

    test_types();
    test_capability_registry();
    test_module_registry();
    test_task_analyzer();
    test_composition_planner();
    test_mosaic_engine();
    test_determinism();
    test_resource_awareness();
    test_cycle_detection();
    test_cache();

    std::cout << "\n=========================================\n";
    std::cout << "Results: " << g_passed << "/" << g_tests << " passed";
    if (g_failed > 0)
        std::cout << "  (" << g_failed << " FAILED)";
    std::cout << "\n";
    return g_failed > 0 ? 1 : 0;
}
