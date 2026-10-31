#pragma once
// reasoning_engine.hpp — PRIME Phase 5: Hybrid Reasoning Engine orchestrator.
//
// Consumes a UniversalProblem (from ATLAS) and produces a ReasoningResult.
// Pipeline:
//   1. Build ReasoningContext (facts, goals, constraints, expressions from ATLAS)
//   2. Initialize facts from ATLAS entities
//   3. (HYBRID) Run symbolic evaluation on ATLAS expressions
//   4. (RULE)   Run forward chaining over rule base
//   5. (HYBRID) Re-run rules after symbolic evaluation
//   6. Check constraints
//   7. Check goals
//   8. Validate result
//   9. Return ReasoningResult with trace

#include "reasoning_types.hpp"
#include "expression_evaluator.hpp"
#include "rule_engine.hpp"
#include "constraint_engine.hpp"
#include "symbolic_engine.hpp"

#include <sstream>
#include <chrono>
#include <algorithm>
#include <mutex>

namespace prime::reasoning {

// ---------------------------------------------------------------------------
// ReasoningContext — per-call isolated state
// ---------------------------------------------------------------------------
struct ReasoningContext {
    std::unordered_map<std::string, Fact>   facts;
    std::vector<Rule>                        rules;
    std::vector<ReasoningConstraint>         constraints;
    std::vector<Goal>                        goals;
    std::vector<AtlasExpression>             expressions;
    ReasoningConfig                          config;
    ReasoningTrace                           trace;
    std::string                              problem_domain;
    std::string                              problem_id;
};

// ---------------------------------------------------------------------------
// ATLAS → ReasoningContext converter
// ---------------------------------------------------------------------------
// These helpers convert the Python-side dict representation into C++ structs.
// They are called from the Python binding layer.

// Fact: from ATLAS entity dict
inline Fact entity_to_fact(
        const std::string& entity_id,
        const std::string& entity_name,
        const std::string& entity_type,
        bool               has_quantity,
        double             quantity_value,
        const std::string& quantity_unit,
        double             confidence)
{
    Fact f;
    f.id         = entity_id;
    f.name       = entity_name;
    f.confidence = confidence;
    f.source     = "atlas";
    f.unit       = quantity_unit;

    if (has_quantity) {
        f.value = FactValue{quantity_value};
        f.bound = true;
    } else {
        f.value = FactValue{entity_name}; // unbound: store name as string
        f.bound = false;
    }
    (void)entity_type; // may be used for richer typing in later phases
    return f;
}

// Constraint: from ATLAS constraint dict
inline ReasoningConstraint atlas_constraint_to_reasoning(
        const std::string& id,
        const std::string& left,
        const std::string& op_str,
        const std::string& right,
        double confidence)
{
    ReasoningConstraint c;
    c.id         = id;
    c.left       = left;
    c.right      = right;
    c.confidence = confidence;

    // Parse op
    if      (op_str == "="  || op_str == "==") c.op = CondOp::EQ;
    else if (op_str == "!=" || op_str == "<>") c.op = CondOp::NEQ;
    else if (op_str == "<")                     c.op = CondOp::LT;
    else if (op_str == "<=")                    c.op = CondOp::LTE;
    else if (op_str == ">")                     c.op = CondOp::GT;
    else if (op_str == ">=")                    c.op = CondOp::GTE;
    else                                         c.op = CondOp::EQ;

    // Try to parse right as numeric
    try {
        size_t idx;
        double d = std::stod(right, &idx);
        if (idx == right.size()) {
            c.right_is_numeric = true;
            c.right_numeric    = d;
        }
    } catch (...) {}

    return c;
}

// ---------------------------------------------------------------------------
// ReasoningEngine
// ---------------------------------------------------------------------------
class ReasoningEngine {
public:
    ReasoningEngine() = default;

    // The main entrypoint. Accepts a fully-populated context.
    ReasoningResult solve(ReasoningContext ctx) {
        auto start = std::chrono::steady_clock::now();
        ReasoningResult result;
        result.mode_used = mode_to_string(ctx.config.mode);

        // Validate input
        if (ctx.goals.empty() && ctx.expressions.empty() && ctx.rules.empty() && ctx.constraints.empty()) {
            result.status        = ReasoningStatus::INVALID_INPUT;
            result.error_message = "No goals, expressions, rules, or constraints to reason over";
            return result;
        }

        // --- Step 1: initialize facts from context ---
        init_facts(ctx);

        // --- Step 2: symbolic evaluation (expressions from ATLAS) ---
        bool use_symbolic = ctx.config.use_symbolic &&
                            (ctx.config.mode == ReasoningMode::SYMBOLIC ||
                             ctx.config.mode == ReasoningMode::HYBRID   ||
                             ctx.config.mode == ReasoningMode::LLM_ASSISTED);
        if (use_symbolic) {
            run_symbolic(ctx);
        }

        // --- Step 3: rule-based forward chaining ---
        bool use_rules = ctx.config.use_rules &&
                         (ctx.config.mode == ReasoningMode::RULE   ||
                          ctx.config.mode == ReasoningMode::HYBRID ||
                          ctx.config.mode == ReasoningMode::LLM_ASSISTED);
        if (use_rules && !ctx.rules.empty()) {
            run_rules(ctx);
        }

        // --- Step 4: second pass symbolic after rules (hybrid) ---
        if (use_symbolic && use_rules && !ctx.rules.empty()) {
            run_symbolic(ctx); // re-run: rules may have added new facts
        }

        // --- Step 5: constraint checking ---
        result.constraint_results = constraint_engine_.check_all(
            ctx.constraints, ctx.facts, ctx.trace);

        // --- Step 6: goal checking ---
        check_goals(ctx, result);

        // --- Step 7: limit check ---
        auto elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (ctx.trace.total_steps >= ctx.config.max_steps) {
            result.status = ReasoningStatus::REASONING_LIMIT_REACHED;
        } else if (elapsed_ms > ctx.config.timeout_ms) {
            result.status = ReasoningStatus::REASONING_LIMIT_REACHED;
            result.error_message = "Timeout exceeded";
        } else if (result.status == ReasoningStatus::UNRESOLVED) {
            // keep UNRESOLVED unless overridden by constraint violation
            if (constraint_engine_.any_violated(result.constraint_results))
                result.status = ReasoningStatus::CONSTRAINT_VIOLATION;
        }

        // --- Step 8: collect derived facts ---
        for (auto& [k, v] : ctx.facts) {
            if (v.source != "atlas" && v.source != "initial")
                result.derived_facts[k] = v;
        }

        // Copy trace
        result.trace = std::move(ctx.trace);
        return result;
    }

    // Add user-defined rules to the engine's persistent rule base.
    bool add_rule(Rule rule) { return rule_engine_.add_rule(std::move(rule)); }
    void clear_rules()       { rule_engine_.clear_rules(); }
    size_t rule_count() const { return rule_engine_.rule_count(); }

    const ExpressionEngine& expr_engine() const { return expr_engine_; }

private:
    RuleEngine       rule_engine_;
    ConstraintEngine constraint_engine_;
    SymbolicEngine   symbolic_engine_;
    ExpressionEngine expr_engine_;

    // --- Initialize facts from ATLAS entities and any pre-bound values ---
    void init_facts(ReasoningContext& ctx) {
        // Any already-bound facts get a FACT_INIT step
        for (auto& [name, fact] : ctx.facts) {
            if (fact.bound && fact_value_is_numeric(fact.value)) {
                ReasoningStep step;
                step.type        = StepType::FACT_INIT;
                step.operation   = "INIT";
                step.output_facts = { name + "=" + fact_value_to_string(fact.value) +
                                      (fact.unit.empty() ? "" : " " + fact.unit) };
                step.confidence   = fact.confidence;
                step.description  = "Initialize: " + name + " = " +
                                    fact_value_to_string(fact.value) +
                                    (fact.unit.empty() ? "" : " " + fact.unit);
                ctx.trace.add_step(step);
            }
        }
    }

    // --- Run symbolic evaluation over all ATLAS expressions ---
    void run_symbolic(ReasoningContext& ctx) {
        for (auto& expr : ctx.expressions) {
            std::string err;
            symbolic_engine_.evaluate_expression(expr, ctx.facts, ctx.trace, err);
            // If it failed, add an error step but continue
            if (!err.empty()) {
                ReasoningStep step;
                step.type        = StepType::ERROR;
                step.operation   = "SYMBOLIC_FAIL";
                step.description = "Expression '" + expr.id + "' failed: " + err;
                step.success     = false;
                ctx.trace.add_step(step);
            }
        }
    }

    // --- Run rule engine forward chaining ---
    void run_rules(ReasoningContext& ctx) {
        // Load per-call rules into a local engine
        RuleEngine local_engine = rule_engine_; // copy global rules
        for (auto& r : ctx.rules)
            local_engine.add_rule(r);

        local_engine.forward_chain(ctx.facts, ctx.trace, ctx.config);
    }

    // --- Check goals against derived facts ---
    void check_goals(ReasoningContext& ctx, ReasoningResult& result) {
        bool all_satisfied = !ctx.goals.empty();
        bool any_satisfied = false;

        for (auto& goal : ctx.goals) {
            GoalResult gr;
            gr.id     = goal.id;
            gr.target = goal.target;

            // Look for the goal target in facts
            bool found = false;

            // Try direct name match
            auto it = ctx.facts.find(goal.target);
            if (it != ctx.facts.end() && it->second.bound) {
                gr.derived_value = fact_value_to_string(it->second.value);
                gr.unit          = it->second.unit;
                gr.confidence    = it->second.confidence;
                gr.satisfied     = true;
                found            = true;
            }

            // Try by scanning fact names (ATLAS may use entity IDs as target)
            if (!found) {
                for (auto& [k, v] : ctx.facts) {
                    if (v.name == goal.target && v.bound) {
                        gr.derived_value = fact_value_to_string(v.value);
                        gr.unit          = v.unit;
                        gr.confidence    = v.confidence;
                        gr.satisfied     = true;
                        found            = true;
                        break;
                    }
                }
            }

            // If still not found, try fuzzy match (substring)
            if (!found) {
                std::string tgt_lower = goal.target;
                std::transform(tgt_lower.begin(), tgt_lower.end(), tgt_lower.begin(), ::tolower);
                for (auto& [k, v] : ctx.facts) {
                    if (!v.bound) continue;
                    std::string kl = k;
                    std::transform(kl.begin(), kl.end(), kl.begin(), ::tolower);
                    std::string nl = v.name;
                    std::transform(nl.begin(), nl.end(), nl.begin(), ::tolower);
                    if (kl.find(tgt_lower) != std::string::npos ||
                        nl.find(tgt_lower) != std::string::npos ||
                        tgt_lower.find(kl) != std::string::npos ||
                        tgt_lower.find(nl) != std::string::npos) {
                        gr.derived_value = fact_value_to_string(v.value);
                        gr.unit          = v.unit;
                        gr.confidence    = v.confidence;
                        gr.satisfied     = true;
                        found            = true;
                        break;
                    }
                }
            }

            if (!gr.satisfied) all_satisfied = false;
            else               any_satisfied = true;

            // Add goal check step
            ReasoningStep step;
            step.type        = StepType::GOAL_CHECK;
            step.operation   = "GOAL_CHECK";
            step.confidence  = gr.confidence;
            step.success     = gr.satisfied;
            step.output_facts = { goal.target };
            if (gr.satisfied)
                step.description = "Goal '" + goal.target + "' satisfied: " +
                                   gr.derived_value +
                                   (gr.unit.empty() ? "" : " " + gr.unit);
            else
                step.description = "Goal '" + goal.target + "' not established";
            ctx.trace.add_step(step);

            result.goal_results.push_back(gr);
        }

        // Set answer from the first satisfied goal
        for (auto& gr : result.goal_results) {
            if (gr.satisfied) {
                result.answer            = gr.derived_value;
                result.answer_unit       = gr.unit;
                result.confidence        = gr.confidence;
                try {
                    size_t idx;
                    double d = std::stod(gr.derived_value, &idx);
                    if (idx == gr.derived_value.size()) {
                        result.answer_numeric    = d;
                        result.answer_is_numeric = true;
                    }
                } catch (...) {}
                break;
            }
        }

        // Set final status
        if (ctx.goals.empty()) {
            // No explicit goals: SOLVED if any derivation happened OR
            // if we only had constraints and all are satisfied.
            bool constraints_ok = !ctx.constraints.empty() &&
                                  !constraint_engine_.any_violated(result.constraint_results);
            bool has_unknowns   = false;
            for (auto& cr : result.constraint_results)
                if (cr.result == ConstraintResult::UNKNOWN) { has_unknowns = true; break; }

            if (!result.derived_facts.empty() || ctx.trace.expressions_evaluated > 0) {
                result.status = ReasoningStatus::SOLVED;
                // Set answer from derived facts
                if (result.answer.empty() && !ctx.facts.empty()) {
                    for (auto& [k, v] : ctx.facts) {
                        if (v.source != "atlas" && v.source != "initial" && v.bound) {
                            result.answer      = fact_value_to_string(v.value);
                            result.answer_unit = v.unit;
                            try {
                                result.answer_numeric    = fact_value_to_double(v.value);
                                result.answer_is_numeric = fact_value_is_numeric(v.value);
                            } catch (...) {}
                            break;
                        }
                    }
                }
            } else if (ctx.trace.rules_fired > 0) {
                result.status = ReasoningStatus::SOLVED;
            } else if (constraints_ok && !has_unknowns) {
                result.status = ReasoningStatus::SOLVED;
                result.answer = "constraints_satisfied";
            } else {
                result.status = ReasoningStatus::UNRESOLVED;
            }
        } else if (all_satisfied) {
            result.status = ReasoningStatus::SOLVED;
        } else if (any_satisfied) {
            result.status = ReasoningStatus::PARTIALLY_SOLVED;
        } else {
            result.status = ReasoningStatus::UNRESOLVED;
        }

        if (constraint_engine_.any_violated(result.constraint_results) &&
            result.status == ReasoningStatus::SOLVED)
            result.status = ReasoningStatus::CONSTRAINT_VIOLATION;
    }
};

} // namespace prime::reasoning
