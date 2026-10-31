#pragma once
// constraint_engine.hpp — Constraint satisfaction checker for PRIME Phase 5.
//
// Evaluates ATLAS constraints against derived facts.
// Does NOT perform constraint propagation (that is Phase 9+).

#include "reasoning_types.hpp"
#include <sstream>

namespace prime::reasoning {

class ConstraintEngine {
public:
    // Check a single constraint against the fact map.
    ConstraintSatisfactionResult check(
            const ReasoningConstraint& c,
            const std::unordered_map<std::string, Fact>& facts) const
    {
        ConstraintSatisfactionResult res;
        res.id = c.id;

        auto lit = facts.find(c.left);
        if (lit == facts.end()) {
            res.result      = ConstraintResult::UNKNOWN;
            res.description = "Fact '" + c.left + "' not found";
            return res;
        }

        if (!fact_value_is_numeric(lit->second.value)) {
            // String/bool equality constraints
            std::string lv = fact_value_to_string(lit->second.value);
            bool satisfied = false;
            switch (c.op) {
                case CondOp::EQ:  satisfied = (lv == c.right); break;
                case CondOp::NEQ: satisfied = (lv != c.right); break;
                default:
                    res.result      = ConstraintResult::UNKNOWN;
                    res.description = "Non-numeric comparison '" + cond_op_to_string(c.op)
                                      + "' unsupported for string fact";
                    return res;
            }
            res.result      = satisfied ? ConstraintResult::SATISFIED
                                        : ConstraintResult::VIOLATED;
            res.description = c.left + " " + cond_op_to_string(c.op) + " " + c.right
                              + (satisfied ? " — satisfied" : " — violated");
            return res;
        }

        double lv = fact_value_to_double(lit->second.value);
        double rv = c.right_is_numeric ? c.right_numeric : parse_rhs(c.right, facts);

        bool satisfied = false;
        switch (c.op) {
            case CondOp::EQ:  satisfied = (lv == rv); break;
            case CondOp::NEQ: satisfied = (lv != rv); break;
            case CondOp::LT:  satisfied = (lv <  rv); break;
            case CondOp::LTE: satisfied = (lv <= rv); break;
            case CondOp::GT:  satisfied = (lv >  rv); break;
            case CondOp::GTE: satisfied = (lv >= rv); break;
            default: break;
        }

        res.result = satisfied ? ConstraintResult::SATISFIED : ConstraintResult::VIOLATED;
        char buf[128];
        snprintf(buf, sizeof(buf), "%g %s %g — %s",
                 lv, cond_op_to_string(c.op).c_str(), rv,
                 satisfied ? "satisfied" : "violated");
        res.description = c.left + ": " + buf;
        return res;
    }

    // Check all constraints and append steps to trace.
    std::vector<ConstraintSatisfactionResult> check_all(
            const std::vector<ReasoningConstraint>& constraints,
            const std::unordered_map<std::string, Fact>& facts,
            ReasoningTrace& trace) const
    {
        std::vector<ConstraintSatisfactionResult> results;
        for (auto& c : constraints) {
            auto r = check(c, facts);
            results.push_back(r);

            ReasoningStep step;
            step.type       = StepType::CONSTRAINT_CHECK;
            step.operation  = "CONSTRAINT_CHECK";
            step.input_facts = { c.left };
            step.confidence  = c.confidence;
            step.success     = (r.result != ConstraintResult::VIOLATED);
            step.description = r.description;
            trace.add_step(step);
        }
        return results;
    }

    bool any_violated(const std::vector<ConstraintSatisfactionResult>& results) const {
        for (auto& r : results)
            if (r.result == ConstraintResult::VIOLATED) return true;
        return false;
    }

private:
    double parse_rhs(const std::string& s,
                     const std::unordered_map<std::string, Fact>& facts) const {
        // Check if it's a fact reference
        auto it = facts.find(s);
        if (it != facts.end() && fact_value_is_numeric(it->second.value))
            return fact_value_to_double(it->second.value);
        // Parse as literal
        try { return std::stod(s); } catch (...) {}
        return 0.0;
    }
};

} // namespace prime::reasoning
