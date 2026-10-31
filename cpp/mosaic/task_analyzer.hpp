#pragma once
// task_analyzer.hpp — Analyzes an ATLAS problem and extracts TaskRequirements.
//
// MOSAIC uses the task requirements to drive module selection.
// This component does NOT solve the problem — it only identifies what
// capabilities are needed.

#include "mosaic_types.hpp"
#include <algorithm>
#include <cctype>

namespace prime::mosaic {

class TaskAnalyzer {
public:
    // Analyze an ATLAS problem dict (passed as structured fields) and return
    // TaskRequirements. All inputs are plain strings/lists for binding simplicity.
    TaskRequirements analyze(
            const std::string& task_id,
            const std::string& domain,
            bool  has_expressions,
            bool  has_rules,
            bool  has_constraints,
            bool  has_goals,
            bool  has_entities,
            bool  has_logic_clauses,
            bool  has_relationships,
            const std::vector<std::string>& operation_types,
            const std::vector<std::string>& goal_targets,
            int   entity_count,
            bool  has_memory_context) const
    {
        TaskRequirements req;
        req.task_id    = task_id;
        req.domain     = domain;
        req.expected_input_types  = {"atlas_problem"};
        req.expected_output_types = {"reasoning_result"};

        // ATLAS representation is always required
        req.required_capabilities.push_back("atlas_representation");

        // Expressions → symbolic/arithmetic reasoning
        if (has_expressions || has_arithmetic_ops(operation_types) || entity_count >= 2) {
            req.required_capabilities.push_back("symbolic_reasoning");
            req.required_capabilities.push_back("arithmetic");
            req.has_arithmetic = true;
        }

        // Rules / logic clauses → rule reasoning
        if (has_rules || has_logic_clauses) {
            req.required_capabilities.push_back("rule_reasoning");
            req.has_rules = true;
        }

        // Constraints → constraint checking
        if (has_constraints) {
            req.required_capabilities.push_back("constraint_checking");
            req.has_constraints = true;
        }

        // Memory context
        if (has_memory_context) {
            req.required_capabilities.push_back("memory_retrieval");
            req.needs_memory = true;
        }

        // Validation is always useful when there are goals
        if (has_goals || !goal_targets.empty()) {
            req.optional_capabilities.push_back("validation");
        }

        // Language-heavy domains without structure → might benefit from LLM
        bool is_language_task = (domain == "general" || domain == "unknown") &&
                                !has_expressions && !has_rules && entity_count < 2;
        if (is_language_task) {
            req.optional_capabilities.push_back("text_generation");
            req.needs_llm = true;
        }

        // General reasoning always required
        req.required_capabilities.push_back("reasoning");

        // Deduplicate
        deduplicate(req.required_capabilities);
        deduplicate(req.optional_capabilities);

        return req;
    }

private:
    static bool has_arithmetic_ops(const std::vector<std::string>& ops) {
        static const std::unordered_set<std::string> arith_ops = {
            "calculate", "compute", "find", "divide", "multiply", "add",
            "subtract", "average", "sum", "product", "evaluate"
        };
        for (auto& op : ops) {
            std::string lo = op;
            std::transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
            if (arith_ops.count(lo)) return true;
        }
        return false;
    }

    static void deduplicate(std::vector<std::string>& v) {
        std::unordered_set<std::string> seen;
        std::vector<std::string> out;
        for (auto& s : v)
            if (seen.insert(s).second) out.push_back(s);
        v = std::move(out);
    }
};

} // namespace prime::mosaic
