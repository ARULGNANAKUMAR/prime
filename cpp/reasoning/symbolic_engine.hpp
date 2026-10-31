#pragma once
// symbolic_engine.hpp — Symbolic arithmetic reasoning for PRIME Phase 5.
//
// Evaluates ATLAS expressions (op + operand_ids → output_id) against a
// fact map, supporting multi-step chained evaluation.

#include "reasoning_types.hpp"
#include "expression_evaluator.hpp"
#include <sstream>
#include <cmath>

namespace prime::reasoning {

// ---------------------------------------------------------------------------
// AtlasExpression — mirrors the ATLAS dict structure for one expression
// ---------------------------------------------------------------------------
struct AtlasExpression {
    std::string              id;
    std::string              op;          // "add", "subtract", "multiply", "divide", etc.
    std::string              output_id;   // entity ID that receives the result
    std::vector<std::string> operand_ids; // entity IDs used as inputs
    std::string              text;        // human description
};

// ---------------------------------------------------------------------------
// SymbolicEngine
// ---------------------------------------------------------------------------
class SymbolicEngine {
public:
    // Evaluate a single AtlasExpression, resolving operands from facts.
    // Returns true if the expression could be evaluated.
    bool evaluate_expression(
            const AtlasExpression& expr,
            std::unordered_map<std::string, Fact>& facts,
            ReasoningTrace& trace,
            std::string& error_out) const
    {
        // Try to resolve each operand to a numeric fact
        std::vector<double> operand_values;
        std::vector<std::string> operand_names;

        for (auto& oid : expr.operand_ids) {
            // First: try parsing as a numeric literal
            try {
                size_t idx;
                double d = std::stod(oid, &idx);
                if (idx == oid.size()) {
                    operand_values.push_back(d);
                    operand_names.push_back(oid);
                    continue;
                }
            } catch (...) {}

            auto it = facts.find(oid);
            if (it == facts.end()) {
                // Try by name (ATLAS sometimes uses "distance", "time" as operand IDs)
                bool found = false;
                for (auto& [k, v] : facts) {
                    if (v.name == oid && fact_value_is_numeric(v.value)) {
                        operand_values.push_back(fact_value_to_double(v.value));
                        operand_names.push_back(k);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    error_out = "Operand '" + oid + "' not found in facts";
                    return false;
                }
            } else {
                if (!fact_value_is_numeric(it->second.value)) {
                    error_out = "Operand '" + oid + "' is not numeric";
                    return false;
                }
                operand_values.push_back(fact_value_to_double(it->second.value));
                operand_names.push_back(oid);
            }
        }

        if (operand_values.empty()) {
            error_out = "No operands resolved for expression '" + expr.id + "'";
            return false;
        }

        // Perform the operation
        double result = 0.0;
        bool   ok     = true;

        std::string op = expr.op;
        std::transform(op.begin(), op.end(), op.begin(), ::tolower);

        if (op == "add" || op == "sum" || op == "plus" || op == "+") {
            result = 0.0;
            for (double v : operand_values) result += v;
        } else if (op == "subtract" || op == "minus" || op == "difference" || op == "-") {
            if (operand_values.size() < 2) { error_out = "subtract requires 2 operands"; return false; }
            result = operand_values[0];
            for (size_t i = 1; i < operand_values.size(); ++i) result -= operand_values[i];
        } else if (op == "multiply" || op == "product" || op == "times" || op == "*") {
            result = 1.0;
            for (double v : operand_values) result *= v;
        } else if (op == "divide" || op == "quotient" || op == "/") {
            if (operand_values.size() < 2) { error_out = "divide requires 2 operands"; return false; }
            if (operand_values[1] == 0.0) { error_out = "Division by zero"; return false; }
            result = operand_values[0] / operand_values[1];
        } else if (op == "power" || op == "pow" || op == "^") {
            if (operand_values.size() < 2) { error_out = "power requires 2 operands"; return false; }
            result = std::pow(operand_values[0], operand_values[1]);
        } else if (op == "sqrt") {
            if (operand_values.empty()) { error_out = "sqrt requires 1 operand"; return false; }
            if (operand_values[0] < 0) { error_out = "sqrt of negative number"; return false; }
            result = std::sqrt(operand_values[0]);
        } else if (op == "modulo" || op == "mod" || op == "%") {
            if (operand_values.size() < 2) { error_out = "modulo requires 2 operands"; return false; }
            if (operand_values[1] == 0.0) { error_out = "Modulo by zero"; return false; }
            result = std::fmod(operand_values[0], operand_values[1]);
        } else if (op == "average" || op == "avg" || op == "mean") {
            if (operand_values.empty()) { error_out = "average requires operands"; return false; }
            result = 0.0;
            for (double v : operand_values) result += v;
            result /= static_cast<double>(operand_values.size());
        } else if (op == "max") {
            result = *std::max_element(operand_values.begin(), operand_values.end());
        } else if (op == "min") {
            result = *std::min_element(operand_values.begin(), operand_values.end());
        } else if (op == "abs") {
            if (operand_values.empty()) { error_out = "abs requires 1 operand"; return false; }
            result = std::abs(operand_values[0]);
        } else if (op == "negate" || op == "neg") {
            if (operand_values.empty()) { error_out = "negate requires 1 operand"; return false; }
            result = -operand_values[0];
        } else if (op == "calculate" || op == "compute" || op == "find" || op == "evaluate") {
            // Generic: try divide (most common: distance/time)
            if (operand_values.size() == 2 && operand_values[1] != 0.0)
                result = operand_values[0] / operand_values[1];
            else if (operand_values.size() == 1)
                result = operand_values[0];
            else {
                error_out = "Cannot determine operation for op='" + expr.op + "'";
                return false;
            }
        } else {
            // Try the ExpressionEngine as a last resort
            try {
                ExpressionEngine ee;
                std::unordered_map<std::string, double> bindings;
                // Bind operand names to their values
                for (size_t i = 0; i < expr.operand_ids.size() && i < operand_values.size(); ++i)
                    bindings[expr.operand_ids[i]] = operand_values[i];
                auto tree = ee.compile(expr.op);
                result = ee.evaluate(tree, bindings);
            } catch (...) {
                error_out = "Unknown operation: '" + expr.op + "'";
                return false;
            }
        }

        if (!ok) return false;

        // Store result
        Fact out_fact;
        out_fact.name       = expr.output_id;
        out_fact.id         = "sym_" + expr.id;
        out_fact.value      = FactValue{result};
        out_fact.source     = "expression:" + expr.id;
        out_fact.confidence = 0.95;
        out_fact.bound      = true;
        // Inherit unit from first operand if available
        for (auto& oid : operand_names) {
            auto it = facts.find(oid);
            if (it != facts.end() && !it->second.unit.empty()) {
                out_fact.unit = it->second.unit;
                break;
            }
        }

        facts[expr.output_id] = out_fact;
        // Also store under entity name for lookup
        if (!expr.output_id.empty())
            facts[out_fact.name] = out_fact;

        // Build trace step
        ReasoningStep step;
        step.type      = StepType::EXPRESSION_EVAL;
        step.operation = expr.op;
        step.input_facts = operand_names;
        step.output_facts = { expr.output_id };
        step.confidence   = 0.95;
        step.success      = true;

        // Build human-readable description
        std::ostringstream desc;
        desc << "Evaluate: ";
        for (size_t i = 0; i < operand_names.size(); ++i) {
            if (i) desc << " " << expr.op << " ";
            desc << operand_names[i] << "=" << operand_values[i];
        }
        char rbuf[64];
        snprintf(rbuf, sizeof(rbuf), "%g", result);
        desc << " → " << expr.output_id << "=" << rbuf;
        step.description = desc.str();

        trace.add_step(step);
        return true;
    }

    // Evaluate a free-form expression string with bound variables.
    bool evaluate_string_expression(
            const std::string& expr_str,
            const std::string& output_name,
            const std::unordered_map<std::string, double>& bindings,
            double& result_out,
            std::string& error_out) const
    {
        try {
            ExpressionEngine ee;
            result_out = ee.eval(expr_str, bindings);
            return true;
        } catch (const ExprParseError& e) {
            error_out = e.what();
        } catch (const ExprEvalError& e) {
            error_out = e.what();
        } catch (const std::exception& e) {
            error_out = e.what();
        }
        return false;
    }

    // Build a binding map from the current fact store (numeric only).
    static std::unordered_map<std::string, double>
    build_bindings(const std::unordered_map<std::string, Fact>& facts) {
        std::unordered_map<std::string, double> bindings;
        for (auto& [k, v] : facts) {
            if (fact_value_is_numeric(v.value))
                bindings[k] = fact_value_to_double(v.value);
            // Also bind by human name
            if (!v.name.empty() && fact_value_is_numeric(v.value))
                bindings[v.name] = fact_value_to_double(v.value);
        }
        return bindings;
    }
};

} // namespace prime::reasoning
