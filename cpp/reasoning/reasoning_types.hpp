#pragma once
// reasoning_types.hpp — Core data types for the PRIME Phase 5 Reasoning Engine.
//
// All types are plain C++17 structs with no external dependencies.
// The reasoning engine consumes UniversalProblem representations from ATLAS
// and produces ReasoningResult objects.

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <optional>
#include <cstdint>
#include <functional>
#include <cstdio>

namespace prime::reasoning {

// ---------------------------------------------------------------------------
// Reasoning status
// ---------------------------------------------------------------------------
enum class ReasoningStatus : uint8_t {
    SOLVED                 = 0,
    PARTIALLY_SOLVED       = 1,
    UNRESOLVED             = 2,
    INVALID_INPUT          = 3,
    CONSTRAINT_VIOLATION   = 4,
    REASONING_LIMIT_REACHED = 5,
    CIRCULAR_REASONING     = 6,
    LLM_UNAVAILABLE        = 7,
    INTERNAL_ERROR         = 8,
};

inline std::string status_to_string(ReasoningStatus s) {
    switch (s) {
        case ReasoningStatus::SOLVED:                  return "SOLVED";
        case ReasoningStatus::PARTIALLY_SOLVED:        return "PARTIALLY_SOLVED";
        case ReasoningStatus::UNRESOLVED:              return "UNRESOLVED";
        case ReasoningStatus::INVALID_INPUT:           return "INVALID_INPUT";
        case ReasoningStatus::CONSTRAINT_VIOLATION:    return "CONSTRAINT_VIOLATION";
        case ReasoningStatus::REASONING_LIMIT_REACHED: return "REASONING_LIMIT_REACHED";
        case ReasoningStatus::CIRCULAR_REASONING:      return "CIRCULAR_REASONING";
        case ReasoningStatus::LLM_UNAVAILABLE:         return "LLM_UNAVAILABLE";
        case ReasoningStatus::INTERNAL_ERROR:          return "INTERNAL_ERROR";
        default:                                       return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Reasoning mode
// ---------------------------------------------------------------------------
enum class ReasoningMode : uint8_t {
    RULE        = 0,
    SYMBOLIC    = 1,
    HYBRID      = 2,
    LLM_ASSISTED = 3,
};

inline std::string mode_to_string(ReasoningMode m) {
    switch (m) {
        case ReasoningMode::RULE:         return "RULE";
        case ReasoningMode::SYMBOLIC:     return "SYMBOLIC";
        case ReasoningMode::HYBRID:       return "HYBRID";
        case ReasoningMode::LLM_ASSISTED: return "LLM_ASSISTED";
        default:                          return "HYBRID";
    }
}

// ---------------------------------------------------------------------------
// Value — typed fact value
// ---------------------------------------------------------------------------
using FactValue = std::variant<double, std::string, bool>;

inline std::string fact_value_to_string(const FactValue& v) {
    if (std::holds_alternative<double>(v)) {
        double d = std::get<double>(v);
        // Format cleanly: no trailing zeros for whole numbers
        char buf[64];
        if (d == static_cast<long long>(d) && d >= -1e15 && d <= 1e15) {
            snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(d));
        } else {
            snprintf(buf, sizeof(buf), "%g", d);
        }
        return buf;
    } else if (std::holds_alternative<std::string>(v)) {
        return std::get<std::string>(v);
    } else {
        return std::get<bool>(v) ? "true" : "false";
    }
}

inline bool fact_value_is_numeric(const FactValue& v) {
    return std::holds_alternative<double>(v);
}

inline double fact_value_to_double(const FactValue& v) {
    if (std::holds_alternative<double>(v)) return std::get<double>(v);
    if (std::holds_alternative<bool>(v))   return std::get<bool>(v) ? 1.0 : 0.0;
    return 0.0;
}

// ---------------------------------------------------------------------------
// Fact — a named value with optional unit and confidence
// ---------------------------------------------------------------------------
struct Fact {
    std::string  id;
    std::string  name;
    FactValue    value;
    std::string  unit;
    double       confidence = 1.0;
    std::string  source;    // "atlas", "rule", "expression", "initial", "derived"
    bool         bound  = true;
};

// ---------------------------------------------------------------------------
// Condition — one side of a rule condition
// ---------------------------------------------------------------------------
enum class CondOp : uint8_t {
    EQ, NEQ, LT, LTE, GT, GTE,
    AND, OR, NOT,
    EXISTS,
};

inline std::string cond_op_to_string(CondOp op) {
    switch (op) {
        case CondOp::EQ:     return "=";
        case CondOp::NEQ:    return "!=";
        case CondOp::LT:     return "<";
        case CondOp::LTE:    return "<=";
        case CondOp::GT:     return ">";
        case CondOp::GTE:    return ">=";
        case CondOp::AND:    return "AND";
        case CondOp::OR:     return "OR";
        case CondOp::NOT:    return "NOT";
        case CondOp::EXISTS: return "EXISTS";
        default:             return "?";
    }
}

struct Condition {
    std::string left;         // fact name or nested condition reference
    CondOp      op   = CondOp::EQ;
    std::string right;        // literal or fact name
    bool        right_is_fact = false;
    std::vector<Condition> sub_conditions; // for AND/OR/NOT
};

// ---------------------------------------------------------------------------
// Conclusion — what a rule derives when it fires
// ---------------------------------------------------------------------------
struct Conclusion {
    std::string name;         // fact name to set
    std::string value_expr;   // literal value or simple expression
    bool        is_numeric = false;
    double      numeric_value = 0.0;
    std::string unit;
    double      confidence = 1.0;
};

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------
struct Rule {
    std::string            id;
    std::string            name;
    std::vector<Condition> conditions;
    std::vector<Conclusion> conclusions;
    int                    priority   = 0;
    double                 confidence = 1.0;
    bool                   fired      = false;
    std::string            metadata;
};

// ---------------------------------------------------------------------------
// Expression AST — safe arithmetic expression tree
// ---------------------------------------------------------------------------
enum class ExprNodeType : uint8_t {
    NUMBER,
    VARIABLE,
    BINOP,
    UNOP,
    FUNCTION,
};

enum class BinOp : uint8_t {
    ADD, SUB, MUL, DIV, MOD, POW,
};

inline std::string binop_to_string(BinOp op) {
    switch (op) {
        case BinOp::ADD: return "+";
        case BinOp::SUB: return "-";
        case BinOp::MUL: return "*";
        case BinOp::DIV: return "/";
        case BinOp::MOD: return "%";
        case BinOp::POW: return "^";
        default:         return "?";
    }
}

enum class UnOp : uint8_t { NEG, ABS };

// Forward declaration
struct ExprNode;
using ExprNodePtr = std::shared_ptr<ExprNode>;

struct ExprNode {
    ExprNodeType type;
    // NUMBER
    double       number = 0.0;
    // VARIABLE
    std::string  var_name;
    // BINOP
    BinOp        binop = BinOp::ADD;
    ExprNodePtr  left;
    ExprNodePtr  right;
    // UNOP
    UnOp         unop = UnOp::NEG;
    ExprNodePtr  child;
    // FUNCTION
    std::string  func_name;
    std::vector<ExprNodePtr> func_args;
};

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------
struct ReasoningConstraint {
    std::string id;
    std::string left;
    CondOp      op = CondOp::LTE;
    std::string right;
    bool        right_is_numeric = false;
    double      right_numeric    = 0.0;
    double      confidence       = 1.0;
};

enum class ConstraintResult : uint8_t {
    SATISFIED, VIOLATED, UNKNOWN
};

// ---------------------------------------------------------------------------
// Goal
// ---------------------------------------------------------------------------
struct Goal {
    std::string id;
    std::string target;
    std::string description;
    int         priority  = 0;
    double      confidence = 1.0;
};

// ---------------------------------------------------------------------------
// Reasoning step (node in the reasoning graph / trace)
// ---------------------------------------------------------------------------
enum class StepType : uint8_t {
    FACT_INIT,
    RULE_FIRE,
    EXPRESSION_EVAL,
    CONSTRAINT_CHECK,
    GOAL_CHECK,
    DEDUCTION,
    BINDING,
    ERROR,
};

inline std::string step_type_to_string(StepType t) {
    switch (t) {
        case StepType::FACT_INIT:        return "FACT_INIT";
        case StepType::RULE_FIRE:        return "RULE_FIRE";
        case StepType::EXPRESSION_EVAL:  return "EXPRESSION_EVAL";
        case StepType::CONSTRAINT_CHECK: return "CONSTRAINT_CHECK";
        case StepType::GOAL_CHECK:       return "GOAL_CHECK";
        case StepType::DEDUCTION:        return "DEDUCTION";
        case StepType::BINDING:          return "BINDING";
        case StepType::ERROR:            return "ERROR";
        default:                         return "STEP";
    }
}

struct ReasoningStep {
    int                      step_id    = 0;
    StepType                 type       = StepType::FACT_INIT;
    std::vector<std::string> input_facts;
    std::vector<std::string> output_facts;
    std::string              operation;
    std::string              rule_id;
    std::string              description;
    double                   confidence = 1.0;
    bool                     success    = true;
};

// ---------------------------------------------------------------------------
// Goal result
// ---------------------------------------------------------------------------
struct GoalResult {
    std::string id;
    std::string target;
    bool        satisfied = false;
    std::string derived_value;
    std::string unit;
    double      confidence = 1.0;
};

// ---------------------------------------------------------------------------
// Constraint satisfaction result
// ---------------------------------------------------------------------------
struct ConstraintSatisfactionResult {
    std::string    id;
    ConstraintResult result = ConstraintResult::UNKNOWN;
    std::string    description;
};

// ---------------------------------------------------------------------------
// ReasoningTrace — full record of reasoning steps
// ---------------------------------------------------------------------------
struct ReasoningTrace {
    std::vector<ReasoningStep> steps;
    int                        total_steps    = 0;
    int                        rules_fired    = 0;
    int                        expressions_evaluated = 0;
    int                        constraints_checked   = 0;

    void add_step(ReasoningStep s) {
        s.step_id = static_cast<int>(steps.size()) + 1;
        total_steps++;
        if (s.type == StepType::RULE_FIRE && s.success) rules_fired++;
        if (s.type == StepType::EXPRESSION_EVAL) expressions_evaluated++;
        if (s.type == StepType::CONSTRAINT_CHECK) constraints_checked++;
        steps.push_back(std::move(s));
    }
};

// ---------------------------------------------------------------------------
// ReasoningResult — top-level output
// ---------------------------------------------------------------------------
struct ReasoningResult {
    ReasoningStatus  status   = ReasoningStatus::UNRESOLVED;
    std::string      answer;
    std::string      answer_unit;
    double           answer_numeric = 0.0;
    bool             answer_is_numeric = false;
    double           confidence = 1.0;
    ReasoningTrace   trace;
    std::vector<GoalResult> goal_results;
    std::vector<ConstraintSatisfactionResult> constraint_results;
    std::unordered_map<std::string, Fact> derived_facts;
    std::string      error_message;
    std::string      mode_used;
};

// ---------------------------------------------------------------------------
// ReasoningConfig
// ---------------------------------------------------------------------------
struct ReasoningConfig {
    ReasoningMode mode               = ReasoningMode::HYBRID;
    int           max_steps          = 1000;
    int           max_depth          = 50;
    int           max_rule_firings   = 500;
    int           max_expression_size= 256;
    double        timeout_ms         = 5000.0;
    bool          use_rules          = true;
    bool          use_symbolic       = true;
    bool          use_llm            = false;
    double        confidence_threshold = 0.0;
    bool          detect_circular    = true;
};

} // namespace prime::reasoning
