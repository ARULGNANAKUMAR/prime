#pragma once
// mosaic_types.hpp — Core data types for PRIME Phase 6 MOSAIC.
//
// MOSAIC = Modular Orchestration / Selection and Adaptive Intelligence Composition
//
// Answers: "Which available intelligence modules are required for this task,
//           and in what order should they operate?"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <chrono>
#include <cstdint>
#include <variant>
#include <optional>

namespace prime::mosaic {

// ---------------------------------------------------------------------------
// Capability category
// ---------------------------------------------------------------------------
enum class CapabilityCategory : uint8_t {
    REPRESENTATION,    // ATLAS, input parsing
    RETRIEVAL,         // Memory lookup
    ARITHMETIC,        // Symbolic / numeric computation
    RULE_REASONING,    // Rule-based inference
    SYMBOLIC_REASONING,// Symbolic manipulation
    CONSTRAINT_CHECKING,// Constraint satisfaction
    LANGUAGE,          // Text generation / interpretation (LLM)
    VALIDATION,        // Output verification
    STORAGE,           // Memory write
    GENERAL,           // Generic / uncategorised
};

inline std::string category_to_string(CapabilityCategory c) {
    switch (c) {
        case CapabilityCategory::REPRESENTATION:     return "representation";
        case CapabilityCategory::RETRIEVAL:          return "retrieval";
        case CapabilityCategory::ARITHMETIC:         return "arithmetic";
        case CapabilityCategory::RULE_REASONING:     return "rule_reasoning";
        case CapabilityCategory::SYMBOLIC_REASONING: return "symbolic_reasoning";
        case CapabilityCategory::CONSTRAINT_CHECKING:return "constraint_checking";
        case CapabilityCategory::LANGUAGE:           return "language";
        case CapabilityCategory::VALIDATION:         return "validation";
        case CapabilityCategory::STORAGE:            return "storage";
        default:                                      return "general";
    }
}

// ---------------------------------------------------------------------------
// Capability
// ---------------------------------------------------------------------------
struct Capability {
    std::string          name;
    CapabilityCategory   category      = CapabilityCategory::GENERAL;
    std::vector<std::string> input_types;   // accepted input type names
    std::vector<std::string> output_types;  // produced output type names
    double               confidence    = 1.0;
    std::string          description;
};

// ---------------------------------------------------------------------------
// Resource cost estimate
// ---------------------------------------------------------------------------
struct ResourceCost {
    double cpu_relative    = 1.0;   // relative CPU units (1 = minimal)
    double memory_relative = 1.0;   // relative memory units
    double latency_ms      = 0.0;   // estimated latency
    bool   requires_model  = false; // requires a loaded LLM
};

// ---------------------------------------------------------------------------
// Module status
// ---------------------------------------------------------------------------
enum class ModuleStatus : uint8_t {
    AVAILABLE,
    UNAVAILABLE,       // e.g. LLM not loaded
    DEGRADED,
    DISABLED,
};

inline std::string module_status_to_string(ModuleStatus s) {
    switch (s) {
        case ModuleStatus::AVAILABLE:   return "available";
        case ModuleStatus::UNAVAILABLE: return "unavailable";
        case ModuleStatus::DEGRADED:    return "degraded";
        case ModuleStatus::DISABLED:    return "disabled";
        default:                         return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Module metadata
// ---------------------------------------------------------------------------
struct ModuleMetadata {
    std::string              id;
    std::string              name;
    std::string              version       = "1.0";
    std::vector<Capability>  capabilities;
    std::vector<std::string> input_types;
    std::vector<std::string> output_types;
    std::vector<std::string> dependencies;  // IDs of modules this depends on
    ResourceCost             cost;
    int                      priority      = 0;  // higher = preferred
    bool                     deterministic = true;
    bool                     is_fallback   = false;
    std::string              fallback_for;        // module ID this can fall back for
    ModuleStatus             status        = ModuleStatus::AVAILABLE;
    std::unordered_map<std::string, std::string> metadata;

    bool has_capability(const std::string& cap_name) const {
        for (auto& c : capabilities)
            if (c.name == cap_name) return true;
        return false;
    }

    bool accepts_input(const std::string& type) const {
        for (auto& t : input_types)
            if (t == type || t == "*") return true;
        return false;
    }

    bool produces_output(const std::string& type) const {
        for (auto& t : output_types)
            if (t == type || t == "*") return true;
        return false;
    }
};

// ---------------------------------------------------------------------------
// TaskRequirements — what MOSAIC extracts from the ATLAS problem
// ---------------------------------------------------------------------------
struct TaskRequirements {
    std::string              task_id;
    std::string              domain;
    std::vector<std::string> required_capabilities;
    std::vector<std::string> optional_capabilities;
    std::vector<std::string> expected_input_types;
    std::vector<std::string> expected_output_types;
    bool                     needs_memory   = false;
    bool                     needs_llm      = false;
    bool                     has_arithmetic = false;
    bool                     has_rules      = false;
    bool                     has_constraints= false;
    int                      priority       = 0;
};

// ---------------------------------------------------------------------------
// CompositionStatus
// ---------------------------------------------------------------------------
enum class CompositionStatus : uint8_t {
    VALID,
    INVALID,           // general invalidity
    COMPOSITION_CYCLE, // circular dependency detected
    MISSING_CAPABILITY,// no module satisfies a required capability
    INPUT_MISMATCH,    // module input/output types incompatible
    RESOURCE_EXCEEDED, // resource limits exceeded
    EMPTY_PLAN,        // no modules selected
};

inline std::string composition_status_to_string(CompositionStatus s) {
    switch (s) {
        case CompositionStatus::VALID:               return "VALID";
        case CompositionStatus::INVALID:             return "COMPOSITION_INVALID";
        case CompositionStatus::COMPOSITION_CYCLE:   return "COMPOSITION_CYCLE";
        case CompositionStatus::MISSING_CAPABILITY:  return "MISSING_CAPABILITY";
        case CompositionStatus::INPUT_MISMATCH:      return "INPUT_MISMATCH";
        case CompositionStatus::RESOURCE_EXCEEDED:   return "RESOURCE_EXCEEDED";
        case CompositionStatus::EMPTY_PLAN:          return "EMPTY_PLAN";
        default:                                      return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// CompositionPlan — the selected, ordered pipeline
// ---------------------------------------------------------------------------
struct CompositionPlan {
    std::string              task_id;
    std::string              domain;
    std::vector<std::string> module_ids;        // ordered execution sequence
    std::vector<std::string> selection_reasons; // one per module
    double                   estimated_cost     = 0.0;
    CompositionStatus        status             = CompositionStatus::EMPTY_PLAN;
    std::string              status_detail;
    TaskRequirements         requirements;
    std::unordered_map<std::string, std::string> metadata;

    bool valid() const { return status == CompositionStatus::VALID; }
};

// ---------------------------------------------------------------------------
// ExecutionStatus
// ---------------------------------------------------------------------------
enum class ExecutionStatus : uint8_t {
    SUCCESS,
    PARTIAL_SUCCESS,
    FAILURE,
    MODULE_ERROR,
    FALLBACK_USED,
    LIMIT_EXCEEDED,
    INVALID_PLAN,
};

inline std::string execution_status_to_string(ExecutionStatus s) {
    switch (s) {
        case ExecutionStatus::SUCCESS:         return "SUCCESS";
        case ExecutionStatus::PARTIAL_SUCCESS: return "PARTIAL_SUCCESS";
        case ExecutionStatus::FAILURE:         return "FAILURE";
        case ExecutionStatus::MODULE_ERROR:    return "MODULE_ERROR";
        case ExecutionStatus::FALLBACK_USED:   return "FALLBACK_USED";
        case ExecutionStatus::LIMIT_EXCEEDED:  return "LIMIT_EXCEEDED";
        case ExecutionStatus::INVALID_PLAN:    return "INVALID_PLAN";
        default:                               return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// CompositionTraceStep
// ---------------------------------------------------------------------------
struct CompositionTraceStep {
    int         step_index   = 0;
    std::string module_id;
    std::string module_name;
    std::string status;           // "success", "failure", "fallback", "skipped"
    std::string input_summary;
    std::string output_summary;
    double      latency_ms   = 0.0;
    std::string error;
    bool        fallback_used = false;
    std::string fallback_module_id;
};

// ---------------------------------------------------------------------------
// CompositionTrace — full record of a MOSAIC execution
// ---------------------------------------------------------------------------
struct CompositionTrace {
    std::string                    task_id;
    std::string                    domain;
    TaskRequirements               requirements;
    CompositionPlan                plan;
    std::vector<CompositionTraceStep> steps;
    ExecutionStatus                execution_status = ExecutionStatus::FAILURE;
    double                         total_latency_ms = 0.0;
    int                            fallback_count   = 0;
    int                            error_count      = 0;
    std::string                    final_output_type;
};

// ---------------------------------------------------------------------------
// MOSAIC configuration
// ---------------------------------------------------------------------------
struct MOSAICConfig {
    int    max_modules          = 10;
    int    max_execution_steps  = 50;
    double max_latency_ms       = 30000.0;
    int    max_retries          = 1;
    bool   allow_llm            = false;   // false by default (model may not be loaded)
    bool   allow_fallback       = true;
    bool   prefer_lightweight   = true;    // prefer low-cost modules
    bool   use_cache            = true;
    std::vector<std::string> preferred_modules;
    std::vector<std::string> excluded_modules;
    double max_total_cost       = 100.0;
};

// ---------------------------------------------------------------------------
// Execution context — shared state passed through the pipeline
// ---------------------------------------------------------------------------
// We keep it as a stringly-typed map for maximum flexibility across modules.
using ContextValue = std::variant<std::string, double, bool>;

struct ExecutionContext {
    std::string task_id;
    std::string raw_input;
    std::unordered_map<std::string, ContextValue> values;
    std::vector<std::string>                      errors;
    CompositionTrace*                             trace = nullptr; // non-owning

    void set(const std::string& key, ContextValue val) { values[key] = std::move(val); }

    bool has(const std::string& key) const { return values.count(key) > 0; }

    std::string get_str(const std::string& key, const std::string& dflt = "") const {
        auto it = values.find(key);
        if (it == values.end()) return dflt;
        if (std::holds_alternative<std::string>(it->second))
            return std::get<std::string>(it->second);
        if (std::holds_alternative<double>(it->second)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", std::get<double>(it->second));
            return buf;
        }
        return std::get<bool>(it->second) ? "true" : "false";
    }

    double get_double(const std::string& key, double dflt = 0.0) const {
        auto it = values.find(key);
        if (it == values.end()) return dflt;
        if (std::holds_alternative<double>(it->second))
            return std::get<double>(it->second);
        return dflt;
    }

    bool get_bool(const std::string& key, bool dflt = false) const {
        auto it = values.find(key);
        if (it == values.end()) return dflt;
        if (std::holds_alternative<bool>(it->second))
            return std::get<bool>(it->second);
        return dflt;
    }
};

// ---------------------------------------------------------------------------
// ExecutionResult — what the MOSAIC executor returns
// ---------------------------------------------------------------------------
struct ExecutionResult {
    ExecutionStatus  status        = ExecutionStatus::FAILURE;
    std::string      answer;
    std::string      answer_unit;
    double           answer_numeric = 0.0;
    bool             answer_is_numeric = false;
    double           confidence    = 0.0;
    std::string      error_message;
    CompositionTrace trace;
};

} // namespace prime::mosaic
