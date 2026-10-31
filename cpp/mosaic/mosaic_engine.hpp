#pragma once
// mosaic_engine.hpp — MOSAIC Engine: top-level orchestrator for Phase 6.
//
// Coordinates:
//   TaskAnalyzer → CompositionPlanner → CompositionPlan
//   (Actual module execution is done by Python-side adapters.)
//
// The C++ engine handles:
//   - Module & capability registry management
//   - Task analysis
//   - Composition planning (capability matching + dependency resolution)
//   - Plan validation
//   - Explanation generation
//
// Module execution is delegated to Python because the actual PRIME modules
// (ATLAS, Memory, Reasoning) are Python objects with C extensions — bridging
// them through a second C layer would only add friction.

#include "mosaic_types.hpp"
#include "module_registry.hpp"
#include "task_analyzer.hpp"
#include "composition_planner.hpp"

#include <sstream>
#include <chrono>
#include <unordered_map>

namespace prime::mosaic {

// ---------------------------------------------------------------------------
// Plan cache entry
// ---------------------------------------------------------------------------
struct CacheEntry {
    CompositionPlan plan;
    std::string     cache_key;
};

// ---------------------------------------------------------------------------
// MOSAICEngine
// ---------------------------------------------------------------------------
class MOSAICEngine {
public:
    explicit MOSAICEngine(MOSAICConfig cfg = {})
        : cfg_(std::move(cfg)) {}

    // ------------------------------------------------------------------
    // Module registration
    // ------------------------------------------------------------------
    bool register_module(ModuleMetadata meta) {
        bool ok = registry_.register_module(meta);
        if (ok) cache_.clear(); // invalidate plan cache on registration changes
        return ok;
    }

    void update_module(ModuleMetadata meta) {
        registry_.update_module(std::move(meta));
        cache_.clear();
    }

    bool remove_module(const std::string& id) {
        bool ok = registry_.remove(id);
        if (ok) cache_.clear();
        return ok;
    }

    bool set_module_status(const std::string& id, ModuleStatus status) {
        bool ok = registry_.set_status(id, status);
        if (ok) cache_.clear();
        return ok;
    }

    const ModuleMetadata* get_module(const std::string& id) const {
        return registry_.get(id);
    }

    std::vector<std::string> list_modules() const {
        return registry_.list_ids();
    }

    std::vector<const ModuleMetadata*> available_modules() const {
        return registry_.available_modules();
    }

    size_t module_count() const { return registry_.size(); }

    // ------------------------------------------------------------------
    // Task analysis
    // ------------------------------------------------------------------
    TaskRequirements analyze_task(
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
        return analyzer_.analyze(
            task_id, domain,
            has_expressions, has_rules, has_constraints,
            has_goals, has_entities, has_logic_clauses, has_relationships,
            operation_types, goal_targets, entity_count, has_memory_context);
    }

    // ------------------------------------------------------------------
    // Composition planning
    // ------------------------------------------------------------------
    CompositionPlan compose(const TaskRequirements& req) const {
        return compose(req, cfg_);
    }

    CompositionPlan compose(const TaskRequirements& req,
                            const MOSAICConfig& cfg) const
    {
        // Cache lookup
        if (cfg_.use_cache) {
            std::string key = cache_key(req, cfg);
            auto it = cache_.find(key);
            if (it != cache_.end()) return it->second.plan;
        }

        auto plan = planner_.plan(req, registry_, cfg);

        // Cache store
        if (cfg_.use_cache && plan.status == CompositionStatus::VALID) {
            std::string key = cache_key(req, cfg);
            if (cache_.size() >= MAX_CACHE) cache_.clear();
            cache_[key] = CacheEntry{plan, key};
        }

        return plan;
    }

    // Validate an existing plan
    CompositionStatus validate_plan(const CompositionPlan& plan) const {
        return planner_.validate(plan, registry_);
    }

    // ------------------------------------------------------------------
    // Explanation generation
    // ------------------------------------------------------------------
    std::string explain_plan(const CompositionPlan& plan) const {
        std::ostringstream oss;
        oss << "========================================\n";
        oss << "MOSAIC Composition Plan\n";
        oss << "========================================\n";
        oss << "Task:    " << plan.task_id << "\n";
        oss << "Domain:  " << plan.domain  << "\n";
        oss << "Status:  " << composition_status_to_string(plan.status) << "\n";

        if (!plan.valid()) {
            oss << "Error:   " << plan.status_detail << "\n";
            return oss.str();
        }

        oss << "\nRequired capabilities:\n";
        for (auto& cap : plan.requirements.required_capabilities)
            oss << "  - " << cap << "\n";

        oss << "\nSelected modules (execution order):\n";
        for (size_t i = 0; i < plan.module_ids.size(); ++i) {
            auto* m = registry_.get(plan.module_ids[i]);
            std::string name = m ? m->name : plan.module_ids[i];
            oss << "  " << (i+1) << ". " << name << " [" << plan.module_ids[i] << "]\n";
            if (i < plan.selection_reasons.size())
                oss << "     Reason: " << plan.selection_reasons[i] << "\n";
        }

        oss << "\nEstimated cost: " << plan.estimated_cost << "\n";
        oss << "========================================\n";
        return oss.str();
    }

    std::string explain_trace(const CompositionTrace& trace) const {
        std::ostringstream oss;
        oss << "========================================\n";
        oss << "MOSAIC Execution Trace\n";
        oss << "========================================\n";
        oss << "Task:       " << trace.task_id << "\n";
        oss << "Status:     " << execution_status_to_string(trace.execution_status) << "\n";
        oss << "Total time: " << trace.total_latency_ms << " ms\n";
        oss << "Fallbacks:  " << trace.fallback_count << "\n";
        oss << "Errors:     " << trace.error_count << "\n\n";

        oss << "Execution steps:\n";
        for (auto& step : trace.steps) {
            oss << "  [" << (step.step_index+1) << "] "
                << step.module_name << " — " << step.status;
            if (step.latency_ms > 0)
                oss << " (" << step.latency_ms << " ms)";
            if (step.fallback_used)
                oss << " [fallback: " << step.fallback_module_id << "]";
            if (!step.error.empty())
                oss << "\n      Error: " << step.error;
            if (!step.output_summary.empty())
                oss << "\n      Output: " << step.output_summary;
            oss << "\n";
        }
        oss << "========================================\n";
        return oss.str();
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------
    MOSAICConfig&       config()       { return cfg_; }
    const MOSAICConfig& config() const { return cfg_; }
    void set_config(MOSAICConfig cfg)  { cfg_ = std::move(cfg); cache_.clear(); }

    void clear_cache() { cache_.clear(); }

    const ModuleRegistry& registry() const { return registry_; }

private:
    MOSAICConfig       cfg_;
    ModuleRegistry     registry_;
    TaskAnalyzer       analyzer_;
    CompositionPlanner planner_;
    mutable std::unordered_map<std::string, CacheEntry> cache_;

    static constexpr size_t MAX_CACHE = 256;

    std::string cache_key(const TaskRequirements& req, const MOSAICConfig& cfg) const {
        // Deterministic key from requirements + relevant config flags
        std::string key = req.domain + "|" + req.task_id;
        key += "|llm=" + std::to_string(cfg.allow_llm);
        key += "|lw="  + std::to_string(cfg.prefer_lightweight);
        for (auto& c : req.required_capabilities) key += "|r:" + c;
        for (auto& c : req.optional_capabilities) key += "|o:" + c;
        for (auto& ex : cfg.excluded_modules)      key += "|ex:" + ex;
        for (auto& pr : cfg.preferred_modules)     key += "|pr:" + pr;
        return key;
    }
};

} // namespace prime::mosaic
