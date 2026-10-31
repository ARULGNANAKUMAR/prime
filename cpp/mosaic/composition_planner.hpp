#pragma once
// composition_planner.hpp — MOSAIC Composition Planner.
//
// Responsibilities:
//   1. Match required capabilities to available modules
//   2. Resolve module dependencies
//   3. Detect circular dependencies
//   4. Order the execution pipeline
//   5. Validate input/output type compatibility
//   6. Apply resource-aware, deterministic selection

#include "mosaic_types.hpp"
#include "module_registry.hpp"
#include <sstream>
#include <cassert>

namespace prime::mosaic {

class CompositionPlanner {
public:
    // Build a CompositionPlan from task requirements and the module registry.
    CompositionPlan plan(
            const TaskRequirements& req,
            const ModuleRegistry&   registry,
            const MOSAICConfig&     cfg) const
    {
        CompositionPlan plan;
        plan.task_id     = req.task_id;
        plan.domain      = req.domain;
        plan.requirements = req;

        // --- Step 1: Select one module per required capability ---
        std::vector<std::string>  selected_ids;
        std::vector<std::string>  reasons;
        std::unordered_set<std::string> covered_caps;
        std::unordered_set<std::string> id_set;

        // Always start with atlas_representation if not already in the required list
        auto add_module = [&](const std::string& cap) -> bool {
            if (covered_caps.count(cap)) return true; // already covered

            auto candidates = registry.find_capable(cap);
            if (candidates.empty()) {
                plan.status       = CompositionStatus::MISSING_CAPABILITY;
                plan.status_detail = "No module satisfies capability: " + cap;
                return false;
            }

            // Filter: excluded, unavailable, LLM policy, cost limit
            std::vector<const ModuleMetadata*> eligible;
            for (auto* m : candidates) {
                if (m->status != ModuleStatus::AVAILABLE) continue;
                if (is_excluded(m->id, cfg)) continue;
                if (m->cost.requires_model && !cfg.allow_llm) continue;
                if (id_set.count(m->id)) continue; // already selected
                eligible.push_back(m);
            }

            if (eligible.empty()) {
                // Capability unmet — not a hard error if optional
                return false;
            }

            // Select: prefer preferred_modules, then by score
            const ModuleMetadata* chosen = select(eligible, cfg);
            if (!chosen) return false;

            selected_ids.push_back(chosen->id);
            id_set.insert(chosen->id);
            std::string reason = "Module '" + chosen->name +
                                 "' selected for capability '" + cap + "'";
            if (cfg.prefer_lightweight && !chosen->cost.requires_model)
                reason += " (lightweight preferred)";
            reasons.push_back(reason);

            // Mark all capabilities this module covers
            for (auto& c : chosen->capabilities)
                covered_caps.insert(c.name);

            // Check module count limit
            if ((int)selected_ids.size() > cfg.max_modules) {
                plan.status        = CompositionStatus::RESOURCE_EXCEEDED;
                plan.status_detail = "max_modules limit reached";
                return false;
            }
            return true;
        };

        // Process required capabilities
        for (auto& cap : req.required_capabilities) {
            if (!add_module(cap)) {
                if (plan.status != CompositionStatus::VALID &&
                    plan.status != CompositionStatus::EMPTY_PLAN) {
                    return plan; // already set to error
                }
                // If required and not found, fail
                if (!covered_caps.count(cap)) {
                    plan.status        = CompositionStatus::MISSING_CAPABILITY;
                    plan.status_detail = "Required capability unmet: " + cap;
                    return plan;
                }
            }
        }

        // Process optional capabilities (best-effort, no failure)
        for (auto& cap : req.optional_capabilities) {
            add_module(cap);
        }

        if (selected_ids.empty()) {
            plan.status        = CompositionStatus::EMPTY_PLAN;
            plan.status_detail = "No modules selected";
            return plan;
        }

        // --- Step 2: Topological sort respecting declared dependencies ---
        auto ordered = topological_sort(selected_ids, registry);
        if (!ordered.has_value()) {
            plan.status        = CompositionStatus::COMPOSITION_CYCLE;
            plan.status_detail = "Circular dependency detected in module graph";
            return plan;
        }
        selected_ids = ordered.value();

        // --- Step 3: Input/output compatibility check ---
        {
            // atlas_representation module (first in pipeline) takes atlas_problem.
            // Each subsequent module should accept what the previous produces.
            // We do a simple check: if a module's input_types is non-empty and
            // doesn't include "*", verify there's overlap with the prior outputs.
            std::unordered_set<std::string> available_types = {"atlas_problem", "raw_text"};
            for (auto& mid : selected_ids) {
                auto* m = registry.get(mid);
                if (!m) continue;
                if (!m->input_types.empty()) {
                    bool has_wildcard = false;
                    for (auto& t : m->input_types) if (t == "*") has_wildcard = true;
                    if (!has_wildcard) {
                        bool ok = false;
                        for (auto& t : m->input_types)
                            if (available_types.count(t)) { ok = true; break; }
                        if (!ok) {
                            plan.status        = CompositionStatus::INPUT_MISMATCH;
                            plan.status_detail = "Module '" + m->name +
                                                 "' requires input " +
                                                 join(m->input_types) +
                                                 " but pipeline only provides " +
                                                 set_to_string(available_types);
                            return plan;
                        }
                    }
                }
                for (auto& t : m->output_types)
                    available_types.insert(t);
            }
        }

        // --- Step 4: Cost sum ---
        double total_cost = 0.0;
        for (auto& mid : selected_ids) {
            auto* m = registry.get(mid);
            if (m) total_cost += m->cost.cpu_relative + m->cost.memory_relative;
        }
        if (total_cost > cfg.max_total_cost) {
            plan.status        = CompositionStatus::RESOURCE_EXCEEDED;
            plan.status_detail = "Estimated cost " + std::to_string(total_cost) +
                                 " exceeds limit " + std::to_string(cfg.max_total_cost);
            return plan;
        }

        plan.module_ids        = selected_ids;
        plan.selection_reasons = reasons;
        plan.estimated_cost    = total_cost;
        plan.status            = CompositionStatus::VALID;
        return plan;
    }

    // Validate an existing plan (can be called independently)
    CompositionStatus validate(const CompositionPlan& plan,
                               const ModuleRegistry&  registry) const
    {
        if (plan.module_ids.empty())
            return CompositionStatus::EMPTY_PLAN;

        std::unordered_set<std::string> available_types = {"atlas_problem", "raw_text"};
        for (auto& mid : plan.module_ids) {
            if (!registry.has(mid))
                return CompositionStatus::MISSING_CAPABILITY;
            auto* m = registry.get(mid);
            if (m->status != ModuleStatus::AVAILABLE)
                return CompositionStatus::MISSING_CAPABILITY;
            for (auto& t : m->output_types)
                available_types.insert(t);
        }

        // Cycle check
        auto ordered = topological_sort(plan.module_ids, registry);
        if (!ordered.has_value())
            return CompositionStatus::COMPOSITION_CYCLE;

        return CompositionStatus::VALID;
    }

private:
    // ---------------------------------------------------------------------------
    // Module selection scoring
    // ---------------------------------------------------------------------------
    const ModuleMetadata* select(const std::vector<const ModuleMetadata*>& candidates,
                                 const MOSAICConfig& cfg) const
    {
        if (candidates.empty()) return nullptr;

        // Score each candidate deterministically
        // score = priority * 10 + preferred_bonus - cost_penalty - model_penalty
        auto score = [&](const ModuleMetadata* m) -> double {
            double s = m->priority * 10.0;
            // Preferred modules get a large bonus
            for (auto& pref : cfg.preferred_modules)
                if (pref == m->id) s += 100.0;
            // Resource cost penalty
            if (cfg.prefer_lightweight) {
                s -= m->cost.cpu_relative;
                s -= m->cost.memory_relative;
                s -= m->cost.latency_ms * 0.001;
            }
            if (m->cost.requires_model) s -= 20.0;
            // Deterministic tie-break: alphabetical by id (lower id = higher score)
            return s;
        };

        const ModuleMetadata* best = candidates[0];
        double best_score = score(candidates[0]);
        for (size_t i = 1; i < candidates.size(); ++i) {
            double s = score(candidates[i]);
            if (s > best_score ||
                (s == best_score && candidates[i]->id < best->id)) {
                best       = candidates[i];
                best_score = s;
            }
        }
        return best;
    }

    bool is_excluded(const std::string& id, const MOSAICConfig& cfg) const {
        for (auto& ex : cfg.excluded_modules)
            if (ex == id) return true;
        return false;
    }

    // ---------------------------------------------------------------------------
    // Topological sort (Kahn's algorithm) — detects cycles
    // Returns nullopt if cycle detected.
    // ---------------------------------------------------------------------------
    std::optional<std::vector<std::string>>
    topological_sort(const std::vector<std::string>& ids,
                     const ModuleRegistry& registry) const
    {
        // Build in-degree map and adjacency list restricted to the selected set
        std::unordered_set<std::string> id_set(ids.begin(), ids.end());
        std::unordered_map<std::string, std::vector<std::string>> edges; // dep → dependents
        std::unordered_map<std::string, int> in_degree;

        for (auto& id : ids) in_degree[id] = 0;

        for (auto& id : ids) {
            auto* m = registry.get(id);
            if (!m) continue;
            for (auto& dep : m->dependencies) {
                if (!id_set.count(dep)) continue; // dep not in this plan
                edges[dep].push_back(id);
                in_degree[id]++;
            }
        }

        // Kahn's: start with zero in-degree nodes, sorted for determinism
        std::vector<std::string> queue;
        for (auto& [id, deg] : in_degree)
            if (deg == 0) queue.push_back(id);
        std::sort(queue.begin(), queue.end());

        std::vector<std::string> result;
        while (!queue.empty()) {
            std::string cur = queue.front();
            queue.erase(queue.begin());
            result.push_back(cur);
            auto it = edges.find(cur);
            if (it == edges.end()) continue;
            std::vector<std::string> next_queue;
            for (auto& nb : it->second) {
                if (--in_degree[nb] == 0)
                    next_queue.push_back(nb);
            }
            std::sort(next_queue.begin(), next_queue.end());
            for (auto& n : next_queue) queue.push_back(n);
        }

        if (result.size() != ids.size()) return std::nullopt; // cycle
        return result;
    }

    static std::string join(const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += "|";
            s += v[i];
        }
        return s;
    }

    static std::string set_to_string(const std::unordered_set<std::string>& s) {
        std::vector<std::string> v(s.begin(), s.end());
        std::sort(v.begin(), v.end());
        return join(v);
    }
};

} // namespace prime::mosaic
