#pragma once
// module_registry.hpp — ModuleRegistry and CapabilityRegistry for MOSAIC.
//
// Thread-compatible (no shared mutable state between calls if used from one thread).

#include "mosaic_types.hpp"
#include <algorithm>
#include <stdexcept>

namespace prime::mosaic {

// ---------------------------------------------------------------------------
// CapabilityRegistry — lookup of capability names
// ---------------------------------------------------------------------------
class CapabilityRegistry {
public:
    // Register a capability. Returns false if name already exists.
    bool register_capability(Capability cap) {
        if (caps_.count(cap.name)) return false;
        caps_[cap.name] = std::move(cap);
        return true;
    }

    // Ensure a capability exists (idempotent — updates if already present).
    void ensure(Capability cap) {
        caps_[cap.name] = std::move(cap);
    }

    bool has(const std::string& name) const { return caps_.count(name) > 0; }

    const Capability* get(const std::string& name) const {
        auto it = caps_.find(name);
        return it == caps_.end() ? nullptr : &it->second;
    }

    std::vector<std::string> list_names() const {
        std::vector<std::string> out;
        out.reserve(caps_.size());
        for (auto& [k, _] : caps_) out.push_back(k);
        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<Capability> find_by_category(CapabilityCategory cat) const {
        std::vector<Capability> out;
        for (auto& [_, c] : caps_)
            if (c.category == cat) out.push_back(c);
        return out;
    }

    size_t size() const { return caps_.size(); }
    void clear() { caps_.clear(); }

private:
    std::unordered_map<std::string, Capability> caps_;
};

// ---------------------------------------------------------------------------
// ModuleRegistry — stores and queries module metadata
// ---------------------------------------------------------------------------
class ModuleRegistry {
public:
    // Register a module. Returns false if ID already exists.
    bool register_module(ModuleMetadata meta) {
        if (modules_.count(meta.id)) return false;
        const std::string id = meta.id;
        modules_[id] = std::move(meta);
        // Index by capabilities
        for (auto& cap : modules_[id].capabilities)
            cap_index_[cap.name].push_back(id);
        return true;
    }

    // Re-register (update). Always succeeds.
    void update_module(ModuleMetadata meta) {
        // Remove old index entries
        auto it = modules_.find(meta.id);
        if (it != modules_.end()) {
            for (auto& cap : it->second.capabilities) {
                auto& v = cap_index_[cap.name];
                v.erase(std::remove(v.begin(), v.end(), meta.id), v.end());
            }
        }
        const std::string id = meta.id;
        modules_[id] = std::move(meta);
        for (auto& cap : modules_[id].capabilities)
            cap_index_[cap.name].push_back(id);
    }

    bool remove(const std::string& id) {
        auto it = modules_.find(id);
        if (it == modules_.end()) return false;
        for (auto& cap : it->second.capabilities) {
            auto& v = cap_index_[cap.name];
            v.erase(std::remove(v.begin(), v.end(), id), v.end());
        }
        modules_.erase(it);
        return true;
    }

    bool has(const std::string& id) const { return modules_.count(id) > 0; }

    const ModuleMetadata* get(const std::string& id) const {
        auto it = modules_.find(id);
        return it == modules_.end() ? nullptr : &it->second;
    }

    ModuleMetadata* get_mutable(const std::string& id) {
        auto it = modules_.find(id);
        return it == modules_.end() ? nullptr : &it->second;
    }

    // All module IDs, sorted for determinism
    std::vector<std::string> list_ids() const {
        std::vector<std::string> out;
        out.reserve(modules_.size());
        for (auto& [k, _] : modules_) out.push_back(k);
        std::sort(out.begin(), out.end());
        return out;
    }

    // All modules with a given capability (sorted by priority desc, then id)
    std::vector<const ModuleMetadata*>
    find_capable(const std::string& capability) const {
        std::vector<const ModuleMetadata*> out;
        auto it = cap_index_.find(capability);
        if (it != cap_index_.end()) {
            for (auto& id : it->second) {
                auto mit = modules_.find(id);
                if (mit != modules_.end() &&
                    mit->second.status == ModuleStatus::AVAILABLE)
                    out.push_back(&mit->second);
            }
        }
        // Deterministic sort: priority desc, then id asc
        std::sort(out.begin(), out.end(),
            [](const ModuleMetadata* a, const ModuleMetadata* b) {
                if (a->priority != b->priority) return a->priority > b->priority;
                return a->id < b->id;
            });
        return out;
    }

    // All available modules
    std::vector<const ModuleMetadata*> available_modules() const {
        std::vector<const ModuleMetadata*> out;
        for (auto& [_, m] : modules_)
            if (m.status == ModuleStatus::AVAILABLE)
                out.push_back(&m);
        std::sort(out.begin(), out.end(),
            [](const ModuleMetadata* a, const ModuleMetadata* b) {
                return a->id < b->id;
            });
        return out;
    }

    // Mark a module unavailable (e.g. LLM not loaded)
    bool set_status(const std::string& id, ModuleStatus status) {
        auto it = modules_.find(id);
        if (it == modules_.end()) return false;
        it->second.status = status;
        return true;
    }

    size_t size() const { return modules_.size(); }
    void clear() {
        modules_.clear();
        cap_index_.clear();
    }

private:
    std::unordered_map<std::string, ModuleMetadata> modules_;
    std::unordered_map<std::string, std::vector<std::string>> cap_index_;
};

} // namespace prime::mosaic
