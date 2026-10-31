#pragma once

/*
 * memory_store.hpp — Storage backend abstraction for the PRIME Memory Engine.
 *
 * Architecture:
 *
 *   MemoryEngine
 *       ↓
 *   MemoryStore  (interface)
 *       ├── InMemoryStore   — hash map + ordered list; no persistence
 *       └── SqliteStore     — SQLite3 via POSIX; see memory_store.cpp
 *
 * Only InMemoryStore lives here (header-only).
 * SqliteStore is in memory_store.cpp / declared below.
 */

#include "memory_types.hpp"
#include "../core/status.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <algorithm>
#include <functional>
#include <optional>
#include <cstdint>

namespace prime {
namespace memory {

// ---------------------------------------------------------------------------
// SearchQuery — parameters for memory.search()
// ---------------------------------------------------------------------------
struct SearchQuery {
    std::string query_text;

    // Optional filters (empty string / MemoryType sentinel = no filter)
    std::optional<MemoryType> filter_type;
    Metadata                  filter_metadata;  // all keys must match

    std::size_t limit = 10;

    // Scoring weights (passed through to MemoryItem::score)
    double w_text       = 0.50;
    double w_recency    = 0.20;
    double w_importance = 0.20;
    double w_freq       = 0.10;
};

// ---------------------------------------------------------------------------
// MemoryStore — pure-virtual backend interface
// ---------------------------------------------------------------------------
class MemoryStore {
public:
    virtual ~MemoryStore() = default;

    // Lifecycle
    virtual Status open(const std::string& path)  = 0;
    virtual Status close()                         = 0;

    // CRUD
    virtual Status put(const MemoryItem& item)     = 0;
    virtual Status get(uint64_t id, MemoryItem& out) const = 0;
    virtual Status update(const MemoryItem& item)  = 0;
    virtual Status remove(uint64_t id)             = 0;
    virtual Status clear(std::optional<MemoryType> type_filter) = 0;

    // Query
    virtual std::vector<MemoryItem> search(const SearchQuery& q,
                                           double half_life) const = 0;
    virtual std::size_t count(std::optional<MemoryType> type_filter) const = 0;
    virtual bool        exists(uint64_t id) const                          = 0;

    // Returns all IDs in insertion order for the given tier
    virtual std::vector<uint64_t> ids_by_insertion(MemoryType type) const = 0;
};

// ---------------------------------------------------------------------------
// Text similarity helpers (lightweight — no embeddings)
// ---------------------------------------------------------------------------
namespace text_util {

// Lower-case a UTF-8 string (ASCII only for Phase 3 — sufficient)
inline std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) out += static_cast<char>(std::tolower(c));
    return out;
}

// Simple bag-of-words Jaccard similarity between query and text.
// Tokenises by whitespace and punctuation, lower-cases.
inline double jaccard(const std::string& query, const std::string& text) {
    auto tokenise = [](const std::string& s) -> std::vector<std::string> {
        std::vector<std::string> tokens;
        std::string cur;
        for (unsigned char c : s) {
            if (std::isalnum(c)) {
                cur += static_cast<char>(std::tolower(c));
            } else if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) tokens.push_back(cur);
        return tokens;
    };

    auto qt = tokenise(query);
    auto tt = tokenise(text);

    if (qt.empty() || tt.empty()) return 0.0;

    // Convert to sets
    std::unordered_map<std::string, int> freq;
    for (auto& w : qt) freq[w]++;
    int intersect = 0, total = static_cast<int>(freq.size());
    std::unordered_map<std::string, bool> seen;
    for (auto& w : tt) {
        if (freq.count(w) && !seen[w]) {
            intersect++;
            seen[w] = true;
        } else if (!freq.count(w) && !seen[w]) {
            total++;
            seen[w] = true;
        }
    }
    return (total > 0) ? static_cast<double>(intersect) / total : 0.0;
}

// Substring presence (normalised): 1.0 if query text occurs in content.
inline double contains_score(const std::string& query, const std::string& text) {
    std::string q_lc = to_lower(query);
    std::string t_lc = to_lower(text);
    if (q_lc.empty()) return 0.0;
    return (t_lc.find(q_lc) != std::string::npos) ? 1.0 : 0.0;
}

// Combined text similarity: max(Jaccard, contains_score)
inline double similarity(const std::string& query, const std::string& text) {
    return std::max(jaccard(query, text), contains_score(query, text));
}

} // namespace text_util

// ---------------------------------------------------------------------------
// InMemoryStore — fast, non-persistent backend
// ---------------------------------------------------------------------------
class InMemoryStore final : public MemoryStore {
public:
    InMemoryStore() = default;
    ~InMemoryStore() override = default;

    Status open(const std::string& /*path*/) override { return Status::ok_status(); }
    Status close()                            override { return Status::ok_status(); }

    // --- CRUD ---------------------------------------------------------------

    Status put(const MemoryItem& item) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (items_.count(item.id)) {
            return Status::error(StatusCode::ERROR_INVALID_CONFIG,
                                 "Duplicate memory ID: " + std::to_string(item.id));
        }
        items_[item.id] = item;
        insertion_order_[item.type].push_back(item.id);
        return Status::ok_status();
    }

    Status get(uint64_t id, MemoryItem& out) const override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = items_.find(id);
        if (it == items_.end()) {
            return Status::error(StatusCode::ERROR_MODULE_NOT_FOUND,
                                 "Memory ID not found: " + std::to_string(id));
        }
        out = it->second;
        return Status::ok_status();
    }

    Status update(const MemoryItem& item) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = items_.find(item.id);
        if (it == items_.end()) {
            return Status::error(StatusCode::ERROR_MODULE_NOT_FOUND,
                                 "Memory ID not found: " + std::to_string(item.id));
        }
        // If type changes, update insertion_order_ bookkeeping
        if (it->second.type != item.type) {
            auto& old_vec = insertion_order_[it->second.type];
            old_vec.erase(std::remove(old_vec.begin(), old_vec.end(), item.id),
                           old_vec.end());
            insertion_order_[item.type].push_back(item.id);
        }
        it->second = item;
        return Status::ok_status();
    }

    Status remove(uint64_t id) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = items_.find(id);
        if (it == items_.end()) {
            return Status::error(StatusCode::ERROR_MODULE_NOT_FOUND,
                                 "Memory ID not found: " + std::to_string(id));
        }
        auto& vec = insertion_order_[it->second.type];
        vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
        items_.erase(it);
        return Status::ok_status();
    }

    Status clear(std::optional<MemoryType> type_filter) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!type_filter) {
            items_.clear();
            insertion_order_.clear();
        } else {
            auto& vec = insertion_order_[*type_filter];
            for (uint64_t id : vec) items_.erase(id);
            vec.clear();
        }
        return Status::ok_status();
    }

    // --- Query --------------------------------------------------------------

    std::vector<MemoryItem> search(const SearchQuery& q, double half_life) const override {
        std::lock_guard<std::mutex> lk(mu_);
        double now = now_seconds();

        std::vector<std::pair<double, MemoryItem>> scored;
        scored.reserve(items_.size());

        for (auto& [id, item] : items_) {
            if (item.is_expired(now)) continue;
            if (q.filter_type && item.type != *q.filter_type) continue;
            if (!q.filter_metadata.empty()) {
                bool match = true;
                for (auto& [k, v] : q.filter_metadata) {
                    auto mit = item.metadata.find(k);
                    if (mit == item.metadata.end() || mit->second != v) {
                        match = false;
                        break;
                    }
                }
                if (!match) continue;
            }

            double text_sim = 0.0;
            if (!q.query_text.empty()) {
                // Search in content + metadata values
                text_sim = text_util::similarity(q.query_text, item.content);
                if (text_sim < 0.001) {
                    for (auto& [k, v] : item.metadata) {
                        double ms = text_util::similarity(q.query_text, v);
                        if (ms > text_sim) text_sim = ms;
                    }
                }
            } else {
                // No text query: all items pass
                text_sim = 1.0;
            }

            double s = item.score(text_sim, half_life,
                                  q.w_text, q.w_recency, q.w_importance, q.w_freq);
            scored.emplace_back(s, item);
        }

        // Sort descending by score
        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        std::size_t limit = (q.limit == 0) ? scored.size()
                          : std::min(q.limit, scored.size());

        std::vector<MemoryItem> result;
        result.reserve(limit);
        for (std::size_t i = 0; i < limit; ++i) result.push_back(scored[i].second);
        return result;
    }

    std::size_t count(std::optional<MemoryType> type_filter) const override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!type_filter) return items_.size();
        auto it = insertion_order_.find(*type_filter);
        if (it == insertion_order_.end()) return 0;
        return it->second.size();
    }

    bool exists(uint64_t id) const override {
        std::lock_guard<std::mutex> lk(mu_);
        return items_.count(id) != 0;
    }

    std::vector<uint64_t> ids_by_insertion(MemoryType type) const override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = insertion_order_.find(type);
        if (it == insertion_order_.end()) return {};
        return std::vector<uint64_t>(it->second.begin(), it->second.end());
    }

    // Mutable access for tracking (called by MemoryEngine after search)
    Status touch(uint64_t id, double at) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = items_.find(id);
        if (it == items_.end()) return Status::ok_status();
        it->second.access_count++;
        it->second.last_accessed = at;
        return Status::ok_status();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, MemoryItem> items_;
    std::unordered_map<MemoryType, std::vector<uint64_t>> insertion_order_;
};

} // namespace memory
} // namespace prime
