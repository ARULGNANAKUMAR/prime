#pragma once

/*
 * memory_types.hpp — Core types for the PRIME Memory Engine (Phase 3).
 *
 * All memory items share a common MemoryItem representation.
 * The three logical tiers (WORKING, SHORT_TERM, LONG_TERM) are modelled via
 * MemoryType rather than separate classes, keeping the engine simple and
 * allowing easy promotion between tiers.
 */

#include <string>
#include <unordered_map>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace prime {
namespace memory {

// ---------------------------------------------------------------------------
// MemoryType — the logical tier a memory item belongs to
// ---------------------------------------------------------------------------
enum class MemoryType : uint8_t {
    WORKING    = 0,  // temporary; current task context
    SHORT_TERM = 1,  // recent; capped ring, auto-evicted
    LONG_TERM  = 2,  // persistent; survives process restart
};

inline const char* memory_type_name(MemoryType t) {
    switch (t) {
        case MemoryType::WORKING:    return "working";
        case MemoryType::SHORT_TERM: return "short_term";
        case MemoryType::LONG_TERM:  return "long_term";
    }
    return "unknown";
}

inline MemoryType memory_type_from_string(const std::string& s) {
    if (s == "working")    return MemoryType::WORKING;
    if (s == "short_term") return MemoryType::SHORT_TERM;
    if (s == "long_term")  return MemoryType::LONG_TERM;
    throw std::invalid_argument("Unknown memory type: " + s);
}

// ---------------------------------------------------------------------------
// Metadata — free-form string key/value pairs
// ---------------------------------------------------------------------------
using Metadata = std::unordered_map<std::string, std::string>;

// ---------------------------------------------------------------------------
// Timestamp helpers (seconds since epoch, stored as double for precision)
// ---------------------------------------------------------------------------
inline double now_seconds() {
    using namespace std::chrono;
    auto tp = system_clock::now().time_since_epoch();
    return duration<double>(tp).count();
}

// ---------------------------------------------------------------------------
// MemoryItem — the single memory representation used throughout Phase 3
// ---------------------------------------------------------------------------
struct MemoryItem {
    // Identity
    uint64_t    id            = 0;
    MemoryType  type          = MemoryType::LONG_TERM;

    // Payload
    std::string content;
    Metadata    metadata;

    // Scoring signals
    double      importance    = 0.5;   // [0.0, 1.0]
    double      confidence    = 1.0;   // [0.0, 1.0]

    // Timestamps (seconds since epoch)
    double      created_at    = 0.0;
    double      last_accessed = 0.0;

    // Usage tracking
    uint64_t    access_count  = 0;

    // Expiration (0 = never)
    double      expires_at    = 0.0;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------
    bool is_expired(double at = 0.0) const {
        if (expires_at <= 0.0) return false;
        double t = (at > 0.0) ? at : now_seconds();
        return t >= expires_at;
    }

    // Recency score: exponential decay with configurable half-life.
    // half_life_seconds = 0 means no decay (returns 1.0).
    double recency_score(double half_life_seconds, double at = 0.0) const {
        if (half_life_seconds <= 0.0) return 1.0;
        double t = (at > 0.0) ? at : now_seconds();
        double age = t - created_at;
        if (age < 0.0) age = 0.0;
        double lambda = 0.693147 / half_life_seconds;  // ln(2) / half_life
        return std::exp(-lambda * age);
    }

    // Composite relevance score for search ranking.
    // text_sim:  [0,1] text similarity to the query (caller provides)
    // half_life: decay constant for recency (0 = no decay)
    double score(double text_sim, double half_life_seconds,
                 double w_text = 0.5, double w_recency = 0.2,
                 double w_importance = 0.2, double w_freq = 0.1) const {
        double rec = recency_score(half_life_seconds);
        // frequency: saturating at 20 accesses → normalise to [0,1]
        double freq_norm = (access_count < 20)
            ? static_cast<double>(access_count) / 20.0
            : 1.0;
        return w_text * text_sim
             + w_recency * rec
             + w_importance * importance
             + w_freq * freq_norm;
    }
};

// ---------------------------------------------------------------------------
// EvictionPolicy enum
// ---------------------------------------------------------------------------
enum class EvictionPolicy : uint8_t {
    FIFO            = 0,  // oldest first
    LRU             = 1,  // least recently accessed first
    IMPORTANCE_ASC  = 2,  // lowest importance first
};

inline EvictionPolicy eviction_policy_from_string(const std::string& s) {
    if (s == "fifo")             return EvictionPolicy::FIFO;
    if (s == "lru")              return EvictionPolicy::LRU;
    if (s == "importance_asc")   return EvictionPolicy::IMPORTANCE_ASC;
    throw std::invalid_argument("Unknown eviction policy: " + s);
}

// ---------------------------------------------------------------------------
// MemoryConfig — per-engine limits and tuning knobs
// ---------------------------------------------------------------------------
struct MemoryConfig {
    // Tier capacities (0 = unlimited for long_term)
    std::size_t max_working    = 16;
    std::size_t max_short_term = 128;
    std::size_t max_long_term  = 0;    // 0 = unlimited

    // Max content size per item (bytes; 0 = unlimited)
    std::size_t max_content_bytes = 65536;  // 64 KiB

    // Max results returned by search
    std::size_t max_search_results = 20;

    // Eviction policy for working/short_term when at capacity
    EvictionPolicy working_eviction    = EvictionPolicy::LRU;
    EvictionPolicy short_term_eviction = EvictionPolicy::LRU;

    // Recency decay half-life in seconds for scoring (0 = no decay)
    double recency_half_life_seconds = 3600.0 * 24;  // 1 day

    // Consolidation: automatically promote from STM to LTM when importance >= threshold
    bool   auto_consolidate               = false;
    double consolidation_importance_threshold = 0.8;

    // Persistence path (empty = no persistence / in-memory only)
    std::string db_path;

    // Validate returns false and sets error_msg if a field is illegal.
    bool validate(std::string& error_msg) const {
        if (max_content_bytes > (1ULL << 24)) {
            error_msg = "max_content_bytes too large (> 16 MiB)";
            return false;
        }
        if (importance_in_range(-0.01) || !importance_in_range(1.0)) {
            // just checking the helper compiles; actual validation is in add()
        }
        return true;
    }

private:
    static bool importance_in_range(double v) { return v >= 0.0 && v <= 1.0; }
};

} // namespace memory
} // namespace prime
