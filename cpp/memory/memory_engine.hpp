#pragma once

/*
 * memory_engine.hpp — PRIME Memory Engine (Phase 3).
 *
 * MemoryEngine is the single developer-facing C++ class.  It manages:
 *
 *   - ID generation (monotonic counter, thread-safe)
 *   - Three logical memory tiers (WORKING / SHORT_TERM / LONG_TERM)
 *   - Capacity enforcement + eviction
 *   - Memory consolidation (STM → LTM)
 *   - Access tracking
 *   - Expiration
 *   - Search / retrieve_context
 *   - Embedding provider hook (stub for Phase 3)
 *
 * The engine is NOT a Module subclass — it is a self-contained subsystem
 * that PRIME.memory() returns.  Future phases may wrap it in a Module.
 */

#include "memory_types.hpp"
#include "memory_store.hpp"
#include "sqlite_store.hpp"
#include "../core/status.hpp"

#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <functional>

namespace prime {
namespace memory {

// ---------------------------------------------------------------------------
// EmbeddingProvider — stub interface for future semantic retrieval (Phase 8+)
// ---------------------------------------------------------------------------
class EmbeddingProvider {
public:
    virtual ~EmbeddingProvider() = default;
    // Returns an embedding vector for text, or empty if unsupported.
    virtual std::vector<float> embed(const std::string& /*text*/) const { return {}; }
};

// ---------------------------------------------------------------------------
// MemoryEngine
// ---------------------------------------------------------------------------
class MemoryEngine {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    explicit MemoryEngine(MemoryConfig config = {})
        : config_(std::move(config))
    {}

    ~MemoryEngine() { (void)shutdown(); }

    MemoryEngine(const MemoryEngine&) = delete;
    MemoryEngine& operator=(const MemoryEngine&) = delete;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------
    Status initialize() {
        std::lock_guard<std::mutex> lk(mu_);
        if (initialized_) {
            return Status::error(StatusCode::ERROR_ALREADY_INITIALIZED,
                                 "MemoryEngine already initialized");
        }

        // Choose backend based on config_.db_path
        if (config_.db_path.empty()) {
            store_ = std::make_unique<InMemoryStore>();
        } else {
            auto sqlite = std::make_unique<SqliteStore>();
            Status s = sqlite->open(config_.db_path);
            if (!s.ok()) {
                // Graceful fallback to in-memory if SQLite unavailable
                store_ = std::make_unique<InMemoryStore>();
            } else {
                store_ = std::move(sqlite);
            }
        }

        Status s = store_->open("");  // InMemoryStore open() is a no-op
        (void)s;

        initialized_ = true;
        return Status::ok_status();
    }

    Status shutdown() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!initialized_) return Status::ok_status();
        if (store_) store_->close();
        initialized_ = false;
        return Status::ok_status();
    }

    bool is_initialized() const { return initialized_; }

    const MemoryConfig& config() const { return config_; }

    // -----------------------------------------------------------------------
    // Core CRUD
    // -----------------------------------------------------------------------

    // store() — create a new memory item, returns assigned ID.
    // Throws std::invalid_argument on validation failure.
    uint64_t store(const std::string& content,
                   MemoryType         type       = MemoryType::LONG_TERM,
                   double             importance  = 0.5,
                   double             confidence  = 1.0,
                   const Metadata&    metadata    = {},
                   double             expires_at  = 0.0)
    {
        require_initialized();
        validate_content(content);
        validate_importance(importance);
        validate_confidence(confidence);

        MemoryItem item;
        item.id           = next_id_.fetch_add(1, std::memory_order_relaxed);
        item.type         = type;
        item.content      = content;
        item.metadata     = metadata;
        item.importance   = importance;
        item.confidence   = confidence;
        item.created_at   = now_seconds();
        item.last_accessed= item.created_at;
        item.expires_at   = expires_at;

        // Enforce capacity + evict before inserting
        enforce_capacity(type);

        Status s = store_->put(item);
        if (!s.ok()) throw std::runtime_error(s.message);

        // Consolidation check
        if (config_.auto_consolidate && type == MemoryType::SHORT_TERM
            && importance >= config_.consolidation_importance_threshold) {
            consolidate_item(item);
        }

        return item.id;
    }

    // get() — retrieve by ID; throws if not found.
    MemoryItem get(uint64_t id) {
        require_initialized();
        MemoryItem item;
        Status s = store_->get(id, item);
        if (!s.ok()) throw std::runtime_error(s.message);
        touch(id);
        return item;
    }

    // update() — update fields on an existing item.
    void update(uint64_t id,
                std::optional<std::string>  content     = std::nullopt,
                std::optional<double>       importance  = std::nullopt,
                std::optional<double>       confidence  = std::nullopt,
                std::optional<Metadata>     metadata    = std::nullopt,
                std::optional<MemoryType>   new_type    = std::nullopt,
                std::optional<double>       expires_at  = std::nullopt)
    {
        require_initialized();
        MemoryItem item;
        Status s = store_->get(id, item);
        if (!s.ok()) throw std::runtime_error(s.message);

        if (content)    { validate_content(*content);     item.content    = *content; }
        if (importance) { validate_importance(*importance); item.importance = *importance; }
        if (confidence) { validate_confidence(*confidence); item.confidence = *confidence; }
        if (metadata)   item.metadata   = *metadata;
        if (new_type)   item.type       = *new_type;
        if (expires_at) item.expires_at = *expires_at;

        s = store_->update(item);
        if (!s.ok()) throw std::runtime_error(s.message);
    }

    // delete_memory() — remove by ID.
    void delete_memory(uint64_t id) {
        require_initialized();
        Status s = store_->remove(id);
        if (!s.ok()) throw std::runtime_error(s.message);
    }

    // -----------------------------------------------------------------------
    // Working Memory helpers
    // -----------------------------------------------------------------------
    uint64_t working_add(const std::string& content,
                         double importance = 0.5,
                         const Metadata& metadata = {}) {
        return store(content, MemoryType::WORKING, importance, 1.0, metadata);
    }

    void working_clear() {
        require_initialized();
        Status s = store_->clear(MemoryType::WORKING);
        if (!s.ok()) throw std::runtime_error(s.message);
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------
    std::vector<MemoryItem> search(const std::string& query_text,
                                   std::size_t                     limit           = 10,
                                   std::optional<MemoryType>       filter_type     = std::nullopt,
                                   const Metadata&                 filter_metadata = {})
    {
        require_initialized();
        SearchQuery q;
        q.query_text      = query_text;
        q.limit           = limit;
        q.filter_type     = filter_type;
        q.filter_metadata = filter_metadata;

        auto results = store_->search(q, config_.recency_half_life_seconds);

        // Track access for returned items
        double now = now_seconds();
        for (auto& item : results) {
            store_touch(item.id, now);
        }

        return results;
    }

    // retrieve_context() — clean interface for future phases to get context.
    // Returns at most `limit` relevant MemoryItems for a given query.
    std::vector<MemoryItem> retrieve_context(const std::string& query,
                                              std::size_t limit = 5) {
        return search(query, limit);
    }

    // -----------------------------------------------------------------------
    // Bookkeeping
    // -----------------------------------------------------------------------
    std::size_t count(std::optional<MemoryType> type_filter = std::nullopt) const {
        require_initialized();
        return store_->count(type_filter);
    }

    bool exists(uint64_t id) const {
        require_initialized();
        return store_->exists(id);
    }

    void clear(std::optional<MemoryType> type_filter = std::nullopt) {
        require_initialized();
        Status s = store_->clear(type_filter);
        if (!s.ok()) throw std::runtime_error(s.message);
    }

    // expire_all() — remove every expired item across all tiers.
    std::size_t expire_all() {
        require_initialized();
        // Search all without text filter to get every item
        SearchQuery q;
        q.query_text = "";
        q.limit      = 0;
        // We need raw access; gather IDs of expired items manually.
        // Perform a full scan by getting all ids from each tier.
        std::size_t removed = 0;
        double now = now_seconds();

        for (auto tier : {MemoryType::WORKING, MemoryType::SHORT_TERM, MemoryType::LONG_TERM}) {
            auto ids = store_->ids_by_insertion(tier);
            for (uint64_t id : ids) {
                MemoryItem item;
                if (!store_->get(id, item).ok()) continue;
                if (item.is_expired(now)) {
                    store_->remove(id);
                    ++removed;
                }
            }
        }
        return removed;
    }

    // consolidate() — promote qualifying STM items into LTM.
    std::size_t consolidate(double importance_threshold = -1.0) {
        require_initialized();
        double thresh = (importance_threshold >= 0.0)
            ? importance_threshold
            : config_.consolidation_importance_threshold;

        auto ids = store_->ids_by_insertion(MemoryType::SHORT_TERM);
        std::size_t promoted = 0;
        for (uint64_t id : ids) {
            MemoryItem item;
            if (!store_->get(id, item).ok()) continue;
            if (item.importance >= thresh) {
                item.type = MemoryType::LONG_TERM;
                store_->update(item);
                ++promoted;
            }
        }
        return promoted;
    }

    // -----------------------------------------------------------------------
    // Embedding hook (Phase 3: stub only)
    // -----------------------------------------------------------------------
    void set_embedding_provider(std::shared_ptr<EmbeddingProvider> provider) {
        embedding_provider_ = std::move(provider);
    }

    // -----------------------------------------------------------------------
    // Access to raw store (for Python binding layer)
    // -----------------------------------------------------------------------
    MemoryStore* store() const { return store_.get(); }

private:
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    void require_initialized() const {
        if (!initialized_)
            throw std::runtime_error("MemoryEngine not initialized");
    }

    static void validate_content(const std::string& content) {
        if (content.empty())
            throw std::invalid_argument("Memory content must not be empty");
    }

    static void validate_importance(double v) {
        if (v < 0.0 || v > 1.0)
            throw std::invalid_argument("importance must be in [0.0, 1.0]");
    }

    static void validate_confidence(double v) {
        if (v < 0.0 || v > 1.0)
            throw std::invalid_argument("confidence must be in [0.0, 1.0]");
    }

    // enforce_capacity evicts items from `type` tier if at/over limit.
    void enforce_capacity(MemoryType type) {
        std::size_t max_cap = 0;
        EvictionPolicy policy = EvictionPolicy::LRU;

        if (type == MemoryType::WORKING) {
            max_cap = config_.max_working;
            policy  = config_.working_eviction;
        } else if (type == MemoryType::SHORT_TERM) {
            max_cap = config_.max_short_term;
            policy  = config_.short_term_eviction;
        } else {
            max_cap = config_.max_long_term;
            policy  = EvictionPolicy::FIFO;
        }

        if (max_cap == 0) return;  // unlimited

        while (store_->count(type) >= max_cap) {
            evict_one(type, policy);
        }
    }

    // evict_one removes the single best eviction candidate from `type`.
    void evict_one(MemoryType type, EvictionPolicy policy) {
        auto ids = store_->ids_by_insertion(type);
        if (ids.empty()) return;

        uint64_t victim = ids.front();  // default: FIFO

        if (policy == EvictionPolicy::LRU) {
            // Find least recently accessed
            double oldest_access = std::numeric_limits<double>::max();
            for (uint64_t id : ids) {
                MemoryItem item;
                if (!store_->get(id, item).ok()) continue;
                if (item.last_accessed < oldest_access) {
                    oldest_access = item.last_accessed;
                    victim = id;
                }
            }
        } else if (policy == EvictionPolicy::IMPORTANCE_ASC) {
            double lowest = std::numeric_limits<double>::max();
            for (uint64_t id : ids) {
                MemoryItem item;
                if (!store_->get(id, item).ok()) continue;
                if (item.importance < lowest) {
                    lowest = item.importance;
                    victim = id;
                }
            }
        }
        // FIFO: victim is already ids.front()

        store_->remove(victim);
    }

    // consolidate_item: copy a single item into LTM (called from store()).
    void consolidate_item(const MemoryItem& item) {
        MemoryItem lt = item;
        lt.id   = next_id_.fetch_add(1, std::memory_order_relaxed);
        lt.type = MemoryType::LONG_TERM;
        store_->put(lt);
    }

    // touch: update access_count and last_accessed for an item.
    void touch(uint64_t id) {
        double now = now_seconds();
        store_touch(id, now);
    }

    void store_touch(uint64_t id, double at) {
        MemoryItem item;
        if (!store_->get(id, item).ok()) return;
        item.access_count++;
        item.last_accessed = at;
        store_->update(item);
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    mutable std::mutex mu_;
    MemoryConfig       config_;
    bool               initialized_ = false;

    std::unique_ptr<MemoryStore> store_;
    std::atomic<uint64_t>        next_id_{1};

    std::shared_ptr<EmbeddingProvider> embedding_provider_;
};

} // namespace memory
} // namespace prime
