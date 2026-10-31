/*
 * test_memory.cpp — C++ unit tests for the PRIME Memory Engine (Phase 3).
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
 *       -I cpp tests/cpp/test_memory.cpp -o tests/cpp/test_memory
 *   (Add -DPRIME_HAVE_SQLITE -lsqlite3 if SQLite is available.)
 *
 * Run:
 *   tests/cpp/test_memory
 */

#include "../../cpp/memory/memory_types.hpp"
#include "../../cpp/memory/memory_store.hpp"
#include "../../cpp/memory/sqlite_store.hpp"
#include "../../cpp/memory/memory_engine.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <thread>
#include <atomic>

using namespace prime;
using namespace prime::memory;

// ---------------------------------------------------------------------------
// Minimal test framework (same style as Phase 1 / Phase 2 tests)
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;
// Per-test fail flag; reset in run() before each test.
static int g_cur_fail = 0;

#define ASSERT_TRUE(expr) do {                                    \
    if (!(expr)) {                                                \
        fprintf(stderr, "  [FAIL] %s:%d: %s\n",                  \
                __FILE__, __LINE__, #expr);                       \
        ++g_cur_fail; return;                                     \
    }                                                             \
} while (0)

#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))
#define ASSERT_GT(a, b) ASSERT_TRUE((a) > (b))
#define ASSERT_GE(a, b) ASSERT_TRUE((a) >= (b))
#define ASSERT_LT(a, b) ASSERT_TRUE((a) < (b))

#define ASSERT_THROWS(stmt) do {                                  \
    bool threw = false;                                           \
    try { stmt; } catch (...) { threw = true; }                   \
    if (!threw) {                                                 \
        fprintf(stderr, "  [FAIL] %s:%d: expected exception from: %s\n", \
                __FILE__, __LINE__, #stmt);                       \
        ++g_cur_fail; return;                                     \
    }                                                             \
} while (0)

static void run(const char* name, void(*fn)()) {
    g_cur_fail = 0;
    fn();
    if (g_cur_fail == 0) {
        printf("  [PASS] %s\n", name);
        ++g_pass;
    } else {
        printf("  [FAIL] %s\n", name);
        ++g_fail;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// Returns heap-allocated engine to avoid copy (MemoryEngine is move-only).
// Tests use it via reference.
static std::unique_ptr<MemoryEngine> make_engine_ptr(
        std::size_t max_w = 16, std::size_t max_s = 128) {
    MemoryConfig cfg;
    cfg.max_working    = max_w;
    cfg.max_short_term = max_s;
    cfg.max_long_term  = 0;
    cfg.recency_half_life_seconds = 0;  // no decay for deterministic tests
    auto eng = std::make_unique<MemoryEngine>(cfg);
    Status s = eng->initialize();
    assert(s.ok());
    return eng;
}
// Convenience macro: declare a local MemoryEngine& from the heap ptr.
#define MAKE_ENGINE(name, ...) \
    auto name##_ptr = make_engine_ptr(__VA_ARGS__); \
    MemoryEngine& name = *name##_ptr

// ---------------------------------------------------------------------------
// Tests — MemoryTypes
// ---------------------------------------------------------------------------
static void test_memory_type_names() {
    ASSERT_EQ(std::string(memory_type_name(MemoryType::WORKING)),    "working");
    ASSERT_EQ(std::string(memory_type_name(MemoryType::SHORT_TERM)), "short_term");
    ASSERT_EQ(std::string(memory_type_name(MemoryType::LONG_TERM)),  "long_term");
}

static void test_memory_type_from_string() {
    ASSERT_EQ(memory_type_from_string("working"),    MemoryType::WORKING);
    ASSERT_EQ(memory_type_from_string("short_term"), MemoryType::SHORT_TERM);
    ASSERT_EQ(memory_type_from_string("long_term"),  MemoryType::LONG_TERM);
}

static void test_memory_type_invalid() {
    ASSERT_THROWS(memory_type_from_string("bogus"));
}

static void test_eviction_policy_from_string() {
    ASSERT_EQ(eviction_policy_from_string("fifo"),           EvictionPolicy::FIFO);
    ASSERT_EQ(eviction_policy_from_string("lru"),            EvictionPolicy::LRU);
    ASSERT_EQ(eviction_policy_from_string("importance_asc"), EvictionPolicy::IMPORTANCE_ASC);
    ASSERT_THROWS(eviction_policy_from_string("invalid"));
}

// ---------------------------------------------------------------------------
// Tests — MemoryItem
// ---------------------------------------------------------------------------
static void test_memory_item_expiration() {
    MemoryItem item;
    item.expires_at = 0.0;
    ASSERT_TRUE(!item.is_expired());  // never expires

    // Set expiry in the past
    item.expires_at = now_seconds() - 1.0;
    ASSERT_TRUE(item.is_expired());

    // Set expiry in the future
    item.expires_at = now_seconds() + 3600.0;
    ASSERT_TRUE(!item.is_expired());
}

static void test_memory_item_recency() {
    MemoryItem item;
    item.created_at = now_seconds();
    // With zero half-life: no decay
    ASSERT_EQ(item.recency_score(0.0), 1.0);

    // Newly created item: nearly 1.0
    double r = item.recency_score(3600.0);
    ASSERT_GT(r, 0.99);  // just created

    // Item created 1 day ago with 1-day half-life: should be ~0.5
    MemoryItem old_item;
    old_item.created_at = now_seconds() - 86400.0;
    double r2 = old_item.recency_score(86400.0);
    ASSERT_GT(r2, 0.45);
    ASSERT_LT(r2, 0.55);
}

static void test_memory_item_score() {
    MemoryItem item;
    item.importance   = 1.0;
    item.access_count = 0;
    item.created_at   = now_seconds();

    double s = item.score(1.0, 0.0);
    ASSERT_GT(s, 0.0);
    ASSERT_LT(s, 1.5);

    // More importance → higher score
    MemoryItem item2;
    item2.importance   = 0.1;
    item2.access_count = 0;
    item2.created_at   = now_seconds();
    double s2 = item2.score(1.0, 0.0);
    ASSERT_GT(s, s2);
}

// ---------------------------------------------------------------------------
// Tests — text_util
// ---------------------------------------------------------------------------
static void test_text_util_jaccard() {
    using namespace text_util;
    ASSERT_GT(jaccard("machine learning", "machine learning is great"), 0.0);
    ASSERT_EQ(jaccard("foo", "bar baz"), 0.0);
    ASSERT_GT(jaccard("python", "Python is a language"), 0.0);  // case-insensitive
}

static void test_text_util_contains() {
    using namespace text_util;
    ASSERT_EQ(contains_score("python", "Python is great"), 1.0);
    ASSERT_EQ(contains_score("rust",   "Python is great"), 0.0);
}

static void test_text_util_similarity() {
    using namespace text_util;
    double s = similarity("machine learning", "machine learning subset of AI");
    ASSERT_GT(s, 0.0);
}

// ---------------------------------------------------------------------------
// Tests — InMemoryStore
// ---------------------------------------------------------------------------
static MemoryItem make_item(uint64_t id, MemoryType t,
                             const std::string& content, double imp = 0.5) {
    MemoryItem item;
    item.id           = id;
    item.type         = t;
    item.content      = content;
    item.importance   = imp;
    item.created_at   = now_seconds();
    item.last_accessed= item.created_at;
    return item;
}

static void test_store_put_get() {
    InMemoryStore s;
    auto item = make_item(1, MemoryType::LONG_TERM, "hello");
    ASSERT_TRUE(s.put(item).ok());
    MemoryItem out;
    ASSERT_TRUE(s.get(1, out).ok());
    ASSERT_EQ(out.content, "hello");
}

static void test_store_duplicate_id() {
    InMemoryStore s;
    auto item = make_item(1, MemoryType::LONG_TERM, "a");
    ASSERT_TRUE(s.put(item).ok());
    auto dup = make_item(1, MemoryType::LONG_TERM, "b");
    ASSERT_TRUE(!s.put(dup).ok());
}

static void test_store_get_missing() {
    InMemoryStore s;
    MemoryItem out;
    ASSERT_TRUE(!s.get(999, out).ok());
}

static void test_store_update() {
    InMemoryStore s;
    auto item = make_item(1, MemoryType::LONG_TERM, "original");
    s.put(item);
    item.content = "updated";
    ASSERT_TRUE(s.update(item).ok());
    MemoryItem out;
    s.get(1, out);
    ASSERT_EQ(out.content, "updated");
}

static void test_store_remove() {
    InMemoryStore s;
    s.put(make_item(1, MemoryType::LONG_TERM, "a"));
    ASSERT_TRUE(s.remove(1).ok());
    ASSERT_TRUE(!s.exists(1));
    ASSERT_TRUE(!s.remove(1).ok());
}

static void test_store_clear_all() {
    InMemoryStore s;
    s.put(make_item(1, MemoryType::LONG_TERM,  "a"));
    s.put(make_item(2, MemoryType::SHORT_TERM, "b"));
    s.clear(std::nullopt);
    ASSERT_EQ(s.count(std::nullopt), 0u);
}

static void test_store_clear_typed() {
    InMemoryStore s;
    s.put(make_item(1, MemoryType::LONG_TERM,  "a"));
    s.put(make_item(2, MemoryType::SHORT_TERM, "b"));
    s.clear(MemoryType::SHORT_TERM);
    ASSERT_EQ(s.count(std::nullopt), 1u);
    ASSERT_TRUE(s.exists(1));
    ASSERT_TRUE(!s.exists(2));
}

static void test_store_count() {
    InMemoryStore s;
    ASSERT_EQ(s.count(std::nullopt), 0u);
    s.put(make_item(1, MemoryType::LONG_TERM,  "a"));
    s.put(make_item(2, MemoryType::SHORT_TERM, "b"));
    s.put(make_item(3, MemoryType::WORKING,    "c"));
    ASSERT_EQ(s.count(std::nullopt), 3u);
    ASSERT_EQ(s.count(MemoryType::LONG_TERM),  1u);
    ASSERT_EQ(s.count(MemoryType::SHORT_TERM), 1u);
    ASSERT_EQ(s.count(MemoryType::WORKING),    1u);
}

static void test_store_search_text() {
    InMemoryStore s;
    s.put(make_item(1, MemoryType::LONG_TERM, "Python is great"));
    s.put(make_item(2, MemoryType::LONG_TERM, "Rust is fast"));
    SearchQuery q;
    q.query_text = "Python";
    q.limit = 10;
    auto r = s.search(q, 0.0);
    ASSERT_GE(r.size(), 1u);
    ASSERT_EQ(r[0].id, 1u);
}

static void test_store_search_type_filter() {
    InMemoryStore s;
    s.put(make_item(1, MemoryType::LONG_TERM,  "Python long_term"));
    s.put(make_item(2, MemoryType::SHORT_TERM, "Python short_term"));
    SearchQuery q;
    q.query_text  = "Python";
    q.filter_type = MemoryType::LONG_TERM;
    q.limit       = 10;
    auto r = s.search(q, 0.0);
    ASSERT_EQ(r.size(), 1u);
    ASSERT_EQ(r[0].id, 1u);
}

static void test_store_search_metadata_filter() {
    InMemoryStore s;
    auto item1 = make_item(1, MemoryType::LONG_TERM, "item one");
    item1.metadata["topic"] = "AI";
    s.put(item1);
    auto item2 = make_item(2, MemoryType::LONG_TERM, "item two");
    item2.metadata["topic"] = "cooking";
    s.put(item2);

    SearchQuery q;
    q.query_text       = "";
    q.filter_metadata["topic"] = "AI";
    q.limit            = 10;
    auto r = s.search(q, 0.0);
    ASSERT_EQ(r.size(), 1u);
    ASSERT_EQ(r[0].id, 1u);
}

static void test_store_search_expired_excluded() {
    InMemoryStore s;
    auto item = make_item(1, MemoryType::LONG_TERM, "expiring");
    item.expires_at = now_seconds() - 1.0;  // already expired
    s.put(item);
    SearchQuery q; q.query_text = ""; q.limit = 10;
    auto r = s.search(q, 0.0);
    ASSERT_EQ(r.size(), 0u);
}

static void test_store_ids_by_insertion() {
    InMemoryStore s;
    s.put(make_item(10, MemoryType::LONG_TERM, "a"));
    s.put(make_item(20, MemoryType::LONG_TERM, "b"));
    s.put(make_item(30, MemoryType::LONG_TERM, "c"));
    auto ids = s.ids_by_insertion(MemoryType::LONG_TERM);
    ASSERT_EQ(ids.size(), 3u);
    ASSERT_EQ(ids[0], 10u);
    ASSERT_EQ(ids[2], 30u);
}

// ---------------------------------------------------------------------------
// Tests — MemoryEngine
// ---------------------------------------------------------------------------
static void test_engine_initialize() {
    MemoryEngine eng;
    ASSERT_TRUE(!eng.is_initialized());
    ASSERT_TRUE(eng.initialize().ok());
    ASSERT_TRUE(eng.is_initialized());
}

static void test_engine_double_initialize() {
    MemoryEngine eng;
    eng.initialize();
    ASSERT_TRUE(!eng.initialize().ok());
}

static void test_engine_store_get() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("hello", MemoryType::LONG_TERM, 0.8);
    ASSERT_GT(id, 0u);
    auto item = eng.get(id);
    ASSERT_EQ(item.content, "hello");
    ASSERT_EQ(item.importance, 0.8);
}

static void test_engine_store_empty_content() {
    MAKE_ENGINE(eng);
    ASSERT_THROWS(eng.store("", MemoryType::LONG_TERM, 0.5));
}

static void test_engine_store_invalid_importance() {
    MAKE_ENGINE(eng);
    ASSERT_THROWS(eng.store("test", MemoryType::LONG_TERM, 1.5));
    ASSERT_THROWS(eng.store("test", MemoryType::LONG_TERM, -0.1));
}

static void test_engine_get_missing() {
    MAKE_ENGINE(eng);
    ASSERT_THROWS(eng.get(9999));
}

static void test_engine_update() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("original", MemoryType::LONG_TERM, 0.5);
    eng.update(id, "updated", 0.9);
    auto item = eng.get(id);
    ASSERT_EQ(item.content, "updated");
    ASSERT_EQ(item.importance, 0.9);
}

static void test_engine_update_missing() {
    MAKE_ENGINE(eng);
    ASSERT_THROWS(eng.update(9999, "x", 0.5));
}

static void test_engine_delete() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("bye", MemoryType::LONG_TERM, 0.5);
    eng.delete_memory(id);
    ASSERT_TRUE(!eng.exists(id));
    ASSERT_THROWS(eng.get(id));
}

static void test_engine_delete_missing() {
    MAKE_ENGINE(eng);
    ASSERT_THROWS(eng.delete_memory(9999));
}

static void test_engine_count() {
    MAKE_ENGINE(eng);
    ASSERT_EQ(eng.count(), 0u);
    eng.store("a", MemoryType::LONG_TERM);
    eng.store("b", MemoryType::SHORT_TERM);
    eng.store("c", MemoryType::WORKING);
    ASSERT_EQ(eng.count(), 3u);
    ASSERT_EQ(eng.count(MemoryType::LONG_TERM),  1u);
    ASSERT_EQ(eng.count(MemoryType::SHORT_TERM), 1u);
    ASSERT_EQ(eng.count(MemoryType::WORKING),    1u);
}

static void test_engine_exists() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("x", MemoryType::LONG_TERM);
    ASSERT_TRUE(eng.exists(id));
    eng.delete_memory(id);
    ASSERT_TRUE(!eng.exists(id));
}

static void test_engine_clear_all() {
    MAKE_ENGINE(eng);
    eng.store("a", MemoryType::LONG_TERM);
    eng.store("b", MemoryType::SHORT_TERM);
    eng.clear();
    ASSERT_EQ(eng.count(), 0u);
}

static void test_engine_clear_typed() {
    MAKE_ENGINE(eng);
    eng.store("a", MemoryType::LONG_TERM);
    eng.store("b", MemoryType::SHORT_TERM);
    eng.clear(MemoryType::SHORT_TERM);
    ASSERT_EQ(eng.count(), 1u);
    ASSERT_EQ(eng.count(MemoryType::LONG_TERM), 1u);
}

static void test_engine_search() {
    MAKE_ENGINE(eng);
    eng.store("Python is a programming language", MemoryType::LONG_TERM, 0.8);
    eng.store("Rust is systems programming",      MemoryType::LONG_TERM, 0.7);
    eng.store("Machine learning basics",           MemoryType::LONG_TERM, 0.6);

    auto results = eng.search("Python programming");
    ASSERT_GE(results.size(), 1u);
    ASSERT_EQ(results[0].content, "Python is a programming language");
}

static void test_engine_search_empty_query() {
    MAKE_ENGINE(eng);
    eng.store("a", MemoryType::LONG_TERM);
    eng.store("b", MemoryType::LONG_TERM);
    auto results = eng.search("", 10);
    ASSERT_EQ(results.size(), 2u);
}

static void test_engine_retrieve_context() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("Deep learning uses neural networks",
                              MemoryType::LONG_TERM, 0.9);
    eng.store("The weather is sunny", MemoryType::LONG_TERM, 0.3);
    auto ctx = eng.retrieve_context("neural networks", 3);
    ASSERT_GE(ctx.size(), 1u);
    ASSERT_EQ(ctx[0].id, id);
}

static void test_engine_working_memory() {
    MAKE_ENGINE(eng, 4);
    eng.working_add("step1");
    eng.working_add("step2");
    eng.working_add("step3");
    ASSERT_EQ(eng.count(MemoryType::WORKING), 3u);
    eng.working_clear();
    ASSERT_EQ(eng.count(MemoryType::WORKING), 0u);
}

static void test_engine_working_capacity_fifo() {
    MemoryConfig cfg;
    cfg.max_working        = 3;
    cfg.working_eviction   = EvictionPolicy::FIFO;
    cfg.recency_half_life_seconds = 0;
    MemoryEngine eng(cfg); eng.initialize();

    uint64_t id1 = eng.working_add("a");
    uint64_t id2 = eng.working_add("b");
    uint64_t id3 = eng.working_add("c");
    // Adding a 4th evicts the first (FIFO)
    eng.working_add("d");
    ASSERT_EQ(eng.count(MemoryType::WORKING), 3u);
    ASSERT_TRUE(!eng.exists(id1));
    ASSERT_TRUE(eng.exists(id2));
    ASSERT_TRUE(eng.exists(id3));
}

static void test_engine_short_term_lru() {
    MemoryConfig cfg;
    cfg.max_short_term       = 3;
    cfg.short_term_eviction  = EvictionPolicy::LRU;
    cfg.recency_half_life_seconds = 0;
    MemoryEngine eng(cfg); eng.initialize();

    uint64_t id1 = eng.store("a", MemoryType::SHORT_TERM);
    (void)eng.store("b", MemoryType::SHORT_TERM);
    uint64_t id3 = eng.store("c", MemoryType::SHORT_TERM);

    // Access id1 to make it recently used
    eng.get(id1);

    // Adding a 4th should evict the LRU (id2, which was never accessed after store)
    eng.store("d", MemoryType::SHORT_TERM);
    ASSERT_EQ(eng.count(MemoryType::SHORT_TERM), 3u);
    // id1 was accessed more recently, so it should survive
    ASSERT_TRUE(eng.exists(id1));
    ASSERT_TRUE(eng.exists(id3));
}

static void test_engine_importance_eviction() {
    MemoryConfig cfg;
    cfg.max_working        = 3;
    cfg.working_eviction   = EvictionPolicy::IMPORTANCE_ASC;
    cfg.recency_half_life_seconds = 0;
    MemoryEngine eng(cfg); eng.initialize();

    eng.working_add("low",  0.1);
    uint64_t hi = eng.working_add("high", 0.9);
    eng.working_add("mid",  0.5);
    // Adding a 4th evicts the lowest-importance item ("low")
    eng.working_add("new", 0.5);
    ASSERT_EQ(eng.count(MemoryType::WORKING), 3u);
    ASSERT_TRUE(eng.exists(hi));
}

static void test_engine_expiration() {
    MAKE_ENGINE(eng);
    // Store an already-expired item and a valid item
    double past = now_seconds() - 1.0;
    uint64_t exp_id = eng.store("only expired content", MemoryType::LONG_TERM,
                                 0.5, 1.0, {}, past);
    eng.store("valid content",   MemoryType::LONG_TERM, 0.5);
    ASSERT_EQ(eng.count(), 2u);

    // expire_all should remove the expired item
    std::size_t removed = eng.expire_all();
    ASSERT_EQ(removed, 1u);
    ASSERT_EQ(eng.count(), 1u);

    // The expired item should no longer exist by ID
    ASSERT_TRUE(!eng.exists(exp_id));

    // Search for the expired item's unique text should yield nothing
    // (the store's search skips expired items; after expire_all the item
    //  has been fully removed, so it won't appear at all)
    auto all = eng.search("", 100);
    for (auto& r : all) {
        ASSERT_NE(r.id, exp_id);
    }
    ASSERT_EQ(all.size(), 1u);  // only the valid item remains
}

static void test_engine_metadata() {
    MAKE_ENGINE(eng);
    Metadata meta;
    meta["topic"] = "AI";
    meta["lang"]  = "en";
    uint64_t id = eng.store("neural nets", MemoryType::LONG_TERM, 0.7, 1.0, meta);
    auto item = eng.get(id);
    ASSERT_EQ(item.metadata["topic"], "AI");
    ASSERT_EQ(item.metadata["lang"],  "en");
}

static void test_engine_metadata_search() {
    MAKE_ENGINE(eng);
    Metadata m1; m1["topic"] = "cooking";
    Metadata m2; m2["topic"] = "AI";
    eng.store("recipe",   MemoryType::LONG_TERM, 0.5, 1.0, m1);
    uint64_t id = eng.store("AI paper", MemoryType::LONG_TERM, 0.9, 1.0, m2);

    Metadata filter; filter["topic"] = "AI";
    auto results = eng.search("", 10, std::nullopt, filter);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].id, id);
}

static void test_engine_consolidation() {
    MemoryConfig cfg;
    cfg.consolidation_importance_threshold = 0.7;
    cfg.recency_half_life_seconds = 0;
    MemoryEngine eng(cfg); eng.initialize();

    eng.store("important", MemoryType::SHORT_TERM, 0.9);
    eng.store("less",      MemoryType::SHORT_TERM, 0.3);

    std::size_t promoted = eng.consolidate(0.7);
    ASSERT_EQ(promoted, 1u);
    ASSERT_EQ(eng.count(MemoryType::LONG_TERM), 1u);
}

static void test_engine_access_tracking() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("tracked", MemoryType::LONG_TERM, 0.5);
    // Initial access_count after store is 0 in the store itself; get() bumps it
    eng.get(id);
    eng.get(id);
    auto item = eng.get(id);
    ASSERT_GE(item.access_count, 2u);
}

static void test_engine_update_type() {
    MAKE_ENGINE(eng);
    uint64_t id = eng.store("promote me", MemoryType::SHORT_TERM);
    ASSERT_EQ(eng.count(MemoryType::SHORT_TERM), 1u);
    eng.update(id, std::nullopt, std::nullopt, std::nullopt,
               std::nullopt, MemoryType::LONG_TERM);
    ASSERT_EQ(eng.count(MemoryType::SHORT_TERM), 0u);
    ASSERT_EQ(eng.count(MemoryType::LONG_TERM),  1u);
}

static void test_engine_shutdown() {
    MemoryEngine eng; eng.initialize();
    ASSERT_TRUE(eng.shutdown().ok());
    ASSERT_TRUE(!eng.is_initialized());
    // Double shutdown is OK
    ASSERT_TRUE(eng.shutdown().ok());
}

// ---------------------------------------------------------------------------
// Tests — thread safety (basic)
// ---------------------------------------------------------------------------
static void test_engine_concurrent_store() {
    MAKE_ENGINE(eng, 0, 0);  // no capacity limit
    constexpr int N = 50;
    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&, i]() {
            try {
                eng.store("item " + std::to_string(i), MemoryType::LONG_TERM, 0.5);
            } catch (...) { errors++; }
        });
    }
    for (auto& t : threads) t.join();

    ASSERT_EQ(errors.load(), 0);
    ASSERT_EQ(eng.count(), static_cast<std::size_t>(N));
}

// ---------------------------------------------------------------------------
// Tests — JSON metadata serialization
// ---------------------------------------------------------------------------
static void test_json_roundtrip() {
    Metadata m;
    m["key1"] = "value1";
    m["key2"] = "hello world";
    m["key3"] = "with \"quotes\"";
    m["key4"] = "newline\n";

    std::string json = json_util::serialise(m);
    Metadata out = json_util::parse(json);

    ASSERT_EQ(out["key1"], "value1");
    ASSERT_EQ(out["key2"], "hello world");
    ASSERT_EQ(out["key3"], "with \"quotes\"");
    ASSERT_EQ(out["key4"], "newline\n");
}

static void test_json_empty() {
    Metadata m;
    std::string json = json_util::serialise(m);
    ASSERT_EQ(json, "{}");
    Metadata out = json_util::parse(json);
    ASSERT_EQ(out.size(), 0u);
}

// ---------------------------------------------------------------------------
// Tests — resource lifecycle (RAII)
// ---------------------------------------------------------------------------
static void test_engine_raii() {
    // Create and destroy multiple engines without leaking
    for (int i = 0; i < 10; ++i) {
        MemoryEngine eng;
        eng.initialize();
        eng.store("hello", MemoryType::LONG_TERM);
        // destructor calls shutdown()
    }
    ASSERT_TRUE(true);
}

// ---------------------------------------------------------------------------
// Tests — EmbeddingProvider stub
// ---------------------------------------------------------------------------
static void test_embedding_provider_stub() {
    MAKE_ENGINE(eng);
    auto provider = std::make_shared<EmbeddingProvider>();
    // Just verify we can set it without errors
    eng.set_embedding_provider(provider);
    // Default embed() returns empty vector
    auto v = provider->embed("test");
    ASSERT_EQ(v.size(), 0u);
}

// ---------------------------------------------------------------------------
// Tests — SQLite (only if compiled in)
// ---------------------------------------------------------------------------
#ifdef PRIME_HAVE_SQLITE
static void test_sqlite_basic() {
    const char* db_path = "/tmp/prime_test_memory.db";
    std::remove(db_path);

    {
        MemoryConfig cfg;
        cfg.db_path = db_path;
        cfg.recency_half_life_seconds = 0;
        MemoryEngine eng(cfg);
        ASSERT_TRUE(eng.initialize().ok());
        uint64_t id = eng.store("persistent fact", MemoryType::LONG_TERM, 0.9);
        (void)id;
        // engine destructor saves and closes
    }

    // Reopen and verify persistence
    {
        MemoryConfig cfg;
        cfg.db_path = db_path;
        cfg.recency_half_life_seconds = 0;
        MemoryEngine eng(cfg);
        ASSERT_TRUE(eng.initialize().ok());
        ASSERT_EQ(eng.count(), 1u);
        auto results = eng.search("persistent");
        ASSERT_GE(results.size(), 1u);
        ASSERT_EQ(results[0].content, "persistent fact");
    }
    std::remove(db_path);
}

static void test_sqlite_search() {
    const char* db_path = "/tmp/prime_test_search.db";
    std::remove(db_path);
    MemoryConfig cfg;
    cfg.db_path = db_path;
    cfg.recency_half_life_seconds = 0;
    MemoryEngine eng(cfg);
    eng.initialize();
    eng.store("Machine learning basics",     MemoryType::LONG_TERM, 0.8);
    eng.store("Deep learning neural nets",   MemoryType::LONG_TERM, 0.7);
    eng.store("The capital of France",       MemoryType::LONG_TERM, 0.5);
    auto r = eng.search("learning", 5);
    ASSERT_GE(r.size(), 2u);
    std::remove(db_path);
}
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("\n=== PRIME Phase 3 — Memory Engine C++ Tests ===\n\n");

    // MemoryTypes
    run("memory_type_names",             test_memory_type_names);
    run("memory_type_from_string",       test_memory_type_from_string);
    run("memory_type_invalid",           test_memory_type_invalid);
    run("eviction_policy_from_string",   test_eviction_policy_from_string);

    // MemoryItem
    run("memory_item_expiration",        test_memory_item_expiration);
    run("memory_item_recency",           test_memory_item_recency);
    run("memory_item_score",             test_memory_item_score);

    // text_util
    run("text_util_jaccard",             test_text_util_jaccard);
    run("text_util_contains",            test_text_util_contains);
    run("text_util_similarity",          test_text_util_similarity);

    // InMemoryStore
    run("store_put_get",                 test_store_put_get);
    run("store_duplicate_id",            test_store_duplicate_id);
    run("store_get_missing",             test_store_get_missing);
    run("store_update",                  test_store_update);
    run("store_remove",                  test_store_remove);
    run("store_clear_all",               test_store_clear_all);
    run("store_clear_typed",             test_store_clear_typed);
    run("store_count",                   test_store_count);
    run("store_search_text",             test_store_search_text);
    run("store_search_type_filter",      test_store_search_type_filter);
    run("store_search_metadata_filter",  test_store_search_metadata_filter);
    run("store_search_expired_excluded", test_store_search_expired_excluded);
    run("store_ids_by_insertion",        test_store_ids_by_insertion);

    // MemoryEngine
    run("engine_initialize",             test_engine_initialize);
    run("engine_double_initialize",      test_engine_double_initialize);
    run("engine_store_get",              test_engine_store_get);
    run("engine_store_empty_content",    test_engine_store_empty_content);
    run("engine_store_invalid_importance",test_engine_store_invalid_importance);
    run("engine_get_missing",            test_engine_get_missing);
    run("engine_update",                 test_engine_update);
    run("engine_update_missing",         test_engine_update_missing);
    run("engine_delete",                 test_engine_delete);
    run("engine_delete_missing",         test_engine_delete_missing);
    run("engine_count",                  test_engine_count);
    run("engine_exists",                 test_engine_exists);
    run("engine_clear_all",              test_engine_clear_all);
    run("engine_clear_typed",            test_engine_clear_typed);
    run("engine_search",                 test_engine_search);
    run("engine_search_empty_query",     test_engine_search_empty_query);
    run("engine_retrieve_context",       test_engine_retrieve_context);
    run("engine_working_memory",         test_engine_working_memory);
    run("engine_working_capacity_fifo",  test_engine_working_capacity_fifo);
    run("engine_short_term_lru",         test_engine_short_term_lru);
    run("engine_importance_eviction",    test_engine_importance_eviction);
    run("engine_expiration",             test_engine_expiration);
    run("engine_metadata",               test_engine_metadata);
    run("engine_metadata_search",        test_engine_metadata_search);
    run("engine_consolidation",          test_engine_consolidation);
    run("engine_access_tracking",        test_engine_access_tracking);
    run("engine_update_type",            test_engine_update_type);
    run("engine_shutdown",               test_engine_shutdown);

    // Thread safety
    run("engine_concurrent_store",       test_engine_concurrent_store);

    // JSON
    run("json_roundtrip",                test_json_roundtrip);
    run("json_empty",                    test_json_empty);

    // RAII
    run("engine_raii",                   test_engine_raii);

    // EmbeddingProvider stub
    run("embedding_provider_stub",       test_embedding_provider_stub);

#ifdef PRIME_HAVE_SQLITE
    printf("\n--- SQLite persistence tests ---\n");
    run("sqlite_basic",                  test_sqlite_basic);
    run("sqlite_search",                 test_sqlite_search);
#endif

    printf("\nResults: %d/%d passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}
