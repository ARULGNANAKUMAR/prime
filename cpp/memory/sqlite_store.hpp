#pragma once

/*
 * sqlite_store.hpp — SQLite3-backed persistent MemoryStore for PRIME Phase 3.
 *
 * Uses the system SQLite3 library (libsqlite3-dev on Debian/Ubuntu).
 * Falls back gracefully: if SQLite is unavailable at build time, the class
 * still compiles but open() returns an error status.
 *
 * Schema (v1):
 *   CREATE TABLE memories (
 *       id            INTEGER PRIMARY KEY,
 *       type          INTEGER NOT NULL,
 *       content       TEXT    NOT NULL,
 *       metadata_json TEXT    NOT NULL DEFAULT '{}',
 *       importance    REAL    NOT NULL DEFAULT 0.5,
 *       confidence    REAL    NOT NULL DEFAULT 1.0,
 *       created_at    REAL    NOT NULL,
 *       last_accessed REAL    NOT NULL,
 *       access_count  INTEGER NOT NULL DEFAULT 0,
 *       expires_at    REAL    NOT NULL DEFAULT 0.0
 *   );
 *   CREATE TABLE schema_version (version INTEGER);
 */

#include "memory_store.hpp"

#ifdef PRIME_HAVE_SQLITE
#include <sqlite3.h>
#endif

#include <string>
#include <optional>
#include <stdexcept>
#include <cstring>

namespace prime {
namespace memory {

// ---------------------------------------------------------------------------
// Minimal JSON helpers for metadata serialisation (no external lib)
// ---------------------------------------------------------------------------
namespace json_util {

// Escape a JSON string value (only handles common chars for metadata)
inline std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Serialise Metadata → {"k":"v",...}
inline std::string serialise(const Metadata& m) {
    std::string out = "{";
    bool first = true;
    for (auto& [k, v] : m) {
        if (!first) out += ',';
        first = false;
        out += '"';
        out += escape(k);
        out += "\":\"";
        out += escape(v);
        out += '"';
    }
    out += '}';
    return out;
}

// Unescape a JSON string value read between quotes.
inline std::string unescape(const char* s, std::size_t len) {
    std::string out;
    out.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        if (s[i] == '\\' && i + 1 < len) {
            ++i;
            switch (s[i]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

// Parse {"k":"v",...} into Metadata.  Accepts only string values.
inline Metadata parse(const std::string& json) {
    Metadata out;
    const char* p = json.c_str();
    const char* end = p + json.size();

    auto skip_ws = [&]() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; };

    auto read_string = [&](std::string& result) -> bool {
        if (p >= end || *p != '"') return false;
        ++p;
        const char* start = p;
        while (p < end && *p != '"') {
            if (*p == '\\') ++p;  // skip escaped char
            ++p;
        }
        if (p >= end) return false;
        result = unescape(start, static_cast<std::size_t>(p - start));
        ++p;  // consume closing "
        return true;
    };

    skip_ws();
    if (p >= end || *p != '{') return out;
    ++p;

    while (true) {
        skip_ws();
        if (p >= end || *p == '}') break;
        if (*p == ',') { ++p; continue; }

        std::string key, val;
        if (!read_string(key)) break;
        skip_ws();
        if (p >= end || *p != ':') break;
        ++p;
        skip_ws();
        if (!read_string(val)) break;
        out[key] = val;
    }
    return out;
}

} // namespace json_util

// ---------------------------------------------------------------------------
// SqliteStore
// ---------------------------------------------------------------------------
class SqliteStore final : public MemoryStore {
public:
    SqliteStore() = default;
    ~SqliteStore() override { (void)close(); }

    SqliteStore(const SqliteStore&) = delete;
    SqliteStore& operator=(const SqliteStore&) = delete;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------
    Status open(const std::string& path) override {
#ifndef PRIME_HAVE_SQLITE
        (void)path;
        return Status::error(StatusCode::ERROR_INIT_FAILED,
                             "SQLite support not compiled in (PRIME_HAVE_SQLITE not defined)");
#else
        if (db_) return Status::error(StatusCode::ERROR_ALREADY_INITIALIZED,
                                      "SqliteStore already open");
        int rc = sqlite3_open(path.c_str(), &db_);
        if (rc != SQLITE_OK) {
            std::string msg = "sqlite3_open failed: ";
            if (db_) { msg += sqlite3_errmsg(db_); sqlite3_close(db_); db_ = nullptr; }
            else msg += "unknown error";
            return Status::error(StatusCode::ERROR_INIT_FAILED, msg);
        }
        return apply_schema();
#endif
    }

    Status close() override {
#ifdef PRIME_HAVE_SQLITE
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
#endif
        return Status::ok_status();
    }

    // -----------------------------------------------------------------------
    // CRUD
    // -----------------------------------------------------------------------
    Status put(const MemoryItem& item) override {
#ifndef PRIME_HAVE_SQLITE
        return no_sqlite();
#else
        if (!db_) return no_db();
        const char* sql =
            "INSERT INTO memories "
            "(id,type,content,metadata_json,importance,confidence,"
            " created_at,last_accessed,access_count,expires_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return db_error("prepare INSERT");
        bind_item(stmt, item);
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc == SQLITE_CONSTRAINT)
            return Status::error(StatusCode::ERROR_INVALID_CONFIG,
                                 "Duplicate memory ID: " + std::to_string(item.id));
        if (rc != SQLITE_DONE) return db_error("INSERT step");
        return Status::ok_status();
#endif
    }

    Status get(uint64_t id, MemoryItem& out) const override {
#ifndef PRIME_HAVE_SQLITE
        (void)id; (void)out; return no_sqlite();
#else
        if (!db_) return no_db();
        const char* sql = "SELECT * FROM memories WHERE id=?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return db_error("prepare SELECT");
        sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
        int rc = sqlite3_step(stmt);
        if (rc == SQLITE_ROW) {
            out = row_to_item(stmt);
            sqlite3_finalize(stmt);
            return Status::ok_status();
        }
        sqlite3_finalize(stmt);
        return Status::error(StatusCode::ERROR_MODULE_NOT_FOUND,
                             "Memory ID not found: " + std::to_string(id));
#endif
    }

    Status update(const MemoryItem& item) override {
#ifndef PRIME_HAVE_SQLITE
        return no_sqlite();
#else
        if (!db_) return no_db();
        const char* sql =
            "UPDATE memories SET type=?,content=?,metadata_json=?,importance=?,"
            "confidence=?,created_at=?,last_accessed=?,access_count=?,expires_at=? "
            "WHERE id=?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return db_error("prepare UPDATE");
        // Bind columns 1-9 for the SET fields
        sqlite3_bind_int(stmt, 1, static_cast<int>(item.type));
        sqlite3_bind_text(stmt, 2, item.content.c_str(), -1, SQLITE_TRANSIENT);
        auto meta_json = json_util::serialise(item.metadata);
        sqlite3_bind_text(stmt, 3, meta_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 4, item.importance);
        sqlite3_bind_double(stmt, 5, item.confidence);
        sqlite3_bind_double(stmt, 6, item.created_at);
        sqlite3_bind_double(stmt, 7, item.last_accessed);
        sqlite3_bind_int64(stmt, 8, static_cast<sqlite3_int64>(item.access_count));
        sqlite3_bind_double(stmt, 9, item.expires_at);
        sqlite3_bind_int64(stmt, 10, static_cast<sqlite3_int64>(item.id));
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) return db_error("UPDATE step");
        if (sqlite3_changes(db_) == 0)
            return Status::error(StatusCode::ERROR_MODULE_NOT_FOUND,
                                 "Memory ID not found: " + std::to_string(item.id));
        return Status::ok_status();
#endif
    }

    Status remove(uint64_t id) override {
#ifndef PRIME_HAVE_SQLITE
        return no_sqlite();
#else
        if (!db_) return no_db();
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "DELETE FROM memories WHERE id=?",
                               -1, &stmt, nullptr) != SQLITE_OK)
            return db_error("prepare DELETE");
        sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) return db_error("DELETE step");
        if (sqlite3_changes(db_) == 0)
            return Status::error(StatusCode::ERROR_MODULE_NOT_FOUND,
                                 "Memory ID not found: " + std::to_string(id));
        return Status::ok_status();
#endif
    }

    Status clear(std::optional<MemoryType> type_filter) override {
#ifndef PRIME_HAVE_SQLITE
        return no_sqlite();
#else
        if (!db_) return no_db();
        const char* sql = type_filter
            ? "DELETE FROM memories WHERE type=?"
            : "DELETE FROM memories";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return db_error("prepare CLEAR");
        if (type_filter)
            sqlite3_bind_int(stmt, 1, static_cast<int>(*type_filter));
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_DONE) return db_error("CLEAR step");
        return Status::ok_status();
#endif
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------
    std::vector<MemoryItem> search(const SearchQuery& q, double half_life) const override {
#ifndef PRIME_HAVE_SQLITE
        return {};
#else
        if (!db_) return {};
        // Load all (optionally type-filtered) non-expired items then score in C++
        double now = now_seconds();

        std::string sql = "SELECT * FROM memories WHERE (expires_at <= 0.0 OR expires_at > ?)";
        if (q.filter_type) sql += " AND type=?";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
            return {};
        sqlite3_bind_double(stmt, 1, now);
        if (q.filter_type)
            sqlite3_bind_int(stmt, 2, static_cast<int>(*q.filter_type));

        std::vector<std::pair<double, MemoryItem>> scored;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            MemoryItem item = row_to_item(stmt);
            // metadata filter
            if (!q.filter_metadata.empty()) {
                bool match = true;
                for (auto& [k, v] : q.filter_metadata) {
                    auto mit = item.metadata.find(k);
                    if (mit == item.metadata.end() || mit->second != v) {
                        match = false; break;
                    }
                }
                if (!match) continue;
            }
            double text_sim = q.query_text.empty()
                ? 1.0
                : text_util::similarity(q.query_text, item.content);
            if (text_sim < 0.001 && !q.query_text.empty()) {
                for (auto& [k, v] : item.metadata) {
                    double ms = text_util::similarity(q.query_text, v);
                    if (ms > text_sim) text_sim = ms;
                }
            }
            double s = item.score(text_sim, half_life,
                                  q.w_text, q.w_recency, q.w_importance, q.w_freq);
            scored.emplace_back(s, item);
        }
        sqlite3_finalize(stmt);

        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        std::size_t limit = (q.limit == 0) ? scored.size()
                          : std::min(q.limit, scored.size());
        std::vector<MemoryItem> result;
        result.reserve(limit);
        for (std::size_t i = 0; i < limit; ++i) result.push_back(scored[i].second);
        return result;
#endif
    }

    std::size_t count(std::optional<MemoryType> type_filter) const override {
#ifndef PRIME_HAVE_SQLITE
        return 0;
#else
        if (!db_) return 0;
        const char* sql = type_filter
            ? "SELECT COUNT(*) FROM memories WHERE type=?"
            : "SELECT COUNT(*) FROM memories";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;
        if (type_filter) sqlite3_bind_int(stmt, 1, static_cast<int>(*type_filter));
        std::size_t n = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            n = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
        sqlite3_finalize(stmt);
        return n;
#endif
    }

    bool exists(uint64_t id) const override {
#ifndef PRIME_HAVE_SQLITE
        return false;
#else
        if (!db_) return false;
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_,
                "SELECT 1 FROM memories WHERE id=? LIMIT 1", -1, &stmt, nullptr)
            != SQLITE_OK) return false;
        sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
        bool found = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        return found;
#endif
    }

    std::vector<uint64_t> ids_by_insertion(MemoryType type) const override {
#ifndef PRIME_HAVE_SQLITE
        return {};
#else
        if (!db_) return {};
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_,
                "SELECT id FROM memories WHERE type=? ORDER BY rowid ASC",
                -1, &stmt, nullptr) != SQLITE_OK) return {};
        sqlite3_bind_int(stmt, 1, static_cast<int>(type));
        std::vector<uint64_t> ids;
        while (sqlite3_step(stmt) == SQLITE_ROW)
            ids.push_back(static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)));
        sqlite3_finalize(stmt);
        return ids;
#endif
    }

#ifdef PRIME_HAVE_SQLITE
    // Expose sqlite3_changes for testing
    int last_changes() const { return db_ ? sqlite3_changes(db_) : 0; }
#endif

private:
#ifdef PRIME_HAVE_SQLITE
    sqlite3* db_ = nullptr;

    Status apply_schema() {
        const char* ddl =
            "PRAGMA journal_mode=WAL;"
            "PRAGMA foreign_keys=ON;"
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);"
            "INSERT OR IGNORE INTO schema_version VALUES (1);"
            "CREATE TABLE IF NOT EXISTS memories ("
            "  id            INTEGER PRIMARY KEY,"
            "  type          INTEGER NOT NULL,"
            "  content       TEXT    NOT NULL,"
            "  metadata_json TEXT    NOT NULL DEFAULT '{}',"
            "  importance    REAL    NOT NULL DEFAULT 0.5,"
            "  confidence    REAL    NOT NULL DEFAULT 1.0,"
            "  created_at    REAL    NOT NULL,"
            "  last_accessed REAL    NOT NULL,"
            "  access_count  INTEGER NOT NULL DEFAULT 0,"
            "  expires_at    REAL    NOT NULL DEFAULT 0.0"
            ");";
        char* errmsg = nullptr;
        int rc = sqlite3_exec(db_, ddl, nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            std::string msg = "Schema init failed: ";
            if (errmsg) { msg += errmsg; sqlite3_free(errmsg); }
            return Status::error(StatusCode::ERROR_INIT_FAILED, msg);
        }
        return Status::ok_status();
    }

    // Bind all 10 positional parameters for INSERT
    void bind_item(sqlite3_stmt* stmt, const MemoryItem& item) const {
        sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(item.id));
        sqlite3_bind_int(stmt,   2, static_cast<int>(item.type));
        sqlite3_bind_text(stmt,  3, item.content.c_str(), -1, SQLITE_TRANSIENT);
        auto meta_json = json_util::serialise(item.metadata);
        sqlite3_bind_text(stmt,  4, meta_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt,5, item.importance);
        sqlite3_bind_double(stmt,6, item.confidence);
        sqlite3_bind_double(stmt,7, item.created_at);
        sqlite3_bind_double(stmt,8, item.last_accessed);
        sqlite3_bind_int64(stmt, 9, static_cast<sqlite3_int64>(item.access_count));
        sqlite3_bind_double(stmt,10,item.expires_at);
    }

    MemoryItem row_to_item(sqlite3_stmt* stmt) const {
        MemoryItem item;
        item.id           = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        item.type         = static_cast<MemoryType>(sqlite3_column_int(stmt, 1));
        auto* content     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        item.content      = content ? content : "";
        auto* meta_raw    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        std::string meta_json = meta_raw ? meta_raw : "{}";
        item.metadata     = json_util::parse(meta_json);
        item.importance   = sqlite3_column_double(stmt, 4);
        item.confidence   = sqlite3_column_double(stmt, 5);
        item.created_at   = sqlite3_column_double(stmt, 6);
        item.last_accessed= sqlite3_column_double(stmt, 7);
        item.access_count = static_cast<uint64_t>(sqlite3_column_int64(stmt, 8));
        item.expires_at   = sqlite3_column_double(stmt, 9);
        return item;
    }

    Status db_error(const char* context) const {
        std::string msg = std::string(context) + ": ";
        msg += db_ ? sqlite3_errmsg(db_) : "no database";
        return Status::error(StatusCode::ERROR_INIT_FAILED, msg);
    }
#endif

    static Status no_sqlite() {
        return Status::error(StatusCode::ERROR_INIT_FAILED,
                             "SQLite support not compiled in");
    }
    static Status no_db() {
        return Status::error(StatusCode::ERROR_NOT_INITIALIZED,
                             "SqliteStore not open");
    }
};

} // namespace memory
} // namespace prime
