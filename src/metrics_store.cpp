#include "orchestrator/metrics_store.h"

#include <stdexcept>

#include "orchestrator/logger.h"

#if defined(ORCH_HAS_SQLITE)
#  include <sqlite3.h>
#endif

namespace orch {

#if defined(ORCH_HAS_SQLITE)

namespace {
void exec_or_throw(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown error";
        sqlite3_free(err);
        throw std::runtime_error("sqlite: " + msg);
    }
}
}  // namespace

MetricsStore::MetricsStore(const std::string& db_path) {
    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
        std::string msg = sqlite3_errmsg(db_);
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("cannot open metrics db: " + msg);
    }
    exec_or_throw(db_,
                  "CREATE TABLE IF NOT EXISTS node_latency ("
                  "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                  "  run_label TEXT NOT NULL,"
                  "  iteration INTEGER NOT NULL,"
                  "  batch INTEGER NOT NULL,"
                  "  node TEXT NOT NULL,"
                  "  latency_ms REAL NOT NULL,"
                  "  created_at TEXT DEFAULT CURRENT_TIMESTAMP);");
}

MetricsStore::~MetricsStore() {
    if (db_) sqlite3_close(db_);
}

bool MetricsStore::enabled() const { return db_ != nullptr; }

void MetricsStore::record(const std::string& run_label, int iteration, int batch,
                          const std::map<std::string, double>& node_ms, double total_ms) {
    exec_or_throw(db_, "BEGIN;");
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
                       "INSERT INTO node_latency(run_label, iteration, batch, node, latency_ms) "
                       "VALUES(?,?,?,?,?);",
                       -1, &stmt, nullptr);
    auto insert = [&](const std::string& node, double ms) {
        sqlite3_bind_text(stmt, 1, run_label.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, iteration);
        sqlite3_bind_int(stmt, 3, batch);
        sqlite3_bind_text(stmt, 4, node.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 5, ms);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    };
    for (const auto& [node, ms] : node_ms) insert(node, ms);
    insert("__total__", total_ms);
    sqlite3_finalize(stmt);
    exec_or_throw(db_, "COMMIT;");
}

std::map<std::string, double> MetricsStore::average_ms(const std::string& run_label) {
    std::map<std::string, double> out;
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_,
                       "SELECT node, AVG(latency_ms) FROM node_latency WHERE run_label = ? GROUP BY node;",
                       -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, run_label.c_str(), -1, SQLITE_TRANSIENT);
    while (sqlite3_step(stmt) == SQLITE_ROW)
        out[reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))] = sqlite3_column_double(stmt, 1);
    sqlite3_finalize(stmt);
    return out;
}

#else  // built without SQLite

MetricsStore::MetricsStore(const std::string&) {
    LOG_WARN("built without SQLite; metrics will not be persisted");
}
MetricsStore::~MetricsStore() = default;
bool MetricsStore::enabled() const { return false; }
void MetricsStore::record(const std::string&, int, int, const std::map<std::string, double>&, double) {}
std::map<std::string, double> MetricsStore::average_ms(const std::string&) { return {}; }

#endif

}  // namespace orch
