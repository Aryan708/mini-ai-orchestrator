#pragma once

#include <map>
#include <string>

#include "orchestrator/export.h"

struct sqlite3;

namespace orch {

// Persists per-node latency samples to SQLite so runs can be compared over time.
// When built without SQLite (ORCH_WITH_SQLITE=OFF) every call is a no-op.
class ORCH_API MetricsStore {
public:
    explicit MetricsStore(const std::string& db_path);
    ~MetricsStore();
    MetricsStore(const MetricsStore&) = delete;
    MetricsStore& operator=(const MetricsStore&) = delete;

    bool enabled() const;
    void record(const std::string& run_label, int iteration, int batch,
                const std::map<std::string, double>& node_ms, double total_ms);

    // Average latency per node for one run label (empty when disabled).
    std::map<std::string, double> average_ms(const std::string& run_label);

private:
    sqlite3* db_ = nullptr;
};

}  // namespace orch
