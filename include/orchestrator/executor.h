#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "orchestrator/node.h"
#include "orchestrator/pipeline.h"
#include "orchestrator/thread_pool.h"

namespace orch {

struct RunState;

struct ORCH_API RunResult {
    std::map<std::string, Tensor> outputs;     // every node's output, keyed by node name
    std::map<std::string, double> node_ms;     // wall-clock latency per node
    std::vector<std::string> completion_order; // order nodes finished in
    double total_ms = 0.0;
};

// Executes a pipeline DAG. A node is scheduled on the thread pool as soon as all
// of its dependencies have finished, so independent branches run in parallel.
// Nodes with several dependencies receive their inputs concatenated on the last axis.
class ORCH_API Executor {
public:
    Executor(std::vector<NodeSpec> specs, std::map<std::string, std::unique_ptr<Node>> nodes,
             size_t num_threads);
    ~Executor();

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    RunResult run(const Tensor& input);

    const std::vector<NodeSpec>& specs() const { return specs_; }
    const std::vector<std::string>& sinks() const { return sinks_; }

private:
    void schedule(const std::shared_ptr<RunState>& st, const std::string& name);

    std::vector<NodeSpec> specs_;
    std::map<std::string, std::unique_ptr<Node>> nodes_;
    std::map<std::string, std::vector<std::string>> children_;
    std::map<std::string, const NodeSpec*> by_name_;
    std::vector<std::string> sinks_;
    std::unique_ptr<ThreadPool> pool_;
};

// Builds an Executor from specs, creating OnnxNode / PluginNode instances.
ORCH_API std::unique_ptr<Executor> build_executor(const std::vector<NodeSpec>& specs,
                                                  const std::string& plugin_dir, size_t num_threads);

}  // namespace orch
