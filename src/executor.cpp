#include "orchestrator/executor.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>

#include "orchestrator/logger.h"
#include "orchestrator/onnx_node.h"
#include "orchestrator/plugin.h"

namespace orch {

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

Executor::Executor(std::vector<NodeSpec> specs, std::map<std::string, std::unique_ptr<Node>> nodes,
                   size_t num_threads)
    : specs_(std::move(specs)), nodes_(std::move(nodes)) {
    topological_order(specs_);  // throws on invalid graphs
    for (const auto& s : specs_) {
        if (!nodes_.count(s.name)) throw std::runtime_error("no implementation for node '" + s.name + "'");
        by_name_[s.name] = &s;
        for (const auto& d : s.deps) children_[d].push_back(s.name);
    }
    sinks_ = sink_nodes(specs_);
    pool_ = std::make_unique<ThreadPool>(num_threads);
}

Executor::~Executor() = default;

// Per-run state lives on the heap and is co-owned by every task, so a worker that is
// still unwinding (unlocking the mutex, notifying) never touches freed memory after
// run() has returned.
struct RunState {
    explicit RunState(const Tensor& in) : input(in) {}
    const Tensor& input;
    RunResult result;
    std::map<std::string, int> pending_deps;
    std::mutex mu;
    std::condition_variable done_cv;
    size_t remaining = 0;
    size_t in_flight = 0;
    std::exception_ptr error;
};

void Executor::schedule(const std::shared_ptr<RunState>& st, const std::string& name) {
    // Caller holds st->mu.
    ++st->in_flight;
    pool_->submit([this, st, name] {
        const NodeSpec& spec = *by_name_.at(name);
        try {
            // Gather inputs. std::map nodes never move, so pointers stay valid.
            std::vector<const Tensor*> inputs;
            {
                std::lock_guard<std::mutex> lock(st->mu);
                if (st->error) throw std::runtime_error("aborted");
                for (const auto& d : spec.deps) inputs.push_back(&st->result.outputs.at(d));
            }
            Tensor merged;
            const Tensor* in = &st->input;
            if (inputs.size() == 1) in = inputs[0];
            else if (inputs.size() > 1) { merged = concat_last_dim(inputs); in = &merged; }

            const auto t0 = Clock::now();
            Tensor out = nodes_.at(name)->run(*in);
            const double elapsed = ms_since(t0);
            LOG_DEBUG("node " << name << " -> " << out.shape_str() << " in " << elapsed << " ms");

            std::lock_guard<std::mutex> lock(st->mu);
            st->result.outputs.emplace(name, std::move(out));
            st->result.node_ms[name] = elapsed;
            st->result.completion_order.push_back(name);
            --st->remaining;
            if (!st->error) {
                auto it = children_.find(name);
                if (it != children_.end())
                    for (const auto& child : it->second)
                        if (--st->pending_deps[child] == 0) schedule(st, child);
            }
            --st->in_flight;
            st->done_cv.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> lock(st->mu);
            if (!st->error) st->error = std::current_exception();
            --st->in_flight;
            st->done_cv.notify_all();
        }
    });
}

RunResult Executor::run(const Tensor& input) {
    auto st = std::make_shared<RunState>(input);
    for (const auto& s : specs_) st->pending_deps[s.name] = static_cast<int>(s.deps.size());
    st->remaining = specs_.size();

    const auto run_start = Clock::now();
    std::unique_lock<std::mutex> lock(st->mu);
    for (const auto& s : specs_)
        if (s.deps.empty()) schedule(st, s.name);
    // Wait for completion, or for an error plus all in-flight tasks to drain
    // (they read `input`, which belongs to the caller).
    st->done_cv.wait(lock, [&] { return st->in_flight == 0 && (st->remaining == 0 || st->error); });
    if (st->error) std::rethrow_exception(st->error);

    RunResult result = std::move(st->result);
    result.total_ms = ms_since(run_start);
    return result;
}

std::unique_ptr<Executor> build_executor(const std::vector<NodeSpec>& specs, const std::string& plugin_dir,
                                         size_t num_threads) {
    std::map<std::string, std::unique_ptr<Node>> nodes;
    for (const auto& s : specs) {
        if (s.type == "onnx") {
            nodes[s.name] = std::make_unique<OnnxNode>(s.path);
        } else {
            nodes[s.name] = std::make_unique<PluginNode>(plugin_file_path(plugin_dir, s.path));
        }
        LOG_INFO("node '" << s.name << "' ready (" << s.type << ": " << s.path << ")");
    }
    return std::make_unique<Executor>(specs, std::move(nodes), num_threads);
}

}  // namespace orch
