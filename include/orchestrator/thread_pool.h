#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "orchestrator/export.h"

namespace orch {

// Fixed-size worker pool. Tasks run in FIFO order; the destructor drains the queue and joins.
class ORCH_API ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task);
    size_t size() const { return workers_.size(); }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

}  // namespace orch
