#pragma once

#include <functional>
#include <memory>
#include <string>

#include "orchestrator/export.h"
#include "orchestrator/tensor.h"

namespace orch {

// A unit of work in the pipeline graph. Implementations must be safe to call
// from multiple threads, because independent branches run concurrently.
class Node {
public:
    virtual ~Node() = default;
    virtual Tensor run(const Tensor& input) = 0;
    virtual std::string kind() const = 0;
};

// Wraps a plain function; handy for tests and lightweight pre/post-processing.
class FunctionNode : public Node {
public:
    explicit FunctionNode(std::function<Tensor(const Tensor&)> fn) : fn_(std::move(fn)) {}
    Tensor run(const Tensor& input) override { return fn_(input); }
    std::string kind() const override { return "function"; }

private:
    std::function<Tensor(const Tensor&)> fn_;
};

}  // namespace orch
