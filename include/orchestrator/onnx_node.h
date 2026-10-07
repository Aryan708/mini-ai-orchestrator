#pragma once

#include <memory>
#include <string>

#include "orchestrator/node.h"

namespace orch {

// Runs a single-input, single-output float32 ONNX model through ONNX Runtime.
// Ort::Session::Run is thread-safe, so one session serves all concurrent callers.
class ORCH_API OnnxNode : public Node {
public:
    OnnxNode(const std::string& model_path, int intra_op_threads = 1);
    ~OnnxNode() override;

    Tensor run(const Tensor& input) override;
    std::string kind() const override { return "onnx"; }

    const std::string& input_name() const;
    const std::string& output_name() const;

private:
    struct Impl;  // hides ONNX Runtime headers from library users
    std::unique_ptr<Impl> impl_;
};

}  // namespace orch
