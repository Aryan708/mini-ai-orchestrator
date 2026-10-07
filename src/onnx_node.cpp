#include "orchestrator/onnx_node.h"

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <stdexcept>

#include "orchestrator/logger.h"

namespace orch {
namespace {

// One ONNX Runtime environment per process.
Ort::Env& shared_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mini-ai-orchestrator");
    return env;
}

}  // namespace

struct OnnxNode::Impl {
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

OnnxNode::OnnxNode(const std::string& model_path, int intra_op_threads) : impl_(std::make_unique<Impl>()) {
    if (!std::filesystem::exists(model_path))
        throw std::runtime_error("model not found: " + model_path);

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(intra_op_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(_WIN32)
    const std::wstring wpath = std::filesystem::path(model_path).wstring();
    impl_->session = std::make_unique<Ort::Session>(shared_env(), wpath.c_str(), opts);
#else
    impl_->session = std::make_unique<Ort::Session>(shared_env(), model_path.c_str(), opts);
#endif

    if (impl_->session->GetInputCount() != 1 || impl_->session->GetOutputCount() != 1)
        throw std::runtime_error("only single-input/single-output models are supported: " + model_path);

    Ort::AllocatorWithDefaultOptions alloc;
    impl_->input_name = impl_->session->GetInputNameAllocated(0, alloc).get();
    impl_->output_name = impl_->session->GetOutputNameAllocated(0, alloc).get();
    LOG_DEBUG("loaded " << model_path << " (in=" << impl_->input_name << ", out=" << impl_->output_name << ")");
}

OnnxNode::~OnnxNode() = default;

const std::string& OnnxNode::input_name() const { return impl_->input_name; }
const std::string& OnnxNode::output_name() const { return impl_->output_name; }

Tensor OnnxNode::run(const Tensor& input) {
    if (input.data.size() != input.numel())
        throw std::invalid_argument("tensor data does not match shape " + input.shape_str());

    // ONNX Runtime does not write to input buffers, so the const_cast is safe.
    Ort::Value in = Ort::Value::CreateTensor<float>(
        impl_->mem_info, const_cast<float*>(input.data.data()), input.data.size(),
        input.shape.data(), input.shape.size());

    const char* in_names[] = {impl_->input_name.c_str()};
    const char* out_names[] = {impl_->output_name.c_str()};
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("model output is not float32");

    Tensor out;
    out.shape = info.GetShape();
    const float* p = outputs[0].GetTensorData<float>();
    out.data.assign(p, p + info.GetElementCount());
    return out;
}

}  // namespace orch
