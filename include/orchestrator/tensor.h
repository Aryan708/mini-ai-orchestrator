#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "orchestrator/export.h"

namespace orch {

// A dense float32 tensor in row-major order.
struct ORCH_API Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    size_t numel() const;
    std::string shape_str() const;
};

// Number of elements implied by a shape.
ORCH_API size_t shape_numel(const std::vector<int64_t>& shape);

// Concatenates tensors along their last dimension. All leading dimensions must match.
ORCH_API Tensor concat_last_dim(const std::vector<const Tensor*>& parts);

// Index of the largest value in each row of a 2-D tensor.
ORCH_API std::vector<int64_t> argmax_rows(const Tensor& t);

}  // namespace orch
