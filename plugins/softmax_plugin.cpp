// Example operator plugin: numerically stable softmax over the last dimension.
// Built as a standalone shared library and loaded by the orchestrator at runtime.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "orchestrator/export.h"

extern "C" {

ORCH_PLUGIN_API const char* orch_plugin_name() { return "softmax"; }

ORCH_PLUGIN_API int orch_plugin_run(const float* input, const int64_t* shape, size_t rank, float* output) {
    if (!input || !shape || !output || rank == 0) return 1;
    const int64_t cols = shape[rank - 1];
    if (cols <= 0) return 2;
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) rows *= shape[i];

    for (int64_t r = 0; r < rows; ++r) {
        const float* in = input + r * cols;
        float* out = output + r * cols;
        const float max_v = *std::max_element(in, in + cols);
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            out[c] = std::exp(in[c] - max_v);
            sum += out[c];
        }
        for (int64_t c = 0; c < cols; ++c) out[c] = static_cast<float>(out[c] / sum);
    }
    return 0;
}

}  // extern "C"
