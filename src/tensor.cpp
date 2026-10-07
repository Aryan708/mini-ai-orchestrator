#include "orchestrator/tensor.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace orch {

size_t shape_numel(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dimension in shape");
        n *= static_cast<size_t>(d);
    }
    return n;
}

size_t Tensor::numel() const { return shape_numel(shape); }

std::string Tensor::shape_str() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
    os << "]";
    return os.str();
}

Tensor concat_last_dim(const std::vector<const Tensor*>& parts) {
    if (parts.empty()) throw std::invalid_argument("concat_last_dim: no inputs");
    if (parts.size() == 1) return *parts[0];

    const auto& ref = parts[0]->shape;
    if (ref.empty()) throw std::invalid_argument("concat_last_dim: scalar input");
    std::vector<int64_t> lead(ref.begin(), ref.end() - 1);
    const size_t rows = shape_numel(lead);

    int64_t total_last = 0;
    for (const Tensor* t : parts) {
        if (t->shape.size() != ref.size() ||
            !std::equal(lead.begin(), lead.end(), t->shape.begin()))
            throw std::invalid_argument("concat_last_dim: leading dims differ: " +
                                        parts[0]->shape_str() + " vs " + t->shape_str());
        total_last += t->shape.back();
    }

    Tensor out;
    out.shape = lead;
    out.shape.push_back(total_last);
    out.data.reserve(rows * static_cast<size_t>(total_last));
    for (size_t r = 0; r < rows; ++r) {
        for (const Tensor* t : parts) {
            const size_t w = static_cast<size_t>(t->shape.back());
            auto begin = t->data.begin() + static_cast<std::ptrdiff_t>(r * w);
            out.data.insert(out.data.end(), begin, begin + static_cast<std::ptrdiff_t>(w));
        }
    }
    return out;
}

std::vector<int64_t> argmax_rows(const Tensor& t) {
    if (t.shape.size() != 2) throw std::invalid_argument("argmax_rows expects a 2-D tensor");
    const size_t rows = static_cast<size_t>(t.shape[0]);
    const size_t cols = static_cast<size_t>(t.shape[1]);
    std::vector<int64_t> out(rows, 0);
    for (size_t r = 0; r < rows; ++r) {
        size_t best = 0;
        for (size_t c = 1; c < cols; ++c)
            if (t.data[r * cols + c] > t.data[r * cols + best]) best = c;
        out[r] = static_cast<int64_t>(best);
    }
    return out;
}

}  // namespace orch
