// Dependency-free unit tests. Run with `ctest` or the orchestrator_tests binary directly.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "orchestrator/executor.h"
#include "orchestrator/onnx_node.h"
#include "orchestrator/pipeline.h"
#include "orchestrator/plugin.h"
#include "orchestrator/tensor.h"
#include "orchestrator/thread_pool.h"

namespace fs = std::filesystem;
using namespace orch;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(cond)) {                                                               \
            ++g_failures;                                                            \
            std::cerr << "  FAILED: " #cond " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        }                                                                            \
    } while (0)

#define CHECK_THROWS(expr)                         \
    do {                                           \
        bool threw_ = false;                       \
        try { expr; } catch (...) { threw_ = true; } \
        CHECK(threw_);                             \
    } while (0)

static Tensor make(std::vector<int64_t> shape, std::vector<float> data) { return Tensor{std::move(shape), std::move(data)}; }

static void test_tensor_ops() {
    Tensor a = make({2, 2}, {1, 2, 3, 4});
    Tensor b = make({2, 1}, {9, 8});
    Tensor c = concat_last_dim({&a, &b});
    CHECK((c.shape == std::vector<int64_t>{2, 3}));
    CHECK((c.data == std::vector<float>{1, 2, 9, 3, 4, 8}));

    Tensor bad = make({3, 1}, {0, 0, 0});
    CHECK_THROWS(concat_last_dim({&a, &bad}));

    auto idx = argmax_rows(make({2, 3}, {0.1f, 0.7f, 0.2f, 0.9f, 0.05f, 0.05f}));
    CHECK((idx == std::vector<int64_t>{1, 0}));
}

static void test_pipeline_parsing() {
    const char* text =
        "# comment line\n"
        "node a type=onnx path=m/a.onnx\n"
        "node b type=onnx path=m/b.onnx deps=a\n"
        "node c type=plugin path=softmax_plugin deps=a\n"
        "node d type=onnx path=m/d.onnx deps=b,c   # trailing comment\n";
    auto specs = parse_pipeline(text);
    CHECK(specs.size() == 4);
    CHECK(specs[3].deps.size() == 2);
    CHECK(specs[2].type == "plugin");

    auto order = topological_order(specs);
    auto pos = [&](const std::string& n) { return std::find(order.begin(), order.end(), n) - order.begin(); };
    CHECK(pos("a") < pos("b"));
    CHECK(pos("a") < pos("c"));
    CHECK(pos("b") < pos("d"));
    CHECK(pos("c") < pos("d"));
    CHECK((sink_nodes(specs) == std::vector<std::string>{"d"}));

    CHECK_THROWS(parse_pipeline("node a type=onnx path=x deps=b\nnode b type=onnx path=y deps=a\n"));  // cycle
    CHECK_THROWS(parse_pipeline("node a type=onnx path=x deps=missing\n"));                          // unknown dep
    CHECK_THROWS(parse_pipeline("node a type=onnx path=x\nnode a type=onnx path=y\n"));              // duplicate
    CHECK_THROWS(parse_pipeline("node a type=gpu path=x\n"));                                        // bad type
    CHECK_THROWS(parse_pipeline("edge a b\n"));                                                      // bad keyword
}

static void test_thread_pool() {
    std::atomic<int> sum{0};
    {
        ThreadPool pool(4);
        for (int i = 1; i <= 1000; ++i) pool.submit([&sum, i] { sum += i; });
    }  // destructor drains the queue
    CHECK(sum == 500500);
}

static void test_executor_parallel_branches() {
    // root -> {slow_a, slow_b} -> join. The two branches must overlap in time.
    std::vector<NodeSpec> specs = {
        {"root", "onnx", "-", {}},
        {"slow_a", "onnx", "-", {"root"}},
        {"slow_b", "onnx", "-", {"root"}},
        {"join", "onnx", "-", {"slow_a", "slow_b"}},
    };
    auto add = [](float k) {
        return [k](const Tensor& t) { Tensor o = t; for (auto& x : o.data) x += k; return o; };
    };
    auto slow = [](float k) {
        return [k](const Tensor& t) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            Tensor o = t; for (auto& x : o.data) x *= k; return o;
        };
    };
    std::map<std::string, std::unique_ptr<Node>> nodes;
    nodes["root"] = std::make_unique<FunctionNode>(add(1.0f));
    nodes["slow_a"] = std::make_unique<FunctionNode>(slow(2.0f));
    nodes["slow_b"] = std::make_unique<FunctionNode>(slow(3.0f));
    nodes["join"] = std::make_unique<FunctionNode>([](const Tensor& t) { return t; });

    Executor exec(specs, std::move(nodes), 4);
    RunResult r = exec.run(make({1, 2}, {0, 1}));

    // root: [1,2]; slow_a: [2,4]; slow_b: [3,6]; join concatenates -> [2,4,3,6]
    CHECK((r.outputs.at("join").data == std::vector<float>{2, 4, 3, 6}));
    CHECK((r.outputs.at("join").shape == std::vector<int64_t>{1, 4}));
    CHECK(r.completion_order.front() == "root");
    CHECK(r.completion_order.back() == "join");
    CHECK(r.total_ms < 180.0);  // sequential would take >= 200 ms
    CHECK((exec.sinks() == std::vector<std::string>{"join"}));
}

static void test_executor_propagates_errors() {
    std::vector<NodeSpec> specs = {{"a", "onnx", "-", {}}, {"b", "onnx", "-", {"a"}}};
    std::map<std::string, std::unique_ptr<Node>> nodes;
    nodes["a"] = std::make_unique<FunctionNode>([](const Tensor&) -> Tensor { throw std::runtime_error("boom"); });
    nodes["b"] = std::make_unique<FunctionNode>([](const Tensor& t) { return t; });
    Executor exec(specs, std::move(nodes), 2);
    CHECK_THROWS(exec.run(make({1, 1}, {0})));
    // The executor stays usable after a failed run.
    CHECK_THROWS(exec.run(make({1, 1}, {0})));
}

static void test_softmax_plugin() {
    PluginNode node(plugin_file_path(ORCH_PLUGIN_DIR, "softmax_plugin"));
    CHECK(node.plugin_name() == "softmax");
    Tensor out = node.run(make({2, 3}, {1, 2, 3, 1000, 1000, 1000}));
    for (int r = 0; r < 2; ++r) {
        float s = out.data[r * 3] + out.data[r * 3 + 1] + out.data[r * 3 + 2];
        CHECK(std::fabs(s - 1.0f) < 1e-5f);
    }
    CHECK(out.data[2] > out.data[1] && out.data[1] > out.data[0]);
    CHECK(std::fabs(out.data[3] - 1.0f / 3.0f) < 1e-5f);  // stable for large logits
    CHECK_THROWS(PluginNode(plugin_file_path(ORCH_PLUGIN_DIR, "does_not_exist")));
}

static void test_onnx_end_to_end() {
    const fs::path models = ORCH_MODELS_DIR;
    if (!fs::exists(models / "normalize.onnx") || !fs::exists(models / "classifier_int8.onnx")) {
        std::cout << "  (skipped: run python/make_models.py and python/quantize.py first)\n";
        return;
    }
    OnnxNode norm((models / "normalize.onnx").string());
    Tensor x = make({2, 16}, std::vector<float>(32, 0.5f));
    Tensor y = norm.run(x);
    CHECK((y.shape == std::vector<int64_t>{2, 16}));

    const std::string text =
        "node normalize  type=onnx   path=normalize.onnx\n"
        "node clf_fp32   type=onnx   path=classifier_fp32.onnx deps=normalize\n"
        "node clf_int8   type=onnx   path=classifier_int8.onnx deps=normalize\n"
        "node probs_fp32 type=plugin path=softmax_plugin deps=clf_fp32\n"
        "node probs_int8 type=plugin path=softmax_plugin deps=clf_int8\n";
    auto exec = build_executor(parse_pipeline(text, models.string()), ORCH_PLUGIN_DIR, 4);
    RunResult r = exec->run(x);
    const Tensor& p = r.outputs.at("probs_fp32");
    CHECK(p.shape.size() == 2 && p.shape[0] == 2);
    float s = 0;
    for (int64_t c = 0; c < p.shape[1]; ++c) s += p.data[static_cast<size_t>(c)];
    CHECK(std::fabs(s - 1.0f) < 1e-4f);
    CHECK(r.outputs.at("probs_int8").shape == p.shape);
}

int main() {
    struct { const char* name; void (*fn)(); } tests[] = {
        {"tensor ops", test_tensor_ops},
        {"pipeline parsing & topological sort", test_pipeline_parsing},
        {"thread pool", test_thread_pool},
        {"executor runs branches in parallel", test_executor_parallel_branches},
        {"executor propagates errors", test_executor_propagates_errors},
        {"softmax plugin (.so/.dll loading)", test_softmax_plugin},
        {"ONNX Runtime end to end", test_onnx_end_to_end},
    };
    for (auto& t : tests) {
        std::cout << "[ RUN  ] " << t.name << "\n";
        const int before = g_failures;
        try { t.fn(); } catch (const std::exception& e) {
            ++g_failures;
            std::cerr << "  FAILED with exception: " << e.what() << "\n";
        }
        std::cout << (g_failures == before ? "[  OK  ] " : "[ FAIL ] ") << t.name << "\n";
    }
    std::cout << "\n" << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures == 0 ? 0 : 1;
}
