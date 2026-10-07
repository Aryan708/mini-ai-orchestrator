// Command-line runner: loads a pipeline, feeds it random input, and reports
// per-node latency statistics (optionally persisted to SQLite).

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "orchestrator/executor.h"
#include "orchestrator/logger.h"
#include "orchestrator/metrics_store.h"
#include "orchestrator/pipeline.h"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string pipeline = "pipelines/demo.pipeline";
    std::string plugin_dir;
    std::string metrics_db;
    std::string label = "default";
    int batch = 8;
    int features = 16;
    int iterations = 50;
    int warmup = 5;
    size_t threads = std::max(2u, std::thread::hardware_concurrency());
    bool verbose = false;
};

void usage() {
    std::cout << "Usage: orchestrator_cli [options]\n"
                 "  --pipeline <file>     pipeline definition (default pipelines/demo.pipeline)\n"
                 "  --plugin-dir <dir>    directory with plugin libraries (default: next to executable)\n"
                 "  --batch <n>           batch size (default 8)\n"
                 "  --features <n>        input feature width (default 16)\n"
                 "  --iterations <n>      timed iterations (default 50)\n"
                 "  --warmup <n>          untimed warm-up iterations (default 5)\n"
                 "  --threads <n>         scheduler worker threads\n"
                 "  --metrics-db <file>   persist latencies to SQLite\n"
                 "  --label <name>        run label stored with metrics\n"
                 "  --verbose             debug logging\n";
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--pipeline") o.pipeline = next();
        else if (a == "--plugin-dir") o.plugin_dir = next();
        else if (a == "--batch") o.batch = std::stoi(next());
        else if (a == "--features") o.features = std::stoi(next());
        else if (a == "--iterations") o.iterations = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--threads") o.threads = static_cast<size_t>(std::stoul(next()));
        else if (a == "--metrics-db") o.metrics_db = next();
        else if (a == "--label") o.label = next();
        else if (a == "--verbose") o.verbose = true;
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown option " << a << "\n"; usage(); std::exit(2); }
    }
    return o;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    if (opt.verbose) orch::Logger::instance().set_level(orch::LogLevel::Debug);
    if (opt.plugin_dir.empty()) opt.plugin_dir = fs::absolute(fs::path(argv[0])).parent_path().string();

    try {
        auto specs = orch::load_pipeline_file(opt.pipeline);
        LOG_INFO("pipeline '" << opt.pipeline << "': " << specs.size() << " nodes, " << opt.threads
                              << " worker threads");
        auto executor = orch::build_executor(specs, opt.plugin_dir, opt.threads);

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        orch::Tensor input;
        input.shape = {opt.batch, opt.features};
        input.data.resize(static_cast<size_t>(opt.batch) * static_cast<size_t>(opt.features));
        for (auto& x : input.data) x = dist(rng);

        for (int i = 0; i < opt.warmup; ++i) executor->run(input);

        std::unique_ptr<orch::MetricsStore> store;
        if (!opt.metrics_db.empty()) store = std::make_unique<orch::MetricsStore>(opt.metrics_db);

        std::map<std::string, std::vector<double>> samples;
        orch::RunResult last;
        for (int i = 0; i < opt.iterations; ++i) {
            last = executor->run(input);
            for (const auto& [node, ms] : last.node_ms) samples[node].push_back(ms);
            samples["(end-to-end)"].push_back(last.total_ms);
            if (store && store->enabled()) store->record(opt.label, i, opt.batch, last.node_ms, last.total_ms);
        }

        std::cout << "\nLatency over " << opt.iterations << " iterations (batch " << opt.batch << ")\n";
        std::cout << std::left << std::setw(16) << "node" << std::right << std::setw(10) << "p50 ms"
                  << std::setw(10) << "p95 ms" << std::setw(10) << "max ms" << "\n";
        std::cout << std::string(46, '-') << "\n" << std::fixed << std::setprecision(3);
        for (const auto& [node, v] : samples)
            std::cout << std::left << std::setw(16) << node << std::right << std::setw(10) << percentile(v, 0.5)
                      << std::setw(10) << percentile(v, 0.95) << std::setw(10)
                      << *std::max_element(v.begin(), v.end()) << "\n";

        std::cout << "\nPipeline outputs (first 8 rows, argmax class)\n";
        for (const auto& sink : executor->sinks()) {
            const auto& t = last.outputs.at(sink);
            std::cout << std::left << std::setw(16) << sink << t.shape_str() << "  ";
            if (t.shape.size() == 2) {
                auto cls = orch::argmax_rows(t);
                for (size_t r = 0; r < std::min<size_t>(8, cls.size()); ++r) std::cout << cls[r] << ' ';
            }
            std::cout << "\n";
        }

        // When the demo has fp32 and int8 heads, report how often they agree.
        if (last.outputs.count("probs_fp32") && last.outputs.count("probs_int8")) {
            auto a = orch::argmax_rows(last.outputs.at("probs_fp32"));
            auto b = orch::argmax_rows(last.outputs.at("probs_int8"));
            size_t agree = 0;
            for (size_t i = 0; i < a.size(); ++i) agree += (a[i] == b[i]);
            std::cout << "\nFP32 vs INT8 top-1 agreement: " << agree << "/" << a.size() << "\n";
        }
        if (store && store->enabled()) std::cout << "\nMetrics written to " << opt.metrics_db << "\n";
    } catch (const std::exception& e) {
        LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
