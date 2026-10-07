#pragma once

#include <string>
#include <vector>

#include "orchestrator/export.h"

namespace orch {

// One step in a pipeline, as declared in a .pipeline file.
struct ORCH_API NodeSpec {
    std::string name;
    std::string type;               // "onnx" or "plugin"
    std::string path;               // model file (onnx) or plugin name (plugin)
    std::vector<std::string> deps;  // upstream nodes; empty = consumes the pipeline input
};

// Parses the line-based pipeline format:
//   # comment
//   node <name> type=<onnx|plugin> path=<file-or-plugin> [deps=a,b]
// Relative onnx paths are resolved against `base_dir`.
ORCH_API std::vector<NodeSpec> parse_pipeline(const std::string& text, const std::string& base_dir = "");
ORCH_API std::vector<NodeSpec> load_pipeline_file(const std::string& path);

// Kahn's algorithm. Throws std::runtime_error on duplicate names, unknown deps or cycles.
ORCH_API std::vector<std::string> topological_order(const std::vector<NodeSpec>& nodes);

// Nodes that no other node depends on (the pipeline's outputs).
ORCH_API std::vector<std::string> sink_nodes(const std::vector<NodeSpec>& nodes);

}  // namespace orch
