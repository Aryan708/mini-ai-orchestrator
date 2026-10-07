#include "orchestrator/pipeline.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace orch {
namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream is(s);
    while (std::getline(is, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

}  // namespace

std::vector<NodeSpec> parse_pipeline(const std::string& text, const std::string& base_dir) {
    std::vector<NodeSpec> nodes;
    std::istringstream in(text);
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
        std::istringstream ls(line);
        std::string keyword;
        if (!(ls >> keyword)) continue;  // blank line
        if (keyword != "node")
            throw std::runtime_error("line " + std::to_string(line_no) + ": expected 'node', got '" + keyword + "'");

        NodeSpec spec;
        if (!(ls >> spec.name))
            throw std::runtime_error("line " + std::to_string(line_no) + ": missing node name");

        std::string kv;
        while (ls >> kv) {
            const auto eq = kv.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error("line " + std::to_string(line_no) + ": bad attribute '" + kv + "'");
            const std::string key = kv.substr(0, eq);
            const std::string value = kv.substr(eq + 1);
            if (key == "type") spec.type = value;
            else if (key == "path") spec.path = value;
            else if (key == "deps") spec.deps = split(value, ',');
            else throw std::runtime_error("line " + std::to_string(line_no) + ": unknown attribute '" + key + "'");
        }
        if (spec.type != "onnx" && spec.type != "plugin")
            throw std::runtime_error("node '" + spec.name + "': type must be onnx or plugin");
        if (spec.path.empty())
            throw std::runtime_error("node '" + spec.name + "': missing path");
        if (spec.type == "onnx" && !base_dir.empty() && fs::path(spec.path).is_relative())
            spec.path = (fs::path(base_dir) / spec.path).lexically_normal().string();
        nodes.push_back(std::move(spec));
    }
    topological_order(nodes);  // validate early
    return nodes;
}

std::vector<NodeSpec> load_pipeline_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open pipeline file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_pipeline(ss.str(), fs::path(path).parent_path().string());
}

std::vector<std::string> topological_order(const std::vector<NodeSpec>& nodes) {
    std::map<std::string, int> indegree;
    std::map<std::string, std::vector<std::string>> children;
    for (const auto& n : nodes) {
        if (!indegree.emplace(n.name, 0).second)
            throw std::runtime_error("duplicate node name: " + n.name);
    }
    for (const auto& n : nodes) {
        std::set<std::string> seen;
        for (const auto& d : n.deps) {
            if (!indegree.count(d))
                throw std::runtime_error("node '" + n.name + "' depends on unknown node '" + d + "'");
            if (!seen.insert(d).second)
                throw std::runtime_error("node '" + n.name + "' lists dependency '" + d + "' twice");
            children[d].push_back(n.name);
            ++indegree[n.name];
        }
    }

    // Preserve declaration order among ready nodes for deterministic output.
    std::queue<std::string> ready;
    for (const auto& n : nodes)
        if (indegree[n.name] == 0) ready.push(n.name);

    std::vector<std::string> order;
    while (!ready.empty()) {
        std::string cur = ready.front();
        ready.pop();
        order.push_back(cur);
        for (const auto& c : children[cur])
            if (--indegree[c] == 0) ready.push(c);
    }
    if (order.size() != nodes.size()) throw std::runtime_error("pipeline contains a cycle");
    return order;
}

std::vector<std::string> sink_nodes(const std::vector<NodeSpec>& nodes) {
    std::set<std::string> has_child;
    for (const auto& n : nodes)
        for (const auto& d : n.deps) has_child.insert(d);
    std::vector<std::string> sinks;
    for (const auto& n : nodes)
        if (!has_child.count(n.name)) sinks.push_back(n.name);
    return sinks;
}

}  // namespace orch
