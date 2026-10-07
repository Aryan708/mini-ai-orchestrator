#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "orchestrator/node.h"

// ---- Plugin C ABI ---------------------------------------------------------
// A plugin is a shared library (.so / .dll) exporting these C symbols.
// Output has the same shape as the input. Return 0 on success.
extern "C" {
typedef const char* (*orch_plugin_name_fn)();
typedef int (*orch_plugin_run_fn)(const float* input, const int64_t* shape, size_t rank, float* output);
}

namespace orch {

// RAII wrapper around dlopen/dlsym (Linux) and LoadLibrary/GetProcAddress (Windows).
class ORCH_API DynamicLibrary {
public:
    explicit DynamicLibrary(const std::string& path);
    ~DynamicLibrary();
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    void* symbol(const char* name) const;  // throws if missing
    const std::string& path() const { return path_; }

private:
    void* handle_ = nullptr;
    std::string path_;
};

// Maps a plugin name to its platform file name in `dir`: libfoo.so or foo.dll.
ORCH_API std::string plugin_file_path(const std::string& dir, const std::string& name);

// Pipeline node backed by a runtime-loaded plugin.
class ORCH_API PluginNode : public Node {
public:
    explicit PluginNode(const std::string& library_path);
    ~PluginNode() override;

    Tensor run(const Tensor& input) override;
    std::string kind() const override { return "plugin"; }
    std::string plugin_name() const;

private:
    DynamicLibrary* lib_;  // raw pointer keeps the class layout simple across the DLL boundary
    orch_plugin_name_fn name_fn_;
    orch_plugin_run_fn run_fn_;
};

}  // namespace orch
