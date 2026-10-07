#include "orchestrator/plugin.h"

#include <filesystem>
#include <stdexcept>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include "orchestrator/logger.h"

namespace orch {

DynamicLibrary::DynamicLibrary(const std::string& path) : path_(path) {
#if defined(_WIN32)
    handle_ = reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
    if (!handle_)
        throw std::runtime_error("LoadLibrary failed for " + path + " (error " +
                                 std::to_string(GetLastError()) + ")");
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
#endif
    LOG_DEBUG("loaded shared library " << path);
}

DynamicLibrary::~DynamicLibrary() {
    if (!handle_) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
}

void* DynamicLibrary::symbol(const char* name) const {
#if defined(_WIN32)
    void* sym = reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle_), name));
#else
    void* sym = dlsym(handle_, name);
#endif
    if (!sym) throw std::runtime_error(std::string("symbol '") + name + "' not found in " + path_);
    return sym;
}

std::string plugin_file_path(const std::string& dir, const std::string& name) {
#if defined(_WIN32)
    const std::string file = name + ".dll";
#elif defined(__APPLE__)
    const std::string file = "lib" + name + ".dylib";
#else
    const std::string file = "lib" + name + ".so";
#endif
    return (std::filesystem::path(dir) / file).string();
}

PluginNode::PluginNode(const std::string& library_path) : lib_(new DynamicLibrary(library_path)) {
    try {
        name_fn_ = reinterpret_cast<orch_plugin_name_fn>(lib_->symbol("orch_plugin_name"));
        run_fn_ = reinterpret_cast<orch_plugin_run_fn>(lib_->symbol("orch_plugin_run"));
    } catch (...) {
        delete lib_;
        throw;
    }
}

PluginNode::~PluginNode() { delete lib_; }

std::string PluginNode::plugin_name() const { return name_fn_(); }

Tensor PluginNode::run(const Tensor& input) {
    Tensor out;
    out.shape = input.shape;
    out.data.resize(input.data.size());
    const int rc = run_fn_(input.data.data(), input.shape.data(), input.shape.size(), out.data.data());
    if (rc != 0)
        throw std::runtime_error("plugin '" + plugin_name() + "' failed with code " + std::to_string(rc));
    return out;
}

}  // namespace orch
