#pragma once

// Symbol visibility for the orchestrator shared library (.so on Linux, .dll on Windows).
#if defined(_WIN32)
#  if defined(ORCH_BUILDING_LIBRARY)
#    define ORCH_API __declspec(dllexport)
#  else
#    define ORCH_API __declspec(dllimport)
#  endif
#else
#  define ORCH_API __attribute__((visibility("default")))
#endif

// Symbol visibility for operator plugins, which are loaded at runtime.
#if defined(_WIN32)
#  define ORCH_PLUGIN_API __declspec(dllexport)
#else
#  define ORCH_PLUGIN_API __attribute__((visibility("default")))
#endif
