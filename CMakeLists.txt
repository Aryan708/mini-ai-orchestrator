cmake_minimum_required(VERSION 3.16)
project(mini_ai_orchestrator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Put every binary (exe, .so/.dll, plugins) in one folder so runtime loading just works.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

option(ORCH_WITH_SQLITE "Persist metrics to SQLite" ON)
option(ORCH_BUILD_TESTS "Build unit tests" ON)

# ---- ONNX Runtime (prebuilt release package) -------------------------------
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime release (contains include/ and lib/)")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=<path to onnxruntime-<os>-x64-<version>>")
endif()
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session NO_DEFAULT_PATH REQUIRED)
find_library(ORT_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH REQUIRED)

# ---- Core shared library ----------------------------------------------------
add_library(orchestrator SHARED
  src/tensor.cpp
  src/logger.cpp
  src/thread_pool.cpp
  src/pipeline.cpp
  src/onnx_node.cpp
  src/plugin.cpp
  src/executor.cpp
  src/metrics_store.cpp)
target_include_directories(orchestrator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(orchestrator PRIVATE ${ORT_INCLUDE_DIR})
target_compile_definitions(orchestrator PRIVATE ORCH_BUILDING_LIBRARY)
target_link_libraries(orchestrator PRIVATE ${ORT_LIBRARY})
find_package(Threads REQUIRED)
target_link_libraries(orchestrator PUBLIC Threads::Threads)
if(UNIX)
  target_link_libraries(orchestrator PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(orchestrator PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;\$ORIGIN")
endif()

if(ORCH_WITH_SQLITE)
  find_package(SQLite3)
  if(SQLite3_FOUND)
    target_link_libraries(orchestrator PRIVATE SQLite::SQLite3)
    target_compile_definitions(orchestrator PRIVATE ORCH_HAS_SQLITE)
  else()
    message(WARNING "SQLite3 not found; metrics persistence disabled")
  endif()
endif()

if(MSVC)
  target_compile_options(orchestrator PRIVATE /W4 /wd4251 /wd4275)
else()
  target_compile_options(orchestrator PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Example plugin (loaded at runtime, not linked) ---------------------------
add_library(softmax_plugin MODULE plugins/softmax_plugin.cpp)
target_include_directories(softmax_plugin PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ---- CLI ---------------------------------------------------------------------
add_executable(orchestrator_cli apps/orchestrator_cli.cpp)
target_link_libraries(orchestrator_cli PRIVATE orchestrator)
add_dependencies(orchestrator_cli softmax_plugin)
if(UNIX)
  set_target_properties(orchestrator_cli PROPERTIES BUILD_RPATH "\$ORIGIN;${ONNXRUNTIME_ROOT}/lib")
endif()

# On Windows, copy onnxruntime.dll next to the executables.
if(WIN32)
  file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
  add_custom_command(TARGET orchestrator_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLLS} $<TARGET_FILE_DIR:orchestrator_cli>)
endif()

# ---- Tests -------------------------------------------------------------------
if(ORCH_BUILD_TESTS)
  enable_testing()
  add_executable(orchestrator_tests tests/test_main.cpp)
  target_link_libraries(orchestrator_tests PRIVATE orchestrator)
  add_dependencies(orchestrator_tests softmax_plugin)
  target_compile_definitions(orchestrator_tests PRIVATE
    ORCH_MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
    ORCH_PLUGIN_DIR="${CMAKE_BINARY_DIR}/bin")
  if(UNIX)
    set_target_properties(orchestrator_tests PROPERTIES BUILD_RPATH "\$ORIGIN;${ONNXRUNTIME_ROOT}/lib")
  endif()
  add_test(NAME orchestrator_tests COMMAND orchestrator_tests)
endif()
