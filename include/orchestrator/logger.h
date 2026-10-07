#pragma once

#include <mutex>
#include <sstream>
#include <string>

#include "orchestrator/export.h"

namespace orch {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

// Minimal thread-safe logger writing timestamped lines to stderr.
class ORCH_API Logger {
public:
    static Logger& instance();
    void set_level(LogLevel level);
    LogLevel level() const;
    void log(LogLevel level, const std::string& msg);

private:
    Logger() = default;
    LogLevel level_ = LogLevel::Info;
    std::mutex mu_;
};

}  // namespace orch

#define ORCH_LOG(lvl, expr)                                                   \
    do {                                                                      \
        if (static_cast<int>(lvl) >=                                          \
            static_cast<int>(::orch::Logger::instance().level())) {           \
            std::ostringstream orch_log_os_;                                  \
            orch_log_os_ << expr;                                             \
            ::orch::Logger::instance().log(lvl, orch_log_os_.str());          \
        }                                                                     \
    } while (0)

#define LOG_DEBUG(expr) ORCH_LOG(::orch::LogLevel::Debug, expr)
#define LOG_INFO(expr) ORCH_LOG(::orch::LogLevel::Info, expr)
#define LOG_WARN(expr) ORCH_LOG(::orch::LogLevel::Warn, expr)
#define LOG_ERROR(expr) ORCH_LOG(::orch::LogLevel::Error, expr)
