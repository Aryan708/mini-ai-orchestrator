#include "orchestrator/logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace orch {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mu_);
    level_ = level;
}

LogLevel Logger::level() const { return level_; }

void Logger::log(LogLevel level, const std::string& msg) {
    static const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::lock_guard<std::mutex> lock(mu_);
    std::cerr << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
              << ms.count() << " [" << names[static_cast<int>(level)] << "] " << msg << '\n';
}

}  // namespace orch
