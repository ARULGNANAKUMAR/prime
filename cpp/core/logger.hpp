#pragma once

#include "config.hpp"
#include <iostream>
#include <string>
#include <mutex>

namespace prime {

// Logger is a lightweight, thread-safe log sink.
// It writes to stderr so it does not interfere with stdout-based Python output.
// The level filter keeps it silent in production without a framework dependency.
class Logger {
public:
    explicit Logger(LogLevel level = LogLevel::INFO) : level_(level) {}

    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    LogLevel level() const { return level_; }

    void error(const std::string& msg) { log(LogLevel::ERROR, "ERROR", msg); }
    void warn(const std::string& msg)  { log(LogLevel::WARN,  "WARN",  msg); }
    void info(const std::string& msg)  { log(LogLevel::INFO,  "INFO",  msg); }
    void debug(const std::string& msg) { log(LogLevel::DEBUG, "DEBUG", msg); }

private:
    void log(LogLevel msg_level, const char* label, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (msg_level <= level_) {
            std::cerr << "[PRIME][" << label << "] " << msg << "\n";
        }
    }

    LogLevel level_;
    std::mutex mutex_;
};

} // namespace prime
