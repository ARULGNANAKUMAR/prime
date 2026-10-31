#pragma once

#include <string>
#include <stdexcept>

namespace prime {

enum class StatusCode {
    OK = 0,
    ERROR_INIT_FAILED,
    ERROR_INVALID_CONFIG,
    ERROR_MODULE_NOT_FOUND,
    ERROR_ALREADY_INITIALIZED,
    ERROR_NOT_INITIALIZED,
    ERROR_SHUTDOWN_FAILED,
};

struct Status {
    StatusCode code;
    std::string message;

    Status() : code(StatusCode::OK), message("OK") {}
    Status(StatusCode c, std::string msg) : code(c), message(std::move(msg)) {}

    bool ok() const { return code == StatusCode::OK; }

    static Status ok_status() { return Status{}; }
    static Status error(StatusCode c, std::string msg) { return Status{c, std::move(msg)}; }
};

// Thrown from C++ when a status error must propagate to Python
class PrimeException : public std::runtime_error {
public:
    explicit PrimeException(const Status& s)
        : std::runtime_error(s.message), code_(s.code) {}

    StatusCode code() const { return code_; }

private:
    StatusCode code_;
};

} // namespace prime
