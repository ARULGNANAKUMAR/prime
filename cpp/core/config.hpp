#pragma once

#include <string>
#include <cstddef>

namespace prime {

// LogLevel controls verbosity of the runtime log output.
enum class LogLevel {
    SILENT = 0,
    ERROR   = 1,
    WARN    = 2,
    INFO    = 3,
    DEBUG   = 4,
};

// RuntimeConfig holds only the configuration options needed by Phase 1.
// Future phases will extend this structure (or add sub-configs) without
// breaking existing fields.
struct RuntimeConfig {
    // Logging
    LogLevel log_level = LogLevel::INFO;

    // Resource limits (advisory; enforced by future phases)
    std::size_t max_memory_bytes = 0;  // 0 = unlimited

    // Backend selection hint for future phases
    std::string compute_backend = "cpu";  // "cpu" | "cuda" (future)

    // Validate returns a non-OK status if any field has an illegal value.
    bool validate(std::string& error_msg) const {
        if (compute_backend != "cpu" && compute_backend != "cuda") {
            error_msg = "compute_backend must be 'cpu' or 'cuda'";
            return false;
        }
        return true;
    }
};

} // namespace prime
