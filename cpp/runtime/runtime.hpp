#pragma once

#include "../core/config.hpp"
#include "../core/status.hpp"
#include "../core/module.hpp"
#include "../core/logger.hpp"

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace prime {

enum class RuntimeState {
    UNINITIALIZED,
    RUNNING,
    SHUTDOWN,
};

// Runtime is the central object that owns all modules and manages the lifecycle.
// There is typically one Runtime per process (owned by the Python PRIME object).
// It is not a singleton so tests can instantiate multiple runtimes.
class Runtime {
public:
    explicit Runtime(RuntimeConfig config = {});
    ~Runtime();

    // Not copyable or movable (owns shared state and logger).
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // initialize starts the runtime. Must be called before any module interaction.
    Status initialize();

    // shutdown stops the runtime and all registered modules in reverse order.
    Status shutdown();

    // register_module adds a module. Must be called before initialize().
    Status register_module(std::shared_ptr<Module> module);

    // get_module returns the named module or nullptr.
    std::shared_ptr<Module> get_module(const std::string& name) const;

    RuntimeState state() const { return state_; }
    const RuntimeConfig& config() const { return config_; }
    Logger& logger() { return logger_; }

    // status_string returns a human-readable summary for Python-side status().
    std::string status_string() const;

private:
    RuntimeConfig config_;
    RuntimeState state_ = RuntimeState::UNINITIALIZED;
    Logger logger_;

    // Modules stored in registration order; shut down in reverse.
    std::vector<std::shared_ptr<Module>> module_order_;
    std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
};

} // namespace prime
