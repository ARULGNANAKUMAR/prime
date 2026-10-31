#include "runtime.hpp"

#include <sstream>
#include <algorithm>

namespace prime {

Runtime::Runtime(RuntimeConfig config) : config_(std::move(config)) {
    logger_.set_level(config_.log_level);
}

Runtime::~Runtime() {
    if (state_ == RuntimeState::RUNNING) {
        // Best-effort shutdown during destruction; errors are logged, not thrown.
        auto s = shutdown();
        if (!s.ok()) {
            logger_.error("Shutdown during destruction failed: " + s.message);
        }
    }
}

Status Runtime::initialize() {
    if (state_ == RuntimeState::RUNNING) {
        return Status::error(StatusCode::ERROR_ALREADY_INITIALIZED,
                             "Runtime is already initialized");
    }
    if (state_ == RuntimeState::SHUTDOWN) {
        return Status::error(StatusCode::ERROR_INIT_FAILED,
                             "Runtime cannot be re-initialized after shutdown");
    }

    std::string config_err;
    if (!config_.validate(config_err)) {
        return Status::error(StatusCode::ERROR_INVALID_CONFIG, config_err);
    }

    logger_.info("Runtime initializing (backend=" + config_.compute_backend + ")");

    for (auto& mod : module_order_) {
        logger_.info("Initializing module: " + mod->name());
        auto s = mod->initialize(config_);
        if (!s.ok()) {
            return Status::error(StatusCode::ERROR_INIT_FAILED,
                                 "Module '" + mod->name() + "' init failed: " + s.message);
        }
        mod->set_initialized(true);
    }

    state_ = RuntimeState::RUNNING;
    logger_.info("Runtime initialized successfully");
    return Status::ok_status();
}

Status Runtime::shutdown() {
    if (state_ != RuntimeState::RUNNING) {
        return Status::error(StatusCode::ERROR_NOT_INITIALIZED,
                             "Runtime is not running");
    }

    logger_.info("Runtime shutting down");

    // Shutdown modules in reverse registration order.
    for (auto it = module_order_.rbegin(); it != module_order_.rend(); ++it) {
        auto& mod = *it;
        if (mod->is_initialized()) {
            logger_.info("Shutting down module: " + mod->name());
            auto s = mod->shutdown();
            if (!s.ok()) {
                // Log but continue; partial shutdown is better than none.
                logger_.error("Module '" + mod->name() + "' shutdown error: " + s.message);
            } else {
                mod->set_initialized(false);
            }
        }
    }

    state_ = RuntimeState::SHUTDOWN;
    logger_.info("Runtime shut down");
    return Status::ok_status();
}

Status Runtime::register_module(std::shared_ptr<Module> module) {
    if (state_ != RuntimeState::UNINITIALIZED) {
        return Status::error(StatusCode::ERROR_INIT_FAILED,
                             "Modules must be registered before initialize()");
    }
    if (!module) {
        return Status::error(StatusCode::ERROR_INVALID_CONFIG, "Module pointer is null");
    }
    const auto& name = module->name();
    if (modules_.count(name)) {
        return Status::error(StatusCode::ERROR_INVALID_CONFIG,
                             "Module '" + name + "' is already registered");
    }
    modules_[name] = module;
    module_order_.push_back(std::move(module));
    return Status::ok_status();
}

std::shared_ptr<Module> Runtime::get_module(const std::string& name) const {
    auto it = modules_.find(name);
    return (it != modules_.end()) ? it->second : nullptr;
}

std::string Runtime::status_string() const {
    std::ostringstream oss;
    oss << "PRIME Runtime | state=";
    switch (state_) {
        case RuntimeState::UNINITIALIZED: oss << "UNINITIALIZED"; break;
        case RuntimeState::RUNNING:       oss << "RUNNING";       break;
        case RuntimeState::SHUTDOWN:      oss << "SHUTDOWN";      break;
    }
    oss << " | backend=" << config_.compute_backend;
    oss << " | modules=" << module_order_.size();
    return oss.str();
}

} // namespace prime
