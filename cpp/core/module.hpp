#pragma once

#include "status.hpp"
#include "config.hpp"
#include <string>
#include <memory>

namespace prime {

// Module is the abstract base for all PRIME subsystems.
// Each future phase (Memory, ATLAS, MOSAIC, etc.) will subclass Module.
//
// Lifecycle:
//   initialize() → execute() [0..N] → shutdown()
//
// Modules are owned by the Runtime via shared_ptr.
class Module {
public:
    explicit Module(std::string name) : name_(std::move(name)) {}
    virtual ~Module() = default;

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    const std::string& name() const { return name_; }
    bool is_initialized() const { return initialized_; }

    // initialize is called once by the Runtime after the module is registered.
    virtual Status initialize(const RuntimeConfig& config) = 0;

    // shutdown is called once by the Runtime during teardown.
    virtual Status shutdown() = 0;

    // set_initialized is called by the Runtime to update module state.
    // Subclasses must not call this themselves.
    void set_initialized(bool value) { initialized_ = value; }

protected:
    bool initialized_ = false;

private:
    std::string name_;
};

} // namespace prime
