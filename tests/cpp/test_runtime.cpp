/*
 * test_runtime.cpp — C++ unit tests for PRIME Phase 1 core.
 *
 * Uses no external test framework. Each test is a function that throws
 * on failure. The main() function runs them all and prints a summary.
 */

#include "../../cpp/runtime/runtime.hpp"
#include "../../cpp/core/config.hpp"
#include "../../cpp/core/status.hpp"
#include "../../cpp/core/module.hpp"
#include "../../cpp/core/logger.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace prime;

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------
static int tests_run = 0;
static int tests_failed = 0;

#define EXPECT_TRUE(expr) do { \
    if (!(expr)) throw std::logic_error(std::string("EXPECT_TRUE failed: ") + #expr \
        + " at line " + std::to_string(__LINE__)); \
} while(0)

#define EXPECT_EQ(a, b) do { \
    if (!((a) == (b))) throw std::logic_error( \
        std::string("EXPECT_EQ failed: ") + #a + " != " + #b \
        + " at line " + std::to_string(__LINE__)); \
} while(0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

static void run_test(const std::string& name, void(*fn)()) {
    tests_run++;
    try {
        fn();
        std::cout << "  [PASS] " << name << "\n";
    } catch (const std::exception& e) {
        tests_failed++;
        std::cerr << "  [FAIL] " << name << ": " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// A trivial concrete module for testing the module abstraction
// ---------------------------------------------------------------------------
class TestModule : public Module {
public:
    explicit TestModule(const std::string& name) : Module(name) {}

    Status initialize(const RuntimeConfig&) override {
        initialized_ = true;
        init_count_++;
        return Status::ok_status();
    }

    Status shutdown() override {
        shutdown_count_++;
        initialized_ = false;
        return Status::ok_status();
    }

    int init_count_     = 0;
    int shutdown_count_ = 0;
};

class FailingModule : public Module {
public:
    explicit FailingModule(const std::string& name) : Module(name) {}

    Status initialize(const RuntimeConfig&) override {
        return Status::error(StatusCode::ERROR_INIT_FAILED, "intentional failure");
    }

    Status shutdown() override { return Status::ok_status(); }
};

// ---------------------------------------------------------------------------
// Tests: Status
// ---------------------------------------------------------------------------
static void test_status_ok() {
    Status s = Status::ok_status();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code, StatusCode::OK);
}

static void test_status_error() {
    Status s = Status::error(StatusCode::ERROR_INIT_FAILED, "test error");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.message, "test error");
}

// ---------------------------------------------------------------------------
// Tests: Config
// ---------------------------------------------------------------------------
static void test_config_default_valid() {
    RuntimeConfig cfg;
    std::string err;
    EXPECT_TRUE(cfg.validate(err));
}

static void test_config_invalid_backend() {
    RuntimeConfig cfg;
    cfg.compute_backend = "invalid";
    std::string err;
    EXPECT_FALSE(cfg.validate(err));
    EXPECT_FALSE(err.empty());
}

// ---------------------------------------------------------------------------
// Tests: Runtime lifecycle
// ---------------------------------------------------------------------------
static void test_runtime_init_shutdown() {
    Runtime rt;
    EXPECT_EQ(rt.state(), RuntimeState::UNINITIALIZED);

    Status s = rt.initialize();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(rt.state(), RuntimeState::RUNNING);

    s = rt.shutdown();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(rt.state(), RuntimeState::SHUTDOWN);
}

static void test_runtime_double_init() {
    Runtime rt;
    rt.initialize();
    Status s = rt.initialize();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::ERROR_ALREADY_INITIALIZED);
    rt.shutdown();
}

static void test_runtime_shutdown_before_init() {
    Runtime rt;
    Status s = rt.shutdown();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::ERROR_NOT_INITIALIZED);
}

static void test_runtime_status_string() {
    Runtime rt;
    rt.initialize();
    std::string status = rt.status_string();
    EXPECT_FALSE(status.empty());
    EXPECT_TRUE(status.find("RUNNING") != std::string::npos);
    rt.shutdown();
}

// ---------------------------------------------------------------------------
// Tests: Module abstraction
// ---------------------------------------------------------------------------
static void test_module_registered_and_initialized() {
    Runtime rt;
    auto mod = std::make_shared<TestModule>("test_mod");
    EXPECT_EQ(mod->init_count_, 0);

    Status s = rt.register_module(mod);
    EXPECT_TRUE(s.ok());

    rt.initialize();
    EXPECT_EQ(mod->init_count_, 1);
    EXPECT_TRUE(mod->is_initialized());

    rt.shutdown();
    EXPECT_EQ(mod->shutdown_count_, 1);
    EXPECT_FALSE(mod->is_initialized());
}

static void test_module_duplicate_registration() {
    Runtime rt;
    auto mod1 = std::make_shared<TestModule>("dup");
    auto mod2 = std::make_shared<TestModule>("dup");
    EXPECT_TRUE(rt.register_module(mod1).ok());
    EXPECT_FALSE(rt.register_module(mod2).ok());
}

static void test_module_register_after_init_fails() {
    Runtime rt;
    rt.initialize();
    auto mod = std::make_shared<TestModule>("late");
    Status s = rt.register_module(mod);
    EXPECT_FALSE(s.ok());
    rt.shutdown();
}

static void test_module_failing_init_propagates() {
    Runtime rt;
    auto mod = std::make_shared<FailingModule>("bad_mod");
    rt.register_module(mod);
    Status s = rt.initialize();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::ERROR_INIT_FAILED);
}

static void test_get_module() {
    Runtime rt;
    auto mod = std::make_shared<TestModule>("findme");
    rt.register_module(mod);
    rt.initialize();

    auto found = rt.get_module("findme");
    EXPECT_TRUE(found != nullptr);

    auto missing = rt.get_module("nothere");
    EXPECT_TRUE(missing == nullptr);

    rt.shutdown();
}

// ---------------------------------------------------------------------------
// Tests: Logger
// ---------------------------------------------------------------------------
static void test_logger_level_filter() {
    Logger log(LogLevel::ERROR);
    EXPECT_EQ(log.level(), LogLevel::ERROR);
    log.set_level(LogLevel::DEBUG);
    EXPECT_EQ(log.level(), LogLevel::DEBUG);
    // Actual output goes to stderr; we only test that calls don't throw.
    log.error("test error");
    log.warn("test warn");
    log.info("test info");
    log.debug("test debug");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "PRIME C++ Tests\n";
    std::cout << "===============\n";

    run_test("status_ok",                        test_status_ok);
    run_test("status_error",                     test_status_error);
    run_test("config_default_valid",             test_config_default_valid);
    run_test("config_invalid_backend",           test_config_invalid_backend);
    run_test("runtime_init_shutdown",            test_runtime_init_shutdown);
    run_test("runtime_double_init",              test_runtime_double_init);
    run_test("runtime_shutdown_before_init",     test_runtime_shutdown_before_init);
    run_test("runtime_status_string",            test_runtime_status_string);
    run_test("module_registered_and_init",       test_module_registered_and_initialized);
    run_test("module_duplicate_registration",    test_module_duplicate_registration);
    run_test("module_register_after_init_fails", test_module_register_after_init_fails);
    run_test("module_failing_init_propagates",   test_module_failing_init_propagates);
    run_test("get_module",                       test_get_module);
    run_test("logger_level_filter",              test_logger_level_filter);

    std::cout << "\n";
    std::cout << "Results: " << (tests_run - tests_failed) << "/" << tests_run << " passed\n";

    return (tests_failed == 0) ? 0 : 1;
}
