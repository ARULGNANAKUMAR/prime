#!/usr/bin/env python3
"""
scripts/test_installed_package.py
----------------------------------
Regression test script for the INSTALLED prime-ai package.

Run this from any directory AFTER installing the package:

    pip install prime_ai-1.0.0-*.whl   # or from TestPyPI
    python scripts/test_installed_package.py

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

import sys
import time

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []


def check(name, fn):
    t0 = time.monotonic()
    try:
        fn()
        elapsed = (time.monotonic() - t0) * 1000
        print(f"  {PASS}  {name}  ({elapsed:.1f} ms)")
        results.append((name, True, None))
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        print(f"  {FAIL}  {name}  ({elapsed:.1f} ms)")
        print(f"         {e}")
        results.append((name, False, str(e)))


# ── 0. Source-tree independence ──────────────────────────────────────────────
def test_not_source_tree():
    import prime
    path = prime.__file__
    assert "PRIME_PROJECT" not in path and "/python/prime" not in path, (
        f"prime imported from source tree: {path}"
    )
    assert "site-packages" in path, f"prime not in site-packages: {path}"


# ── 1. Package metadata ──────────────────────────────────────────────────────
def test_version():
    import prime
    assert prime.__version__ == "1.0.0", f"Bad version: {prime.__version__}"


def test_public_imports():
    from prime import (
        PRIME, ATLASEngine, MemoryEngine, ReasoningEngine,
        MOSAICEngine, ECHOEngine, LearningController,
        AdaptiveOptimizationEngine, PrimeConfig, PrimeResult,
        SolveStatus, FeatureFlags, PrimeTrace,
    )


# ── 2. C++ extensions ────────────────────────────────────────────────────────
def test_cpp_core():
    import _prime
    # Core extension exposes runtime state constants
    assert hasattr(_prime, "STATE_RUNNING")


def test_cpp_llm():
    import _prime_llm
    assert hasattr(_prime_llm, "LLMError")


def test_cpp_memory():
    import _prime_memory
    assert hasattr(_prime_memory, "MemoryError")


def test_cpp_atlas():
    import _prime_atlas
    assert hasattr(_prime_atlas, "ATLASError")


def test_cpp_reasoning():
    import _prime_reasoning
    # Extension exposes low-level functions, not a class
    assert hasattr(_prime_reasoning, "ReasoningError")
    assert hasattr(_prime_reasoning, "VERSION")


def test_cpp_mosaic():
    import _prime_mosaic
    assert hasattr(_prime_mosaic, "MOSAICError")
    assert hasattr(_prime_mosaic, "create_engine")


# ── 3. Runtime lifecycle ─────────────────────────────────────────────────────
def test_init_and_shutdown():
    from prime import PRIME
    p = PRIME()
    assert p.is_running()
    p.shutdown()
    assert not p.is_running()


def test_context_manager():
    from prime import PRIME
    with PRIME() as p:
        assert p.is_running()
    assert not p.is_running()


# ── 4. ATLAS ─────────────────────────────────────────────────────────────────
def test_atlas_parse():
    from prime import PRIME
    with PRIME() as p:
        atlas = p.atlas()
        prob = atlas.parse(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        assert len(prob.entities) > 0
        assert len(prob.goals) > 0


# ── 5. Reasoning ─────────────────────────────────────────────────────────────
def test_reasoning_solves_speed():
    from prime import PRIME
    with PRIME() as p:
        atlas = p.atlas()
        reasoner = p.reasoner()
        prob = atlas.parse(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        res = reasoner.solve(prob)
        assert res.status == "SOLVED", f"Expected SOLVED, got {res.status}"
        assert str(res.answer) == "50", f"Expected 50, got {res.answer!r}"


def test_reasoning_unresolved_status():
    from prime import PRIME
    with PRIME() as p:
        atlas = p.atlas()
        reasoner = p.reasoner()
        prob = atlas.parse("Calculate 6 multiplied by 7.")
        res = reasoner.solve(prob)
        assert res.status in ("UNRESOLVED", "SOLVED")


# ── 6. Memory ────────────────────────────────────────────────────────────────
def test_memory_store_retrieve():
    from prime import PRIME
    with PRIME() as p:
        mem = p.memory()
        mem.store("install_test_content", memory_type="short_term",
                  importance=0.8)
        ctx = mem.retrieve_context("install_test_content")
        assert ctx is not None


# ── 7. MOSAIC ────────────────────────────────────────────────────────────────
def test_mosaic_compose():
    from prime import PRIME
    with PRIME() as p:
        atlas = p.atlas()
        mosaic = p.mosaic()
        prob = atlas.parse(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        plan = mosaic.compose(prob)
        assert plan is not None
        assert len(plan.module_ids) > 0


# ── 8. ECHO ──────────────────────────────────────────────────────────────────
def test_echo_record():
    from prime import PRIME
    with PRIME() as p:
        echo = p.echo()
        exp = echo.record(problem_text="install_test_task")
        assert exp is not None
        assert hasattr(exp, "id")


# ── 9. Learning ──────────────────────────────────────────────────────────────
def test_learning_controller():
    from prime import PRIME
    with PRIME() as p:
        lc = p.learning()
        assert lc is not None
        # get_state() returns a dict with learning state info
        state = lc.get_state()
        assert isinstance(state, dict)
        # State dict contains mode, version, counts — verify it's non-empty
        assert "mode" in state and "version" in state


# ── 10. Optimizer ────────────────────────────────────────────────────────────
def test_optimizer():
    from prime import PRIME
    with PRIME() as p:
        opt = p.optimizer()
        assert opt is not None
        snap = opt.get_resource_snapshot()
        # ResourceSnapshot has cpu and memory sub-objects
        assert hasattr(snap, "cpu")
        assert hasattr(snap, "memory")


# ── 11. End-to-end solve ─────────────────────────────────────────────────────
def test_solve_speed_problem():
    from prime import PRIME, SolveStatus
    with PRIME() as p:
        r = p.solve(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        assert r.status == SolveStatus.SUCCESS, f"Status: {r.status}"
        assert r.answer == "50", f"Answer: {r.answer!r}"
        assert r.latency_ms > 0


def test_solve_partial_success():
    from prime import PRIME, SolveStatus
    with PRIME() as p:
        r = p.solve("Calculate 6 multiplied by 7.")
        assert r.status in (SolveStatus.PARTIAL_SUCCESS, SolveStatus.SUCCESS), (
            f"Unexpected: {r.status}"
        )


def test_solve_returns_prime_result():
    from prime import PRIME, PrimeResult, SolveStatus
    with PRIME() as p:
        r = p.solve(
            "A car travels 150 km in 3 hours. Calculate average speed.")
        assert isinstance(r, PrimeResult)
        assert isinstance(r.status, SolveStatus)
        assert isinstance(r.answer, str)
        assert isinstance(r.latency_ms, float)
        assert r.latency_ms < 5000


# ── 12. Offline network check ────────────────────────────────────────────────
def test_offline_solve():
    import socket as _socket
    from prime import PRIME, SolveStatus

    _orig = _socket.socket

    class _Blocked(Exception):
        pass

    def _no_net(*a, **kw):
        raise _Blocked("Network access during solve!")

    _socket.socket = _no_net
    try:
        with PRIME() as p:
            r = p.solve(
                "A car travels 150 km in 3 hours. Calculate average speed.")
            assert r.status == SolveStatus.SUCCESS
    finally:
        _socket.socket = _orig


# ── Run all checks ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("PRIME — Installed Package Regression Tests")
    print("=" * 60)

    print("\n[0] Source-tree independence")
    check("not_from_source_tree", test_not_source_tree)

    print("\n[1] Package metadata")
    check("version_is_1.0.0", test_version)
    check("public_imports", test_public_imports)

    print("\n[2] C++ extensions")
    check("cpp_core (_prime)", test_cpp_core)
    check("cpp_llm (_prime_llm)", test_cpp_llm)
    check("cpp_memory (_prime_memory)", test_cpp_memory)
    check("cpp_atlas (_prime_atlas)", test_cpp_atlas)
    check("cpp_reasoning (_prime_reasoning)", test_cpp_reasoning)
    check("cpp_mosaic (_prime_mosaic)", test_cpp_mosaic)

    print("\n[3] Runtime lifecycle")
    check("init_and_shutdown", test_init_and_shutdown)
    check("context_manager", test_context_manager)

    print("\n[4] ATLAS")
    check("atlas_parse", test_atlas_parse)

    print("\n[5] Reasoning")
    check("reasoning_solves_speed", test_reasoning_solves_speed)
    check("reasoning_unresolved_status", test_reasoning_unresolved_status)

    print("\n[6] Memory")
    check("memory_store_retrieve", test_memory_store_retrieve)

    print("\n[7] MOSAIC")
    check("mosaic_compose", test_mosaic_compose)

    print("\n[8] ECHO")
    check("echo_record", test_echo_record)

    print("\n[9] Learning")
    check("learning_controller", test_learning_controller)

    print("\n[10] Optimizer")
    check("optimizer_snapshot", test_optimizer)

    print("\n[11] End-to-end solve")
    check("solve_speed_problem", test_solve_speed_problem)
    check("solve_partial_success", test_solve_partial_success)
    check("solve_returns_prime_result", test_solve_returns_prime_result)

    print("\n[12] Offline runtime")
    check("offline_solve", test_offline_solve)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    print()
    print("=" * 60)
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print()
    print("=" * 60)
    print()

    sys.exit(0 if failed == 0 else 1)
