"""
benchmark_phase1.py — Phase 1 baseline performance measurements.

Measures: init time, shutdown time, C++/Python call overhead, memory usage.
These numbers establish a baseline for future phases to compare against.
"""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from prime import PRIME, RuntimeConfig


def measure_init_shutdown(n: int = 50) -> dict:
    """Measure average init and shutdown times over n iterations."""
    init_times = []
    shutdown_times = []

    silent_cfg = RuntimeConfig(log_level="silent")

    for _ in range(n):
        t0 = time.perf_counter()
        brain = PRIME(silent_cfg)
        init_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        brain.shutdown()
        shutdown_times.append(time.perf_counter() - t1)

    return {
        "init_mean_us":     statistics.mean(init_times) * 1e6,
        "init_median_us":   statistics.median(init_times) * 1e6,
        "shutdown_mean_us": statistics.mean(shutdown_times) * 1e6,
    }


def measure_call_overhead(n: int = 1000) -> dict:
    """Measure Python→C++ call overhead via repeated status() calls."""
    brain = PRIME(RuntimeConfig(log_level="silent"))
    times = []

    for _ in range(n):
        t0 = time.perf_counter()
        _ = brain.status()
        times.append(time.perf_counter() - t0)

    brain.shutdown()

    return {
        "call_mean_us":   statistics.mean(times) * 1e6,
        "call_median_us": statistics.median(times) * 1e6,
        "call_p99_us":    sorted(times)[int(n * 0.99)] * 1e6,
    }


def measure_memory() -> dict:
    """Estimate process memory with and without an active runtime."""
    try:
        import resource
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        brain = PRIME(RuntimeConfig(log_level="silent"))
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        brain.shutdown()
        # On Linux ru_maxrss is in kB; on macOS in bytes.
        unit = 1024 if sys.platform == "linux" else 1
        return {
            "before_kb": before * unit // 1024,
            "after_kb":  after  * unit // 1024,
            "delta_kb":  (after - before) * unit // 1024,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=== PRIME Phase 1 Baseline Benchmark ===\n")

    print("Init / shutdown (50 iterations):")
    r = measure_init_shutdown()
    print(f"  Init   mean:   {r['init_mean_us']:.1f} µs")
    print(f"  Init   median: {r['init_median_us']:.1f} µs")
    print(f"  Shutdown mean: {r['shutdown_mean_us']:.1f} µs")

    print()
    print("C++/Python call overhead (1000 status() calls):")
    r = measure_call_overhead()
    print(f"  Mean:   {r['call_mean_us']:.2f} µs")
    print(f"  Median: {r['call_median_us']:.2f} µs")
    print(f"  p99:    {r['call_p99_us']:.2f} µs")

    print()
    print("Memory footprint:")
    r = measure_memory()
    if "error" in r:
        print(f"  (unavailable: {r['error']})")
    else:
        print(f"  Before runtime: {r['before_kb']} kB")
        print(f"  After  runtime: {r['after_kb']} kB")
        print(f"  Delta:          {r['delta_kb']} kB")

    print("\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
