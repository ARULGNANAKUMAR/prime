"""
examples/benchmark_phase3.py — Phase 3 Memory Engine benchmark.

Measures real throughput and latency for:
    - Insert (store) operations
    - Retrieval by ID (get)
    - Search (bag-of-words + scoring)
    - retrieve_context
    - expire_all
    - Memory footprint (approximate)

Run:
    PYTHONPATH=python python3 examples/benchmark_phase3.py

Results are printed to stdout. No numbers are invented; all measurements
are from the actual implementation running on this machine.
"""

from __future__ import annotations

import os
import sys
import time
import platform
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from prime import MemoryConfig, MemoryEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hline(char="-", width=60):
    print(char * width)


def header(title):
    hline("=")
    print(f"  {title}")
    hline("=")


def row(label, value):
    print(f"  {label:<35} {value}")


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def print_system_info():
    header("System Information")
    import subprocess
    row("OS",            platform.platform())
    row("Python",        sys.version.split()[0])
    try:
        cc = subprocess.check_output(["g++", "--version"],
                                      stderr=subprocess.STDOUT).decode().split("\n")[0]
    except Exception:
        cc = "unknown"
    row("Compiler",      cc)
    row("CPU",           platform.processor() or "unknown")
    row("C++ standard",  "C++17")
    print()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

CONTENT_TEMPLATE = (
    "Memory item {i}: The quick brown fox jumps over the lazy dog. "
    "This sentence is used as filler to give realistic content length."
)

def make_engine(max_items=0):
    cfg = MemoryConfig(
        max_working=max_items,
        max_short_term=max_items,
        max_long_term=max_items,
        recency_half_life=86400.0,
    )
    return MemoryEngine(cfg)


def bench_insert(n: int) -> tuple[float, list[int]]:
    """Returns (elapsed_seconds, list_of_ids)."""
    mem = make_engine()
    ids: list[int] = []
    t0 = time.perf_counter()
    for i in range(n):
        mid = mem.store(
            CONTENT_TEMPLATE.format(i=i),
            memory_type="long_term",
            importance=round((i % 10 + 1) / 10.0, 1),
            metadata={"batch": str(i % 5), "index": str(i)},
        )
        ids.append(mid)
    elapsed = time.perf_counter() - t0
    return elapsed, ids, mem


def bench_get(mem: MemoryEngine, ids: list[int]) -> float:
    t0 = time.perf_counter()
    for mid in ids:
        mem.get(mid)
    return time.perf_counter() - t0


def bench_search(mem: MemoryEngine, n_queries: int) -> float:
    queries = [
        "quick brown fox",
        "lazy dog jumps",
        "memory item filler",
        "realistic content length",
        "sentence used filler",
    ]
    t0 = time.perf_counter()
    for i in range(n_queries):
        mem.search(queries[i % len(queries)], limit=10)
    return time.perf_counter() - t0


def bench_retrieve_context(mem: MemoryEngine, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        mem.retrieve_context("quick brown fox", limit=5)
    return time.perf_counter() - t0


def bench_expire(n: int) -> float:
    """Store n already-expired items then run expire_all."""
    mem = make_engine()
    past = time.time() - 1.0
    for i in range(n):
        mem.store(f"expired {i}", expires_at=past)
    t0 = time.perf_counter()
    removed = mem.expire_all()
    elapsed = time.perf_counter() - t0
    assert removed == n, f"expected {n} removals, got {removed}"
    return elapsed


def bench_memory_footprint(n: int) -> int:
    """Returns approximate peak memory delta in bytes."""
    tracemalloc.start()
    mem = make_engine()
    for i in range(n):
        mem.store(CONTENT_TEMPLATE.format(i=i), memory_type="long_term",
                  importance=0.5)
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    return sum(s.size for s in snapshot.statistics("lineno"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_system_info()

    SIZES = [1_000, 10_000]

    for N in SIZES:
        header(f"Insert / Retrieve / Search  (N = {N:,} items)")

        # Insert
        elapsed_ins, ids, mem = bench_insert(N)
        row("Insert time (ms)",         f"{elapsed_ins*1000:.1f}")
        row("Throughput (writes/sec)",   f"{N/elapsed_ins:,.0f}")
        row("Items in store",            f"{mem.count():,}")
        print()

        # Get by ID
        # Benchmark a sample to keep the benchmark fast
        sample = ids[:min(N, 1000)]
        elapsed_get = bench_get(mem, sample)
        row("Get by ID (1K sample, ms)", f"{elapsed_get*1000:.2f}")
        row("Avg get latency (µs)",      f"{elapsed_get/len(sample)*1e6:.1f}")
        print()

        # Search
        N_QUERIES = 100
        elapsed_search = bench_search(mem, N_QUERIES)
        row(f"Search ({N_QUERIES} queries, ms)", f"{elapsed_search*1000:.1f}")
        row("Avg search latency (ms)",   f"{elapsed_search/N_QUERIES*1000:.2f}")
        row("Searches/sec",              f"{N_QUERIES/elapsed_search:,.0f}")
        print()

        # retrieve_context
        elapsed_ctx = bench_retrieve_context(mem, N_QUERIES)
        row(f"retrieve_context ({N_QUERIES}x, ms)", f"{elapsed_ctx*1000:.1f}")
        row("Avg latency (ms)",          f"{elapsed_ctx/N_QUERIES*1000:.2f}")
        print()

        del mem

    # Expiration
    header("expire_all  (1K expired items)")
    elapsed_exp = bench_expire(1000)
    row("expire_all 1K items (ms)",   f"{elapsed_exp*1000:.2f}")
    print()

    # Memory footprint
    header("Memory footprint (approximate, via tracemalloc)")
    for N in [1_000, 10_000]:
        total_bytes = bench_memory_footprint(N)
        row(f"Tracemalloc peak for {N:>6,} items", f"{total_bytes/1024:.1f} KiB")
        row(f"  Per item",                          f"{total_bytes/N:.0f} bytes")
        print()

    hline("=")
    print("  Benchmark complete.")
    hline("=")


if __name__ == "__main__":
    main()
