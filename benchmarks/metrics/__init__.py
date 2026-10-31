"""Benchmark metrics utilities."""
import time
import resource
import statistics
import math


def get_rss_kb():
    """Return current process RSS in KB (Linux: ru_maxrss is already KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def timed_call(fn, n=1):
    """Call fn() n times; return (results_list, latencies_ms_list)."""
    results, latencies = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        t1 = time.perf_counter()
        results.append(r)
        latencies.append((t1 - t0) * 1000.0)
    return results, latencies


def summarise_latencies(latencies_ms):
    if not latencies_ms:
        return {}
    return {
        "n": len(latencies_ms),
        "mean_ms": round(statistics.mean(latencies_ms), 4),
        "median_ms": round(statistics.median(latencies_ms), 4),
        "min_ms": round(min(latencies_ms), 4),
        "max_ms": round(max(latencies_ms), 4),
        "stdev_ms": round(statistics.stdev(latencies_ms), 4) if len(latencies_ms) > 1 else 0.0,
    }


def pct_change(baseline, experimental):
    """Return % change from baseline to experimental (negative = improvement in latency)."""
    if baseline == 0:
        return None
    return round((experimental - baseline) / baseline * 100.0, 2)
