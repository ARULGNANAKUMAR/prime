"""
Category G — Adaptive Optimization Benchmark

Hypothesis H9: Adaptive optimization can improve computational efficiency or
decision quality.

Compares PRIME solve() with default (adaptive) configuration vs a simulation
of static configuration using repeated identical queries.

Note: PRIME does not expose a config flag to disable adaptive optimization
directly; we approximate by comparing:
  - First-call latency (cold, no optimization context built up)
  - Repeated-call latency (warm, optimization context established)
This measures within-session adaptation rather than a true A/B config test.
"""
import time
import statistics


TASKS = [
    "A car travels 150 km in 3 hours. Calculate average speed.",
    "Calculate 1000 divided by 25.",
    "A train travels 480 km in 4 hours. Calculate average speed.",
    "Revenue is 500 units at price 20 each. Calculate total revenue.",
    "A population of 200 doubles every 3 hours. What is the count after 6 hours?",
]


def run(prime_instance, cold_n=5, warm_n=100):
    """
    For each task measure:
      - cold_latency: first cold_n calls
      - warm_latency: next warm_n calls
    Improvement (if any) suggests session-level adaptive optimization.
    """
    results = []

    for text in TASKS:
        cold_lats, warm_lats = [], []
        cold_statuses, warm_statuses = [], []

        # Cold phase
        for _ in range(cold_n):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(text)
                cold_lats.append((time.perf_counter() - t0) * 1000.0)
                cold_statuses.append(str(r.status))
            except Exception:
                cold_lats.append((time.perf_counter() - t0) * 1000.0)
                cold_statuses.append("EXCEPTION")

        # Warm phase
        for _ in range(warm_n):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(text)
                warm_lats.append((time.perf_counter() - t0) * 1000.0)
                warm_statuses.append(str(r.status))
            except Exception:
                warm_lats.append((time.perf_counter() - t0) * 1000.0)
                warm_statuses.append("EXCEPTION")

        cold_mean = statistics.mean(cold_lats)
        warm_mean = statistics.mean(warm_lats)
        improvement_pct = round((cold_mean - warm_mean) / cold_mean * 100.0, 2) if cold_mean > 0 else None

        results.append({
            "task": text[:60],
            "cold_n": cold_n,
            "warm_n": warm_n,
            "cold_success_rate": round(sum(1 for s in cold_statuses if "SUCCESS" in s.upper()) / cold_n, 4),
            "warm_success_rate": round(sum(1 for s in warm_statuses if "SUCCESS" in s.upper()) / warm_n, 4),
            "cold_latency_ms": {
                "mean": round(cold_mean, 4),
                "min": round(min(cold_lats), 4),
                "max": round(max(cold_lats), 4),
            },
            "warm_latency_ms": {
                "mean": round(warm_mean, 4),
                "median": round(statistics.median(warm_lats), 4),
                "min": round(min(warm_lats), 4),
                "max": round(max(warm_lats), 4),
                "stdev": round(statistics.stdev(warm_lats), 4),
            },
            "latency_improvement_pct": improvement_pct,
            "improvement_observed": improvement_pct is not None and improvement_pct > 2.0,
        })

    all_improvements = [r["latency_improvement_pct"] for r in results if r["latency_improvement_pct"] is not None]
    aggregate = {
        "total_tasks": len(results),
        "tasks_with_improvement": sum(1 for r in results if r["improvement_observed"]),
        "mean_improvement_pct": round(statistics.mean(all_improvements), 2) if all_improvements else None,
        "cold_n": cold_n,
        "warm_n": warm_n,
        "limitation": (
            "True adaptive optimization A/B test requires PRIME to expose a "
            "disable_adaptive_optimization flag, which is not part of the current "
            "public API. This benchmark measures warm-up effect as a proxy."
        ),
    }

    return {"tasks": results, "aggregate": aggregate}
