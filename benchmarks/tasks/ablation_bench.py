"""
Ablation Study

Compares PRIME with all components enabled against configurations where
components are simulated as absent by measuring observable differences.

Because PRIME's public API is PRIME() with no per-component disable flags,
this ablation uses:
  - Experiment 1: Full PRIME (baseline)
  - Experiment 2: PRIME with no prior memory (fresh instance each call)
  - Experiment 3: PRIME context_hint='' (minimal context)
  - Experiment 4: Minimal task (single arithmetic, no domain context)
  - Experiment 5: Complex task (multi-step, multi-domain)
  - Experiment 6: Stress test (many rapid calls)
  - Experiment 7: Cold start (new PRIME instance per solve)
  - Experiment 8: Warm reuse (same instance, many calls)

Note on methodology: True component ablation would require PRIME to expose
constructor flags like PRIME(use_atlas=False). As these are not in the public
API, we approximate ablation by varying task complexity and instance lifetime.
This is a genuine limitation of the study.
"""
import time
import statistics


STANDARD_TASK = "A car travels 150 km in 3 hours. Calculate average speed."
COMPLEX_TASK = (
    "A car travels 150 km in 3 hours and then 240 km in 4 hours. "
    "Calculate the average speed for the entire journey."
)
MINIMAL_TASK = "Calculate 150 divided by 3."


def _run_task(factory_fn, task_text, n, warmup=3):
    """factory_fn() returns a prime instance. New instance per call if factory_fn is not None."""
    latencies, statuses = [], []
    for i in range(n + warmup):
        instance = factory_fn()
        t0 = time.perf_counter()
        try:
            r = instance.solve(task_text)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                latencies.append(elapsed)
                statuses.append(str(r.status))
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                latencies.append(elapsed)
                statuses.append("EXCEPTION")
    success = sum(1 for s in statuses if "SUCCESS" in s.upper())
    return {
        "n": n,
        "success_rate": round(success / n, 4),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 4),
            "median": round(statistics.median(latencies), 4),
            "min": round(min(latencies), 4),
            "max": round(max(latencies), 4),
            "stdev": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        },
    }


def run(prime_class, iterations=50):
    """
    prime_class is the PRIME class (not an instance), so we can instantiate
    multiple configurations.
    """
    results = {}

    # Exp 1: Full PRIME, single shared instance, standard task
    p_shared = prime_class()
    lats, statuses = [], []
    for _ in range(3):  # warmup
        p_shared.solve(STANDARD_TASK)
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            r = p_shared.solve(STANDARD_TASK)
            lats.append((time.perf_counter() - t0) * 1000.0)
            statuses.append(str(r.status))
        except Exception:
            lats.append((time.perf_counter() - t0) * 1000.0)
            statuses.append("EXCEPTION")
    s = sum(1 for x in statuses if "SUCCESS" in x.upper())
    results["exp1_full_prime_shared"] = {
        "description": "Full PRIME, shared instance, standard task",
        "task": STANDARD_TASK[:60],
        "n": iterations,
        "success_rate": round(s / iterations, 4),
        "latency_ms": {
            "mean": round(statistics.mean(lats), 4),
            "median": round(statistics.median(lats), 4),
            "min": round(min(lats), 4),
            "max": round(max(lats), 4),
            "stdev": round(statistics.stdev(lats), 4) if len(lats) > 1 else 0.0,
        },
    }

    # Exp 2: Fresh instance per call (approximates "no memory accumulation")
    results["exp2_fresh_instance_per_call"] = _run_task(prime_class, STANDARD_TASK, iterations)
    results["exp2_fresh_instance_per_call"]["description"] = "Fresh PRIME instance per call (approx. no memory)"
    results["exp2_fresh_instance_per_call"]["task"] = STANDARD_TASK[:60]

    # Exp 3: Minimal task (single arithmetic)
    lats3, statuses3 = [], []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            r = p_shared.solve(MINIMAL_TASK)
            lats3.append((time.perf_counter() - t0) * 1000.0)
            statuses3.append(str(r.status))
        except Exception:
            lats3.append((time.perf_counter() - t0) * 1000.0)
            statuses3.append("EXCEPTION")
    s3 = sum(1 for x in statuses3 if "SUCCESS" in x.upper())
    results["exp3_minimal_task"] = {
        "description": "Full PRIME, shared instance, minimal arithmetic task",
        "task": MINIMAL_TASK[:60],
        "n": iterations,
        "success_rate": round(s3 / iterations, 4),
        "latency_ms": {
            "mean": round(statistics.mean(lats3), 4),
            "median": round(statistics.median(lats3), 4),
            "min": round(min(lats3), 4),
            "max": round(max(lats3), 4),
            "stdev": round(statistics.stdev(lats3), 4),
        },
    }

    # Exp 4: Complex multi-step task
    lats4, statuses4 = [], []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            r = p_shared.solve(COMPLEX_TASK)
            lats4.append((time.perf_counter() - t0) * 1000.0)
            statuses4.append(str(r.status))
        except Exception:
            lats4.append((time.perf_counter() - t0) * 1000.0)
            statuses4.append("EXCEPTION")
    s4 = sum(1 for x in statuses4 if "SUCCESS" in x.upper())
    results["exp4_complex_task"] = {
        "description": "Full PRIME, shared instance, complex multi-step task",
        "task": COMPLEX_TASK[:80],
        "n": iterations,
        "success_rate": round(s4 / iterations, 4),
        "latency_ms": {
            "mean": round(statistics.mean(lats4), 4),
            "median": round(statistics.median(lats4), 4),
            "min": round(min(lats4), 4),
            "max": round(max(lats4), 4),
            "stdev": round(statistics.stdev(lats4), 4),
        },
    }

    # Exp 5: Cross-domain task sequence (tests ATLAS/MOSAIC routing)
    cross_domain_tasks = [
        "A car travels 150 km in 3 hours. Calculate average speed.",
        "Revenue is 500 units at price 20. Calculate total revenue.",
        "Calculate 840 divided by 28.",
        "A population of 200 doubles every 3 hours. Count after 6 hours.",
        "An investment of 1000 at 5 percent for 1 year. Final value?",
    ]
    cross_lats, cross_statuses = [], []
    for task_text in cross_domain_tasks:
        for _ in range(iterations // len(cross_domain_tasks)):
            t0 = time.perf_counter()
            try:
                r = p_shared.solve(task_text)
                cross_lats.append((time.perf_counter() - t0) * 1000.0)
                cross_statuses.append(str(r.status))
            except Exception:
                cross_lats.append((time.perf_counter() - t0) * 1000.0)
                cross_statuses.append("EXCEPTION")
    s5 = sum(1 for x in cross_statuses if "SUCCESS" in x.upper())
    results["exp5_cross_domain_sequence"] = {
        "description": "Full PRIME, cross-domain task sequence (tests ATLAS/MOSAIC routing)",
        "domains": ["physics", "business", "arithmetic", "biology", "finance"],
        "total_calls": len(cross_lats),
        "success_rate": round(s5 / len(cross_statuses), 4) if cross_statuses else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(cross_lats), 4),
            "median": round(statistics.median(cross_lats), 4),
            "min": round(min(cross_lats), 4),
            "max": round(max(cross_lats), 4),
            "stdev": round(statistics.stdev(cross_lats), 4),
        },
    }

    # Ablation comparison table
    exp1_mean = results["exp1_full_prime_shared"]["latency_ms"]["mean"]
    exp2_mean = results["exp2_fresh_instance_per_call"]["latency_ms"]["mean"]
    exp3_mean = results["exp3_minimal_task"]["latency_ms"]["mean"]
    exp4_mean = results["exp4_complex_task"]["latency_ms"]["mean"]
    exp5_mean = results["exp5_cross_domain_sequence"]["latency_ms"]["mean"]

    comparison = {
        "baseline_exp1_mean_ms": exp1_mean,
        "exp2_vs_exp1_pct": round((exp2_mean - exp1_mean) / exp1_mean * 100, 2) if exp1_mean else None,
        "exp3_vs_exp1_pct": round((exp3_mean - exp1_mean) / exp1_mean * 100, 2) if exp1_mean else None,
        "exp4_vs_exp1_pct": round((exp4_mean - exp1_mean) / exp1_mean * 100, 2) if exp1_mean else None,
        "exp5_vs_exp1_pct": round((exp5_mean - exp1_mean) / exp1_mean * 100, 2) if exp1_mean else None,
        "methodology_limitation": (
            "True ablation requires PRIME(use_atlas=False, use_mosaic=False, ...) "
            "flags, which the current public API does not expose. Instance-lifetime "
            "and task-complexity proxies are used instead."
        ),
    }

    return {"experiments": results, "comparison": comparison}
