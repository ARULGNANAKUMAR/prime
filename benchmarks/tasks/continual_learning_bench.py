"""
Category F — Continual Learning Benchmark

Tests whether PRIME can incorporate new information without full retraining.

Hypothesis H8: Continual learning can incorporate new information without full
system retraining.

Design:
  Stage 0: Baseline — solve tasks from Dataset A (speed problems)
  Stage 1: Introduce Dataset B (revenue problems via memory/solve)
  Stage 2: Verify Dataset B knowledge
  Stage 3: Re-verify Dataset A knowledge still works (retention check)

Metrics:
  Acquisition rate  = success on B tasks after introduction
  Retention rate    = success on A tasks after B introduction
"""
import time
import statistics


DATASET_A = [
    "A car travels 150 km in 3 hours. Calculate average speed.",
    "A train travels 480 km in 4 hours. Calculate average speed.",
    "A ship sails 500 km in 25 hours. Calculate average speed.",
]

DATASET_B = [
    "Revenue is 500 units at price 20 each. Calculate total revenue.",
    "Revenue is 100 units at price 50 each. Calculate total revenue.",
    "Revenue is 250 units at price 8 each. Calculate total revenue.",
]


def run(prime_instance, iterations=30):
    """Execute the four-stage continual learning experiment."""

    def solve_batch(tasks, n):
        batch_results = []
        for text in tasks:
            lats, statuses = [], []
            for _ in range(n):
                t0 = time.perf_counter()
                try:
                    r = prime_instance.solve(text)
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    statuses.append(str(r.status))
                except Exception:
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    statuses.append("EXCEPTION")
            s = sum(1 for x in statuses if "SUCCESS" in x.upper())
            batch_results.append({
                "task": text[:60],
                "success_rate": round(s / n, 4),
                "mean_latency_ms": round(statistics.mean(lats), 4),
            })
        return batch_results

    # Stage 0: Baseline on A
    stage0_A = solve_batch(DATASET_A, iterations)

    # Stage 1: Introduce B (no explicit training step — PRIME learns through solve())
    # Run B tasks to expose PRIME to new domain
    _ = solve_batch(DATASET_B, 5)  # exposure pass

    # Stage 2: Test B knowledge acquisition
    stage2_B = solve_batch(DATASET_B, iterations)

    # Stage 3: Re-test A knowledge retention
    stage3_A = solve_batch(DATASET_A, iterations)

    # Metrics
    baseline_A_mean = statistics.mean(r["success_rate"] for r in stage0_A)
    acquisition_B_mean = statistics.mean(r["success_rate"] for r in stage2_B)
    retention_A_mean = statistics.mean(r["success_rate"] for r in stage3_A)

    retention_rate = round(retention_A_mean / baseline_A_mean, 4) if baseline_A_mean > 0 else None
    acquisition_rate = acquisition_B_mean

    aggregate = {
        "dataset_A_tasks": len(DATASET_A),
        "dataset_B_tasks": len(DATASET_B),
        "iterations_per_stage": iterations,
        "baseline_A_success_rate": round(baseline_A_mean, 4),
        "acquisition_B_success_rate": round(acquisition_rate, 4),
        "retention_A_success_rate": round(retention_A_mean, 4),
        "retention_ratio": retention_rate,
        "note": (
            "PRIME's continual learning operates via internal C++ components. "
            "This benchmark measures whether task B can be solved after exposure, "
            "and whether task A performance is retained. It does not measure "
            "weight-level updates (PRIME has no neural weights)."
        ),
    }

    return {
        "stage0_baseline_A": stage0_A,
        "stage2_acquisition_B": stage2_B,
        "stage3_retention_A": stage3_A,
        "aggregate": aggregate,
    }
