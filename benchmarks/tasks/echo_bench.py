"""
Category D — ECHO Benchmark

Tests whether previous successful experiences improve later task performance.

Hypothesis H5: ECHO can convert previous successful experiences into reusable
knowledge.

Design:
  Phase 1: Solve a set of TRAINING tasks (experience generation)
  Phase 2: Solve TRANSFER tasks — structurally similar but not identical
  Compare latency Phase 2 iteration 1 vs iteration N (within-session learning)
"""
import time
import statistics


TRAINING_TASKS = [
    "A car travels 150 km in 3 hours. Calculate average speed.",
    "A train travels 480 km in 4 hours. Calculate average speed.",
    "A bicycle travels 90 km in 3 hours. Calculate average speed.",
    "A ship sails 500 km in 25 hours. Calculate average speed.",
    "A runner covers 10 km in 50 minutes. Calculate average speed in km per minute.",
]

TRANSFER_TASKS = [
    "A plane flies 1200 km in 2 hours. Calculate average speed.",
    "A boat travels 60 km in 4 hours. What is the average speed?",
    "Calculate 360 divided by 12.",
    "A motorcycle covers 240 km in 3 hours. What is its speed?",
    "A swimmer completes 2 km in 40 minutes. Calculate speed in km per minute.",
]


def run(prime_instance, iterations=50):
    """
    Phase 1: Run training tasks (experience accumulation).
    Phase 2: Run transfer tasks, measuring early vs late iterations to detect
             any within-session improvement that ECHO may provide.
    """
    # --- Phase 1: Training ---
    training_results = []
    for task_text in TRAINING_TASKS:
        t0 = time.perf_counter()
        try:
            r = prime_instance.solve(task_text)
            elapsed = (time.perf_counter() - t0) * 1000.0
            training_results.append({
                "task": task_text[:60],
                "status": str(r.status),
                "latency_ms": round(elapsed, 4),
            })
        except Exception as e:
            training_results.append({
                "task": task_text[:60],
                "status": "EXCEPTION",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            })

    # --- Phase 2: Transfer tasks with early/late measurement ---
    transfer_results = []
    for task_text in TRANSFER_TASKS:
        early_latencies = []
        late_latencies = []
        statuses = []

        split = iterations // 2
        for i in range(iterations):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(task_text)
                elapsed = (time.perf_counter() - t0) * 1000.0
                statuses.append(str(r.status))
            except Exception:
                elapsed = (time.perf_counter() - t0) * 1000.0
                statuses.append("EXCEPTION")

            if i < split:
                early_latencies.append(elapsed)
            else:
                late_latencies.append(elapsed)

        success_count = sum(1 for s in statuses if "SUCCESS" in s.upper())
        early_mean = statistics.mean(early_latencies)
        late_mean = statistics.mean(late_latencies)
        # Negative value = improvement (lower latency later)
        latency_delta_pct = round((late_mean - early_mean) / early_mean * 100.0, 2) if early_mean > 0 else None

        transfer_results.append({
            "task": task_text[:60],
            "iterations": iterations,
            "success_rate": round(success_count / iterations, 4),
            "early_latency_ms": round(early_mean, 4),
            "late_latency_ms": round(late_mean, 4),
            "latency_delta_pct": latency_delta_pct,
            "improvement_observed": latency_delta_pct is not None and latency_delta_pct < -2.0,
        })

    training_success = sum(1 for r in training_results if "SUCCESS" in r["status"].upper())
    transfer_success = sum(1 for r in transfer_results if r["success_rate"] > 0.9)
    improvements = sum(1 for r in transfer_results if r["improvement_observed"])

    aggregate = {
        "training_tasks": len(TRAINING_TASKS),
        "training_success_count": training_success,
        "transfer_tasks": len(TRANSFER_TASKS),
        "transfer_tasks_high_success": transfer_success,
        "within_session_improvements_observed": improvements,
        "echo_latency_improvement_rate": round(improvements / len(TRANSFER_TASKS), 4),
        "note": (
            "ECHO operates as an internal C++ component. Measurable within-session "
            "latency improvement would support H5; absence does not rule out internal "
            "experience accumulation that becomes relevant across longer task sequences."
        ),
    }

    return {
        "training_results": training_results,
        "transfer_results": transfer_results,
        "aggregate": aggregate,
    }
