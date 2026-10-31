"""
Category E — MOSAIC Benchmark

Tests module selection and architecture composition across varied task types.

Hypothesis H6: MOSAIC can select/compose different cognitive modules based on
task requirements.
"""
import time
import statistics


TASK_TYPES = [
    {"id": "mo01", "type": "arithmetic",    "text": "Calculate 840 divided by 28."},
    {"id": "mo02", "type": "physics",       "text": "A car travels 150 km in 3 hours. Calculate average speed."},
    {"id": "mo03", "type": "business",      "text": "Revenue is 500 units at price 20 each. Calculate total revenue."},
    {"id": "mo04", "type": "biology",       "text": "A population of 200 doubles every 3 hours. What is the count after 6 hours?"},
    {"id": "mo05", "type": "programming",   "text": "An algorithm processes N items in O(N) time. How many operations for N=100?"},
    {"id": "mo06", "type": "engineering",   "text": "A beam supports 500 kg over 2 metres. Calculate force per metre."},
    {"id": "mo07", "type": "finance",       "text": "An investment of 1000 at 5 percent grows for 1 year. What is the final value?"},
]


def run(prime_instance, iterations=50):
    """
    Run each task type through PRIME and measure:
    - success rate per task type
    - latency per task type
    - whether module_plan is available in result metadata
    """
    results = []

    for task in TASK_TYPES:
        task_id = task["id"]
        task_type = task["type"]
        text = task["text"]

        latencies = []
        statuses = []
        has_module_plan = []

        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(text)
                elapsed = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed)
                statuses.append(str(r.status))

                # Check for MOSAIC module plan in result
                plan_present = False
                if hasattr(r, "module_plan") and r.module_plan:
                    plan_present = True
                elif hasattr(r, "metadata") and isinstance(r.metadata, dict):
                    plan_present = "module_plan" in r.metadata or "mosaic" in str(r.metadata).lower()
                has_module_plan.append(plan_present)

            except Exception:
                latencies.append((time.perf_counter() - t0) * 1000.0)
                statuses.append("EXCEPTION")
                has_module_plan.append(False)

        success_count = sum(1 for s in statuses if "SUCCESS" in s.upper())
        plan_count = sum(1 for v in has_module_plan if v)

        results.append({
            "task_id": task_id,
            "task_type": task_type,
            "iterations": iterations,
            "success_count": success_count,
            "success_rate": round(success_count / iterations, 4),
            "module_plan_observed_rate": round(plan_count / iterations, 4),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 4),
                "median": round(statistics.median(latencies), 4),
                "min": round(min(latencies), 4),
                "max": round(max(latencies), 4),
                "stdev": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
            },
        })

    types_tested = [r["task_type"] for r in results]
    all_success = [r["success_rate"] for r in results]
    all_latency = [r["latency_ms"]["mean"] for r in results]
    latency_range = max(all_latency) - min(all_latency)

    aggregate = {
        "total_tasks": len(results),
        "task_types_tested": types_tested,
        "mean_success_rate": round(statistics.mean(all_success), 4),
        "mean_latency_ms": round(statistics.mean(all_latency), 4),
        "latency_range_ms": round(latency_range, 4),
        "latency_variance_pct": round(latency_range / statistics.mean(all_latency) * 100.0, 2) if all_latency else 0.0,
        "note": (
            "Latency variance across task types may indicate adaptive module "
            "selection, but cannot be distinguished from general solve() variance "
            "without direct MOSAIC API exposure."
        ),
    }

    return {"tasks": results, "aggregate": aggregate}
