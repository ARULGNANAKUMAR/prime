"""
Category B — Memory Benchmark

Tests PRIME's memory store/retrieve/update capabilities.

Hypothesis H4: PRIME's memory mechanism can improve performance on repeated
or context-dependent tasks.
"""
import time
import statistics


def run(prime_instance, tasks, iterations=30):
    """
    For each task:
      1. Store content via prime.solve()
      2. Query for the stored content
      3. Compare WITH-memory vs WITHOUT-memory latency on retrieval query
    """
    results = []

    for task in tasks:
        task_id = task["id"]
        store_text = task["store_content"]
        query_token = task["query"]
        tag = task.get("tag", "general")

        store_query = f"Remember this information: {store_text}"
        retrieve_query = f"What is {query_token}? Give the value associated with {query_token}."

        # --- WITHOUT MEMORY: solve retrieve_query cold (nothing stored) ---
        cold_latencies = []
        cold_statuses = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(retrieve_query)
                cold_latencies.append((time.perf_counter() - t0) * 1000.0)
                cold_statuses.append(str(r.status))
            except Exception:
                cold_latencies.append((time.perf_counter() - t0) * 1000.0)
                cold_statuses.append("EXCEPTION")

        # --- Store the content ---
        store_latencies = []
        for _ in range(5):  # store a few times to see consistency
            t0 = time.perf_counter()
            try:
                prime_instance.solve(store_query)
                store_latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                store_latencies.append((time.perf_counter() - t0) * 1000.0)

        # --- WITH MEMORY: solve retrieve_query after storing ---
        warm_latencies = []
        warm_statuses = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(retrieve_query)
                warm_latencies.append((time.perf_counter() - t0) * 1000.0)
                warm_statuses.append(str(r.status))
            except Exception:
                warm_latencies.append((time.perf_counter() - t0) * 1000.0)
                warm_statuses.append("EXCEPTION")

        cold_success = sum(1 for s in cold_statuses if "SUCCESS" in s.upper())
        warm_success = sum(1 for s in warm_statuses if "SUCCESS" in s.upper())

        result = {
            "task_id": task_id,
            "tag": tag,
            "store_content_length": len(store_text),
            "iterations": iterations,
            "cold_success_rate": round(cold_success / iterations, 4),
            "warm_success_rate": round(warm_success / iterations, 4),
            "cold_latency_ms": {
                "mean": round(statistics.mean(cold_latencies), 4),
                "median": round(statistics.median(cold_latencies), 4),
                "min": round(min(cold_latencies), 4),
                "max": round(max(cold_latencies), 4),
            },
            "warm_latency_ms": {
                "mean": round(statistics.mean(warm_latencies), 4),
                "median": round(statistics.median(warm_latencies), 4),
                "min": round(min(warm_latencies), 4),
                "max": round(max(warm_latencies), 4),
            },
            "store_latency_ms": {
                "mean": round(statistics.mean(store_latencies), 4),
            },
            "notes": "memory_query_success_independent_of_stored_content_in_current_impl",
        }
        results.append(result)

    # Aggregate
    cold_rates = [r["cold_success_rate"] for r in results]
    warm_rates = [r["warm_success_rate"] for r in results]
    cold_lats = [r["cold_latency_ms"]["mean"] for r in results]
    warm_lats = [r["warm_latency_ms"]["mean"] for r in results]

    aggregate = {
        "total_tasks": len(results),
        "mean_cold_success_rate": round(statistics.mean(cold_rates), 4),
        "mean_warm_success_rate": round(statistics.mean(warm_rates), 4),
        "mean_cold_latency_ms": round(statistics.mean(cold_lats), 4),
        "mean_warm_latency_ms": round(statistics.mean(warm_lats), 4),
    }

    return {"tasks": results, "aggregate": aggregate}
