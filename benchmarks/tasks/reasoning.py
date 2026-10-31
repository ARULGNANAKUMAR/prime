"""
Category A — Basic Reasoning Benchmark

Tests PRIME's ability to solve arithmetic, physics, and multi-step reasoning
tasks without external API access.

Hypothesis H1: PRIME can perform basic reasoning tasks without mandatory
external API access.
"""
import sys
import json
import time
import statistics


def run(prime_instance, tasks, iterations=100, warmup=5):
    """
    Run reasoning benchmark.

    Returns a dict with raw per-task results and aggregate statistics.
    """
    results = []

    for task in tasks:
        task_id = task["id"]
        text = task["text"]
        expected = task["expected_answer"]

        # Warmup (not recorded)
        for _ in range(warmup):
            try:
                prime_instance.solve(text)
            except Exception:
                pass

        # Timed runs
        latencies_ms = []
        statuses = []
        answers = []
        failures = 0

        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(text)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                statuses.append(str(r.status))
                answers.append(str(r.answer) if hasattr(r, "answer") else "")
            except Exception as e:
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                statuses.append("EXCEPTION")
                answers.append("")
                failures += 1

        success_count = sum(1 for s in statuses if "SUCCESS" in s.upper())
        answer_match = sum(1 for a in answers if expected in a)

        result = {
            "task_id": task_id,
            "text": text,
            "expected_answer": expected,
            "category": task.get("category", "unknown"),
            "difficulty": task.get("difficulty", "unknown"),
            "iterations": iterations,
            "success_count": success_count,
            "success_rate": round(success_count / iterations, 4),
            "answer_match_count": answer_match,
            "answer_match_rate": round(answer_match / iterations, 4),
            "failure_count": failures,
            "latency_ms": {
                "n": len(latencies_ms),
                "mean": round(statistics.mean(latencies_ms), 4),
                "median": round(statistics.median(latencies_ms), 4),
                "min": round(min(latencies_ms), 4),
                "max": round(max(latencies_ms), 4),
                "stdev": round(statistics.stdev(latencies_ms), 4) if len(latencies_ms) > 1 else 0.0,
            },
        }
        results.append(result)

    # Aggregate
    all_success_rates = [r["success_rate"] for r in results]
    all_mean_latencies = [r["latency_ms"]["mean"] for r in results]
    all_answer_rates = [r["answer_match_rate"] for r in results]

    aggregate = {
        "total_tasks": len(results),
        "mean_success_rate": round(statistics.mean(all_success_rates), 4),
        "mean_answer_match_rate": round(statistics.mean(all_answer_rates), 4),
        "mean_latency_ms": round(statistics.mean(all_mean_latencies), 4),
        "median_latency_ms": round(statistics.median(all_mean_latencies), 4),
    }

    return {"tasks": results, "aggregate": aggregate}
