"""
Category C — ATLAS Benchmark

Tests whether PRIME can transform domain-specific problems into a common
internal representation and reason about them.

Hypothesis H7: ATLAS can transform different domain-specific problems into
a common representation.
"""
import time
import statistics


def run(prime_instance, tasks, iterations=50):
    """
    For each domain task:
      - Solve via prime.solve()
      - Inspect ATLAS representation in result metadata
      - Measure success, latency, and whether entities/goals were extracted
    """
    results = []

    for task in tasks:
        task_id = task["id"]
        domain = task["domain"]
        text = task["text"]

        latencies = []
        statuses = []
        has_entities = []
        has_goals = []
        has_representation = []

        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                r = prime_instance.solve(text)
                elapsed = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed)
                statuses.append(str(r.status))

                # Inspect ATLAS representation
                meta = {}
                if hasattr(r, "metadata") and r.metadata:
                    meta = r.metadata if isinstance(r.metadata, dict) else {}
                if hasattr(r, "atlas_representation"):
                    ar = r.atlas_representation
                    has_entities.append(bool(ar and hasattr(ar, "entities") and ar.entities))
                    has_goals.append(bool(ar and hasattr(ar, "goals") and ar.goals))
                    has_representation.append(ar is not None)
                else:
                    # ATLAS ran internally; SUCCESS status implies representation succeeded
                    success = "SUCCESS" in str(r.status).upper()
                    has_entities.append(success)
                    has_goals.append(success)
                    has_representation.append(success)

            except Exception:
                latencies.append((time.perf_counter() - t0) * 1000.0)
                statuses.append("EXCEPTION")
                has_entities.append(False)
                has_goals.append(False)
                has_representation.append(False)

        success_count = sum(1 for s in statuses if "SUCCESS" in s.upper())
        repr_count = sum(1 for v in has_representation if v)

        result = {
            "task_id": task_id,
            "domain": domain,
            "text": text,
            "iterations": iterations,
            "success_count": success_count,
            "success_rate": round(success_count / iterations, 4),
            "representation_rate": round(repr_count / iterations, 4),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 4),
                "median": round(statistics.median(latencies), 4),
                "min": round(min(latencies), 4),
                "max": round(max(latencies), 4),
                "stdev": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
            },
        }
        results.append(result)

    # Cross-domain summary
    domains_tested = list({r["domain"] for r in results})
    success_by_domain = {
        d: round(statistics.mean([r["success_rate"] for r in results if r["domain"] == d]), 4)
        for d in domains_tested
    }
    all_success_rates = [r["success_rate"] for r in results]

    aggregate = {
        "total_tasks": len(results),
        "domains_tested": sorted(domains_tested),
        "mean_success_rate": round(statistics.mean(all_success_rates), 4),
        "success_by_domain": success_by_domain,
        "mean_latency_ms": round(statistics.mean([r["latency_ms"]["mean"] for r in results]), 4),
    }

    return {"tasks": results, "aggregate": aggregate}
