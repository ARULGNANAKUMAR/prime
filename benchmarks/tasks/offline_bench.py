"""
Offline Operation Benchmark

Hypothesis H1 (partial): PRIME operates without mandatory network access.

Blocks socket.socket at the OS level during solve() calls and verifies
that PRIME continues to function correctly.
"""
import socket
import time
import statistics


TASKS = [
    "A car travels 150 km in 3 hours. Calculate average speed.",
    "Calculate 840 divided by 28.",
    "Revenue is 500 units at price 20. Calculate total revenue.",
    "A train travels 480 km in 4 hours. Calculate average speed.",
    "Calculate 1000 divided by 25.",
]


class _BlockedSocket:
    def __init__(self, *a, **kw):
        raise OSError("OFFLINE_TEST: network access blocked")


def run(prime_instance, iterations=20):
    original_socket = socket.socket
    results = []

    try:
        socket.socket = _BlockedSocket

        for text in TASKS:
            lats, statuses = [], []
            errors = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                try:
                    r = prime_instance.solve(text)
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    statuses.append(str(r.status))
                except Exception as e:
                    lats.append((time.perf_counter() - t0) * 1000.0)
                    err_str = str(e)
                    if "OFFLINE_TEST" in err_str:
                        statuses.append("NETWORK_ATTEMPTED")
                        errors.append("NETWORK_CALL_DETECTED")
                    else:
                        statuses.append("OTHER_EXCEPTION")
                        errors.append(err_str[:80])

            success = sum(1 for s in statuses if "SUCCESS" in s.upper())
            network_attempts = sum(1 for s in statuses if s == "NETWORK_ATTEMPTED")

            results.append({
                "task": text[:60],
                "iterations": iterations,
                "success_count": success,
                "success_rate": round(success / iterations, 4),
                "network_attempts": network_attempts,
                "latency_ms": {
                    "mean": round(statistics.mean(lats), 4),
                    "min": round(min(lats), 4),
                    "max": round(max(lats), 4),
                },
            })

    finally:
        socket.socket = original_socket

    all_success = [r["success_rate"] for r in results]
    total_network = sum(r["network_attempts"] for r in results)

    aggregate = {
        "tasks_tested": len(results),
        "mean_success_rate": round(statistics.mean(all_success), 4),
        "total_network_attempts_blocked": total_network,
        "offline_capable": total_network == 0 and statistics.mean(all_success) > 0.9,
    }

    return {"tasks": results, "aggregate": aggregate}
