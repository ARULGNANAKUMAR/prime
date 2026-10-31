"""
PRIME Benchmark Runner

Usage:
    python -m benchmarks.runners.run_benchmarks [--suite SUITE]

Suites: reasoning, memory, atlas, echo, mosaic, continual_learning,
        optimization, ablation, dependency, offline, all (default)
"""
import sys
import os
import json
import time
import platform
import resource
import argparse
import datetime

# Ensure we import from the installed package, not the source tree
# (this file is run from outside site-packages)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from benchmarks.tasks import (
    reasoning, memory_bench, atlas_bench, echo_bench,
    mosaic_bench, continual_learning_bench, optimization_bench,
    ablation_bench, dependency_bench, offline_bench,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_tasks():
    ds_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "datasets", "benchmark_tasks.json"
    )
    with open(ds_path) as f:
        return json.load(f)


def env_info():
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "prime_version": "1.0.0",
        "random_seed": 42,
        "rss_at_start_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def save(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → saved {path}")
    return path


def run_suite(suite_name, tasks, prime_class, prime_instance, cfg):
    print(f"\n{'='*60}")
    print(f"  SUITE: {suite_name.upper()}")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    if suite_name == "reasoning":
        data = reasoning.run(prime_instance, tasks["reasoning_tasks"],
                             iterations=cfg["iterations"], warmup=cfg["warmup_iterations"])

    elif suite_name == "memory":
        data = memory_bench.run(prime_instance, tasks["memory_tasks"],
                                iterations=cfg["iterations"] // 3)

    elif suite_name == "atlas":
        data = atlas_bench.run(prime_instance, tasks["atlas_tasks"],
                               iterations=cfg["iterations"] // 2)

    elif suite_name == "echo":
        data = echo_bench.run(prime_instance, iterations=cfg["iterations"] // 2)

    elif suite_name == "mosaic":
        data = mosaic_bench.run(prime_instance, iterations=cfg["iterations"] // 2)

    elif suite_name == "continual_learning":
        data = continual_learning_bench.run(prime_instance,
                                            iterations=cfg["iterations"] // 3)

    elif suite_name == "optimization":
        data = optimization_bench.run(prime_instance,
                                      cold_n=5, warm_n=cfg["iterations"])

    elif suite_name == "ablation":
        data = ablation_bench.run(prime_class, iterations=cfg["iterations"] // 2)

    elif suite_name == "dependency":
        data = dependency_bench.run()

    elif suite_name == "offline":
        data = offline_bench.run(prime_instance, iterations=20)

    else:
        print(f"  Unknown suite: {suite_name}")
        return None

    elapsed = (time.perf_counter() - t0) * 1000.0
    data["_suite"] = suite_name
    data["_elapsed_ms"] = round(elapsed, 2)
    save(f"{suite_name}_results", data)
    print(f"  Suite completed in {elapsed:.1f} ms")
    return data


def main():
    parser = argparse.ArgumentParser(description="PRIME Benchmark Runner")
    parser.add_argument("--suite", default="all",
                        help="Suite name or 'all'")
    args = parser.parse_args()

    # Load config
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "benchmark_config.json"
    )
    with open(cfg_path) as f:
        cfg = json.load(f)

    print("\n" + "="*60)
    print("  PRIME BENCHMARK RUNNER v1.0.0")
    print("="*60)

    env = env_info()
    print(f"  Python  : {env['python_version']}")
    print(f"  Platform: {env['platform']}")
    print(f"  CPUs    : {env['cpu_count']}")
    print(f"  Time    : {env['timestamp']}")

    # Import PRIME from installed package
    try:
        import prime as prime_mod
        from prime import PRIME
        print(f"  PRIME   : {prime_mod.__version__} from {prime_mod.__file__}")
        assert "site-packages" in prime_mod.__file__, "PRIME imported from source tree!"
    except Exception as e:
        print(f"  ERROR: Could not import PRIME: {e}")
        sys.exit(1)

    tasks = load_tasks()
    prime_instance = PRIME()

    all_suites = cfg["suites"]
    if args.suite == "all":
        suites_to_run = all_suites
    else:
        suites_to_run = [args.suite]

    all_results = {"env": env, "suites": {}}
    failed = []

    for suite_name in suites_to_run:
        try:
            result = run_suite(suite_name, tasks, PRIME, prime_instance, cfg)
            if result:
                all_results["suites"][suite_name] = result
        except Exception as e:
            print(f"  ERROR in suite {suite_name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append({"suite": suite_name, "error": str(e)})

    # Save combined results
    all_results["failed_suites"] = failed
    all_results["env"]["rss_at_end_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save("all_results", all_results)

    print(f"\n{'='*60}")
    print(f"  Suites completed: {len(all_results['suites'])}/{len(suites_to_run)}")
    if failed:
        print(f"  FAILED: {[f['suite'] for f in failed]}")
        sys.exit(1)
    else:
        print("  All suites OK")

    return all_results


if __name__ == "__main__":
    main()
