"""
phase1_demo.py — PRIME Phase 1 integration demo.

Proves the full Python → C++ → Python communication path.
Does not call any LLM or external service.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from prime import PRIME, RuntimeConfig


def main():
    print("=== PRIME Phase 1 Integration Demo ===\n")

    # Default initialization
    brain = PRIME()
    print(f"Status: {brain.status()}")
    print(f"Running: {brain.is_running()}")
    brain.shutdown()
    print(f"Running after shutdown: {brain.is_running()}")

    print()

    # Context manager usage
    print("Context manager:")
    with PRIME(RuntimeConfig(log_level="warn")) as brain:
        print(f"  Status: {brain.status()}")
        print(f"  Config backend: {brain.config.compute_backend}")
    print(f"  Running after context exit: {brain.is_running()}")

    print()

    # Repr
    brain = PRIME(RuntimeConfig(log_level="silent"))
    print(f"Repr (running): {brain!r}")
    brain.shutdown()
    print(f"Repr (stopped): {brain!r}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
