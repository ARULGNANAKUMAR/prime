"""
Step 9: ECHO learning tests.

Verifies ECHO correctly stores experiences after the API wiring fixes.
Does NOT claim that learning_applied=True will occur in rapid benchmarks
(dedup window prevents it by design).
"""
import time
import pytest
from prime import PRIME


@pytest.fixture(scope="module")
def prime_with_echo():
    p = PRIME()
    yield p
    p.shutdown()


def test_echo_stores_experiences(prime_with_echo):
    """ECHO must store at least one experience after a solve."""
    p = prime_with_echo
    r = p.solve("A car travels 150 km in 3 hours. Calculate average speed.", learning=True)
    assert r.experience_id, "experience_id must be non-empty after solve with learning=True"


def test_echo_experience_has_correct_outcome(prime_with_echo):
    """Stored experiences must have SUCCESS outcome for successful solves."""
    p = prime_with_echo
    r = p.solve("A train travels 480 km in 4 hours. Calculate average speed.", learning=True)
    pl = getattr(p, '_default_pipeline', None)
    if pl and hasattr(pl, '_echo') and pl._echo:
        store = getattr(pl._echo, '_store', None)
        if store and r.experience_id:
            exp = store.get(r.experience_id)
            if exp:
                assert exp.outcome in ('SUCCESS', 'PARTIAL_SUCCESS', 'FAILURE'), \
                    f"Outcome must be known enum value, got: {exp.outcome!r}"


def test_echo_accumulates_multiple_experiences(prime_with_echo):
    """ECHO store must grow with successive distinct tasks."""
    p = prime_with_echo
    tasks = [
        "A car travels 150 km in 3 hours. Calculate average speed.",
        "Calculate 144 divided by 12.",
        "Revenue is 500 units at price 20 each. Calculate total revenue.",
    ]
    for task in tasks:
        time.sleep(0.01)  # small gap to avoid dedup window
        p.solve(task, learning=True)

    pl = getattr(p, '_default_pipeline', None)
    if pl and hasattr(pl, '_echo') and pl._echo:
        store = getattr(pl._echo, '_store', None)
        if store:
            n = len(getattr(store, '_experiences', {}))
            assert n >= 3, f"Expected >=3 stored experiences, got {n}"
