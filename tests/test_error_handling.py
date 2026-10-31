"""
Regression tests for error handling — Step 8 stabilization.

Bug fixed: solve(int) raised AttributeError because int has no .strip()
Fix: pipeline.py coerces non-str input to str before .strip() check
"""
import pytest
from prime import PRIME, SolveStatus


@pytest.fixture(scope="module")
def prime():
    p = PRIME()
    yield p
    p.shutdown()


def test_solve_empty_string(prime):
    r = prime.solve("")
    assert r.status == SolveStatus.INVALID_INPUT
    assert not r.succeeded


def test_solve_whitespace(prime):
    r = prime.solve("   ")
    assert r.status == SolveStatus.INVALID_INPUT
    assert not r.succeeded


def test_solve_integer_no_crash(prime):
    """Regression: solve(int) must not raise AttributeError (Step 8 bug fix)."""
    r = prime.solve(123)
    # 123 coerces to "123" — a valid string, no crash
    assert r is not None
    assert hasattr(r, 'status')


def test_solve_none_no_crash(prime):
    """None input must return INVALID_INPUT, not raise."""
    r = prime.solve(None)
    assert r.status == SolveStatus.INVALID_INPUT
    assert not r.succeeded


def test_solve_list_no_crash(prime):
    """List input must not crash."""
    r = prime.solve([])
    assert r is not None
    assert hasattr(r, 'status')


def test_solve_dict_no_crash(prime):
    """Dict input must not crash."""
    r = prime.solve({})
    assert r is not None
    assert hasattr(r, 'status')


def test_solve_nonsense_no_success(prime):
    """Nonsense input must not report SUCCESS."""
    r = prime.solve("xkqzwmv florbinated")
    assert r.status != SolveStatus.SUCCESS


def test_solve_normal_still_works(prime):
    """Ensure fix does not break normal operation."""
    r = prime.solve("A car travels 150 km in 3 hours. Calculate average speed.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer == "50"
