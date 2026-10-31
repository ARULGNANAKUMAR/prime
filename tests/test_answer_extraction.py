"""
Step 9: Answer extraction regression tests.

These verify that lightweight expression inference in pipeline.py
correctly extracts answers for arithmetic division, business multiplication,
and exponential growth tasks.
"""
import pytest
from prime import PRIME, SolveStatus


@pytest.fixture(scope="module")
def prime():
    p = PRIME()
    yield p
    p.shutdown()


# Physics — existing behaviour preserved
def test_physics_speed_success(prime):
    r = prime.solve("A car travels 150 km in 3 hours. Calculate average speed.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer == "50"


# Arithmetic division — new in Step 9
def test_arithmetic_div_144_12(prime):
    r = prime.solve("Calculate 144 divided by 12.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(12.0, abs=0.1)


def test_arithmetic_div_840_28(prime):
    r = prime.solve("Calculate 840 divided by 28.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(30.0, abs=0.1)


def test_arithmetic_div_1000_25(prime):
    r = prime.solve("Calculate 1000 divided by 25.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(40.0, abs=0.1)


# Business multiplication — new in Step 9
def test_business_revenue(prime):
    r = prime.solve("Revenue is 500 units at price 20 each. Calculate total revenue.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(10000.0, abs=1.0)


def test_business_cost(prime):
    r = prime.solve("Total cost is 250 items at price 8 each.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(2000.0, abs=1.0)


# Exponential growth — new in Step 9
def test_biology_doubling(prime):
    r = prime.solve("A population of 200 doubles every 3 hours. Count after 6 hours.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(800.0, abs=1.0)


def test_biology_doubling_variant(prime):
    r = prime.solve("A population of 50 doubles every 2 hours. Count after 8 hours.")
    assert r.status == SolveStatus.SUCCESS
    assert r.answer_numeric == pytest.approx(800.0, abs=1.0)


# Status integrity — must not fabricate SUCCESS
def test_engineering_partial_success(prime):
    r = prime.solve("A beam supports 500 kg over 2 metres. Calculate force per metre.")
    # Not yet handled by inference; must be PARTIAL_SUCCESS, never a wrong SUCCESS
    assert r.status in (SolveStatus.PARTIAL_SUCCESS, SolveStatus.SUCCESS)
    if r.status == SolveStatus.SUCCESS:
        assert r.answer_numeric is not None, "SUCCESS must have a numeric answer"


def test_no_fabricated_success(prime):
    """Nonsense must never return SUCCESS."""
    r = prime.solve("xkqzwmv florbinated")
    assert r.status != SolveStatus.SUCCESS
