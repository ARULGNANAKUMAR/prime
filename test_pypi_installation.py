"""
test_pypi_installation.py
=========================
Lightweight post-PyPI-installation smoke test for PRIME 1.0.0.

Run this in a *clean virtual environment* after installing from PyPI:

    pip install prime-ai
    python test_pypi_installation.py

All tests must pass for the release to be considered valid.
No source tree is required or expected.
"""

# This is a standalone post-install smoke-test script, not a pytest module.
# Run it explicitly with: python test_pypi_installation.py
__test__ = False

import importlib
import importlib.metadata
import socket
import sys
import time

PASS = 0
FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as exc:
        print(f"  [FAIL] {name}: {exc}")
        FAIL += 1


# ──────────────────────────────────────────────────────────
# 1. Import
# ──────────────────────────────────────────────────────────
print("\n[1] Import")

def test_import():
    import prime
    assert prime is not None

check("prime package imports", test_import)


# ──────────────────────────────────────────────────────────
# 2. Version
# ──────────────────────────────────────────────────────────
print("\n[2] Version")

def test_version():
    import prime
    assert prime.__version__ == "1.0.0", f"Got {prime.__version__}"

def test_metadata_version():
    v = importlib.metadata.version("prime-ai")
    assert v == "1.0.0", f"Metadata version: {v}"

check("prime.__version__ == 1.0.0", test_version)
check("importlib.metadata version == 1.0.0", test_metadata_version)


# ──────────────────────────────────────────────────────────
# 3. C++ extensions load
# ──────────────────────────────────────────────────────────
print("\n[3] C++ extensions")

for ext in ["_prime", "_prime_llm", "_prime_memory", "_prime_atlas",
            "_prime_reasoning", "_prime_mosaic"]:
    def _make_test(name):
        def _t():
            importlib.import_module(name)
        return _t
    check(f"{ext} loads", _make_test(ext))


# ──────────────────────────────────────────────────────────
# 4. PRIME initialises
# ──────────────────────────────────────────────────────────
print("\n[4] PRIME core")

def test_prime_init():
    from prime import PRIME
    with PRIME() as p:
        assert p is not None

check("PRIME() context manager", test_prime_init)


# ──────────────────────────────────────────────────────────
# 5. ATLAS
# ──────────────────────────────────────────────────────────
print("\n[5] ATLAS")

def test_atlas():
    from prime import ATLASEngine
    a = ATLASEngine()
    prob = a.parse("A car travels 150 km in 3 hours.")
    assert prob is not None

check("ATLASEngine.parse()", test_atlas)


# ──────────────────────────────────────────────────────────
# 6. Memory
# ──────────────────────────────────────────────────────────
print("\n[6] Memory")

def test_memory():
    from prime import MemoryEngine
    mem = MemoryEngine()
    mid = mem.store("speed = 50 km/h", memory_type="long_term")
    item = mem.get(mid)
    assert item is not None

check("MemoryEngine store/get", test_memory)


# ──────────────────────────────────────────────────────────
# 7. Reasoning
# ──────────────────────────────────────────────────────────
print("\n[7] Reasoning")

def test_reasoning():
    from prime import ReasoningEngine
    r = ReasoningEngine()
    val = r.eval_expression("150 / 3")
    assert val == 50.0, f"Expected 50.0, got {val}"

check("ReasoningEngine.eval_expression(150/3) == 50.0", test_reasoning)


# ──────────────────────────────────────────────────────────
# 8. MOSAIC
# ──────────────────────────────────────────────────────────
print("\n[8] MOSAIC")

def test_mosaic():
    from prime import MOSAICEngine
    m = MOSAICEngine()
    mods = m.list_modules()
    assert len(mods) > 0, f"Expected >0 modules, got {len(mods)}"

check("MOSAICEngine.list_modules()", test_mosaic)


# ──────────────────────────────────────────────────────────
# 9. prime.solve() — correctness
# ──────────────────────────────────────────────────────────
print("\n[9] prime.solve()")

def test_solve():
    from prime import PRIME
    with PRIME() as p:
        result = p.solve(
            "A car travels 150 km in 3 hours. Calculate average speed."
        )
        assert result.status.name == "SUCCESS", f"Status: {result.status}"
        assert "50" in str(result.answer), f"Expected 50 in answer: {result.answer}"

check("prime.solve() returns SUCCESS with answer=50", test_solve)


# ──────────────────────────────────────────────────────────
# 10. No unexpected network access (offline runtime)
# ──────────────────────────────────────────────────────────
print("\n[10] Offline runtime")

def test_offline():
    _orig = socket.socket

    class _Blocked:
        def __init__(self, *a, **kw):
            raise OSError("Network access blocked for offline test")

    socket.socket = _Blocked
    try:
        from prime import PRIME
        with PRIME() as p:
            result = p.solve(
                "A car travels 150 km in 3 hours. Calculate average speed."
            )
            assert result.status.name == "SUCCESS"
            assert "50" in str(result.answer)
    finally:
        socket.socket = _orig

check("prime.solve() succeeds with all network sockets blocked", test_offline)


# ──────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'='*56}")
print(f"  prime-ai 1.0.0 — PyPI installation smoke test")
print(f"  Results: {PASS}/{total} PASS  {FAIL} FAIL")
print(f"{'='*56}")

if FAIL > 0:
    print("  STATUS: FAIL — do NOT release")
    sys.exit(1)
else:
    print("  STATUS: PASS — installation verified")
    sys.exit(0)
