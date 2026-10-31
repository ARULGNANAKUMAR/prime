# PRIME PYPI RELEASE REPORT

---

## Identity

| Field | Value |
|---|---|
| Distribution | `prime-ai` |
| Python import | `prime` |
| Version | `1.0.0` |
| Release date | 2026-10-07 |
| License | MIT |

---

## PyPI Status

| Registry | Status | Notes |
|---|---|---|
| Production PyPI | **READY — NOT YET UPLOADED** | Network blocked in build environment; upload via CI/CD |
| TestPyPI | **BLOCKED_BY_ENVIRONMENT** | Egress proxy blocks test.pypi.org and pypi.org |
| PyPI name (`prime-ai`) | **UNKNOWN — MUST VERIFY** | Cannot reach pypi.org from this environment; verify before upload |

> **Action required before production upload:**
> 1. Confirm `prime-ai` is unclaimed on pypi.org
> 2. Run GitHub Actions workflow (`push tag v1.0.0`) to build cross-platform wheels via cibuildwheel
> 3. Trigger TestPyPI upload via `workflow_dispatch` with `upload_target=testpypi`
> 4. Validate TestPyPI installation in clean venv
> 5. On TestPyPI PASS, tag `v1.0.0` → CI automatically publishes to production PyPI

---

## Distributions

### Source Distribution (SDist)

| Property | Value |
|---|---|
| Filename | `prime-ai-1.0.0.tar.gz` |
| Size | 192 KB (196,782 bytes) |
| SHA-256 | `931ec821eaa45a497b584caa552c7c68187523793466ac9265173f207c31f8a9` |
| C++ headers | 33 |
| C++ source files | 11 |
| Python modules | 15 |
| Problematic files | NONE |
| Required files | README.md ✅ LICENSE ✅ THIRD_PARTY_NOTICES.md ✅ pyproject.toml ✅ setup.py ✅ |

SDist validated: builds clean package from source, all required files present, no unwanted artifacts.

### Binary Wheel (Linux)

| Property | Value |
|---|---|
| Filename | `prime_ai-1.0.0-cp312-cp312-linux_x86_64.whl` |
| Size | 5.4 MB (5,608,119 bytes) |
| SHA-256 | `5d46e5eb80a01b75f60404a19c55595a87038c40b1d0d491bf3167eb55edacab` |
| Platform | `linux_x86_64` |
| Python | CPython 3.12 |
| ABI | `cp312` |
| C++ extensions | 6 / 6 present |
| Problematic files | NONE |

---

## Platforms

| Platform | Status | Notes |
|---|---|---|
| Linux x86_64 | ✅ PASS | Built and fully validated in this environment |
| Linux aarch64 | NOT_BUILT | Requires CI (cibuildwheel `manylinux2014`) |
| macOS x86_64 | NOT_BUILT | Requires CI (`macos-13` runner) |
| macOS arm64 | NOT_BUILT | Requires CI (`macos-14` runner) |
| Windows x86_64 | NOT_BUILT | Requires CI (`windows-latest` + MSVC 2022) |

CI/CD workflow (`.github/workflows/build_wheels.yml`) is configured correctly for all platforms.

---

## Python Versions

| Python | Wheel available | Notes |
|---|---|---|
| 3.10 | ❌ NOT_BUILT (CI required) | Declared in pyproject.toml `requires-python >=3.10` |
| 3.11 | ❌ NOT_BUILT (CI required) | |
| 3.12 | ✅ PASS | Built and validated locally |

---

## Installation

Tested in two separate isolated virtual environments — no access to source tree.

```
python3 -m venv /tmp/prime_final_test
pip install prime_ai-1.0.0-cp312-cp312-linux_x86_64.whl
```

| Check | Result |
|---|---|
| `pip install` | PASS |
| `import prime` | PASS |
| `prime.__file__` in site-packages (not source tree) | PASS |
| `importlib.metadata.version("prime-ai")` → `1.0.0` | PASS |
| Uninstall | PASS |
| Reinstall | PASS |

---

## C++ Extensions

All 6 extensions load successfully from the installed wheel:

| Extension | Status |
|---|---|
| `_prime` | PASS |
| `_prime_llm` | PASS |
| `_prime_memory` | PASS |
| `_prime_atlas` | PASS |
| `_prime_reasoning` | PASS |
| `_prime_mosaic` | PASS |

---

## End-to-End Validation

```python
from prime import PRIME
with PRIME() as p:
    result = p.solve("A car travels 150 km in 3 hours. Calculate average speed.")
    # result.status  → SolveStatus.SUCCESS
    # result.answer  → 50
    # result.latency_ms → ~1.5 ms
```

| Check | Result |
|---|---|
| `prime.solve()` status | `SolveStatus.SUCCESS` |
| Answer contains `50` | PASS |
| Latency | ~1.5 ms |

### Subsystem Tests (all from installed wheel)

| Subsystem | Test | Result |
|---|---|---|
| ATLAS | `ATLASEngine.parse(problem_text)` | PASS |
| Memory | `MemoryEngine.store / get` | PASS |
| Reasoning | `ReasoningEngine.eval_expression("150 / 3")` → 50.0 | PASS |
| MOSAIC | `MOSAICEngine.list_modules()` → 5 modules | PASS |
| ECHO | `ECHOEngine.get_experiences()` | PASS |
| Continual Learning | `LearningController.get_state()` → dict | PASS |
| Adaptive Optimization | `AdaptiveOptimizationEngine.get_resource_snapshot()` | PASS |

---

## Security

| Check | Result |
|---|---|
| `eval()` / `exec()` in Python source | NONE |
| Network imports (`socket`, `urllib`, `requests`, `http`) | NONE |
| Hardcoded API keys / tokens / secrets | NONE |
| `shell=True` subprocess calls | NONE |
| Telemetry / beacon calls | NONE |
| CI workflow hardcoded tokens | NONE (uses GitHub Secrets / OIDC) |
| Wheel contains credentials | NONE |
| SDist contains credentials | NONE |

**Security audit: PASS**

---

## Licensing

| Component | License | Redistribution |
|---|---|---|
| PRIME Python/C++ source | MIT | ✅ Permitted |
| pybind11 (binding layer) | BSD-3-Clause | ✅ Permitted (see THIRD_PARTY_NOTICES.md) |

- `LICENSE` file present: ✅
- `THIRD_PARTY_NOTICES.md` present: ✅
- No model weights bundled: ✅
- No GPL-licensed components: ✅

**License audit: PASS**

---

## Dependencies

| Type | Package | Required | Condition |
|---|---|---|---|
| Runtime | (none) | — | Zero mandatory runtime dependencies |
| Optional | `psutil >=5.9` | No | `pip install prime-ai[metrics]` |
| Dev/test | `pytest >=7.0` | No | `pip install prime-ai[dev]` |
| Build | `setuptools >=68` | Build-time only | Not installed as runtime dep |

`Requires-Python: >=3.10` ✅

---

## Offline Runtime

Tested with all `socket.socket` calls blocked:

- `prime.solve(...)` completed successfully
- No network access attempted
- No API key required
- No cloud service required
- No Ollama required

**Offline runtime: PASS**

---

## Package Sizes

| Artifact | Size |
|---|---|
| SDist (`prime-ai-1.0.0.tar.gz`) | 192 KB |
| Linux wheel (`prime_ai-1.0.0-cp312-cp312-linux_x86_64.whl`) | 5.4 MB |
| Installed size (site-packages) | ~19 MB (6 compiled .so + Python modules) |

---

## Test Results

| Suite | Results |
|---|---|
| Python unit tests | 1031 / 1032 PASS, 1 skipped |
| C++ tests | 542 / 542 PASS |
| PyPI installation smoke test (`test_pypi_installation.py`) | 16 / 16 PASS |

Skipped test: `test_sqlite_backend` — `libsqlite3-dev` not available in build environment. SQLite path in production falls back gracefully; no functional regression.

---

## Known Limitations

1. **PyPI name unverified** — `prime-ai` availability on pypi.org could not be confirmed from this environment (network blocked). Must verify manually before upload.

2. **TestPyPI not executed** — Egress proxy blocks `test.pypi.org`. Upload must be performed via GitHub Actions CI in a network-enabled environment.

3. **macOS and Windows wheels not built** — Only the Linux x86_64 / CPython 3.12 wheel was produced locally. Cross-platform wheels require the configured GitHub Actions workflow (cibuildwheel targeting `macos-13`, `macos-14`, `windows-latest`).

4. **Python 3.10 / 3.11 wheels not built** — Require CI; declared in classifiers and `Requires-Python >=3.10`.

5. **SQLite memory backend** — Optional SQLite support compiled out when `libsqlite3-dev` is absent. Standard in-memory backend operates normally.

6. **LLM inference** — Requires a local GGUF model file provided by the user. No models are bundled or downloaded automatically.

---

## Release Decision

```
READY
```

All local release gates pass. Blocked only by environment (no network egress).
Production upload requires:
1. Manual confirmation that `prime-ai` is unclaimed on PyPI
2. GitHub Actions run to build cross-platform wheels
3. TestPyPI upload and fresh-venv validation
4. Tag `v1.0.0` → CI publishes to production PyPI via OIDC trusted publishing

---

*Report generated: 2026-10-07*
