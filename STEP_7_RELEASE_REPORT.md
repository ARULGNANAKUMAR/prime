# PRIME Step 7 Release Report

**Date:** 2026-10-08  
**Package:** `prime-ai`  
**Version:** `1.0.0`

---

## Release Identity

| Field | Value |
|---|---|
| Package | `prime-ai` |
| Version | `1.0.0` |
| Python requirement | `>=3.10` |
| License | MIT |
| Mandatory runtime dependencies | 0 |

---

## Git

| Field | Value |
|---|---|
| Repository | Local — initialized this step |
| Branch | `main` |
| Commit | `096c426` (3 commits total) |
| Tag | `v1.0.0` (annotated) |
| Working tree | Clean |
| Remote | Not configured — requires user action |
| Push | **NOT_EXECUTED — github.com blocked by egress proxy** |

### Commands ready for execution (requires network):

```bash
# From the PRIME directory on a machine with GitHub access:
git remote add origin https://github.com/<your-org>/prime.git
git push -u origin main
git push origin v1.0.0
```

The tag push triggers the GitHub Actions workflow automatically.

---

## Release Preparation Changes (This Step)

| Change | Reason |
|---|---|
| README: added `pip install prime-ai` section | Missing — critical for PyPI discoverability |
| README: removed NumPy from requirements table | NumPy is not in pyproject.toml; inconsistency removed |
| README: fixed source-install instructions | Simplified to `pip install -e .` |
| `pyproject.toml`: classifier `Beta` → `Production/Stable` | 1.0.0 is a stable release |
| Git repository initialized on `main` branch | Required for tagging and CI trigger |
| `v1.0.0` annotated tag created | Required for GitHub Actions release workflow |
| `.gitignore` created | Excludes dist/, build/, venv/, secrets, IDE files |
| 148 source files committed | Complete source tree tracked |

---

## Build Matrix

| Platform | Python | Wheel | Build | Install | Smoke | Notes |
|---|---|---|---|---|---|---|
| Linux x86_64 | 3.12 | `prime_ai-1.0.0-cp312-cp312-linux_x86_64.whl` | PASS | PASS | **16/16 PASS** | Validated locally |
| Linux x86_64 | 3.11 | — | NOT_BUILT | — | — | Requires CI |
| Linux x86_64 | 3.10 | — | NOT_BUILT | — | — | Requires CI |
| Linux aarch64 | 3.12 | — | NOT_BUILT | — | — | Requires CI (QEMU) |
| Linux aarch64 | 3.11 | — | NOT_BUILT | — | — | Requires CI (QEMU) |
| Linux aarch64 | 3.10 | — | NOT_BUILT | — | — | Requires CI (QEMU) |
| macOS x86_64 | 3.12 | — | NOT_BUILT | — | — | Requires CI (macos-13) |
| macOS x86_64 | 3.11 | — | NOT_BUILT | — | — | Requires CI (macos-13) |
| macOS x86_64 | 3.10 | — | NOT_BUILT | — | — | Requires CI (macos-13) |
| macOS arm64 | 3.12 | — | NOT_BUILT | — | — | Requires CI (macos-14) |
| macOS arm64 | 3.11 | — | NOT_BUILT | — | — | Requires CI (macos-14) |
| macOS arm64 | 3.10 | — | NOT_BUILT | — | — | Requires CI (macos-14) |
| Windows x86_64 | 3.12 | — | NOT_BUILT | — | — | Requires CI (MSVC 2022) |
| Windows x86_64 | 3.11 | — | NOT_BUILT | — | — | Requires CI (MSVC 2022) |
| Windows x86_64 | 3.10 | — | NOT_BUILT | — | — | Requires CI (MSVC 2022) |

**Actual wheels built: 1/15**

---

## Release Artifacts

| Artifact | Size | SHA-256 |
|---|---|---|
| `prime-ai-1.0.0.tar.gz` | 192 KB | `931ec821eaa45a497b584caa552c7c68187523793466ac9265173f207c31f8a9` |
| `prime_ai-1.0.0-cp312-cp312-linux_x86_64.whl` | 5.35 MB | `5d46e5eb80a01b75f60404a19c55595a87038c40b1d0d491bf3167eb55edacab` |

---

## Local Release Gate Verification

| Check | Result |
|---|---|
| Import (`prime.__file__` → site-packages) | **PASS** |
| Version (`importlib.metadata.version`) | **PASS** — `1.0.0` |
| C++ extensions | **6/6 PASS** |
| `prime.solve()` | **PASS** — `SUCCESS`, answer=`'50'`, latency=1.78 ms |
| Offline (socket blocked) | **PASS** — 0 network calls |
| Zero mandatory dependencies | **PASS** |
| Source-tree independence | **PASS** — no PYTHONPATH override needed |
| Smoke test | **16/16 PASS** |
| Artifact security | **PASS** — no credentials, secrets, or build caches |
| Workflow security | **12/12 PASS** — OIDC, pinned SHAs, TestPyPI gate |

---

## TestPyPI

| Field | Status |
|---|---|
| Upload | **NOT_EXECUTED — test.pypi.org blocked by egress proxy** |
| Fresh installation | **NOT_EXECUTED** |
| Smoke test | **NOT_EXECUTED** |

---

## Production PyPI

| Field | Status |
|---|---|
| Name availability | **UNKNOWN** — pypi.org blocked |
| Published | **NOT_RELEASED** |
| Public installation | **NOT_EXECUTED** |
| Smoke test | **NOT_EXECUTED** |

---

## Environmental Blockers

All external operations are blocked by the sandbox egress proxy:

| Host | Status |
|---|---|
| `github.com` | BLOCKED |
| `test.pypi.org` | BLOCKED |
| `pypi.org` | BLOCKED |

These are environment constraints, not defects in the package or workflow.

---

## Required Manual Steps to Complete Release

Execute these from a machine with internet access, in order:

**Step 1 — Verify PyPI name availability**
```
Open: https://pypi.org/project/prime-ai/
If claimed by another project: STOP and re-evaluate package name.
```

**Step 2 — Create GitHub repository**
```bash
# On GitHub.com: New repository → "prime" (or similar)
# Then:
git remote add origin https://github.com/<org>/prime.git
git push -u origin main
```

**Step 3 — Configure OIDC Trusted Publishing**
```
On GitHub: Settings → Environments → Create "testpypi"
  PyPI OIDC: publisher = test.pypi.org, project = prime-ai
On GitHub: Settings → Environments → Create "pypi"
  PyPI OIDC: publisher = pypi.org, project = prime-ai
```

**Step 4 — Push the release tag (triggers CI automatically)**
```bash
git push origin v1.0.0
```

**Step 5 — Monitor GitHub Actions**
```
Expected: 15 wheel build jobs + build_sdist + publish_testpypi
         + testpypi_smoke (GATE) + publish_pypi
```

**Step 6 — Post-release verification**
```bash
python3 -m venv /tmp/prime-verify
/tmp/prime-verify/bin/pip install prime-ai
/tmp/prime-verify/bin/python -c "
    import prime, importlib.metadata
    print(importlib.metadata.version('prime-ai'))   # expect: 1.0.0
    from prime import PRIME
    r = PRIME().solve('A car travels 150 km in 3 hours. Calculate average speed.')
    print(r.answer)   # expect: 50
"
/tmp/prime-verify/bin/python test_pypi_installation.py  # expect: 16/16 PASS
```

---

## Final Status

**RELEASE_READY**

All locally executable release gates pass. The package is correctly packaged,
versioned, tested, and committed. The git tag `v1.0.0` is created. The GitHub
Actions workflow is security-hardened with a mandatory TestPyPI gate before
production publishing.

Publication is blocked only by the sandbox environment (no network egress).
The code, packaging, workflow, and git state are production-ready.

**PRIME 1.0.0 IS NOT YET PUBLICLY RELEASED.**  
It will be released when Steps 1–6 above are executed on a machine with
network access.

---

*Report generated: 2026-10-08*
