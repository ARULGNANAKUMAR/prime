# PRIME — Third-Party Notices

This document lists all third-party components used by PRIME and their
respective licenses.

---

## Build Dependencies

### setuptools
- **Version:** ≥ 68
- **License:** MIT
- **Usage:** Build backend for Python packaging and C++ extension compilation
- **URL:** https://github.com/pypa/setuptools
- **Notice:** No notice requirement beyond license text

---

## Optional Runtime Dependencies

### psutil *(optional)*
- **Version:** ≥ 5.9 (when installed via `pip install prime-ai[metrics]`)
- **License:** BSD 3-Clause
- **Usage:** Enhanced CPU/memory metrics in the Phase 9 Adaptive Optimizer.
  PRIME degrades gracefully to stdlib alternatives when psutil is absent.
- **URL:** https://github.com/giampaolo/psutil
- **Notice required:** Yes — BSD 3-Clause requires preservation of copyright
  notice in distributions. Full text:

  ```
  Copyright (c) 2009, Jay Loden, Dave Daley, Giampaolo Rodola'.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
  ```

---

## Optional SQLite (system library, not bundled)

### SQLite *(optional, system-provided)*
- **Version:** ≥ 3.x (detected at build time via `pkg-config` or header check)
- **License:** Public Domain
- **Usage:** Optional persistence backend for Phase 3 Memory (long-term store)
  and Phase 7 ECHO experience store. Falls back to in-memory storage when
  the system SQLite development headers (`libsqlite3-dev`) are absent.
- **URL:** https://www.sqlite.org
- **Notice:** SQLite is released into the public domain; no notice requirement.

---

## C++ Standard Library

PRIME's C++ core uses only the C++17 standard library (no Boost, no third-party
headers). The standard library is provided by the platform toolchain (GCC libstdc++,
LLVM libc++, or MSVC STL) and is subject to its own license, typically GPL with
runtime exception (GCC), LLVM Apache 2.0 (Clang), or proprietary (MSVC).

---

## Python Standard Library

PRIME's Python layer uses only the Python standard library (json, uuid, time,
math, sqlite3, pathlib, collections, enum, dataclasses, typing, abc, platform,
os, sys, statistics). No license notice beyond Python's own PSF license is required.

---

## Model Files (not bundled)

PRIME does **not** bundle any LLM model weights. Model files used with
`PRIME.load_model()` must be supplied by the user. Each model has its own
separate license; the user is responsible for ensuring compliance with that
license.

GGUF format: developed by the llama.cpp project (MIT License).
URL: https://github.com/ggerganov/llama.cpp

---

## No Telemetry

PRIME does not include any telemetry, analytics, or remote-reporting libraries.
It makes no network connections at runtime unless explicitly configured by the user.

---

*Last updated: 2025*
