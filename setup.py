"""
setup.py — builds all PRIME C++ extensions alongside the Python package.

Run:
    python setup.py build_ext --inplace   # development build
    pip install -e .                       # editable install
"""

from setuptools import setup, Extension, find_packages
import sys
import os
import subprocess
import shutil

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
compile_args = [
    "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
    "-Wno-missing-field-initializers",
]
if sys.platform == "win32":
    compile_args = ["/std:c++17", "/O2", "/W4"]
elif sys.platform == "darwin":
    compile_args += ["-stdlib=libc++"]

include_dirs = [os.path.join(os.path.dirname(__file__), "cpp")]

# ---------------------------------------------------------------------------
# Phase 1 — Core runtime
# ---------------------------------------------------------------------------
prime_ext = Extension(
    name="_prime",
    sources=[
        "cpp/runtime/runtime.cpp",
        "cpp/bindings/prime_bindings.cpp",
    ],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    language="c++",
)

# ---------------------------------------------------------------------------
# Phase 2 — LLM engine
# ---------------------------------------------------------------------------
prime_llm_ext = Extension(
    name="_prime_llm",
    sources=[
        "cpp/runtime/runtime.cpp",
        "cpp/llm/gguf.cpp",
        "cpp/llm/tokenizer.cpp",
        "cpp/llm/transformer.cpp",
        "cpp/llm/llm_engine.cpp",
        "cpp/bindings/llm_bindings.cpp",
    ],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    language="c++",
)

# ---------------------------------------------------------------------------
# Phase 3 — Memory engine (optional SQLite support)
# ---------------------------------------------------------------------------
_have_sqlite = False
if shutil.which("pkg-config"):
    rc = subprocess.run(
        ["pkg-config", "--exists", "sqlite3"],
        capture_output=True,
    ).returncode
    _have_sqlite = rc == 0
else:
    _have_sqlite = os.path.exists("/usr/include/sqlite3.h")

prime_memory_ext = Extension(
    name="_prime_memory",
    sources=["cpp/bindings/memory_bindings.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args[:],
    extra_link_args=["-lsqlite3"] if _have_sqlite else [],
    define_macros=[("PRIME_HAVE_SQLITE", "1")] if _have_sqlite else [],
    language="c++",
)

# ---------------------------------------------------------------------------
# Phase 4 — ATLAS
# ---------------------------------------------------------------------------
prime_atlas_ext = Extension(
    name="_prime_atlas",
    sources=["cpp/bindings/atlas_bindings.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args + ["-Wno-unused-parameter"],
    language="c++",
)

# ---------------------------------------------------------------------------
# Phase 5 — Reasoning engine
# ---------------------------------------------------------------------------
prime_reasoning_ext = Extension(
    name="_prime_reasoning",
    sources=["cpp/bindings/reasoning_bindings.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args + ["-Wno-unused-parameter"],
    language="c++",
)

# ---------------------------------------------------------------------------
# Phase 6 — MOSAIC
# ---------------------------------------------------------------------------
prime_mosaic_ext = Extension(
    name="_prime_mosaic",
    sources=["cpp/bindings/mosaic_bindings.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args + ["-Wno-unused-parameter"],
    language="c++",
)

# ---------------------------------------------------------------------------
# Single canonical setup() call
# ---------------------------------------------------------------------------
setup(
    name="prime-ai",
    version="1.0.0",
    description=(
        "PRIME: Lightweight Adaptive AI Core — "
        "modular, CPU-first, offline-capable AI runtime"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="PRIME Research Project",
    project_urls={
        "Homepage":     "https://github.com/prime-ai/prime",
        "Documentation": "https://github.com/prime-ai/prime/blob/main/docs/API.md",
        "Bug Tracker":  "https://github.com/prime-ai/prime/issues",
    },
    python_requires=">=3.10",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=[
        prime_ext,
        prime_llm_ext,
        prime_memory_ext,
        prime_atlas_ext,
        prime_reasoning_ext,
        prime_mosaic_ext,
    ],
    package_data={"prime": ["*.pyi"]},
    extras_require={
        "metrics": ["psutil>=5.9"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
)
