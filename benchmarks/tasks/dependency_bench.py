"""
Dependency and footprint analysis.

Hypothesis H2: PRIME has lower mandatory dependency overhead than heavyweight
AI frameworks.

Measures:
- Mandatory runtime dependencies
- Wheel size
- Installed size (site-packages)
- Import time
- Startup time to first solve()
- Third-party packages imported during prime usage
"""
import os
import sys
import time
import importlib
import importlib.metadata
import importlib.util
import statistics


def run():
    results = {}

    # 1. Declared runtime dependencies
    try:
        meta = importlib.metadata.metadata("prime-ai")
        requires = meta.get_all("Requires-Dist") or []
        mandatory_deps = [r for r in requires if "extra ==" not in r]
        optional_deps = [r for r in requires if "extra ==" in r]
    except Exception as e:
        requires = []
        mandatory_deps = []
        optional_deps = []
        results["dep_error"] = str(e)

    results["declared_mandatory_deps"] = mandatory_deps
    results["declared_optional_deps"] = optional_deps
    results["mandatory_dep_count"] = len(mandatory_deps)

    # 2. Wheel and installed size
    wheel_path = None
    wheel_size = None
    for path_dir in ["/home/claude/PRIME/dist"]:
        for f in os.listdir(path_dir) if os.path.isdir(path_dir) else []:
            if f.endswith(".whl") and "prime_ai" in f:
                wheel_path = os.path.join(path_dir, f)
                wheel_size = os.path.getsize(wheel_path)
                break

    results["wheel_path"] = wheel_path
    results["wheel_size_bytes"] = wheel_size
    results["wheel_size_mb"] = round(wheel_size / 1024 / 1024, 3) if wheel_size else None

    # Installed size via site-packages
    try:
        prime_file = importlib.util.find_spec("prime").origin
        prime_dir = os.path.dirname(prime_file)
        installed_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(prime_dir)
            for f in files
        )
        results["installed_prime_dir_bytes"] = installed_size
        results["installed_prime_dir_mb"] = round(installed_size / 1024 / 1024, 3)
    except Exception as e:
        results["installed_size_error"] = str(e)

    # 3. Import time
    import_times = []
    for _ in range(10):
        # Force reimport
        if "prime" in sys.modules:
            mods_to_remove = [k for k in sys.modules if k == "prime" or k.startswith("prime.")]
            for m in mods_to_remove:
                del sys.modules[m]
        t0 = time.perf_counter()
        import prime  # noqa: F401
        import_times.append((time.perf_counter() - t0) * 1000.0)

    results["import_time_ms"] = {
        "mean": round(statistics.mean(import_times), 4),
        "min": round(min(import_times), 4),
        "max": round(max(import_times), 4),
        "n": len(import_times),
    }

    # 4. Startup time to first solve()
    startup_times = []
    for _ in range(5):
        mods_to_remove = [k for k in sys.modules if k == "prime" or k.startswith("prime.")]
        for m in mods_to_remove:
            del sys.modules[m]
        t0 = time.perf_counter()
        import prime as p_mod
        instance = p_mod.PRIME()
        instance.solve("Calculate 6 divided by 2.")
        startup_times.append((time.perf_counter() - t0) * 1000.0)

    results["startup_to_first_solve_ms"] = {
        "mean": round(statistics.mean(startup_times), 4),
        "min": round(min(startup_times), 4),
        "max": round(max(startup_times), 4),
        "n": len(startup_times),
    }

    # 5. Third-party modules imported when prime is used
    base_mods = set(sys.modules.keys())
    import prime as p_fresh
    p_fresh.PRIME().solve("Calculate 6 divided by 2.")
    after_mods = set(sys.modules.keys())
    new_mods = after_mods - base_mods
    third_party = sorted([
        m for m in new_mods
        if not m.startswith("_")
        and not m.startswith("prime")
        and m not in sys.stdlib_module_names
        and "." not in m  # top-level only
    ])
    results["third_party_modules_imported"] = third_party
    results["third_party_module_count"] = len(third_party)

    # 6. C++ extension file sizes
    try:
        import prime as p_check
        prime_dir = os.path.dirname(p_check.__file__)
        ext_sizes = {}
        for f in os.listdir(prime_dir):
            if f.endswith(".so"):
                path = os.path.join(prime_dir, f)
                ext_sizes[f] = os.path.getsize(path)
        results["cpp_extension_sizes_bytes"] = ext_sizes
        results["cpp_extensions_total_bytes"] = sum(ext_sizes.values())
        results["cpp_extensions_total_mb"] = round(sum(ext_sizes.values()) / 1024 / 1024, 3)
    except Exception as e:
        results["cpp_extension_error"] = str(e)

    return results
