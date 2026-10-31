"""
benchmark_phase2.py — PRIME Phase 2 baseline inference benchmark.

Measures: model load time, first-token latency, tokens/sec, memory.
Run from the project root:
    PYTHONPATH=python python3 examples/benchmark_phase2.py
"""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from prime import PRIME, GenerationConfig

FIXTURE = os.path.join(os.path.dirname(__file__), "../tests/fixtures/tiny_llama.gguf")
PROMPT = "hello world"
MAX_TOKENS = 32
N_ITERS = 10


def measure_load(model_path: str, n: int = 5) -> dict:
    brain = PRIME()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model = brain.load_model(model_path)
        times.append(time.perf_counter() - t0)
        del model
    brain.shutdown()
    return {
        "load_mean_ms":   statistics.mean(times) * 1000,
        "load_median_ms": statistics.median(times) * 1000,
    }


def measure_generation(model_path: str, n: int = N_ITERS) -> dict:
    brain = PRIME()
    model = brain.load_model(model_path)
    cfg = GenerationConfig(max_new_tokens=MAX_TOKENS, temperature=0.0)

    gen_times = []
    token_counts = []

    for _ in range(n):
        t0 = time.perf_counter()
        result = model.generate(PROMPT, cfg)
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed)
        # Count generated tokens (approximate via spaces + length heuristic)
        token_counts.append(max(1, len(result.split())))

    brain.shutdown()

    mean_time = statistics.mean(gen_times)
    mean_toks = statistics.mean(token_counts)
    return {
        "generation_mean_ms": mean_time * 1000,
        "tokens_per_sec": mean_toks / mean_time if mean_time > 0 else 0,
        "max_new_tokens": MAX_TOKENS,
        "prompt": PROMPT,
    }


def measure_memory() -> dict:
    try:
        import resource
        brain = PRIME()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        model = brain.load_model(FIXTURE)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        brain.shutdown()
        unit = 1024 if sys.platform == "linux" else 1
        return {
            "before_kb": before * unit // 1024,
            "after_kb":  after  * unit // 1024,
            "delta_kb":  (after - before) * unit // 1024,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=== PRIME Phase 2 Inference Benchmark ===\n")
    print(f"Model:    {os.path.basename(FIXTURE)}")
    print(f"Prompt:   {PROMPT!r}")
    print(f"Tokens:   {MAX_TOKENS} max new tokens")
    print(f"Iters:    {N_ITERS}")
    print()

    brain = PRIME()
    model = brain.load_model(FIXTURE)
    cfg = GenerationConfig(max_new_tokens=1, temperature=0.0)
    cfg_info = GenerationConfig(max_new_tokens=MAX_TOKENS, temperature=0.0)
    print(f"Arch:     {model.arch}")
    print(f"Vocab:    {model.vocab_size}")
    print(f"Context:  {model.context_length}")
    sample = model.generate(PROMPT, cfg_info)
    print(f"Sample:   {sample!r}")
    brain.shutdown()
    print()

    print(f"Load time ({5} runs):")
    r = measure_load(FIXTURE)
    print(f"  Mean:   {r['load_mean_ms']:.1f} ms")
    print(f"  Median: {r['load_median_ms']:.1f} ms")

    print()
    print(f"Generation ({N_ITERS} runs, greedy, max {MAX_TOKENS} tokens):")
    r = measure_generation(FIXTURE)
    print(f"  Mean generation time: {r['generation_mean_ms']:.1f} ms")
    print(f"  Approx tokens/sec:    {r['tokens_per_sec']:.1f}")

    print()
    print("Memory footprint:")
    r = measure_memory()
    if "error" in r:
        print(f"  (unavailable: {r['error']})")
    else:
        print(f"  Before model load: {r['before_kb']} kB")
        print(f"  After  model load: {r['after_kb']} kB")
        print(f"  Delta:             {r['delta_kb']} kB")

    print("\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
