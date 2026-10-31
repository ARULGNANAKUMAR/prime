"""
phase2_demo.py — PRIME Phase 2 integration demo.

Demonstrates: model loading, greedy generation, streaming, generation config.
Uses the test fixture (tiny_llama.gguf). For real use, point to any GGUF model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

from prime import PRIME, GenerationConfig


FIXTURE = os.path.join(os.path.dirname(__file__), "../tests/fixtures/tiny_llama.gguf")


def main():
    print("=== PRIME Phase 2 Demo ===\n")

    brain = PRIME()
    print(f"Runtime: {brain.status()}\n")

    print("Loading model...")
    model = brain.load_model(FIXTURE)
    print(f"  {model!r}")
    print(f"  Vocabulary size: {model.vocab_size}")
    print(f"  Context length:  {model.context_length}")
    print()

    # Greedy generation
    cfg_greedy = GenerationConfig(max_new_tokens=10, temperature=0.0)
    result = model.generate("hello world", cfg_greedy)
    print(f"Greedy output: {result!r}")
    print()

    # Streaming
    print("Streaming output: ", end="", flush=True)
    cfg_stream = GenerationConfig(max_new_tokens=8, temperature=0.0)
    for piece in model.stream("hello", cfg_stream):
        print(piece, end="", flush=True)
    print()
    print()

    # Sampling with seed
    cfg_seeded = GenerationConfig(max_new_tokens=8, temperature=0.8,
                                   top_k=10, seed=42)
    r1 = model.generate("hello", cfg_seeded)
    r2 = model.generate("hello", cfg_seeded)
    print(f"Seeded sample 1: {r1!r}")
    print(f"Seeded sample 2: {r2!r}")
    print()

    brain.shutdown()
    print("=== Demo complete ===")


if __name__ == "__main__":
    main()
