"""
examples/phase3_demo.py — PRIME Phase 3 Memory Engine demo.

Shows the public API for the three memory tiers, search, retrieval,
eviction, consolidation, and the integration point with the LLM engine.

Run:
    PYTHONPATH=python python3 examples/phase3_demo.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from prime import PRIME, MemoryConfig, MemoryError

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                           '..', 'tests', 'fixtures', 'tiny_llama.gguf')


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main() -> None:
    section("Phase 3 — Memory Engine Demo")

    # ------------------------------------------------------------------
    # 1. Create PRIME runtime and get a MemoryEngine
    # ------------------------------------------------------------------
    prime = PRIME()
    print("[PRIME] Runtime started")

    mem = prime.memory(MemoryConfig(
        max_working    = 4,
        max_short_term = 8,
        max_long_term  = 0,         # unlimited
        recency_half_life = 86400,  # 1 day
        eviction_policy = "lru",
    ))
    print(f"[Memory] Engine ready: {mem}")

    # ------------------------------------------------------------------
    # 2. Long-term memory: store facts
    # ------------------------------------------------------------------
    section("Long-Term Memory — store / get / search")

    facts = [
        ("Machine learning is a subset of artificial intelligence.", 0.9),
        ("Python is a high-level programming language.", 0.8),
        ("Neural networks are inspired by the human brain.", 0.85),
        ("The Transformer architecture underpins modern LLMs.", 0.95),
        ("Gradient descent is used to optimise neural networks.", 0.8),
    ]

    ids = []
    for content, importance in facts:
        mid = mem.store(content, memory_type="long_term",
                        importance=importance,
                        metadata={"domain": "AI", "lang": "en"})
        ids.append(mid)
        print(f"  stored [{mid}] imp={importance:.1f}  {content[:50]}…")

    print(f"\n  Total items: {mem.count()}")

    # Search
    print("\n  Search: 'machine learning neural'")
    results = mem.search("machine learning neural", limit=3)
    for r in results:
        print(f"    [{r.id}] {r.content[:60]}…")

    # Get by ID
    item = mem.get(ids[0])
    print(f"\n  Get [{ids[0]}]: content={item.content[:40]}…")
    print(f"    importance={item.importance}  access_count={item.access_count}")

    # Update
    mem.update(ids[0], importance=1.0)
    print(f"\n  Updated [{ids[0]}] importance → {mem.get(ids[0]).importance}")

    # ------------------------------------------------------------------
    # 3. Working memory
    # ------------------------------------------------------------------
    section("Working Memory — current task context")

    mem.working_add("current task: summarise AI concepts")
    mem.working_add("current input: 'What is a transformer?'")
    mem.working_add("step: retrieve relevant memories")
    mem.working_add("step: build context string")

    print(f"  Working memory items: {mem.count('working')}")
    print("  (4th item evicts 1st via LRU because max_working=4)")

    # Adding a 5th should evict the LRU
    mem.working_add("step: generate response")
    print(f"  After adding 5th: {mem.count('working')} items (LRU evicted)")

    mem.working_clear()
    print("  Cleared working memory")

    # ------------------------------------------------------------------
    # 4. Short-term memory + consolidation
    # ------------------------------------------------------------------
    section("Short-Term Memory — automatic eviction + consolidation")

    stm_data = [
        ("User asked about transformers.", 0.6),
        ("User seemed interested in attention mechanisms.", 0.9),
        ("Session started at 14:32 UTC.", 0.2),
    ]
    for content, imp in stm_data:
        mem.store(content, memory_type="short_term", importance=imp)

    print(f"  Short-term items: {mem.count('short_term')}")

    promoted = mem.consolidate(importance_threshold=0.8)
    print(f"  Consolidated {promoted} STM items → LTM (threshold=0.8)")
    print(f"  Long-term items after consolidation: {mem.count('long_term')}")

    # ------------------------------------------------------------------
    # 5. Expiration
    # ------------------------------------------------------------------
    section("Expiration — time-based forgetting")

    past = time.time() - 1.0
    future = time.time() + 3600.0
    mem.store("already expired fact", expires_at=past,
               memory_type="long_term", importance=0.1)
    mem.store("still valid fact",     expires_at=future,
               memory_type="long_term", importance=0.5)

    before = mem.count()
    removed = mem.expire_all()
    after = mem.count()
    print(f"  Before expire_all: {before} items")
    print(f"  Expired items removed: {removed}")
    print(f"  After expire_all: {after} items")

    # ------------------------------------------------------------------
    # 6. retrieve_context — integration point for future phases
    # ------------------------------------------------------------------
    section("retrieve_context — future reasoning integration point")

    context = mem.retrieve_context("transformer architecture attention", limit=3)
    print("  Top-3 context items for 'transformer architecture attention':")
    for i, c in enumerate(context, 1):
        print(f"    {i}. [{c.id}] {c.content[:60]}…")

    # ------------------------------------------------------------------
    # 7. LLM + Memory integration (if fixture available)
    # ------------------------------------------------------------------
    section("LLM + Memory Integration (optional)")

    if os.path.isfile(MODEL_PATH):
        model = prime.load_model(MODEL_PATH)
        ctx_text = context[0].content[:20] if context else "AI"
        prompt = f"mem:{ctx_text}"
        from prime import GenerationConfig
        out = model.generate(prompt, config=GenerationConfig(max_new_tokens=4,
                                                              temperature=0.0,
                                                              seed=42))
        print(f"  Augmented prompt: {prompt!r}")
        print(f"  LLM output:       {out!r}")
        print("  ✓ LLM engine + Memory Engine pipeline verified")
    else:
        print("  (LLM fixture not found — skipping LLM integration demo)")
        print(f"  Fixture path: {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 8. Error handling examples
    # ------------------------------------------------------------------
    section("Error Handling")

    try:
        mem.get(99999)
    except MemoryError as e:
        print(f"  get(missing_id) → MemoryError: {e}")

    try:
        mem.store("", memory_type="long_term")
    except MemoryError as e:
        print(f"  store('') → MemoryError: {e}")

    try:
        mem.store("x", importance=1.5)
    except MemoryError as e:
        print(f"  store(importance=1.5) → MemoryError: {e}")

    # ------------------------------------------------------------------
    # 9. Shutdown
    # ------------------------------------------------------------------
    section("Shutdown")

    print(f"  Final item count: {mem.count()}")
    prime.shutdown()
    print("[PRIME] Runtime shut down cleanly")


if __name__ == "__main__":
    main()
