"""prime.generate — Generation configuration for PRIME LLM inference."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """Controls text generation behaviour.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 0 means greedy decoding.
        top_k: Keep only top-k tokens. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        seed: RNG seed for reproducibility. 0 = random.
        stop_token_ids: Additional token IDs that stop generation
            (EOS is always a stop token).
    """
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    stop_token_ids: List[int] = field(default_factory=list)

    def validate(self) -> None:
        """Raise ValueError on invalid configuration."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
