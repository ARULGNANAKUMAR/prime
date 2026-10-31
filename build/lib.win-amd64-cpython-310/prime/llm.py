"""prime.llm — Public LLM model API for PRIME Phase 2."""

from __future__ import annotations

import os
from typing import Callable, Iterator, Optional

import _prime_llm

from .generate import GenerationConfig


class LLMError(Exception):
    """Raised when a PRIME LLM operation fails."""


class LLMModel:
    """A loaded local LLM model ready for inference.

    Obtain via :meth:`prime.PRIME.load_model`::

        from prime import PRIME

        brain = PRIME()
        model = brain.load_model("/path/to/model.gguf")

        result = model.generate("Explain transformers briefly.")
        print(result)

        brain.shutdown()

    The model file must be in GGUF format.
    No network access is required during inference.
    """

    def __init__(self, model_path: str) -> None:
        """Load a GGUF model from *model_path*.

        Args:
            model_path: Absolute or relative path to a ``.gguf`` model file.

        Raises:
            FileNotFoundError: If the file does not exist.
            LLMError: If the file cannot be parsed or is unsupported.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self._engine = _prime_llm._LLMEngine(model_path=model_path)
        except _prime_llm.LLMError as exc:
            raise LLMError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Generation API
    # ------------------------------------------------------------------

    def generate(self, prompt: str,
                 config: Optional[GenerationConfig] = None) -> str:
        """Generate text for *prompt* and return the full result.

        Args:
            prompt: The input text to complete.
            config: Optional :class:`GenerationConfig`. Defaults to
                ``GenerationConfig()`` (temperature=1, up to 128 tokens).

        Returns:
            Generated text (not including the prompt).

        Raises:
            LLMError: On inference failure.
            ValueError: On invalid *config*.
        """
        cfg = config or GenerationConfig()
        cfg.validate()
        try:
            return self._engine.generate(
                prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                seed=cfg.seed,
            )
        except _prime_llm.LLMError as exc:
            raise LLMError(str(exc)) from exc

    def stream(self, prompt: str,
               config: Optional[GenerationConfig] = None) -> Iterator[str]:
        """Yield token pieces as they are generated (streaming mode).

        Example::

            for piece in model.stream("Hello"):
                print(piece, end="", flush=True)
            print()

        Args:
            prompt: Input text.
            config: Optional :class:`GenerationConfig`.

        Yields:
            Decoded token pieces (may be partial UTF-8 words).

        Raises:
            LLMError: On inference failure.
        """
        cfg = config or GenerationConfig()
        cfg.validate()

        pieces: list[str] = []
        error_holder: list[Exception] = []

        def _callback(piece: str) -> bool:
            pieces.append(piece)
            return True  # continue

        try:
            self._engine.stream_generate(
                prompt, _callback,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                seed=cfg.seed,
            )
        except _prime_llm.LLMError as exc:
            raise LLMError(str(exc)) from exc

        yield from pieces

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the model's vocabulary."""
        return self._engine.vocab_size()

    @property
    def context_length(self) -> int:
        """Maximum sequence length the model supports."""
        return self._engine.context_length()

    @property
    def model_path(self) -> str:
        """Path to the loaded GGUF file."""
        return self._engine.model_path()

    @property
    def arch(self) -> str:
        """Model architecture name (e.g. 'llama')."""
        return self._engine.arch_name()

    def __repr__(self) -> str:
        return (f"<LLMModel arch={self.arch!r} "
                f"vocab={self.vocab_size} ctx={self.context_length}>")
