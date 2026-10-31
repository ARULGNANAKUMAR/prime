"""
prime.config — Python-side configuration for PRIME.

Only fields used by the Phase 1 runtime are present. Future phases add
sub-configs (e.g. ModelConfig, MemoryConfig) without changing this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import _prime  # C++ extension


LogLevel = Literal["silent", "error", "warn", "info", "debug"]

_LOG_LEVEL_MAP: dict[str, int] = {
    "silent": _prime.LOG_SILENT,
    "error":  _prime.LOG_ERROR,
    "warn":   _prime.LOG_WARN,
    "info":   _prime.LOG_INFO,
    "debug":  _prime.LOG_DEBUG,
}


@dataclass
class RuntimeConfig:
    """Configuration for the PRIME runtime.

    Attributes:
        log_level: Verbosity level for the C++ logger.
        compute_backend: Backend hint for future phases ("cpu" is the only
            supported value in Phase 1).
        max_memory_bytes: Advisory memory cap. 0 means unlimited.
    """

    log_level: LogLevel = "info"
    compute_backend: Literal["cpu"] = "cpu"
    max_memory_bytes: int = 0

    def validate(self) -> None:
        """Raise ValueError if any field has an illegal value."""
        if self.log_level not in _LOG_LEVEL_MAP:
            raise ValueError(
                f"log_level must be one of {list(_LOG_LEVEL_MAP)}, got {self.log_level!r}"
            )
        if self.compute_backend not in ("cpu",):
            raise ValueError(
                f"compute_backend must be 'cpu' in Phase 1, got {self.compute_backend!r}"
            )
        if self.max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must be >= 0")

    def _log_level_int(self) -> int:
        return _LOG_LEVEL_MAP[self.log_level]
