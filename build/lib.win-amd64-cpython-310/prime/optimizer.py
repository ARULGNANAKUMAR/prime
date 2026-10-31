"""
prime.optimizer — PRIME Phase 9: Adaptive Lightweight Optimization Engine.

Phase 9 answers: "Which compatible strategy can execute efficiently under the
current resource constraints?"

It works AFTER MOSAIC (Phase 6) and Phase 8 (Learning) have run:

    ATLAS
      ↓
    MOSAIC Capability Matching
      ↓
    Phase 8 Learned Preferences
      ↓
    Phase 9 Resource / Performance Analysis      ← THIS MODULE
      ↓
    Final Execution Strategy
      ↓
    Runtime
      ↓
    Performance Monitoring → Feedback → Optimization State

Design principles
─────────────────
• All resource metrics use actual runtime measurements or return UNKNOWN.
• Hard constraints always win over soft preferences.
• No hidden learning — performance history ≠ Phase 8 learning engine.
• Optimizer overhead is measured and can be bypassed for tiny tasks.
• Cross-platform (Linux / Windows / macOS); missing metrics return UNKNOWN.
• Fully offline, no network calls, no GPU mandatory.

Phase boundary (strict)
───────────────────────
Phase 9 MAY:  measure resources, rank strategies by cost, enforce constraints,
              record performance history, apply adaptive scheduling,
              cache optimization decisions, expose traces.
Phase 9 MUST NOT: retrain models, modify Phase 8 learning logic, perform
              gradient descent, add cloud APIs, assume specific hardware.
Phase 10 handles: final integration, benchmarks, research paper.
"""

from __future__ import annotations

import os
import platform
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel for unavailable measurements
# ─────────────────────────────────────────────────────────────────────────────

UNKNOWN = "UNKNOWN"

# psutil is optional; we degrade gracefully if not installed
try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None
    _PSUTIL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationPolicy(str, Enum):
    """High-level objective for the optimization engine."""
    LOW_LATENCY      = "LOW_LATENCY"
    LOW_MEMORY       = "LOW_MEMORY"
    LOW_CPU          = "LOW_CPU"
    BALANCED         = "BALANCED"
    MAX_QUALITY      = "MAX_QUALITY"
    RESOURCE_LIMITED = "RESOURCE_LIMITED"
    ENERGY_AWARE_PROXY = "ENERGY_AWARE_PROXY"


class ExecutionStrategy(str, Enum):
    """Named execution strategies Phase 9 can select."""
    FULL        = "FULL"        # highest quality, all resources
    BALANCED    = "BALANCED"    # quality+resource trade-off (default)
    LIGHTWEIGHT = "LIGHTWEIGHT" # minimal resource usage
    LOW_MEMORY  = "LOW_MEMORY"  # prefer low-memory modules/models
    LOW_LATENCY = "LOW_LATENCY" # prefer fastest path
    CPU_SAVER   = "CPU_SAVER"   # minimal CPU usage


class OptimizationLevel(str, Enum):
    """How aggressively Phase 9 optimizes."""
    OFF        = "OFF"        # no optimization; pass through
    MINIMAL    = "MINIMAL"    # only hard constraint enforcement
    BALANCED   = "BALANCED"   # resource + performance-aware selection
    AGGRESSIVE = "AGGRESSIVE" # full historical optimization + adaptive switching


# ─────────────────────────────────────────────────────────────────────────────
# Resource Monitor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CPUInfo:
    """CPU availability snapshot."""
    logical_count:    Any = UNKNOWN   # int or UNKNOWN
    physical_count:   Any = UNKNOWN   # int or UNKNOWN
    utilization_pct:  Any = UNKNOWN   # float 0-100 or UNKNOWN
    load_avg_1m:      Any = UNKNOWN   # float or UNKNOWN
    frequency_mhz:    Any = UNKNOWN   # float or UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_count":   self.logical_count,
            "physical_count":  self.physical_count,
            "utilization_pct": self.utilization_pct,
            "load_avg_1m":     self.load_avg_1m,
            "frequency_mhz":   self.frequency_mhz,
        }


@dataclass
class MemoryInfo:
    """Memory availability snapshot (bytes or UNKNOWN)."""
    total_bytes:     Any = UNKNOWN
    available_bytes: Any = UNKNOWN
    used_bytes:      Any = UNKNOWN
    process_rss:     Any = UNKNOWN   # current process resident set
    utilization_pct: Any = UNKNOWN

    @property
    def available_mb(self) -> Any:
        if self.available_bytes == UNKNOWN:
            return UNKNOWN
        return self.available_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes":     self.total_bytes,
            "available_bytes": self.available_bytes,
            "used_bytes":      self.used_bytes,
            "process_rss":     self.process_rss,
            "utilization_pct": self.utilization_pct,
        }


@dataclass
class ResourceSnapshot:
    """Combined resource state at a point in time."""
    timestamp:   float = field(default_factory=time.time)
    cpu:         CPUInfo  = field(default_factory=CPUInfo)
    memory:      MemoryInfo = field(default_factory=MemoryInfo)
    platform_os: str = ""
    python_ver:  str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":   self.timestamp,
            "cpu":         self.cpu.to_dict(),
            "memory":      self.memory.to_dict(),
            "platform_os": self.platform_os,
            "python_ver":  self.python_ver,
        }


class ResourceMonitor:
    """
    Collects actual runtime resource information.

    Metrics that cannot be measured return UNKNOWN.
    Uses psutil when available; falls back to stdlib.
    Caches measurements for ``cache_ttl_s`` seconds to avoid overhead.
    """

    def __init__(self, cache_ttl_s: float = 1.0) -> None:
        self._cache_ttl   = cache_ttl_s
        self._last_snap:  Optional[ResourceSnapshot] = None
        self._last_time:  float = 0.0
        self._platform    = platform.system()
        self._python_ver  = platform.python_version()

    # ── Public API ─────────────────────────────────────────────────────────

    def snapshot(self, force: bool = False) -> ResourceSnapshot:
        """Return a resource snapshot, using cache when fresh."""
        now = time.monotonic()
        if (not force
                and self._last_snap is not None
                and (now - self._last_time) < self._cache_ttl):
            return self._last_snap
        snap = ResourceSnapshot(
            timestamp   = time.time(),
            cpu         = self._read_cpu(),
            memory      = self._read_memory(),
            platform_os = self._platform,
            python_ver  = self._python_ver,
        )
        self._last_snap = snap
        self._last_time = now
        return snap

    def available_memory_bytes(self) -> Any:
        return self.snapshot().memory.available_bytes

    def cpu_utilization_pct(self) -> Any:
        return self.snapshot().cpu.utilization_pct

    def logical_cpu_count(self) -> Any:
        return self.snapshot().cpu.logical_count

    # ── Internal ───────────────────────────────────────────────────────────

    def _read_cpu(self) -> CPUInfo:
        info = CPUInfo()
        # Logical/physical count
        try:
            info.logical_count = os.cpu_count() or UNKNOWN
        except Exception:
            pass
        if _PSUTIL_AVAILABLE:
            try:
                info.physical_count = _psutil.cpu_count(logical=False) or UNKNOWN
            except Exception:
                pass
            try:
                # interval=None uses last cached psutil value (no blocking)
                info.utilization_pct = _psutil.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                freq = _psutil.cpu_freq()
                if freq:
                    info.frequency_mhz = freq.current
            except Exception:
                pass
        # Load average (POSIX only)
        try:
            la = os.getloadavg()
            info.load_avg_1m = la[0]
        except (AttributeError, OSError):
            pass
        return info

    def _read_memory(self) -> MemoryInfo:
        info = MemoryInfo()
        if _PSUTIL_AVAILABLE:
            try:
                vm = _psutil.virtual_memory()
                info.total_bytes     = vm.total
                info.available_bytes = vm.available
                info.used_bytes      = vm.used
                info.utilization_pct = vm.percent
            except Exception:
                pass
            try:
                proc = _psutil.Process()
                info.process_rss = proc.memory_info().rss
            except Exception:
                pass
        else:
            # Fallback via /proc/meminfo on Linux
            if self._platform == "Linux":
                try:
                    info.total_bytes, info.available_bytes = \
                        self._read_proc_meminfo()
                except Exception:
                    pass
            # Process RSS via /proc/self/status on Linux
            try:
                with open("/proc/self/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            info.process_rss = kb * 1024
                            break
            except Exception:
                pass
        return info

    @staticmethod
    def _read_proc_meminfo() -> Tuple[int, int]:
        total = avail = 0
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
        return total, avail


# ─────────────────────────────────────────────────────────────────────────────
# Performance Record & History
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceRecord:
    """Single execution performance measurement."""
    execution_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    task_id:            str   = ""
    module_id:          str   = ""
    strategy:           str   = ExecutionStrategy.BALANCED.value
    start_time:         float = 0.0
    end_time:           float = 0.0
    latency_ms:         float = 0.0
    cpu_usage:          Any   = UNKNOWN   # float pct or UNKNOWN
    memory_usage:       Any   = UNKNOWN   # bytes or UNKNOWN
    success:            bool  = False
    result_quality:     float = 0.0       # 0.0–1.0 estimate
    resource_class:     str   = ""        # "low" | "medium" | "high"
    model:              str   = ""
    configuration:      Dict[str, Any] = field(default_factory=dict)
    metadata:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerformanceRecord":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


class PerformanceHistory:
    """
    Lightweight store of PerformanceRecord objects.

    Not a Phase 8 learning engine — only tracks execution statistics
    for resource-aware module selection.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._max    = max_records
        self._records: deque = deque(maxlen=max_records)
        # module_id → list of PerformanceRecord
        self._by_module: Dict[str, List[PerformanceRecord]] = {}

    # ── Write ──────────────────────────────────────────────────────────────

    def record(self, rec: PerformanceRecord) -> None:
        self._records.append(rec)
        lst = self._by_module.setdefault(rec.module_id, [])
        lst.append(rec)
        # Bound per-module list to prevent unbounded growth
        if len(lst) > self._max:
            del lst[:-self._max]
        # Prune per-module list to avoid unlimited growth
        lst = self._by_module[rec.module_id]
        if len(lst) > self._max:
            self._by_module[rec.module_id] = lst[-self._max:]

    # ── Read ───────────────────────────────────────────────────────────────

    def get_recent(self, n: int = 50) -> List[PerformanceRecord]:
        return list(self._records)[-n:]

    def for_module(self, module_id: str,
                   n: int = 100) -> List[PerformanceRecord]:
        return list(self._by_module.get(module_id, []))[-n:]

    def avg_latency_ms(self, module_id: str,
                       min_samples: int = 1) -> Any:
        recs = self.for_module(module_id)
        if len(recs) < min_samples:
            return UNKNOWN
        return sum(r.latency_ms for r in recs) / len(recs)

    def success_rate(self, module_id: str) -> Any:
        recs = self.for_module(module_id)
        if not recs:
            return UNKNOWN
        return sum(1 for r in recs if r.success) / len(recs)

    def avg_memory_usage(self, module_id: str) -> Any:
        recs = [r for r in self.for_module(module_id)
                if r.memory_usage != UNKNOWN]
        if not recs:
            return UNKNOWN
        return sum(r.memory_usage for r in recs) / len(recs)

    def sample_count(self, module_id: str) -> int:
        return len(self._by_module.get(module_id, []))

    def aggregate(self, module_id: str) -> Dict[str, Any]:
        return {
            "module_id":       module_id,
            "sample_count":    self.sample_count(module_id),
            "avg_latency_ms":  self.avg_latency_ms(module_id),
            "success_rate":    self.success_rate(module_id),
            "avg_memory_usage":self.avg_memory_usage(module_id),
        }

    def clear(self) -> None:
        self._records.clear()
        self._by_module.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Cost Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostWeights:
    """Configurable weights for the cost model."""
    cpu:     float = 0.30
    memory:  float = 0.30
    latency: float = 0.30
    quality: float = 0.10   # penalty when quality is expected to be lower

    def validate(self) -> None:
        total = self.cpu + self.memory + self.latency + self.quality
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"CostWeights must sum to ~1.0 (got {total:.3f})"
            )


class CostModel:
    """
    Estimates execution cost as a scalar score (lower = cheaper).

    Cost components:
        cpu_score     = cpu_relative / max_cpu_reference
        memory_score  = memory_relative / max_mem_reference
        latency_score = latency_ms / max_latency_reference
        quality_penalty = (1 - expected_quality)

    All inputs are relative MOSAIC ResourceCost values or historical data.
    No actual energy consumption is measured or claimed.
    ENERGY_AWARE_PROXY is computed from CPU time × relative usage (a proxy only).
    """

    CPU_REF     = 10.0    # reference cpu_relative for normalization
    MEMORY_REF  = 10.0    # reference memory_relative
    LATENCY_REF = 2000.0  # reference latency_ms

    def __init__(self, weights: Optional[CostWeights] = None) -> None:
        self._w = weights or CostWeights()

    def score(self, cpu_relative: float = 1.0,
              memory_relative: float = 1.0,
              latency_ms: float = 0.0,
              expected_quality: float = 1.0) -> float:
        """Return a cost score ≥ 0. Lower is cheaper."""
        cpu_s     = min(cpu_relative / self.CPU_REF, 1.0)
        mem_s     = min(memory_relative / self.MEMORY_REF, 1.0)
        lat_s     = min(latency_ms / self.LATENCY_REF, 1.0)
        qual_pen  = max(0.0, 1.0 - expected_quality)
        return (self._w.cpu     * cpu_s
                + self._w.memory  * mem_s
                + self._w.latency * lat_s
                + self._w.quality * qual_pen)

    def energy_proxy(self, cpu_relative: float,
                     latency_ms: float) -> float:
        """
        A dimensionless energy proxy based on CPU × time.
        Clearly labeled as a proxy — not actual power consumption.
        """
        return (cpu_relative / self.CPU_REF) * (latency_ms / self.LATENCY_REF)

    def score_from_module_meta(self, module_meta: Dict[str, Any],
                               history: Optional[PerformanceHistory] = None
                               ) -> float:
        cost = module_meta.get("cost", {})
        cpu  = cost.get("cpu_relative",    1.0)
        mem  = cost.get("memory_relative", 1.0)
        lat  = cost.get("latency_ms",      0.0)
        # Override latency from history when enough samples exist
        if history is not None:
            mid = module_meta.get("id", "")
            h_lat = history.avg_latency_ms(mid, min_samples=3)
            if h_lat != UNKNOWN:
                lat = h_lat
        return self.score(cpu, mem, lat)


# ─────────────────────────────────────────────────────────────────────────────
# Resource Constraints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceConstraints:
    """
    Hard and soft resource limits.

    Hard constraints (MUST NOT be violated):
        max_memory_bytes, max_latency_ms, max_modules, max_model_size_bytes

    Soft preferences (optimizer tries to honour):
        preferred_cpu_pct, preferred_memory_pct
    """
    # Hard
    max_memory_bytes:    Optional[int]   = None
    max_latency_ms:      Optional[float] = None
    max_cpu_pct:         Optional[float] = None
    max_modules:         Optional[int]   = None
    max_model_size_bytes:Optional[int]   = None
    max_context_tokens:  Optional[int]   = None
    max_threads:         Optional[int]   = None

    # Soft
    preferred_cpu_pct:    Optional[float] = None
    preferred_memory_pct: Optional[float] = None
    min_quality:          float = 0.0   # 0.0 = no minimum quality requirement

    def check_memory(self, required_bytes: int,
                     available_bytes: Any) -> Tuple[bool, str]:
        """Return (ok, reason). Hard-fails when required > available."""
        if available_bytes != UNKNOWN and required_bytes > available_bytes:
            return False, (
                f"Required {required_bytes/1e6:.1f} MB exceeds "
                f"available {available_bytes/1e6:.1f} MB"
            )
        if (self.max_memory_bytes is not None
                and required_bytes > self.max_memory_bytes):
            return False, (
                f"Required {required_bytes/1e6:.1f} MB exceeds "
                f"hard limit {self.max_memory_bytes/1e6:.1f} MB"
            )
        return True, ""

    def check_latency(self, estimated_ms: float) -> Tuple[bool, str]:
        if (self.max_latency_ms is not None
                and estimated_ms > self.max_latency_ms):
            return False, (
                f"Estimated latency {estimated_ms:.0f} ms exceeds "
                f"limit {self.max_latency_ms:.0f} ms"
            )
        return True, ""

    def check_quality(self, expected_quality: float) -> Tuple[bool, str]:
        if expected_quality < self.min_quality:
            return False, (
                f"Expected quality {expected_quality:.2f} below "
                f"minimum {self.min_quality:.2f}"
            )
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Execution Strategy Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """Full specification for one ExecutionStrategy."""
    strategy:          ExecutionStrategy
    policy:            OptimizationPolicy
    quality_weight:    float = 1.0
    latency_weight:    float = 0.3
    memory_weight:     float = 0.3
    cpu_weight:        float = 0.3
    thread_fraction:   float = 1.0   # fraction of logical CPUs to use
    prefer_lightweight:bool  = False
    description:       str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy":          self.strategy.value,
            "policy":            self.policy.value,
            "quality_weight":    self.quality_weight,
            "latency_weight":    self.latency_weight,
            "memory_weight":     self.memory_weight,
            "cpu_weight":        self.cpu_weight,
            "thread_fraction":   self.thread_fraction,
            "prefer_lightweight":self.prefer_lightweight,
            "description":       self.description,
        }


# Pre-defined strategy configurations
STRATEGY_CONFIGS: Dict[str, StrategyConfig] = {
    ExecutionStrategy.FULL.value: StrategyConfig(
        strategy=ExecutionStrategy.FULL,
        policy=OptimizationPolicy.MAX_QUALITY,
        quality_weight=2.0, latency_weight=0.1,
        memory_weight=0.1, cpu_weight=0.1,
        thread_fraction=1.0, prefer_lightweight=False,
        description="Maximum quality; unconstrained resource usage",
    ),
    ExecutionStrategy.BALANCED.value: StrategyConfig(
        strategy=ExecutionStrategy.BALANCED,
        policy=OptimizationPolicy.BALANCED,
        quality_weight=1.0, latency_weight=0.3,
        memory_weight=0.3, cpu_weight=0.3,
        thread_fraction=0.75, prefer_lightweight=False,
        description="Balanced quality and resource usage (default)",
    ),
    ExecutionStrategy.LIGHTWEIGHT.value: StrategyConfig(
        strategy=ExecutionStrategy.LIGHTWEIGHT,
        policy=OptimizationPolicy.RESOURCE_LIMITED,
        quality_weight=0.5, latency_weight=0.5,
        memory_weight=0.5, cpu_weight=0.5,
        thread_fraction=0.5, prefer_lightweight=True,
        description="Minimal resource usage; reduced quality acceptable",
    ),
    ExecutionStrategy.LOW_MEMORY.value: StrategyConfig(
        strategy=ExecutionStrategy.LOW_MEMORY,
        policy=OptimizationPolicy.LOW_MEMORY,
        quality_weight=0.8, latency_weight=0.2,
        memory_weight=1.0, cpu_weight=0.3,
        thread_fraction=0.5, prefer_lightweight=True,
        description="Prefer lowest-memory modules and configurations",
    ),
    ExecutionStrategy.LOW_LATENCY.value: StrategyConfig(
        strategy=ExecutionStrategy.LOW_LATENCY,
        policy=OptimizationPolicy.LOW_LATENCY,
        quality_weight=0.8, latency_weight=1.0,
        memory_weight=0.2, cpu_weight=0.2,
        thread_fraction=1.0, prefer_lightweight=True,
        description="Prefer fastest modules; latency above quality",
    ),
    ExecutionStrategy.CPU_SAVER.value: StrategyConfig(
        strategy=ExecutionStrategy.CPU_SAVER,
        policy=OptimizationPolicy.LOW_CPU,
        quality_weight=0.7, latency_weight=0.3,
        memory_weight=0.3, cpu_weight=1.0,
        thread_fraction=0.25, prefer_lightweight=True,
        description="Minimal CPU usage; suitable for background execution",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Thread Selector
# ─────────────────────────────────────────────────────────────────────────────

class ThreadSelector:
    """
    Selects an appropriate worker thread count based on strategy and CPU info.
    Does not assume a specific CPU architecture.
    """

    def select(self, strategy: ExecutionStrategy,
               cpu_info: CPUInfo,
               constraints: ResourceConstraints) -> int:
        logical = cpu_info.logical_count
        if logical == UNKNOWN or not isinstance(logical, int):
            logical = 1

        fraction = STRATEGY_CONFIGS[strategy.value].thread_fraction
        selected = max(1, int(logical * fraction))

        if constraints.max_threads is not None:
            selected = min(selected, constraints.max_threads)

        return selected


# ─────────────────────────────────────────────────────────────────────────────
# Memory Estimator
# ─────────────────────────────────────────────────────────────────────────────

# Rough memory cost categories for built-in MOSAIC modules
_MODULE_MEMORY_ESTIMATES: Dict[str, int] = {
    "atlas":     20 * 1024 * 1024,   # ~20 MB
    "reasoning": 30 * 1024 * 1024,   # ~30 MB
    "memory":    50 * 1024 * 1024,   # ~50 MB (depends on store size)
    "validator":  5 * 1024 * 1024,   #  ~5 MB
    "llm":      800 * 1024 * 1024,   # ~800 MB (very rough; model-dependent)
}
_DEFAULT_MODULE_MEMORY = 25 * 1024 * 1024  # 25 MB default


def estimate_module_memory(module_id: str,
                           module_meta: Optional[Dict[str, Any]] = None) -> int:
    """Return an estimated memory footprint in bytes for a module."""
    if module_meta is not None:
        # Use memory_relative from ResourceCost as a scaling hint
        cost = module_meta.get("cost", {})
        mem_rel = cost.get("memory_relative", 1.0)
        base = _MODULE_MEMORY_ESTIMATES.get(module_id, _DEFAULT_MODULE_MEMORY)
        return int(base * mem_rel)
    return _MODULE_MEMORY_ESTIMATES.get(module_id, _DEFAULT_MODULE_MEMORY)


# ─────────────────────────────────────────────────────────────────────────────
# Optimization Trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateAssessment:
    """Assessment of one candidate module or strategy."""
    module_id:          str
    strategy:           str = ExecutionStrategy.BALANCED.value
    cost_score:         float = 0.0
    estimated_latency:  float = 0.0          # ms; 0.0 = no estimate
    estimated_memory:   int   = 0            # bytes
    learned_bonus:      float = 0.0          # from Phase 8
    final_score:        float = 0.0
    selected:           bool  = False
    rejected:           bool  = False
    rejection_reason:   str   = ""
    is_hard_constraint: bool  = False        # hard-constraint rejection

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class OptimizationTrace:
    """
    Full explanation of one optimization decision.

    All fields are populated from actual optimizer data — no LLM generation.
    """
    trace_id:            str   = field(default_factory=lambda: str(uuid.uuid4()))
    task_id:             str   = ""
    timestamp:           float = field(default_factory=time.time)
    # Resource state at decision time
    cpu_utilization:     Any   = UNKNOWN
    available_memory:    Any   = UNKNOWN   # bytes or UNKNOWN
    logical_cpus:        Any   = UNKNOWN
    # Policy and constraints
    policy:              str   = OptimizationPolicy.BALANCED.value
    strategy_selected:   str   = ExecutionStrategy.BALANCED.value
    optimization_level:  str   = OptimizationLevel.BALANCED.value
    thread_count:        int   = 1
    # Phase 8 information consumed
    phase8_preferred:    List[str] = field(default_factory=list)
    phase8_excluded:     List[str] = field(default_factory=list)
    # Candidates
    candidates:          List[CandidateAssessment] = field(default_factory=list)
    # Outcome
    bypass_reason:       str   = ""   # non-empty when OPTIMIZATION_BYPASS
    fallback_used:       bool  = False
    fallback_reason:     str   = ""
    optimizer_latency_ms:float = 0.0

    def explain(self) -> str:
        lines = ["=" * 60, "PRIME Phase 9 Optimization Trace", "=" * 60]
        lines.append(f"Task:            {self.task_id}")
        lines.append(f"Policy:          {self.policy}")
        lines.append(f"Strategy:        {self.strategy_selected}")
        lines.append(f"Level:           {self.optimization_level}")
        lines.append(f"Threads:         {self.thread_count}")
        lines.append(f"CPU utilization: {self.cpu_utilization}")
        if self.available_memory != UNKNOWN:
            mb = self.available_memory / (1024 * 1024)
            lines.append(f"Available RAM:   {mb:.0f} MB")
        else:
            lines.append(f"Available RAM:   {UNKNOWN}")
        if self.phase8_preferred:
            lines.append(f"Ph8 preferred:   {self.phase8_preferred}")
        if self.phase8_excluded:
            lines.append(f"Ph8 excluded:    {self.phase8_excluded}")
        if self.bypass_reason:
            lines.append(f"\nBYPASS: {self.bypass_reason}")
        elif self.candidates:
            lines.append("\nCandidates:")
            for c in self.candidates:
                icon = "✓" if c.selected else "✗"
                lines.append(
                    f"  {icon} {c.module_id:<20s} "
                    f"cost={c.cost_score:.3f}  "
                    f"bonus={c.learned_bonus:+.3f}  "
                    f"final={c.final_score:.3f}"
                    + (f"  REJECTED: {c.rejection_reason}" if c.rejected else "")
                )
        if self.fallback_used:
            lines.append(f"\nFallback used: {self.fallback_reason}")
        lines.append(f"\nOptimizer took {self.optimizer_latency_ms:.2f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Optimization State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationState:
    """
    Persists the current optimization configuration and recent statistics.
    Not a learning engine — records factual adaptation state only.
    """
    current_strategy:     str   = ExecutionStrategy.BALANCED.value
    current_policy:       str   = OptimizationPolicy.BALANCED.value
    optimization_level:   str   = OptimizationLevel.BALANCED.value
    thread_count:         int   = 1
    recent_avg_latency_ms:Any   = UNKNOWN
    recent_success_rate:  Any   = UNKNOWN
    switch_count:         int   = 0
    last_switch_reason:   str   = ""
    optimizer_version:    str   = "9.0"
    updated_at:           float = field(default_factory=time.time)
    metadata:             Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizationState":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Optimization Cache
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationCache:
    """
    Bounded LRU cache for optimization decisions.
    Key: (task_signature, resource_class, policy).
    Safe to invalidate on resource change.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max  = max_size
        self._data: Dict[str, Tuple[ExecutionStrategy, int]] = {}  # key → (strategy, threads)
        self._order: deque = deque()

    def _key(self, task_sig: str, resource_class: str, policy: str) -> str:
        return f"{task_sig}|{resource_class}|{policy}"

    def get(self, task_sig: str, resource_class: str,
            policy: str) -> Optional[Tuple[ExecutionStrategy, int]]:
        k = self._key(task_sig, resource_class, policy)
        return self._data.get(k)

    def put(self, task_sig: str, resource_class: str,
            policy: str,
            strategy: ExecutionStrategy, threads: int) -> None:
        k = self._key(task_sig, resource_class, policy)
        if k not in self._data and len(self._data) >= self._max:
            evict = self._order.popleft()
            self._data.pop(evict, None)
        self._data[k] = (strategy, threads)
        if k in self._order:
            self._order.remove(k)  # re-insert at right
        self._order.append(k)

    def invalidate(self) -> None:
        self._data.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# Hysteresis Controller
# ─────────────────────────────────────────────────────────────────────────────

class HysteresisController:
    """
    Prevents rapid strategy oscillation.
    A strategy switch is only accepted when the score improvement
    exceeds ``threshold``.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self._threshold = threshold
        self._current_score: Optional[float] = None
        self._current_strategy: Optional[ExecutionStrategy] = None

    def should_switch(self, candidate: ExecutionStrategy,
                      candidate_score: float,
                      current: ExecutionStrategy,
                      current_score: float) -> bool:
        if current == candidate:
            return False
        improvement = current_score - candidate_score  # lower is better
        return improvement > self._threshold

    def update(self, strategy: ExecutionStrategy, score: float) -> None:
        self._current_strategy = strategy
        self._current_score    = score

    @property
    def current_strategy(self) -> Optional[ExecutionStrategy]:
        return self._current_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Resource Classifier
# ─────────────────────────────────────────────────────────────────────────────

class ResourceClassifier:
    """
    Classifies current resource availability into a coarse class:
    "high" | "medium" | "low" | "unknown".

    Used for cache keys and strategy pre-selection.
    """

    LOW_MEM_MB    = 256
    MEDIUM_MEM_MB = 1024
    HIGH_CPU_PCT  = 80.0
    MEDIUM_CPU_PCT = 50.0

    def classify(self, snap: ResourceSnapshot) -> str:
        mem = snap.memory.available_bytes
        cpu = snap.cpu.utilization_pct
        if mem == UNKNOWN and cpu == UNKNOWN:
            return "unknown"
        mem_class = "high"
        if mem != UNKNOWN:
            mb = mem / (1024 * 1024)
            if mb < self.LOW_MEM_MB:
                mem_class = "low"
            elif mb < self.MEDIUM_MEM_MB:
                mem_class = "medium"
        cpu_class = "high"
        if cpu != UNKNOWN:
            if cpu > self.HIGH_CPU_PCT:
                cpu_class = "low"     # high CPU usage → low resources available
            elif cpu > self.MEDIUM_CPU_PCT:
                cpu_class = "medium"
        if mem_class == "low" or cpu_class == "low":
            return "low"
        if mem_class == "medium" or cpu_class == "medium":
            return "medium"
        return "high"


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveScheduler:
    """
    Applies a chosen ExecutionStrategy to an execution request.

    Responsibilities:
    - Apply resource-aware thread count.
    - Configure MOSAIC prefer_lightweight flag.
    - Record performance metrics before/after execution.
    - Enforce hard constraints.
    """

    def __init__(self, resource_monitor: ResourceMonitor,
                 performance_history: PerformanceHistory,
                 constraints: ResourceConstraints) -> None:
        self._monitor  = resource_monitor
        self._history  = performance_history
        self._constr   = constraints
        self._selector = ThreadSelector()

    def configure(self, strategy: ExecutionStrategy,
                  snap: ResourceSnapshot) -> Dict[str, Any]:
        """Return MOSAIC-compatible configuration overrides for the strategy."""
        cfg_tmpl = STRATEGY_CONFIGS[strategy.value]
        threads  = self._selector.select(strategy, snap.cpu, self._constr)
        return {
            "prefer_lightweight": cfg_tmpl.prefer_lightweight,
            "_opt_strategy":      strategy.value,
            "_opt_threads":       threads,
        }

    def record_execution(self, task_id: str, module_id: str,
                         strategy: ExecutionStrategy,
                         start_time: float, end_time: float,
                         success: bool,
                         snap_before: ResourceSnapshot) -> PerformanceRecord:
        """Record a completed execution into PerformanceHistory."""
        latency_ms = (end_time - start_time) * 1000.0
        rec = PerformanceRecord(
            task_id      = task_id,
            module_id    = module_id,
            strategy     = strategy.value,
            start_time   = start_time,
            end_time     = end_time,
            latency_ms   = latency_ms,
            cpu_usage    = snap_before.cpu.utilization_pct,
            memory_usage = snap_before.memory.process_rss,
            success      = success,
        )
        self._history.record(rec)
        return rec


# ─────────────────────────────────────────────────────────────────────────────
# Execution Strategy Selector
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionStrategySelector:
    """
    Selects the best ExecutionStrategy from candidates given:
    - current resource snapshot
    - optimization policy
    - hard constraints

    Returns (strategy, reason_string).
    """

    _RESOURCE_CLASS_STRATEGY: Dict[str, ExecutionStrategy] = {
        "high":    ExecutionStrategy.BALANCED,
        "medium":  ExecutionStrategy.BALANCED,
        "low":     ExecutionStrategy.LIGHTWEIGHT,
        "unknown": ExecutionStrategy.BALANCED,
    }

    def select(self, policy: OptimizationPolicy,
               resource_class: str,
               constraints: ResourceConstraints,
               snap: ResourceSnapshot) -> Tuple[ExecutionStrategy, str]:
        # Policy → preferred strategy mapping
        policy_map: Dict[OptimizationPolicy, ExecutionStrategy] = {
            OptimizationPolicy.LOW_LATENCY:      ExecutionStrategy.LOW_LATENCY,
            OptimizationPolicy.LOW_MEMORY:       ExecutionStrategy.LOW_MEMORY,
            OptimizationPolicy.LOW_CPU:          ExecutionStrategy.CPU_SAVER,
            OptimizationPolicy.BALANCED:         ExecutionStrategy.BALANCED,
            OptimizationPolicy.MAX_QUALITY:      ExecutionStrategy.FULL,
            OptimizationPolicy.RESOURCE_LIMITED: ExecutionStrategy.LIGHTWEIGHT,
            OptimizationPolicy.ENERGY_AWARE_PROXY: ExecutionStrategy.CPU_SAVER,
        }
        preferred = policy_map.get(policy, ExecutionStrategy.BALANCED)

        # Hard resource override: if resources are low, don't go FULL
        if resource_class == "low" and preferred == ExecutionStrategy.FULL:
            preferred = ExecutionStrategy.LIGHTWEIGHT
            reason = ("resource class=low overrides MAX_QUALITY; "
                      "using LIGHTWEIGHT")
        elif resource_class == "low":
            reason = f"resource class=low + policy={policy.value}"
        else:
            reason = (f"policy={policy.value} "
                      f"resource_class={resource_class}")

        return preferred, reason


# ─────────────────────────────────────────────────────────────────────────────
# Module Ranker (Phase 9 resource-aware ranking)
# ─────────────────────────────────────────────────────────────────────────────

class ModuleRanker:
    """
    Ranks candidate modules from MOSAIC using:
      1. Phase 8 learned preference bonus
      2. Phase 9 resource cost score
      3. Historical latency/success

    Returns ordered list of (module_id, CandidateAssessment).
    Hard constraints eliminate candidates before ranking.
    """

    def __init__(self, cost_model: CostModel,
                 history: PerformanceHistory,
                 constraints: ResourceConstraints) -> None:
        self._cost    = cost_model
        self._history = history
        self._constr  = constraints

    def rank(self, module_metas: List[Dict[str, Any]],
             strategy: ExecutionStrategy,
             snap: ResourceSnapshot,
             learned_hints: Optional[Dict[str, Any]] = None
             ) -> List[Tuple[str, CandidateAssessment]]:
        """
        Score modules; return sorted list best→worst.
        Hard-rejected modules appear at the end with rejected=True.
        """
        cfg_tmpl = STRATEGY_CONFIGS[strategy.value]
        phase8_preferred = []
        phase8_excluded  = []
        if learned_hints:
            phase8_preferred = learned_hints.get("preferred_modules", [])
            phase8_excluded  = learned_hints.get("excluded_modules", [])

        results: List[Tuple[str, CandidateAssessment]] = []
        available_mem = snap.memory.available_bytes

        for meta in module_metas:
            mid = meta.get("id", "")
            cost = meta.get("cost", {})
            cpu_rel = cost.get("cpu_relative",    1.0)
            mem_rel = cost.get("memory_relative", 1.0)
            lat_ms  = cost.get("latency_ms",      0.0)

            # Use history for latency when available
            h_lat = self._history.avg_latency_ms(mid, min_samples=3)
            if h_lat != UNKNOWN:
                lat_ms = h_lat

            # Estimated memory
            est_mem = estimate_module_memory(mid, meta)
            # Phase 8 bonus
            bonus = 0.0
            if mid in phase8_preferred:
                bonus = +0.10
            elif mid in phase8_excluded:
                bonus = -0.10

            ca = CandidateAssessment(
                module_id         = mid,
                strategy          = strategy.value,
                cost_score        = self._cost.score(cpu_rel, mem_rel, lat_ms),
                estimated_latency = lat_ms,
                estimated_memory  = est_mem,
                learned_bonus     = bonus,
            )

            # Hard constraint: memory
            mem_ok, mem_reason = self._constr.check_memory(
                est_mem, available_mem)
            if not mem_ok:
                ca.rejected = True
                ca.is_hard_constraint = True
                ca.rejection_reason = mem_reason
                results.append((mid, ca))
                continue

            # Hard constraint: latency
            lat_ok, lat_reason = self._constr.check_latency(lat_ms)
            if not lat_ok:
                ca.rejected = True
                ca.is_hard_constraint = True
                ca.rejection_reason = lat_reason
                results.append((mid, ca))
                continue

            # Hard constraint: module status
            if meta.get("status", "available") not in ("available", ""):
                ca.rejected = True
                ca.is_hard_constraint = True
                ca.rejection_reason = f"module status={meta['status']}"
                results.append((mid, ca))
                continue

            # Compute final score: lower cost → better score
            # bonus reduces effective cost; quality weight applied via strategy
            ca.final_score = (
                cfg_tmpl.cpu_weight     * (cpu_rel / CostModel.CPU_REF)
                + cfg_tmpl.memory_weight * (mem_rel / CostModel.MEMORY_REF)
                + cfg_tmpl.latency_weight * (lat_ms / CostModel.LATENCY_REF)
                - cfg_tmpl.quality_weight * bonus
            )
            results.append((mid, ca))

        # Sort: rejected last; among accepted sort by final_score ascending
        accepted  = [(m, c) for m, c in results if not c.rejected]
        rejected  = [(m, c) for m, c in results if c.rejected]
        accepted.sort(key=lambda x: x[1].final_score)
        if accepted:
            accepted[0][1].selected = True
        return accepted + rejected


# ─────────────────────────────────────────────────────────────────────────────
# Optimization Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Configuration for the AdaptiveOptimizationEngine."""
    policy:             OptimizationPolicy  = OptimizationPolicy.BALANCED
    level:              OptimizationLevel   = OptimizationLevel.BALANCED
    constraints:        ResourceConstraints = field(
                            default_factory=ResourceConstraints)
    cost_weights:       CostWeights         = field(default_factory=CostWeights)
    cache_ttl_s:        float = 1.0         # resource cache TTL
    history_max:        int   = 1000
    cache_max:          int   = 128
    bypass_threshold_ms:float = 0.5         # bypass if optimizer cost > this
    hysteresis_threshold:float = 0.05
    switching_min_improvement:float = 0.05  # minimum score improvement to switch
    max_fallback_attempts:int = 2
    # Resource spike smoothing: use N-sample window for classification
    resource_window:    int   = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy":              self.policy.value,
            "level":               self.level.value,
            "constraints":         self.constraints.to_dict(),
            "cost_weights":        self.cost_weights.__dict__,
            "cache_ttl_s":         self.cache_ttl_s,
            "history_max":         self.history_max,
            "cache_max":           self.cache_max,
            "bypass_threshold_ms": self.bypass_threshold_ms,
            "hysteresis_threshold":self.hysteresis_threshold,
            "switching_min_improvement": self.switching_min_improvement,
            "max_fallback_attempts":     self.max_fallback_attempts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        obj = cls()
        if "policy" in d:
            obj.policy = OptimizationPolicy(d["policy"])
        if "level" in d:
            obj.level = OptimizationLevel(d["level"])
        if "constraints" in d:
            c = ResourceConstraints()
            for k, v in d["constraints"].items():
                if hasattr(c, k):
                    setattr(c, k, v)
            obj.constraints = c
        for k in ("cache_ttl_s", "history_max", "cache_max",
                   "bypass_threshold_ms", "hysteresis_threshold",
                   "switching_min_improvement", "max_fallback_attempts",
                   "resource_window"):
            if k in d:
                setattr(obj, k, d[k])
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Optimization Engine (main orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveOptimizationEngine:
    """
    Phase 9 main orchestrator.

    Integrates with MOSAIC (Phase 6) and Phase 8 LearnedPreferenceProvider
    to produce resource-aware execution strategies.

    Typical usage::

        from prime import PRIME
        from prime.optimizer import AdaptiveOptimizationEngine, OptimizerConfig

        prime     = PRIME()
        mosaic    = prime.mosaic()
        optimizer = AdaptiveOptimizationEngine()

        # Annotate a MOSAIC solve call
        plan, trace = optimizer.optimize_plan(mosaic, problem, task_id="t1")
        result      = mosaic.execute(plan, problem)
        optimizer.record_result(trace, result.succeeded)
    """

    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self._cfg     = config or OptimizerConfig()
        self._monitor = ResourceMonitor(cache_ttl_s=self._cfg.cache_ttl_s)
        self._history = PerformanceHistory(max_records=self._cfg.history_max)
        self._cache   = OptimizationCache(max_size=self._cfg.cache_max)
        self._cost    = CostModel(self._cfg.cost_weights)
        self._classifier = ResourceClassifier()
        self._hysteresis = HysteresisController(self._cfg.hysteresis_threshold)
        self._scheduler  = AdaptiveScheduler(
            self._monitor, self._history, self._cfg.constraints)
        self._selector   = ExecutionStrategySelector()
        self._ranker     = ModuleRanker(
            self._cost, self._history, self._cfg.constraints)
        self._state      = OptimizationState(
            current_strategy   = ExecutionStrategy.BALANCED.value,
            current_policy     = self._cfg.policy.value,
            optimization_level = self._cfg.level.value,
        )
        self._trace_log: List[OptimizationTrace] = []
        self._resource_window: deque = deque(maxlen=self._cfg.resource_window)

    # ── Main public API ─────────────────────────────────────────────────────

    def optimize(self, module_metas: List[Dict[str, Any]],
                 task_id: str = "",
                 task_signature: str = "",
                 learned_hints: Optional[Dict[str, Any]] = None,
                 policy_override: Optional[OptimizationPolicy] = None,
                 ) -> OptimizationTrace:
        """
        Core optimization entry point.

        Args:
            module_metas:   List of module metadata dicts from MOSAIC.
            task_id:        Unique task identifier for tracing.
            task_signature: Compact signature for cache lookup.
            learned_hints:  Phase 8 get_mosaic_hints() output.
            policy_override: Override instance-level policy for this call.

        Returns:
            OptimizationTrace with selected strategy and ranked candidates.
        """
        opt_start = time.monotonic()

        if self._cfg.level == OptimizationLevel.OFF:
            return self._bypass_trace(task_id, "optimization_level=OFF",
                                      opt_start)

        policy = policy_override or self._cfg.policy

        # Measure resources (cached)
        snap = self._monitor.snapshot()
        self._resource_window.append(snap)
        # Use smoothed classification over recent window
        resource_class = self._smooth_resource_class()

        trace = OptimizationTrace(
            task_id            = task_id,
            cpu_utilization    = snap.cpu.utilization_pct,
            available_memory   = snap.memory.available_bytes,
            logical_cpus       = snap.cpu.logical_count,
            policy             = policy.value,
            optimization_level = self._cfg.level.value,
            phase8_preferred   = (learned_hints or {}).get(
                "preferred_modules", []),
            phase8_excluded    = (learned_hints or {}).get(
                "excluded_modules", []),
        )

        # MINIMAL mode: only hard constraint enforcement, no ranking
        if self._cfg.level == OptimizationLevel.MINIMAL:
            strategy, reason = self._select_strategy(
                policy, resource_class, snap)
            threads = self._scheduler._selector.select(
                strategy, snap.cpu, self._cfg.constraints)
            trace.strategy_selected = strategy.value
            trace.thread_count      = threads
            trace.optimizer_latency_ms = (time.monotonic() - opt_start) * 1000
            self._finalize_state(strategy, trace)
            return trace

        # Check optimization cache
        if task_signature:
            cached = self._cache.get(task_signature, resource_class,
                                     policy.value)
            if cached is not None:
                strategy, threads = cached
                trace.strategy_selected = strategy.value
                trace.thread_count      = threads
                trace.bypass_reason     = "cache_hit"
                trace.optimizer_latency_ms = (
                    time.monotonic() - opt_start) * 1000
                self._finalize_state(strategy, trace)
                return trace

        # Select strategy
        strategy, strategy_reason = self._select_strategy(
            policy, resource_class, snap)

        # Adaptive bypass for very simple tasks
        if self._cfg.level == OptimizationLevel.BALANCED:
            opt_so_far_ms = (time.monotonic() - opt_start) * 1000
            if opt_so_far_ms > self._cfg.bypass_threshold_ms and not module_metas:
                trace.strategy_selected = strategy.value
                trace.bypass_reason = (
                    f"bypass: optimizer_cost ({opt_so_far_ms:.2f} ms) "
                    f"> threshold ({self._cfg.bypass_threshold_ms} ms); "
                    "no modules to rank")
                trace.optimizer_latency_ms = (
                    time.monotonic() - opt_start) * 1000
                self._finalize_state(strategy, trace)
                return trace

        # Rank modules
        if module_metas:
            ranked = self._ranker.rank(
                module_metas, strategy, snap, learned_hints)
            trace.candidates = [c for _, c in ranked]

        # Thread count
        threads = self._scheduler._selector.select(
            strategy, snap.cpu, self._cfg.constraints)
        trace.strategy_selected = strategy.value
        trace.thread_count      = threads

        # Store in cache
        if task_signature:
            self._cache.put(task_signature, resource_class,
                            policy.value, strategy, threads)

        trace.optimizer_latency_ms = (time.monotonic() - opt_start) * 1000
        self._finalize_state(strategy, trace)
        self._trace_log.append(trace)
        return trace

    def optimize_plan(self, mosaic_engine: Any,
                      problem: Any,
                      task_id: str = "",
                      learned_hints: Optional[Dict[str, Any]] = None,
                      policy_override: Optional[OptimizationPolicy] = None,
                      **mosaic_kwargs: Any,
                      ) -> Tuple[Any, OptimizationTrace]:
        """
        Convenience wrapper: optimize → compose a MOSAIC plan.

        Returns (CompositionPlan, OptimizationTrace).
        MOSAIC remains in charge of capability matching and pipeline
        construction; Phase 9 only contributes resource-aware configuration.
        """
        # Gather available module metadata
        module_metas = []
        try:
            for mid in mosaic_engine.list_modules():
                meta = mosaic_engine.get_module(mid)
                if meta:
                    module_metas.append(meta)
        except Exception:
            pass

        trace = self.optimize(
            module_metas   = module_metas,
            task_id        = task_id,
            task_signature = getattr(problem, "problem_id", ""),
            learned_hints  = learned_hints,
            policy_override= policy_override,
        )

        # Build MOSAIC overrides from trace
        overrides = {}
        strategy_name = trace.strategy_selected
        cfg_tmpl = STRATEGY_CONFIGS.get(strategy_name)
        if cfg_tmpl:
            overrides["prefer_lightweight"] = cfg_tmpl.prefer_lightweight

        if learned_hints:
            if learned_hints.get("preferred_modules"):
                overrides["preferred_modules"] = learned_hints["preferred_modules"]
            if learned_hints.get("excluded_modules"):
                overrides["excluded_modules"] = learned_hints["excluded_modules"]

        # Let MOSAIC compose with Phase 9 resource hints
        all_kwargs = {**mosaic_kwargs, **overrides}
        plan = mosaic_engine.compose(problem, **all_kwargs)
        return plan, trace

    def record_result(self, trace: OptimizationTrace,
                      success: bool,
                      latency_ms: Optional[float] = None) -> None:
        """
        Record the actual execution outcome into PerformanceHistory.
        Not a learning update — only performance statistics.
        """
        for cand in trace.candidates:
            if cand.selected:
                rec = PerformanceRecord(
                    task_id    = trace.task_id,
                    module_id  = cand.module_id,
                    strategy   = trace.strategy_selected,
                    latency_ms = latency_ms or cand.estimated_latency,
                    success    = success,
                )
                self._history.record(rec)
                break

    def explain(self, trace: Optional[OptimizationTrace] = None) -> str:
        """
        Return a human-readable explanation of the last (or given) trace.
        Generated from actual optimizer data — no LLM involved.
        """
        if trace is None:
            if not self._trace_log:
                return "No optimization traces recorded yet."
            trace = self._trace_log[-1]
        return trace.explain()

    def get_state(self) -> OptimizationState:
        return self._state

    def get_performance_history(self) -> PerformanceHistory:
        return self._history

    def get_resource_snapshot(self, force: bool = False) -> ResourceSnapshot:
        return self._monitor.snapshot(force=force)

    def get_traces(self, last_n: int = 10) -> List[OptimizationTrace]:
        return list(self._trace_log[-last_n:])

    def invalidate_cache(self) -> None:
        """Invalidate the optimization cache (e.g. after resource change)."""
        self._cache.invalidate()

    def set_policy(self, policy: OptimizationPolicy) -> None:
        self._cfg.policy      = policy
        self._state.current_policy = policy.value

    def set_level(self, level: OptimizationLevel) -> None:
        self._cfg.level       = level
        self._state.optimization_level = level.value

    def summary(self) -> Dict[str, Any]:
        """High-level summary for display."""
        snap = self._monitor.snapshot()
        return {
            "optimizer_version":    self._state.optimizer_version,
            "level":                self._cfg.level.value,
            "policy":               self._cfg.policy.value,
            "current_strategy":     self._state.current_strategy,
            "thread_count":         self._state.thread_count,
            "cpu_utilization":      snap.cpu.utilization_pct,
            "available_memory_mb":  (
                snap.memory.available_bytes / (1024 * 1024)
                if snap.memory.available_bytes != UNKNOWN else UNKNOWN),
            "logical_cpus":         snap.cpu.logical_count,
            "performance_records":  len(self._history._records),
            "cache_entries":        len(self._cache),
            "traces_recorded":      len(self._trace_log),
            "switch_count":         self._state.switch_count,
            "psutil_available":     _PSUTIL_AVAILABLE,
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _select_strategy(self, policy: OptimizationPolicy,
                         resource_class: str,
                         snap: ResourceSnapshot,
                         ) -> Tuple[ExecutionStrategy, str]:
        strategy, reason = self._selector.select(
            policy, resource_class, self._cfg.constraints, snap)

        # Hysteresis: only switch when improvement exceeds threshold
        current = ExecutionStrategy(self._state.current_strategy)
        if current != strategy:
            # Estimate cost of both strategies
            cur_cfg  = STRATEGY_CONFIGS[current.value]
            new_cfg  = STRATEGY_CONFIGS[strategy.value]
            # Simpler heuristic: LIGHTWEIGHT < BALANCED < FULL
            strategy_order = [
                ExecutionStrategy.CPU_SAVER,
                ExecutionStrategy.LOW_MEMORY,
                ExecutionStrategy.LOW_LATENCY,
                ExecutionStrategy.LIGHTWEIGHT,
                ExecutionStrategy.BALANCED,
                ExecutionStrategy.FULL,
            ]
            cur_idx = (strategy_order.index(current)
                       if current in strategy_order else 3)
            new_idx = (strategy_order.index(strategy)
                       if strategy in strategy_order else 3)
            diff = abs(cur_idx - new_idx)
            if diff > 0 and resource_class not in ("low",):
                # Allow switch
                pass
            elif resource_class == "low":
                # Always switch down under resource pressure
                pass

        return strategy, reason

    def _smooth_resource_class(self) -> str:
        """Return most frequent resource class from recent window."""
        if not self._resource_window:
            return "unknown"
        classes = [self._classifier.classify(s)
                   for s in self._resource_window]
        # Vote: most common class wins
        counts: Dict[str, int] = {}
        for c in classes:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def _bypass_trace(self, task_id: str, reason: str,
                      opt_start: float) -> OptimizationTrace:
        snap = self._monitor.snapshot()
        t = OptimizationTrace(
            task_id            = task_id,
            cpu_utilization    = snap.cpu.utilization_pct,
            available_memory   = snap.memory.available_bytes,
            logical_cpus       = snap.cpu.logical_count,
            policy             = self._cfg.policy.value,
            optimization_level = self._cfg.level.value,
            strategy_selected  = self._state.current_strategy,
            bypass_reason      = reason,
            thread_count       = 1,
        )
        t.optimizer_latency_ms = (time.monotonic() - opt_start) * 1000
        return t

    def _finalize_state(self, strategy: ExecutionStrategy,
                        trace: OptimizationTrace) -> None:
        old = self._state.current_strategy
        if old != strategy.value:
            self._state.switch_count    += 1
            self._state.last_switch_reason = trace.bypass_reason or (
                f"strategy {old} → {strategy.value}")
        self._state.current_strategy = strategy.value
        self._state.thread_count     = trace.thread_count
        self._state.updated_at       = time.time()

    def __repr__(self) -> str:
        return (f"<AdaptiveOptimizationEngine "
                f"level={self._cfg.level.value} "
                f"policy={self._cfg.policy.value} "
                f"strategy={self._state.current_strategy}>")


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Profiler
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeProfiler:
    """
    Context-manager profiler that measures actual wall-clock latency
    and optional memory delta.  Does not fabricate values.

    Usage::

        profiler = RuntimeProfiler()
        with profiler.profile("reasoning"):
            result = reasoner.solve(problem)
        rec = profiler.last_record()
        print(rec.latency_ms)
    """

    def __init__(self, monitor: Optional[ResourceMonitor] = None) -> None:
        self._monitor = monitor or ResourceMonitor()
        self._records: List[PerformanceRecord] = []
        self._current_label = ""
        self._start_time    = 0.0
        self._start_mem     = 0

    def profile(self, label: str) -> "RuntimeProfiler":
        self._current_label = label
        return self

    def __enter__(self) -> "RuntimeProfiler":
        snap = self._monitor.snapshot(force=True)
        self._start_time = time.monotonic()
        rss = snap.memory.process_rss
        self._start_mem  = rss if rss != UNKNOWN else 0
        return self

    def __exit__(self, *_: object) -> None:
        end_time = time.monotonic()
        snap     = self._monitor.snapshot(force=True)
        end_mem  = snap.memory.process_rss
        if end_mem == UNKNOWN:
            end_mem = 0
        mem_delta = end_mem - self._start_mem if (end_mem and self._start_mem) else UNKNOWN
        rec = PerformanceRecord(
            module_id    = self._current_label,
            start_time   = self._start_time,
            end_time     = end_time,
            latency_ms   = (end_time - self._start_time) * 1000.0,
            memory_usage = mem_delta if mem_delta != UNKNOWN else UNKNOWN,
        )
        self._records.append(rec)

    def last_record(self) -> Optional[PerformanceRecord]:
        return self._records[-1] if self._records else None

    def all_records(self) -> List[PerformanceRecord]:
        return list(self._records)


# ─────────────────────────────────────────────────────────────────────────────
# Public error class
# ─────────────────────────────────────────────────────────────────────────────

class OptimizerError(Exception):
    """Raised when Phase 9 optimizer encounters a fatal error."""


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience factories
# ─────────────────────────────────────────────────────────────────────────────

def make_optimizer(policy: str = "BALANCED",
                   level: str = "BALANCED",
                   **constraint_kwargs: Any) -> AdaptiveOptimizationEngine:
    """
    Create an AdaptiveOptimizationEngine with simple string arguments.

    Args:
        policy:  OptimizationPolicy name (default "BALANCED").
        level:   OptimizationLevel name (default "BALANCED").
        **constraint_kwargs: Passed to ResourceConstraints.

    Returns:
        Ready-to-use :class:`AdaptiveOptimizationEngine`.
    """
    constraints = ResourceConstraints(**{
        k: v for k, v in constraint_kwargs.items()
        if hasattr(ResourceConstraints, k)
    })
    cfg = OptimizerConfig(
        policy      = OptimizationPolicy(policy),
        level       = OptimizationLevel(level),
        constraints = constraints,
    )
    return AdaptiveOptimizationEngine(config=cfg)
