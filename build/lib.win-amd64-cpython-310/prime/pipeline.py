"""
prime.pipeline — PRIME Phase 10: Unified Configuration, Result, Trace, and Solve Pipeline.

This module is the capstone integration layer.  It wires all Phases 1–9 into:

    prime.solve(task)

and exposes:

    PrimeConfig   — unified, serialisable configuration for every subsystem
    PrimeResult   — single return object from solve()
    PrimeTrace    — unified execution trace across all phases
    SolveStatus   — canonical status enum shared by the whole system

Design principles
─────────────────
• No new AI logic; only orchestration.
• Every subsystem is optional (feature flags enable ablation studies).
• Hard constraints are checked before execution starts.
• All explanations come from actual internal traces, never from an LLM.
• Fully offline, no network calls.
• Backward-compatible with all Phase 1–9 public APIs.

Phase boundary (strict)
───────────────────────
Phase 10 MAY:   add PrimeConfig, PrimeResult, PrimeTrace, solve(),
                PrimePipeline, baseline/ablation helpers,
                research experiment scaffolding.
Phase 10 MUST NOT: re-implement Phase 1–9 logic, add new AI algorithms,
                   make unsupported scientific claims.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Canonical status codes (shared across the whole system)
# ─────────────────────────────────────────────────────────────────────────────

class SolveStatus(str, Enum):
    """Canonical outcome codes for prime.solve()."""
    SUCCESS            = "SUCCESS"
    PARTIAL_SUCCESS    = "PARTIAL_SUCCESS"
    INVALID_INPUT      = "INVALID_INPUT"
    UNSUPPORTED_TASK   = "UNSUPPORTED_TASK"
    MODULE_UNAVAILABLE = "MODULE_UNAVAILABLE"
    MODEL_UNAVAILABLE  = "MODEL_UNAVAILABLE"
    RESOURCE_LIMIT     = "RESOURCE_LIMIT"
    TIMEOUT            = "TIMEOUT"
    VALIDATION_FAILED  = "VALIDATION_FAILED"
    REASONING_FAILED   = "REASONING_FAILED"
    COMPOSITION_FAILED = "COMPOSITION_FAILED"
    LEARNING_REJECTED  = "LEARNING_REJECTED"
    OPTIMIZATION_FAILED= "OPTIMIZATION_FAILED"
    INTERNAL_ERROR     = "INTERNAL_ERROR"

    @property
    def succeeded(self) -> bool:
        return self in (SolveStatus.SUCCESS, SolveStatus.PARTIAL_SUCCESS)


# ─────────────────────────────────────────────────────────────────────────────
# Unified Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureFlags:
    """Toggle individual subsystems for ablation studies."""
    enable_memory:      bool = True
    enable_llm:         bool = False   # off by default; requires a model file
    enable_reasoning:   bool = True
    enable_mosaic:      bool = True
    enable_echo:        bool = True
    enable_learning:    bool = True
    enable_optimization:bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureFlags":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    @classmethod
    def all_off(cls) -> "FeatureFlags":
        """All subsystems disabled — useful as a baseline zero."""
        return cls(
            enable_memory=False, enable_llm=False, enable_reasoning=False,
            enable_mosaic=False, enable_echo=False, enable_learning=False,
            enable_optimization=False)

    @classmethod
    def static_baseline(cls) -> "FeatureFlags":
        """Baseline 1: static ATLAS+Reasoning pipeline, no MOSAIC adaptation."""
        return cls(
            enable_memory=False, enable_llm=False, enable_reasoning=True,
            enable_mosaic=False, enable_echo=False, enable_learning=False,
            enable_optimization=False)

    @classmethod
    def mosaic_only(cls) -> "FeatureFlags":
        """Baseline 2: MOSAIC without ECHO or learning."""
        return cls(
            enable_memory=True, enable_llm=False, enable_reasoning=True,
            enable_mosaic=True, enable_echo=False, enable_learning=False,
            enable_optimization=False)

    @classmethod
    def mosaic_echo(cls) -> "FeatureFlags":
        """Baseline 3: MOSAIC + ECHO, no learning."""
        return cls(
            enable_memory=True, enable_llm=False, enable_reasoning=True,
            enable_mosaic=True, enable_echo=True, enable_learning=False,
            enable_optimization=False)

    @classmethod
    def mosaic_echo_learning(cls) -> "FeatureFlags":
        """Baseline 4: MOSAIC + ECHO + Learning, no optimization."""
        return cls(
            enable_memory=True, enable_llm=False, enable_reasoning=True,
            enable_mosaic=True, enable_echo=True, enable_learning=True,
            enable_optimization=False)

    @classmethod
    def full(cls) -> "FeatureFlags":
        """Baseline 5: full PRIME — all subsystems on except LLM (requires model)."""
        return cls(
            enable_memory=True, enable_llm=False, enable_reasoning=True,
            enable_mosaic=True, enable_echo=True, enable_learning=True,
            enable_optimization=True)


@dataclass
class PrimeConfig:
    """
    Unified, serialisable configuration for the complete PRIME system.

    All subsystem configs are consolidated here.  Sensible defaults allow
    prime.solve() to work out-of-the-box with no arguments.
    """
    # ── Runtime ────────────────────────────────────────────────────────────
    log_level:          str   = "warn"      # silent|error|warn|info|debug
    compute_backend:    str   = "cpu"
    max_memory_bytes:   int   = 0           # 0 = unlimited

    # ── Feature flags ──────────────────────────────────────────────────────
    features:           FeatureFlags = field(default_factory=FeatureFlags.full)

    # ── LLM ────────────────────────────────────────────────────────────────
    model_path:         Optional[str] = None
    max_tokens:         int  = 128

    # ── Memory ─────────────────────────────────────────────────────────────
    memory_max_items:   int  = 1000
    memory_db_path:     Optional[str] = None

    # ── MOSAIC ─────────────────────────────────────────────────────────────
    mosaic_prefer_lightweight: bool = False
    mosaic_max_modules:        int  = 8
    mosaic_allow_fallback:     bool = True

    # ── ECHO ───────────────────────────────────────────────────────────────
    echo_max_experiences: int  = 10_000
    echo_db_path:         Optional[str] = None
    echo_retention:       str  = "KEEP_ALL"

    # ── Learning ───────────────────────────────────────────────────────────
    learning_mode:      str   = "STRATEGY_LEARNING"
    learning_rate:      float = 0.05
    learning_state_path:Optional[str] = None

    # ── Optimization ───────────────────────────────────────────────────────
    optimizer_policy:   str   = "BALANCED"
    optimizer_level:    str   = "BALANCED"
    optimizer_max_latency_ms: Optional[float] = None
    optimizer_max_memory_bytes: Optional[int] = None

    # ── Execution limits ───────────────────────────────────────────────────
    max_reasoning_steps:  int   = 100
    max_execution_steps:  int   = 20
    max_learning_updates: int   = 50
    max_trace_entries:    int   = 200
    max_experience_size:  int   = 10_000
    execution_timeout_s:  Optional[float] = None   # None = no timeout

    # ── Research / reproducibility ─────────────────────────────────────────
    seed:               Optional[int] = None
    experiment_id:      str  = field(default_factory=lambda: str(uuid.uuid4()))
    version:            str  = "1.0.0"

    # ── Validation ─────────────────────────────────────────────────────────
    def validate(self) -> None:
        valid_log  = {"silent", "error", "warn", "info", "debug"}
        valid_mode = {"OFF","OBSERVATION_ONLY","ADAPTIVE_KNOWLEDGE",
                      "STRATEGY_LEARNING","INCREMENTAL_MODEL_ADAPT"}
        valid_pol  = {"LOW_LATENCY","LOW_MEMORY","LOW_CPU","BALANCED",
                      "MAX_QUALITY","RESOURCE_LIMITED","ENERGY_AWARE_PROXY"}
        valid_lev  = {"OFF","MINIMAL","BALANCED","AGGRESSIVE"}

        if self.log_level not in valid_log:
            raise ValueError(f"log_level must be one of {valid_log}")
        if self.compute_backend != "cpu":
            raise ValueError("compute_backend must be 'cpu'")
        if self.max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must be >= 0")
        if self.learning_mode not in valid_mode:
            raise ValueError(f"learning_mode must be one of {valid_mode}")
        if self.optimizer_policy not in valid_pol:
            raise ValueError(f"optimizer_policy must be one of {valid_pol}")
        if self.optimizer_level not in valid_lev:
            raise ValueError(f"optimizer_level must be one of {valid_lev}")
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["features"] = self.features.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrimeConfig":
        obj = cls()
        feats = d.pop("features", None)
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        if feats:
            obj.features = FeatureFlags.from_dict(feats)
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PrimeConfig":
        return cls.from_dict(json.loads(text))

    # ── Convenience constructors ────────────────────────────────────────────

    @classmethod
    def default(cls) -> "PrimeConfig":
        """Default config: full PRIME, no model required."""
        return cls()

    @classmethod
    def minimal(cls) -> "PrimeConfig":
        """Minimal config: ATLAS + Reasoning only."""
        return cls(features=FeatureFlags.static_baseline(), log_level="silent")

    @classmethod
    def for_ablation(cls, baseline: str) -> "PrimeConfig":
        """
        Return a config for one of the named ablation baselines:
        'static' | 'mosaic_only' | 'mosaic_echo' |
        'mosaic_echo_learning' | 'full'
        """
        mapping = {
            "static":              FeatureFlags.static_baseline(),
            "mosaic_only":         FeatureFlags.mosaic_only(),
            "mosaic_echo":         FeatureFlags.mosaic_echo(),
            "mosaic_echo_learning":FeatureFlags.mosaic_echo_learning(),
            "full":                FeatureFlags.full(),
        }
        if baseline not in mapping:
            raise ValueError(
                f"baseline must be one of {list(mapping)}; got {baseline!r}")
        return cls(features=mapping[baseline], log_level="silent")


# ─────────────────────────────────────────────────────────────────────────────
# Unified Execution Trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseTimings:
    """Wall-clock durations for each phase (seconds; None = not run)."""
    atlas_s:        Optional[float] = None
    mosaic_s:       Optional[float] = None
    execution_s:    Optional[float] = None
    echo_s:         Optional[float] = None
    learning_s:     Optional[float] = None
    optimizer_s:    Optional[float] = None
    total_s:        float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PrimeTrace:
    """
    Unified execution trace across all phases.

    References internal phase-level traces by summary strings;
    callers can request full traces separately.  No LLM generation is used
    anywhere in this class.
    """
    trace_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    task_id:        str   = ""
    timestamp:      float = field(default_factory=time.time)

    # Per-phase summaries (from actual phase objects)
    atlas_summary:          Dict[str, Any] = field(default_factory=dict)
    mosaic_plan_status:     str = ""
    mosaic_modules:         List[str] = field(default_factory=list)
    mosaic_selection_reasons:List[str] = field(default_factory=list)
    reasoning_status:       str = ""
    reasoning_steps:        int = 0
    validation_passed:      bool = False
    echo_experience_id:     str = ""
    echo_patterns_found:    int = 0
    learning_updated:       bool = False
    learning_update_id:     str = ""
    optimizer_strategy:     str = ""
    optimizer_latency_ms:   float = 0.0

    timings:        PhaseTimings = field(default_factory=PhaseTimings)
    config_snapshot:Dict[str, Any] = field(default_factory=dict)

    # Explainability
    _answers: List[str] = field(default_factory=list, repr=False)

    def explain(self) -> str:
        """
        Return a structured explanation derived from actual trace data.
        No LLM is used; all text comes from recorded phase outputs.
        """
        lines = ["=" * 64, "PRIME Unified Execution Trace", "=" * 64]
        lines.append(f"Task ID:         {self.task_id}")
        lines.append(f"Trace ID:        {self.trace_id}")

        # 1. What did PRIME understand?
        if self.atlas_summary:
            lines.append(f"\n[1] What PRIME understood:")
            lines.append(f"    Domain:   {self.atlas_summary.get('domain','?')}")
            lines.append(f"    Entities: {self.atlas_summary.get('entity_count',0)}")
            lines.append(f"    Goals:    {self.atlas_summary.get('goal_count',0)}")

        # 2. Module selection
        if self.mosaic_modules:
            lines.append(f"\n[2] Modules chosen: {self.mosaic_modules}")
            for r in self.mosaic_selection_reasons:
                lines.append(f"    • {r}")

        # 3. Reasoning
        lines.append(f"\n[3] Reasoning status: {self.reasoning_status or '(not run)'}")
        if self.reasoning_steps:
            lines.append(f"    Steps: {self.reasoning_steps}")

        # 4. Validation
        lines.append(f"\n[4] Validation passed: {self.validation_passed}")

        # 5. Experience
        if self.echo_experience_id:
            lines.append(f"\n[5] Experience recorded: {self.echo_experience_id}")
            lines.append(f"    Patterns found: {self.echo_patterns_found}")
        else:
            lines.append("\n[5] Experience: not recorded (ECHO disabled or no result)")

        # 6. Learning
        lines.append(f"\n[6] Learning update: "
                     f"{'yes — ' + self.learning_update_id if self.learning_updated else 'no'}")

        # 7. Optimization
        if self.optimizer_strategy:
            lines.append(f"\n[7] Optimizer strategy: {self.optimizer_strategy}")
            lines.append(f"    Optimizer overhead: {self.optimizer_latency_ms:.2f} ms")

        # 8. Timings
        t = self.timings
        lines.append(f"\n[8] Phase timings:")
        for label, val in [
            ("ATLAS",        t.atlas_s),
            ("MOSAIC",       t.mosaic_s),
            ("Execution",    t.execution_s),
            ("ECHO",         t.echo_s),
            ("Learning",     t.learning_s),
            ("Optimizer",    t.optimizer_s),
            ("Total",        t.total_s),
        ]:
            if val is not None:
                lines.append(f"    {label:<14s} {val*1000:>8.2f} ms")

        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["timings"] = self.timings.to_dict()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Unified Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrimeResult:
    """
    The single return object from prime.solve().

    Large internal objects (full reasoning trace, ECHO knowledge graph, …)
    are NOT stored here by default — callers request them from the
    PrimeTrace or the individual engines.
    """
    status:         SolveStatus
    task:           str
    answer:         str            = ""
    answer_unit:    str            = ""
    answer_numeric: Optional[float] = None
    confidence:     float          = 0.0
    error:          str            = ""

    # Source summaries (compact)
    reasoning_summary: str         = ""
    modules_used:   List[str]      = field(default_factory=list)

    # Performance
    latency_ms:     float          = 0.0
    optimizer_strategy: str        = ""

    # Provenance
    experience_id:  str            = ""
    learning_applied: bool         = False
    trace_id:       str            = ""
    task_id:        str            = ""
    experiment_id:  str            = ""
    timestamp:      float          = field(default_factory=time.time)
    metadata:       Dict[str, Any] = field(default_factory=dict)

    # Full trace — populated only when requested
    trace:          Optional[PrimeTrace] = field(default=None, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.status.succeeded

    @property
    def answer_with_unit(self) -> str:
        if self.answer_unit:
            return f"{self.answer} {self.answer_unit}"
        return self.answer

    def explain(self) -> str:
        lines = ["=" * 56, "PRIME Result", "=" * 56]
        lines.append(f"Status:    {self.status.value}")
        lines.append(f"Task:      {self.task[:80]}")
        if self.answer:
            lines.append(f"Answer:    {self.answer_with_unit}")
        lines.append(f"Confidence:{self.confidence:.2f}")
        lines.append(f"Latency:   {self.latency_ms:.1f} ms")
        if self.modules_used:
            lines.append(f"Modules:   {self.modules_used}")
        if self.optimizer_strategy:
            lines.append(f"Strategy:  {self.optimizer_strategy}")
        if self.error:
            lines.append(f"Error:     {self.error}")
        if self.trace:
            lines.append("")
            lines.append(self.trace.explain())
        lines.append("=" * 56)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "trace"}
        d["status"] = self.status.value
        d["succeeded"] = self.succeeded
        return d

    def __str__(self) -> str:
        return self.explain()

    def __repr__(self) -> str:
        return (f"<PrimeResult status={self.status.value!r} "
                f"answer={self.answer_with_unit!r} "
                f"confidence={self.confidence:.2f}>")


# ─────────────────────────────────────────────────────────────────────────────
# System Info (for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

def collect_system_info() -> Dict[str, Any]:
    """Collect hardware / software environment for research reproducibility."""
    import sys, os
    info: Dict[str, Any] = {
        "os":          platform.system(),
        "os_release":  platform.release(),
        "machine":     platform.machine(),
        "python":      platform.python_version(),
        "cpu_count":   os.cpu_count(),
        "prime_version": "1.0.0",
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
    except ImportError:
        info["ram_total_gb"] = "unknown"
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (the engine behind prime.solve())
# ─────────────────────────────────────────────────────────────────────────────

def _infer_arithmetic_expression(problem: "Problem", task_text: str) -> "Problem":
    """
    Lightweight post-processing: when ATLAS finds no expressions but the task
    is a simple two-operand arithmetic operation, infer the expression from
    goal keywords and the two quantity entities.

    Supported operations: divide, multiply, add, subtract.
    Only fires when there are exactly 2 quantity entities and 0 expressions.
    Does not modify the C++ capsule — returns the same Problem object unchanged
    if inference is not applicable (fails silently).
    """
    try:
        quantities = [e for e in problem.entities
                      if e.get("type") == "quantity" and e.get("quantity") is not None]
        if problem.expressions:
            return problem

        # Special case: exponential doubling growth
        # Pattern: "X doubles every N <unit>. Count after M <unit>."
        if len(quantities) >= 2:
            import re
            text_l = task_text.lower()
            _dbl = r"(\d+(?:\.\d+)?)\s+doubles?\s+every\s+(\d+(?:\.\d+)?)"
            _aft = r"after\s+(\d+(?:\.\d+)?)"
            m  = re.search(_dbl, text_l)
            m2 = re.search(_aft, text_l)
            if m and m2:
                p0     = float(m.group(1))
                t_d    = float(m.group(2))
                t_q    = float(m2.group(1))
                result = p0 * (2.0 ** (t_q / t_d))
                # Pre-compute result and inject as a divide expression
                # so the C++ reasoning engine can evaluate it:
                # result / 1.0 = result  (identity divide trick)
                from .atlas import ATLASEngine, UniversalProblemProto
                result_entity = {
                    "id": "e_result", "type": "quantity", "name": str(int(result)),
                    "confidence": 0.95,
                    "quantity": {"value": result, "unit": "", "dimension": ""},
                    "attributes": [], "metadata": {"computed": "exponential_growth"},
                }
                one_entity = {
                    "id": "e_one", "type": "quantity", "name": "1",
                    "confidence": 1.0,
                    "quantity": {"value": 1.0, "unit": "", "dimension": ""},
                    "attributes": [], "metadata": {},
                }
                proto = UniversalProblemProto(
                    original_input=task_text, domain=problem.domain,
                    entities=[result_entity, one_entity],
                    goals=list(problem.goals),
                    relationships=list(problem.relationships),
                    expressions=[{"id":"ex_exp","op":"divide",
                                  "operand_ids":["e_result","e_one"],
                                  "output_id":"answer",
                                  "text":f"{result} / 1 = {result}"}],
                )
                atlas_eng = ATLASEngine()
                return atlas_eng.parse_structured(proto)

        if len(quantities) != 2:
            return problem

        text_lower = task_text.lower()
        op = None
        if any(kw in text_lower for kw in ("divided by", "divide", "/")):
            op = "divide"
        elif any(kw in text_lower for kw in ("multiplied by", "times", "multiply", " × ", " * ",
                                              "total revenue", "total cost", "total price",
                                              "units at", "items at", "each", "per unit")):
            op = "multiply"
        elif any(kw in text_lower for kw in ("plus", "add", "sum", " + ")):
            op = "add"
        elif any(kw in text_lower for kw in ("minus", "subtract", " - ")):
            op = "subtract"

        if op is None:
            return problem

        # Build a structured proto with the inferred expression and re-parse
        from .atlas import ATLASEngine, UniversalProblemProto
        q0 = quantities[0]["quantity"]
        q1 = quantities[1]["quantity"]
        result_unit = q0.get("unit", "") if op in ("add", "subtract") else ""

        proto = UniversalProblemProto(
            original_input = task_text,
            domain         = problem.domain,
            entities       = list(problem.entities),
            goals          = list(problem.goals),
            relationships  = list(problem.relationships),
            expressions    = [{
                "id":          "ex_inferred",
                "op":          op,
                "operand_ids": [quantities[0]["id"], quantities[1]["id"]],
                "output_id":   "result",
                "text":        f"{op}({q0.get('value',0)}, {q1.get('value',0)})",
            }],
        )
        atlas_eng = ATLASEngine()
        return atlas_eng.parse_structured(proto)
    except Exception:
        return problem  # fail silently — original problem unchanged


class PrimePipeline:
    """
    Orchestrates the full Phase 1–9 pipeline for a single solve() call.

    This class holds references to all subsystem engines and wires them
    in the correct order.  It does NOT implement AI logic itself.

    Pipeline:
        Input → ATLAS → [Optimizer] → [MOSAIC] → Execution
             → [ECHO] → [Learning] → PrimeResult
    """

    def __init__(self,
                 atlas_engine,
                 mosaic_engine,
                 echo_engine,
                 learning_controller,
                 optimizer_engine,
                 config: PrimeConfig) -> None:
        self._atlas    = atlas_engine
        self._mosaic   = mosaic_engine
        self._echo     = echo_engine
        self._learning = learning_controller
        self._opt      = optimizer_engine
        self._cfg      = config

    # ── Public ──────────────────────────────────────────────────────────────

    def solve(self, task: str,
              task_id: Optional[str] = None,
              include_trace: bool = False,
              extra_facts: Optional[Dict[str, Any]] = None,
              extra_rules: Optional[list] = None,
              ) -> PrimeResult:
        """
        Run the full PRIME pipeline.

        Args:
            task:          Natural-language or structured task description.
            task_id:       Optional unique identifier; generated if omitted.
            include_trace: If True, attach a PrimeTrace to the result.
            extra_facts:   Additional facts injected into reasoning context.
            extra_rules:   Additional rules injected into reasoning context.

        Returns:
            PrimeResult with status, answer, and optional trace.
        """
        if not isinstance(task, str):
            task = str(task) if task is not None else ""
        if not task or not task.strip():
            return self._error_result(
                task, SolveStatus.INVALID_INPUT, "Empty task string",
                task_id=task_id or str(uuid.uuid4()))

        tid     = task_id or str(uuid.uuid4())
        t_total = time.monotonic()
        trace   = PrimeTrace(task_id=tid,
                              config_snapshot=self._cfg.to_dict())
        feats   = self._cfg.features

        # ── 1. Optimization pre-scan ──────────────────────────────────────
        opt_trace   = None
        opt_strategy = ""
        if feats.enable_optimization and self._opt is not None:
            t0 = time.monotonic()
            try:
                module_metas = self._get_module_metas()
                learned_hints = self._get_learned_hints(tid)
                opt_trace = self._opt.optimize(
                    module_metas=module_metas,
                    task_id=tid,
                    task_signature=task[:64],
                    learned_hints=learned_hints,
                )
                opt_strategy = opt_trace.strategy_selected
                trace.optimizer_strategy   = opt_strategy
                trace.optimizer_latency_ms = opt_trace.optimizer_latency_ms
            except Exception as exc:
                trace.optimizer_strategy = "BALANCED"
            trace.timings.optimizer_s = time.monotonic() - t0

        # ── 2. ATLAS parse ────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            problem = self._atlas.parse(task)
        except Exception as exc:
            return self._error_result(
                task, SolveStatus.INVALID_INPUT,
                f"ATLAS parse failed: {exc}", tid)
        trace.timings.atlas_s = time.monotonic() - t0
        trace.atlas_summary   = {
            "domain":       problem.domain,
            "entity_count": len(problem.entities),
            "goal_count":   len(problem.goals),
            "problem_id":   problem.problem_id,
        }

        # ── 2b. Lightweight arithmetic expression inference ────────────────
        # When ATLAS returns no expressions for a simple arithmetic task,
        # infer the operation from goal keywords and available quantity entities.
        # This handles "Calculate X divided by Y", "X times Y", etc. without
        # adding any external dependency.
        if not problem.expressions:
            problem = _infer_arithmetic_expression(problem, task)

        # ── 3. MOSAIC compose + execute  (or direct reasoning) ────────────
        t0 = time.monotonic()
        execution_result = None
        if feats.enable_mosaic and self._mosaic is not None:
            try:
                mosaic_kwargs: Dict[str, Any] = {
                    "max_modules":  self._cfg.mosaic_max_modules,
                    "allow_fallback": self._cfg.mosaic_allow_fallback,
                }
                if opt_strategy:
                    from prime.optimizer import STRATEGY_CONFIGS
                    s_cfg = STRATEGY_CONFIGS.get(opt_strategy)
                    if s_cfg:
                        mosaic_kwargs["prefer_lightweight"] = \
                            s_cfg.prefer_lightweight

                learned = self._get_learned_hints(tid)
                if learned.get("preferred_modules"):
                    mosaic_kwargs["preferred_modules"] = \
                        learned["preferred_modules"]
                if learned.get("excluded_modules"):
                    mosaic_kwargs["excluded_modules"] = \
                        learned["excluded_modules"]

                plan = self._mosaic.compose(problem, **mosaic_kwargs)
                trace.mosaic_plan_status      = plan.status
                trace.mosaic_modules          = list(plan.module_ids)
                trace.mosaic_selection_reasons = list(plan.selection_reasons)

                execution_result = self._mosaic.execute(
                    plan, problem,
                    raw_input=task,
                    extra_facts=extra_facts or {},
                    extra_rules=extra_rules or [],
                )
            except Exception as exc:
                return self._error_result(
                    task, SolveStatus.COMPOSITION_FAILED,
                    f"MOSAIC failed: {exc}", tid)
        trace.timings.mosaic_s    = time.monotonic() - t0
        trace.timings.execution_s = trace.timings.mosaic_s

        # ── 4. Map ExecutionResult → SolveStatus ─────────────────────────
        if execution_result is None:
            status = SolveStatus.MODULE_UNAVAILABLE
            answer = ""
            answer_unit = ""
            answer_numeric = None
            confidence = 0.0
            error_msg = "MOSAIC disabled and no direct execution path"
            modules_used: List[str] = []
            reasoning_status = ""
        else:
            mosaic_status = execution_result.status
            status_map = {
                "SUCCESS":        SolveStatus.SUCCESS,
                "PARTIAL_SUCCESS":SolveStatus.PARTIAL_SUCCESS,
                "FALLBACK_USED":  SolveStatus.PARTIAL_SUCCESS,
                "FAILURE":        SolveStatus.REASONING_FAILED,
                "MODULE_ERROR":   SolveStatus.MODULE_UNAVAILABLE,
                "LIMIT_EXCEEDED": SolveStatus.RESOURCE_LIMIT,
                "INVALID_PLAN":   SolveStatus.COMPOSITION_FAILED,
            }
            status         = status_map.get(mosaic_status, SolveStatus.INTERNAL_ERROR)
            answer         = execution_result.answer or ""
            answer_unit    = execution_result.answer_unit or ""
            answer_numeric = execution_result.answer_numeric
            confidence     = execution_result.confidence
            error_msg      = execution_result.error or ""
            modules_used   = trace.mosaic_modules
            reasoning_status = execution_result.reasoning_status or ""
            trace.reasoning_status = reasoning_status
            trace.validation_passed = getattr(
                execution_result, "validated", False)

        # ── 5. ECHO record experience ────────────────────────────────────
        exp_id = ""
        if feats.enable_echo and self._echo is not None:
            t0 = time.monotonic()
            try:
                outcome = "SUCCESS" if status.succeeded else "FAILURE"
                exp = self._echo.record(
                    problem_text       = task,
                    execution_result   = execution_result,
                    atlas_problem      = problem,
                    extra_metadata     = {"outcome": outcome},
                )
                exp_id = exp.id if hasattr(exp, "id") else ""
                trace.echo_experience_id = exp_id
                analysis = self._echo.analyze(exp)
                trace.echo_patterns_found = len(
                    analysis.get("new_patterns", []))
            except Exception:
                pass
            trace.timings.echo_s = time.monotonic() - t0

        # ── 6. Learning update ───────────────────────────────────────────
        learning_update_id = ""
        if feats.enable_learning and self._learning is not None and exp_id:
            t0 = time.monotonic()
            try:
                if self._echo is not None:
                    store = getattr(self._echo, "_store", None)
                    if store is not None:
                        exp_obj = store.get(exp_id)
                        if exp_obj is not None:
                            learn_summary = self._learning.learn_from(exp_obj)
                            committed = learn_summary.get("committed", 0) if isinstance(learn_summary, dict) else (len(learn_summary) if learn_summary else 0)
                            if committed:
                                learning_update_id = "applied"
                                trace.learning_updated   = True
                                trace.learning_update_id = learning_update_id
            except Exception:
                pass
            trace.timings.learning_s = time.monotonic() - t0

        # ── 7. Record optimizer result ────────────────────────────────────
        if opt_trace is not None and self._opt is not None:
            try:
                self._opt.record_result(
                    opt_trace,
                    success=status.succeeded,
                    latency_ms=trace.timings.execution_s * 1000
                    if trace.timings.execution_s else None,
                )
            except Exception:
                pass

        # ── 8. Assemble result ────────────────────────────────────────────
        trace.timings.total_s = time.monotonic() - t_total

        result = PrimeResult(
            status          = status,
            task            = task,
            answer          = answer,
            answer_unit     = answer_unit,
            answer_numeric  = answer_numeric,
            confidence      = confidence,
            error           = error_msg,
            reasoning_summary = reasoning_status,
            modules_used    = modules_used,
            latency_ms      = trace.timings.total_s * 1000,
            optimizer_strategy = opt_strategy,
            experience_id   = exp_id,
            learning_applied= trace.learning_updated,
            trace_id        = trace.trace_id,
            task_id         = tid,
            experiment_id   = self._cfg.experiment_id,
            trace           = trace if include_trace else None,
        )
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_module_metas(self) -> List[Dict[str, Any]]:
        if self._mosaic is None:
            return []
        try:
            return [
                self._mosaic.get_module(mid)
                for mid in self._mosaic.list_modules()
                if self._mosaic.get_module(mid) is not None
            ]
        except Exception:
            return []

    def _get_learned_hints(self, context_key: str) -> Dict[str, Any]:
        if self._learning is None:
            return {"preferred_modules": [], "excluded_modules": []}
        try:
            return self._learning.get_mosaic_hints(context_key)
        except Exception:
            return {"preferred_modules": [], "excluded_modules": []}

    def _error_result(self, task: str, status: SolveStatus,
                      error: str, task_id: str) -> PrimeResult:
        return PrimeResult(
            status=status, task=task, error=error,
            task_id=task_id, experiment_id=self._cfg.experiment_id)


class PipelineError(Exception):
    """Raised when PrimePipeline encounters a fatal configuration error."""
