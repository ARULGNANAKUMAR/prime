"""
prime.mosaic — PRIME Phase 6: MOSAIC Dynamic Intelligence Composition Engine.

MOSAIC = Modular Orchestration / Selection and Adaptive Intelligence Composition

MOSAIC answers: "Which available intelligence modules are required for this task,
                 and in what order should they operate?"

It does NOT solve the task itself — existing PRIME modules do that.

Typical usage::

    from prime import PRIME

    prime    = PRIME()
    atlas    = prime.atlas()
    mosaic   = prime.mosaic()

    problem  = atlas.parse("A car travels 150 km in 3 hours.")
    plan     = mosaic.compose(problem)
    print(plan.explain())

    result   = mosaic.execute(plan, problem)
    print(result)

    # Or in one call:
    result = mosaic.solve("A car travels 150 km in 3 hours.")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import _prime_mosaic


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class MOSAICError(Exception):
    """Raised when MOSAIC composition or execution fails."""


# ---------------------------------------------------------------------------
# Data classes mirroring C++ structs
# ---------------------------------------------------------------------------
@dataclass
class Capability:
    """A named cognitive capability with optional category and type info."""
    name:        str
    category:    str = "general"
    description: str = ""
    input_types: List[str] = field(default_factory=list)
    output_types:List[str] = field(default_factory=list)
    confidence:  float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "category": self.category,
                "description": self.description,
                "input_types": self.input_types,
                "output_types": self.output_types,
                "confidence": self.confidence}


@dataclass
class ResourceCost:
    cpu_relative:    float = 1.0
    memory_relative: float = 1.0
    latency_ms:      float = 0.0
    requires_model:  bool  = False

    def to_dict(self) -> Dict[str, Any]:
        return {"cpu_relative": self.cpu_relative,
                "memory_relative": self.memory_relative,
                "latency_ms": self.latency_ms,
                "requires_model": self.requires_model}


@dataclass
class ModuleMetadata:
    """Full metadata for a MOSAIC cognitive module."""
    id:           str
    name:         str
    capabilities: List[Capability] = field(default_factory=list)
    input_types:  List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    cost:         ResourceCost = field(default_factory=ResourceCost)
    priority:     int  = 0
    version:      str  = "1.0"
    deterministic:bool = True
    is_fallback:  bool = False
    fallback_for: str  = ""
    status:       str  = "available"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "version": self.version,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "input_types": self.input_types, "output_types": self.output_types,
            "dependencies": self.dependencies, "cost": self.cost.to_dict(),
            "priority": self.priority, "deterministic": self.deterministic,
            "is_fallback": self.is_fallback, "fallback_for": self.fallback_for,
            "status": self.status,
        }


@dataclass
class TaskRequirements:
    task_id:               str
    domain:                str
    required_capabilities: List[str] = field(default_factory=list)
    optional_capabilities: List[str] = field(default_factory=list)
    needs_memory:          bool = False
    needs_llm:             bool = False
    has_arithmetic:        bool = False
    has_rules:             bool = False
    has_constraints:       bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CompositionTraceStep:
    step_index:       int
    module_id:        str
    module_name:      str
    status:           str = "pending"  # "success" | "failure" | "fallback" | "skipped"
    input_summary:    str = ""
    output_summary:   str = ""
    latency_ms:       float = 0.0
    error:            str = ""
    fallback_used:    bool = False
    fallback_module_id: str = ""


class CompositionPlan:
    """
    A validated MOSAIC pipeline plan.

    Attributes:
        task_id:           Unique task identifier.
        domain:            Problem domain from ATLAS.
        status:            ``"VALID"`` or an error status string.
        module_ids:        Ordered list of module IDs to execute.
        selection_reasons: One-line reason per selected module.
        estimated_cost:    Sum of resource costs.
        requirements:      The :class:`TaskRequirements` that drove selection.
        valid:             True iff ``status == "VALID"``.
    """

    def __init__(self, raw: Dict[str, Any], engine: "MOSAICEngine") -> None:
        self._raw     = raw
        self._engine  = engine
        self.task_id            = raw.get("task_id", "")
        self.domain             = raw.get("domain", "")
        self.status             = raw.get("status", "EMPTY_PLAN")
        self.status_detail      = raw.get("status_detail", "")
        self.module_ids         = raw.get("module_ids", [])
        self.selection_reasons  = raw.get("selection_reasons", [])
        self.estimated_cost     = raw.get("estimated_cost", 0.0)
        self.valid:  bool       = raw.get("valid", False)
        req_raw = raw.get("requirements", {})
        self.requirements = TaskRequirements(
            task_id               = req_raw.get("task_id", self.task_id),
            domain                = req_raw.get("domain", self.domain),
            required_capabilities = req_raw.get("required_capabilities", []),
            optional_capabilities = req_raw.get("optional_capabilities", []),
            needs_memory          = req_raw.get("needs_memory", False),
            needs_llm             = req_raw.get("needs_llm", False),
            has_arithmetic        = req_raw.get("has_arithmetic", False),
            has_rules             = req_raw.get("has_rules", False),
            has_constraints       = req_raw.get("has_constraints", False),
        )

    def explain(self) -> str:
        """Generate a human-readable explanation from module metadata and
        selection logic — not from a generative model."""
        return _prime_mosaic.explain_plan(self._engine._capsule, self._raw)

    def __repr__(self) -> str:
        mods = " → ".join(self.module_ids) if self.module_ids else "(none)"
        return f"<CompositionPlan status={self.status!r} modules=[{mods}]>"

    def __str__(self) -> str:
        return self.explain()


class ExecutionResult:
    """Result of a MOSAIC pipeline execution."""

    def __init__(self, status: str, answer: str = "", answer_unit: str = "",
                 answer_numeric: Optional[float] = None,
                 confidence: float = 0.0, error: str = "",
                 trace_steps: Optional[List[CompositionTraceStep]] = None,
                 derived_facts: Optional[Dict[str, Any]] = None,
                 goal_results: Optional[List[Dict]] = None,
                 reasoning_status: str = "") -> None:
        self.status            = status
        self.answer            = answer
        self.answer_unit       = answer_unit
        self.answer_numeric    = answer_numeric
        self.confidence        = confidence
        self.error             = error
        self.trace_steps:  List[CompositionTraceStep] = trace_steps or []
        self.derived_facts:Dict[str, Any]             = derived_facts or {}
        self.goal_results: List[Dict]                 = goal_results or []
        self.reasoning_status  = reasoning_status

    @property
    def answer_with_unit(self) -> str:
        if self.answer_unit:
            return f"{self.answer} {self.answer_unit}"
        return self.answer

    @property
    def succeeded(self) -> bool:
        return self.status in ("SUCCESS", "PARTIAL_SUCCESS", "FALLBACK_USED")

    def explain(self) -> str:
        lines = ["=" * 56, "MOSAIC Execution Result", "=" * 56]
        lines.append(f"Status:     {self.status}")
        if self.answer:
            lines.append(f"Answer:     {self.answer_with_unit}")
        lines.append(f"Confidence: {self.confidence:.2f}")
        if self.error:
            lines.append(f"Error:      {self.error}")
        if self.trace_steps:
            lines.append("\nExecution steps:")
            for s in self.trace_steps:
                icon = "✓" if s.status == "success" else ("↩" if s.fallback_used else "✗")
                lines.append(f"  {icon} [{s.step_index+1}] {s.module_name} — {s.status}"
                             + (f" ({s.latency_ms:.1f} ms)" if s.latency_ms else ""))
                if s.output_summary:
                    lines.append(f"      → {s.output_summary}")
                if s.error:
                    lines.append(f"      Error: {s.error}")
        lines.append("=" * 56)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<ExecutionResult status={self.status!r} "
                f"answer={self.answer_with_unit!r} "
                f"confidence={self.confidence:.2f}>")

    def __str__(self) -> str:
        return self.explain()


# ---------------------------------------------------------------------------
# Built-in module adapters
# ---------------------------------------------------------------------------

# Each adapter is a callable: (context_dict) → updated_context_dict
# context_dict always carries at least:
#   "atlas_problem": the Problem object (or None)
#   "raw_input":     the original string
#   "reasoning_result": a ReasoningResult (after reasoning module runs)

ModuleExecutorFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _make_cap(name: str, category: str = "general",
              inp: Optional[List[str]] = None,
              out: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"name": name, "category": category,
            "input_types": inp or [], "output_types": out or [],
            "description": name, "confidence": 1.0}


def _atlas_module_meta() -> Dict[str, Any]:
    return {
        "id": "atlas", "name": "ATLAS Representation Engine", "version": "4.0",
        "capabilities": [
            _make_cap("atlas_representation", "representation",
                      ["raw_text"], ["atlas_problem"]),
        ],
        "input_types":  ["raw_text"],
        "output_types": ["atlas_problem"],
        "dependencies": [],
        "cost": {"cpu_relative": 1.0, "memory_relative": 1.0,
                 "latency_ms": 5.0, "requires_model": False},
        "priority": 100, "deterministic": True,
    }


def _reasoning_module_meta() -> Dict[str, Any]:
    return {
        "id": "reasoning", "name": "Hybrid Reasoning Engine", "version": "5.0",
        "capabilities": [
            _make_cap("reasoning",           "symbolic_reasoning",
                      ["atlas_problem"], ["reasoning_result"]),
            _make_cap("symbolic_reasoning",  "symbolic_reasoning",
                      ["atlas_problem"], ["reasoning_result"]),
            _make_cap("arithmetic",          "arithmetic",
                      ["atlas_problem"], ["reasoning_result"]),
            _make_cap("rule_reasoning",      "rule_reasoning",
                      ["atlas_problem"], ["reasoning_result"]),
            _make_cap("constraint_checking", "constraint_checking",
                      ["atlas_problem"], ["reasoning_result"]),
        ],
        "input_types":  ["atlas_problem"],
        "output_types": ["reasoning_result"],
        "dependencies": ["atlas"],
        "cost": {"cpu_relative": 2.0, "memory_relative": 2.0,
                 "latency_ms": 50.0, "requires_model": False},
        "priority": 90, "deterministic": True,
    }


def _memory_module_meta() -> Dict[str, Any]:
    return {
        "id": "memory", "name": "Memory Engine", "version": "3.0",
        "capabilities": [
            _make_cap("memory_retrieval", "retrieval",
                      ["*"], ["memory_context"]),
            _make_cap("memory_storage",   "storage",
                      ["*"], ["memory_context"]),
        ],
        "input_types":  ["*"],
        "output_types": ["memory_context", "atlas_problem"],
        "dependencies": [],
        "cost": {"cpu_relative": 0.5, "memory_relative": 1.5,
                 "latency_ms": 2.0, "requires_model": False},
        "priority": 70, "deterministic": True,
    }


def _validation_module_meta() -> Dict[str, Any]:
    return {
        "id": "validator", "name": "Result Validator", "version": "5.0",
        "capabilities": [
            _make_cap("validation", "validation",
                      ["reasoning_result"], ["validated_result"]),
        ],
        "input_types":  ["reasoning_result"],
        "output_types": ["validated_result"],
        "dependencies": ["reasoning"],
        "cost": {"cpu_relative": 0.2, "memory_relative": 0.2,
                 "latency_ms": 1.0, "requires_model": False},
        "priority": 60, "deterministic": True,
    }


def _llm_module_meta(available: bool) -> Dict[str, Any]:
    return {
        "id": "llm", "name": "Local LLM Engine", "version": "2.0",
        "capabilities": [
            _make_cap("text_generation",       "language",
                      ["*"], ["text_output"]),
            _make_cap("language_interpretation","language",
                      ["raw_text"], ["atlas_problem"]),
        ],
        "input_types":  ["*"],
        "output_types": ["text_output", "atlas_problem"],
        "dependencies": [],
        "cost": {"cpu_relative": 10.0, "memory_relative": 8.0,
                 "latency_ms": 2000.0, "requires_model": True},
        "priority": 30, "deterministic": False,
        "status": "available" if available else "unavailable",
    }


# ---------------------------------------------------------------------------
# MOSAIC Execution Engine (Python side)
# ---------------------------------------------------------------------------
class _ExecutionEngine:
    """Executes a CompositionPlan by invoking module adapter functions in order."""

    def __init__(self, adapters: Dict[str, ModuleExecutorFn],
                 fallbacks: Dict[str, str],
                 cfg: Dict[str, Any]) -> None:
        self._adapters  = adapters   # module_id → callable
        self._fallbacks = fallbacks  # module_id → fallback_module_id
        self._cfg       = cfg

    def execute(self, plan: CompositionPlan,
                initial_context: Dict[str, Any]) -> ExecutionResult:
        if not plan.valid:
            return ExecutionResult(
                status="INVALID_PLAN",
                error=f"{plan.status}: {plan.status_detail}")

        ctx         = dict(initial_context)
        trace_steps = []
        max_retries = self._cfg.get("max_retries", 1)
        t_start     = time.perf_counter()

        for i, mod_id in enumerate(plan.module_ids):
            step = CompositionTraceStep(
                step_index  = i,
                module_id   = mod_id,
                module_name = mod_id,
            )
            t0 = time.perf_counter()

            # Try primary adapter, then fallback
            adapter = self._adapters.get(mod_id)
            used_id = mod_id

            if adapter is None:
                step.status = "skipped"
                step.error  = f"No adapter for module '{mod_id}'"
                trace_steps.append(step)
                continue

            last_err = ""
            success  = False
            for attempt in range(max_retries + 1):
                try:
                    ctx     = adapter(ctx)
                    success = True
                    break
                except Exception as exc:
                    last_err = str(exc)
                    # Try fallback on first failure
                    if attempt == 0 and self._cfg.get("allow_fallback", True):
                        fb_id = self._fallbacks.get(used_id)
                        if fb_id and fb_id in self._adapters:
                            step.fallback_used      = True
                            step.fallback_module_id = fb_id
                            used_id  = fb_id
                            adapter  = self._adapters[fb_id]
                            continue
                    break

            step.latency_ms = (time.perf_counter() - t0) * 1000
            step.status     = "success" if success else "failure"
            if not success:
                step.error = last_err

            # Summarise context for trace
            rr = ctx.get("reasoning_result")
            if rr is not None:
                step.output_summary = (
                    f"status={getattr(rr, 'status', '?')} "
                    f"answer={getattr(rr, 'answer', '?')}"
                )

            trace_steps.append(step)

            if not success:
                total_ms = (time.perf_counter() - t_start) * 1000
                return ExecutionResult(
                    status="MODULE_ERROR",
                    error=f"Module '{mod_id}' failed: {last_err}",
                    trace_steps=trace_steps)

        # Collect final result
        total_ms = (time.perf_counter() - t_start) * 1000
        fallback_used = any(s.fallback_used for s in trace_steps)

        rr = ctx.get("reasoning_result")
        if rr is not None:
            ans_numeric = None
            if getattr(rr, "answer_is_numeric", False):
                ans_numeric = getattr(rr, "answer_numeric", None)
            # Map reasoning engine status → ExecutionResult status so that
            # callers can distinguish a genuine answer from a failed attempt.
            _r_status = getattr(rr, "status", "")
            _REASONING_STATUS_MAP = {
                "SOLVED":                  "SUCCESS",
                "PARTIALLY_SOLVED":        "PARTIAL_SUCCESS",
                "UNRESOLVED":              "PARTIAL_SUCCESS",
                "INVALID_INPUT":           "FAILURE",
                "CONSTRAINT_VIOLATION":    "FAILURE",
                "REASONING_LIMIT_REACHED": "LIMIT_EXCEEDED",
                "CIRCULAR_REASONING":      "FAILURE",
            }
            _exec_status = _REASONING_STATUS_MAP.get(_r_status, "SUCCESS")
            if fallback_used and _exec_status == "SUCCESS":
                _exec_status = "FALLBACK_USED"
            return ExecutionResult(
                status=_exec_status,
                answer          = getattr(rr, "answer", ""),
                answer_unit     = getattr(rr, "answer_unit", ""),
                answer_numeric  = ans_numeric,
                confidence      = getattr(rr, "confidence", 0.0),
                trace_steps     = trace_steps,
                derived_facts   = getattr(rr, "derived_facts", {}),
                goal_results    = [{"id": g.id, "target": g.target,
                                    "satisfied": g.satisfied,
                                    "derived_value": g.derived_value,
                                    "unit": g.unit}
                                   for g in getattr(rr, "goal_results", [])],
                reasoning_status= getattr(rr, "status", ""),
            )

        # If no reasoning result but execution completed, return partial
        return ExecutionResult(
            status="PARTIAL_SUCCESS" if ctx.get("memory_retrieved") else "SUCCESS",
            trace_steps=trace_steps,
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# MOSAICEngine — main public class
# ---------------------------------------------------------------------------
class MOSAICEngine:
    """
    PRIME Phase 6 MOSAIC Dynamic Intelligence Composition Engine.

    Dynamically selects and composes available cognitive modules to solve
    a given task represented as an ATLAS :class:`~prime.atlas.Problem`.

    Usage::

        mosaic  = prime.mosaic()
        problem = prime.atlas().parse("A car travels 150 km in 3 hours.")

        plan   = mosaic.compose(problem)
        print(plan.explain())

        result = mosaic.execute(plan, problem)
        print(result)

        # Single-call shortcut:
        result = mosaic.solve("A car travels 150 km in 3 hours.")
    """

    def __init__(self, prime_instance=None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        self._prime   = prime_instance
        self._cfg     = config or {}
        self._capsule = _prime_mosaic.create_engine(self._cfg)
        self._adapters:  Dict[str, ModuleExecutorFn] = {}
        self._fallbacks: Dict[str, str]              = {}
        self._custom_executors: Dict[str, Callable]  = {}
        self._register_builtin_modules()

    # ------------------------------------------------------------------
    # Built-in module registration
    # ------------------------------------------------------------------
    def _register_builtin_modules(self) -> None:
        # Detect LLM availability
        llm_available = False
        if self._prime:
            try:
                llm = self._prime.llm()
                llm_available = llm.is_loaded()
            except Exception:
                pass

        for meta in [
            _atlas_module_meta(),
            _reasoning_module_meta(),
            _memory_module_meta(),
            _validation_module_meta(),
            _llm_module_meta(llm_available),
        ]:
            _prime_mosaic.register_module(self._capsule, meta)

        # Wire up adapter functions
        self._adapters["atlas"]     = self._make_atlas_adapter()
        self._adapters["reasoning"] = self._make_reasoning_adapter()
        self._adapters["memory"]    = self._make_memory_adapter()
        self._adapters["validator"] = self._make_validator_adapter()
        self._adapters["llm"]       = self._make_llm_adapter()

    def _make_atlas_adapter(self) -> ModuleExecutorFn:
        def _run(ctx: Dict[str, Any]) -> Dict[str, Any]:
            if ctx.get("atlas_problem") is not None:
                return ctx  # already parsed
            raw = ctx.get("raw_input", "")
            if not raw:
                raise MOSAICError("ATLAS adapter: no raw_input in context")
            if self._prime:
                atlas   = self._prime.atlas()
                problem = atlas.parse(raw)
                ctx["atlas_problem"] = problem
            return ctx
        return _run

    def _make_reasoning_adapter(self) -> ModuleExecutorFn:
        def _run(ctx: Dict[str, Any]) -> Dict[str, Any]:
            problem = ctx.get("atlas_problem")
            if problem is None:
                raise MOSAICError("Reasoning adapter: no atlas_problem in context")
            if self._prime:
                reasoner = self._prime.reasoner()
                extra_rules  = ctx.get("extra_rules",  [])
                extra_facts  = ctx.get("extra_facts",  {})
                result = reasoner.solve(problem,
                                        extra_rules=extra_rules,
                                        extra_facts=extra_facts)
                ctx["reasoning_result"] = result
            return ctx
        return _run

    def _make_memory_adapter(self) -> ModuleExecutorFn:
        def _run(ctx: Dict[str, Any]) -> Dict[str, Any]:
            query = ctx.get("raw_input", "")
            if self._prime and query:
                try:
                    mem   = self._prime.memory()
                    items = mem.search(query, limit=5)
                    ctx["memory_retrieved"] = True
                    ctx["memory_items"]     = items
                    # Surface retrieved facts into reasoning context
                    if items:
                        extra_facts = ctx.get("extra_facts", {})
                        for item in items:
                            if hasattr(item, "content") and "=" in item.content:
                                parts = item.content.split("=", 1)
                                try:
                                    extra_facts[parts[0].strip()] = float(parts[1].strip())
                                except ValueError:
                                    pass
                        ctx["extra_facts"] = extra_facts
                except Exception:
                    ctx["memory_retrieved"] = False
            return ctx
        return _run

    def _make_validator_adapter(self) -> ModuleExecutorFn:
        def _run(ctx: Dict[str, Any]) -> Dict[str, Any]:
            rr = ctx.get("reasoning_result")
            if rr is None:
                return ctx  # nothing to validate
            ctx["validated"] = getattr(rr, "status", "") == "SOLVED"
            return ctx
        return _run

    def _make_llm_adapter(self) -> ModuleExecutorFn:
        def _run(ctx: Dict[str, Any]) -> Dict[str, Any]:
            if not self._prime:
                raise MOSAICError("LLM adapter: no PRIME instance available")
            llm = self._prime.llm()
            if not llm.is_loaded():
                raise MOSAICError("LLM adapter: no model loaded")
            query = ctx.get("raw_input", "")
            if query:
                from prime.generate import GenerationConfig
                resp = llm.generate(query, config=GenerationConfig(max_tokens=128))
                ctx["llm_output"] = resp.text
            return ctx
        return _run

    # ------------------------------------------------------------------
    # Public module management
    # ------------------------------------------------------------------

    def register_module(self, meta: "ModuleMetadata",
                        executor: Optional[Callable] = None,
                        fallback_for: Optional[str] = None) -> bool:
        """
        Register a custom module.

        Args:
            meta:        Module metadata (use :class:`ModuleMetadata`).
            executor:    Callable ``(ctx: dict) → dict`` that runs the module.
            fallback_for: Module ID this acts as a fallback for.

        Returns:
            True if registered (False if ID already exists).
        """
        d = meta if isinstance(meta, dict) else meta.to_dict()
        ok = _prime_mosaic.register_module(self._capsule, d)
        if ok and executor is not None:
            self._adapters[d["id"]] = executor
        if ok and fallback_for:
            self._fallbacks[fallback_for] = d["id"]
        return ok

    def update_module_status(self, module_id: str, status: str) -> bool:
        """Set a module's status (``"available"``, ``"unavailable"``, …)."""
        return _prime_mosaic.set_module_status(self._capsule, module_id, status)

    def remove_module(self, module_id: str) -> bool:
        """Remove a module from the registry."""
        ok = _prime_mosaic.remove_module(self._capsule, module_id)
        self._adapters.pop(module_id, None)
        return ok

    def list_modules(self) -> List[str]:
        """Return sorted list of registered module IDs."""
        return _prime_mosaic.list_modules(self._capsule)

    def get_module(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Return module metadata dict, or None if not found."""
        return _prime_mosaic.get_module(self._capsule, module_id)

    # ------------------------------------------------------------------
    # Task analysis
    # ------------------------------------------------------------------

    def analyze(self, problem) -> TaskRequirements:
        """
        Analyze an ATLAS :class:`~prime.atlas.Problem` and extract
        :class:`TaskRequirements` — what capabilities are needed.

        Does NOT solve the problem.
        """
        try:
            task_dict = {
                "task_id":           getattr(problem, "problem_id", str(uuid.uuid4())),
                "domain":            getattr(problem, "domain", "general"),
                "has_expressions":   bool(getattr(problem, "expressions", [])),
                "has_rules":         bool(getattr(problem, "rules", [])),
                "has_constraints":   bool(getattr(problem, "constraints", [])),
                "has_goals":         bool(getattr(problem, "goals", [])),
                "has_entities":      bool(getattr(problem, "entities", [])),
                "has_logic_clauses": bool(getattr(problem, "logic_clauses", [])),
                "has_relationships": bool(getattr(problem, "relationships", [])),
                "entity_count":      len(getattr(problem, "entities", [])),
                "has_memory_context":bool(getattr(problem, "metadata", {}).get("memory")),
                "operation_types":   [op.get("op", "") for op in getattr(problem, "expressions", [])],
                "goal_targets":      [g.get("target", "") for g in getattr(problem, "goals", [])],
            }
        except Exception as exc:
            raise MOSAICError(f"analyze() requires an ATLAS Problem: {exc}") from exc

        raw = _prime_mosaic.analyze_task(self._capsule, task_dict)
        return TaskRequirements(**raw)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, problem_or_requirements,
                **kwargs) -> CompositionPlan:
        """
        Build a :class:`CompositionPlan` for the given problem.

        Args:
            problem_or_requirements: An ATLAS :class:`~prime.atlas.Problem`
                or a :class:`TaskRequirements` object.
            **kwargs: MOSAIC configuration overrides (``max_modules``,
                      ``allow_llm``, ``prefer_lightweight``, ``excluded_modules``,
                      ``preferred_modules``, ``max_total_cost``, …).

        Returns:
            A :class:`CompositionPlan`.
        """
        if isinstance(problem_or_requirements, TaskRequirements):
            req = problem_or_requirements
        else:
            req = self.analyze(problem_or_requirements)

        cfg = {**self._cfg, **kwargs}
        raw = _prime_mosaic.compose(self._capsule, req.to_dict(),
                                    cfg if cfg else None)
        return CompositionPlan(raw, self)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, plan: CompositionPlan,
                problem=None,
                raw_input: str = "",
                extra_facts: Optional[Dict[str, Any]] = None,
                extra_rules: Optional[List] = None,
                **kwargs) -> ExecutionResult:
        """
        Execute a :class:`CompositionPlan`.

        Args:
            plan:        A valid :class:`CompositionPlan`.
            problem:     Optional ATLAS :class:`~prime.atlas.Problem`;
                         if omitted, ATLAS module will parse ``raw_input``.
            raw_input:   Raw text input (used if ``problem`` is None).
            extra_facts: Additional facts to inject into the reasoning context.
            extra_rules: Additional rules to inject into the reasoning context.

        Returns:
            An :class:`ExecutionResult`.
        """
        cfg = {**self._cfg, **kwargs}
        executor = _ExecutionEngine(self._adapters, self._fallbacks, cfg)
        ctx: Dict[str, Any] = {
            "atlas_problem": problem,
            "raw_input":     raw_input or (problem.original_input
                                          if problem and hasattr(problem, "original_input")
                                          else ""),
            "extra_facts":   extra_facts or {},
            "extra_rules":   extra_rules or [],
        }
        return executor.execute(plan, ctx)

    # ------------------------------------------------------------------
    # Convenience: solve end-to-end
    # ------------------------------------------------------------------

    def solve(self, input_text: str,
              **kwargs) -> ExecutionResult:
        """
        Full pipeline: parse → analyze → compose → execute.

        Args:
            input_text: Natural-language problem description.
            **kwargs:   MOSAIC configuration overrides.

        Returns:
            An :class:`ExecutionResult`.

        Example::

            result = mosaic.solve("A car travels 150 km in 3 hours.")
            print(result.answer)  # "50"
        """
        if not self._prime:
            raise MOSAICError("solve() requires a PRIME instance")

        atlas   = self._prime.atlas()
        problem = atlas.parse(input_text)
        plan    = self.compose(problem, **kwargs)
        return self.execute(plan, problem, raw_input=input_text, **kwargs)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def explain_plan(self, plan: CompositionPlan) -> str:
        """Alias for ``plan.explain()``."""
        return plan.explain()

    def validate_plan(self, plan: CompositionPlan) -> str:
        """Validate a plan and return its status string."""
        return _prime_mosaic.validate_plan(self._capsule, plan._raw)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._cfg)

    def __repr__(self) -> str:
        n = len(self.list_modules())
        return f"<MOSAICEngine modules={n}>"
