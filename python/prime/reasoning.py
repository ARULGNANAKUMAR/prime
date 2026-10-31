"""
prime.reasoning — Public Reasoning Engine API for PRIME Phase 5.

The Reasoning Engine consumes a structured :class:`~prime.atlas.Problem`
(from ATLAS) and performs symbolic, rule-based, and hybrid reasoning to
produce a :class:`ReasoningResult`.

ATLAS answers: "What is the problem structure?"
Phase 5 answers: "What reasoning operation can be performed over that structure?"

Typical usage::

    from prime import PRIME

    prime = PRIME()
    atlas  = prime.atlas()
    reasoner = prime.reasoner()

    problem = atlas.parse("A car travels 150 km in 3 hours. Find average speed.")
    result  = reasoner.solve(problem)

    print(result.status)        # "SOLVED"
    print(result.answer)        # "50"
    print(result.answer_unit)   # "km"
    print(result.trace)         # ReasoningTrace object
    print(result.explain())     # human-readable explanation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import _prime_reasoning


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ReasoningError(Exception):
    """Raised when a reasoning operation fails."""


# ---------------------------------------------------------------------------
# ReasoningStep
# ---------------------------------------------------------------------------
@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""
    step_id:      int
    type:         str
    operation:    str
    rule_id:      str
    description:  str
    confidence:   float
    success:      bool
    input_facts:  List[str]
    output_facts: List[str]

    def __repr__(self) -> str:
        return (f"<Step #{self.step_id} [{self.type}] {self.description[:60]!r}>")


# ---------------------------------------------------------------------------
# ReasoningTrace
# ---------------------------------------------------------------------------
class ReasoningTrace:
    """Structured record of all reasoning steps."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw            = raw
        self.total_steps:    int = raw.get("total_steps", 0)
        self.rules_fired:    int = raw.get("rules_fired", 0)
        self.expressions_evaluated: int = raw.get("expressions_evaluated", 0)
        self.constraints_checked:   int = raw.get("constraints_checked", 0)
        self.steps: List[ReasoningStep] = [
            ReasoningStep(
                step_id      = s["step_id"],
                type         = s["type"],
                operation    = s["operation"],
                rule_id      = s["rule_id"],
                description  = s["description"],
                confidence   = s["confidence"],
                success      = s["success"],
                input_facts  = s["input_facts"],
                output_facts = s["output_facts"],
            )
            for s in raw.get("steps", [])
        ]

    def __repr__(self) -> str:
        return (f"<ReasoningTrace steps={self.total_steps} "
                f"rules={self.rules_fired} "
                f"expressions={self.expressions_evaluated}>")

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def format(self) -> str:
        """Format the trace as a human-readable multi-line string."""
        lines = []
        for step in self.steps:
            status = "✓" if step.success else "✗"
            lines.append(f"  Step {step.step_id:3d} [{status}] {step.type:<20} {step.description}")
        return "\n".join(lines) if lines else "  (empty trace)"


# ---------------------------------------------------------------------------
# GoalResult
# ---------------------------------------------------------------------------
@dataclass
class GoalResult:
    """Result for a single goal."""
    id:            str
    target:        str
    satisfied:     bool
    derived_value: str
    unit:          str
    confidence:    float

    def __repr__(self) -> str:
        val = f"{self.derived_value}{' ' + self.unit if self.unit else ''}"
        return f"<Goal target={self.target!r} satisfied={self.satisfied} value={val!r}>"


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------
@dataclass
class ConstraintCheckResult:
    """Result for a single constraint check."""
    id:          str
    result:      str   # "satisfied" | "violated" | "unknown"
    description: str

    @property
    def satisfied(self) -> bool:
        return self.result == "satisfied"

    def __repr__(self) -> str:
        return f"<Constraint id={self.id!r} result={self.result}>"


# ---------------------------------------------------------------------------
# ReasoningResult
# ---------------------------------------------------------------------------
class ReasoningResult:
    """
    The result of a reasoning call.

    Attributes:
        status:           One of SOLVED / PARTIALLY_SOLVED / UNRESOLVED /
                          INVALID_INPUT / CONSTRAINT_VIOLATION /
                          REASONING_LIMIT_REACHED / CIRCULAR_REASONING.
        answer:           String representation of the primary answer.
        answer_unit:      Unit of the answer (if numeric), e.g. "km/h".
        answer_numeric:   Numeric value of the answer (if applicable).
        answer_is_numeric: True if the answer is a number.
        confidence:       Confidence score [0, 1].
        trace:            :class:`ReasoningTrace` — the full reasoning trace.
        goal_results:     Per-goal :class:`GoalResult` objects.
        constraint_results: Per-constraint :class:`ConstraintCheckResult` objects.
        derived_facts:    Dict of newly derived facts (name → value).
        error_message:    Non-empty if status is not SOLVED.
        mode_used:        The reasoning mode that was applied.
    """

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw               = raw
        self.status:            str   = raw.get("status", "UNRESOLVED")
        self.answer:            str   = raw.get("answer", "")
        self.answer_unit:       str   = raw.get("answer_unit", "")
        self.answer_numeric:    Optional[float] = raw.get("answer_numeric")
        self.answer_is_numeric: bool  = raw.get("answer_is_numeric", False)
        self.confidence:        float = raw.get("confidence", 0.0)
        self.error_message:     str   = raw.get("error_message", "")
        self.mode_used:         str   = raw.get("mode_used", "")
        self.trace:             ReasoningTrace = ReasoningTrace(raw.get("trace", {}))
        self.goal_results:      List[GoalResult] = [
            GoalResult(**g) for g in raw.get("goal_results", [])
        ]
        self.constraint_results: List[ConstraintCheckResult] = [
            ConstraintCheckResult(**c) for c in raw.get("constraint_results", [])
        ]
        self.derived_facts: Dict[str, Any] = raw.get("derived_facts", {})

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def solved(self) -> bool:
        """True if status is SOLVED."""
        return self.status == "SOLVED"

    @property
    def answer_with_unit(self) -> str:
        """Answer string followed by unit (if any)."""
        if self.answer_unit:
            return f"{self.answer} {self.answer_unit}"
        return self.answer

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def explain(self) -> str:
        """
        Generate a human-readable explanation from the reasoning trace.

        The explanation is derived entirely from the structured trace —
        no generative text is invented.

        Returns:
            A multi-line string describing the reasoning steps.
        """
        lines = ["=" * 60]
        lines.append(f"PRIME Reasoning Explanation")
        lines.append(f"Mode:   {self.mode_used}")
        lines.append(f"Status: {self.status}")
        lines.append("=" * 60)

        # Initialization facts
        init_steps = [s for s in self.trace.steps if s.type == "FACT_INIT"]
        if init_steps:
            lines.append("\nKnown facts:")
            for s in init_steps:
                for out in s.output_facts:
                    lines.append(f"  • {out}")

        # Expression evaluations
        expr_steps = [s for s in self.trace.steps if s.type == "EXPRESSION_EVAL"]
        if expr_steps:
            lines.append("\nComputations:")
            for s in expr_steps:
                status = "✓" if s.success else "✗"
                lines.append(f"  {status} {s.description}")

        # Rule firings
        rule_steps = [s for s in self.trace.steps if s.type == "RULE_FIRE"]
        if rule_steps:
            lines.append("\nRules applied:")
            for s in rule_steps:
                status = "✓" if s.success else "✗"
                lines.append(f"  {status} {s.description}")
                if s.output_facts:
                    lines.append(f"     → {', '.join(s.output_facts)}")

        # Deductions
        deduction_steps = [s for s in self.trace.steps if s.type == "DEDUCTION"]
        if deduction_steps:
            lines.append("\nDeductions:")
            for s in deduction_steps:
                lines.append(f"  • {s.description}")

        # Constraint checks
        cst_steps = [s for s in self.trace.steps if s.type == "CONSTRAINT_CHECK"]
        if cst_steps:
            lines.append("\nConstraint checks:")
            for s in cst_steps:
                status = "✓" if s.success else "✗"
                lines.append(f"  {status} {s.description}")

        # Goal checks
        goal_steps = [s for s in self.trace.steps if s.type == "GOAL_CHECK"]
        if goal_steps:
            lines.append("\nGoal evaluation:")
            for s in goal_steps:
                status = "✓" if s.success else "✗"
                lines.append(f"  {status} {s.description}")

        # Summary
        lines.append("\n" + "-" * 60)
        if self.solved and self.answer:
            lines.append(f"Result: {self.answer_with_unit}")
        elif self.status == "PARTIALLY_SOLVED":
            lines.append("Result: partial — some goals satisfied")
        elif self.status == "CONSTRAINT_VIOLATION":
            lines.append("Result: constraint violated")
        else:
            lines.append(f"Result: {self.status}")
            if self.error_message:
                lines.append(f"Reason: {self.error_message}")

        lines.append(f"Steps:  {self.trace.total_steps}  "
                     f"(rules={self.trace.rules_fired}, "
                     f"expressions={self.trace.expressions_evaluated})")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        ans = self.answer_with_unit if self.answer else "—"
        return (f"<ReasoningResult status={self.status!r} "
                f"answer={ans!r} "
                f"confidence={self.confidence:.2f} "
                f"steps={self.trace.total_steps}>")

    def __str__(self) -> str:
        return self.explain()


# ---------------------------------------------------------------------------
# Rule builder helpers
# ---------------------------------------------------------------------------

def make_condition(left: str, op: str, right: str, *,
                   right_is_fact: bool = False) -> Dict[str, Any]:
    """Build a condition dict for use in a :func:`make_rule` call.

    Args:
        left:         Fact name on the left-hand side.
        op:           Comparison operator: ``=``, ``!=``, ``<``, ``<=``,
                      ``>``, ``>=``, ``AND``, ``OR``, ``NOT``, ``EXISTS``.
        right:        Literal value or fact name on the right-hand side.
        right_is_fact: If True, ``right`` is interpreted as a fact name
                      rather than a literal.
    """
    return {"left": left, "op": op, "right": right,
            "right_is_fact": right_is_fact, "sub_conditions": []}


def make_compound_condition(op: str, *sub_conditions) -> Dict[str, Any]:
    """Build a compound AND/OR/NOT condition.

    Args:
        op:            ``AND``, ``OR``, or ``NOT``.
        *sub_conditions: Child conditions produced by :func:`make_condition`
                         or :func:`make_compound_condition`.
    """
    return {"left": "", "op": op, "right": "", "right_is_fact": False,
            "sub_conditions": list(sub_conditions)}


def make_conclusion(name: str, value: Union[str, float, int, bool], *,
                    unit: str = "", confidence: float = 1.0) -> Dict[str, Any]:
    """Build a conclusion dict for use in a :func:`make_rule` call.

    Args:
        name:       Fact name that this conclusion derives.
        value:      The value to set (string, number, or bool).
        unit:       Optional unit string.
        confidence: Confidence of this conclusion [0, 1].
    """
    conc: Dict[str, Any] = {"name": name, "unit": unit, "confidence": confidence}
    if isinstance(value, bool):
        conc["value"] = "true" if value else "false"
    elif isinstance(value, (int, float)):
        conc["value"]         = str(value)
        conc["numeric_value"] = float(value)
    else:
        conc["value"] = str(value)
    return conc


def make_rule(rule_id: str, conditions: List[Dict], conclusions: List[Dict], *,
              name: str = "", priority: int = 0,
              confidence: float = 1.0) -> Dict[str, Any]:
    """Build a rule dict to pass to :class:`ReasoningEngine`.

    Args:
        rule_id:     Unique rule identifier.
        conditions:  List of condition dicts (from :func:`make_condition`).
        conclusions: List of conclusion dicts (from :func:`make_conclusion`).
        name:        Optional human-readable rule name.
        priority:    Higher priorities fire first (default 0).
        confidence:  Rule confidence score [0, 1].

    Example::

        rule = make_rule(
            "r1",
            conditions=[make_condition("temperature", ">", "100")],
            conclusions=[make_conclusion("warning", True)],
            name="high_temperature_warning",
        )
    """
    return {
        "id": rule_id, "name": name or rule_id,
        "priority": priority, "confidence": confidence,
        "conditions": conditions,
        "conclusions": conclusions,
    }


# ---------------------------------------------------------------------------
# LLMReasoningProvider interface
# ---------------------------------------------------------------------------
class LLMReasoningProvider:
    """
    Abstract interface for LLM-assisted reasoning.

    Concrete implementations can plug in the Phase 2 LLM or a mock.
    The reasoning engine calls these methods and validates the proposals
    using deterministic components — it never blindly trusts the output.
    """

    def propose_reasoning(self, problem_json: str) -> Optional[Dict[str, Any]]:
        """Propose a reasoning path for the given problem JSON.

        Returns a dict with suggested ``rules``, ``expressions``, or
        ``facts``, or None if the LLM cannot assist.
        """
        return None

    def interpret_ambiguity(self, text: str) -> Optional[str]:
        """Return a cleaner interpretation of ambiguous text, or None."""
        return None

    def generate_explanation(self, trace_dict: Dict) -> Optional[str]:
        """Generate a natural-language explanation from a trace dict, or None."""
        return None


class MockLLMProvider(LLMReasoningProvider):
    """Deterministic mock LLM provider for testing the LLM interface."""

    def __init__(self, rules: Optional[List[Dict]] = None,
                 expressions: Optional[List[Dict]] = None) -> None:
        self._rules       = rules or []
        self._expressions = expressions or []

    def propose_reasoning(self, problem_json: str) -> Optional[Dict[str, Any]]:
        return {
            "rules":       self._rules,
            "expressions": self._expressions,
            "source":      "mock_llm",
        }

    def generate_explanation(self, trace_dict: Dict) -> Optional[str]:
        steps = trace_dict.get("steps", [])
        return f"[MockLLM] Reasoning completed in {len(steps)} steps."


# ---------------------------------------------------------------------------
# ReasoningEngine — main public class
# ---------------------------------------------------------------------------
class ReasoningEngine:
    """
    PRIME Phase 5 Hybrid Reasoning Engine.

    Converts :class:`~prime.atlas.Problem` representations into
    :class:`ReasoningResult` objects via rule, symbolic, and hybrid reasoning.

    Usage::

        from prime import PRIME

        prime    = PRIME()
        atlas    = prime.atlas()
        reasoner = prime.reasoner()

        problem = atlas.parse("A car travels 150 km in 3 hours.")
        result  = reasoner.solve(problem)

        print(result)        # explanation
        print(result.trace)  # trace object

    Direct API (without ATLAS)::

        from prime.reasoning import ReasoningEngine, make_rule, make_condition, make_conclusion

        engine = ReasoningEngine()
        result = engine.solve_facts(
            facts={"temperature": 120},
            rules=[
                make_rule("r1",
                    conditions=[make_condition("temperature", ">", "100")],
                    conclusions=[make_conclusion("warning", True)])
            ],
        )
        print(result.status)   # "SOLVED"
        print(result.derived_facts)  # {"warning": True}
    """

    def __init__(self, llm_provider: Optional[LLMReasoningProvider] = None) -> None:
        self._llm = llm_provider
        self._global_rules: List[Dict] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a persistent rule to the engine's global rule base.

        Rules added here are applied to every :meth:`solve` / :meth:`solve_facts`
        call.  Call :meth:`clear_rules` to remove them.

        Args:
            rule: A rule dict (from :func:`make_rule`).
        """
        self._global_rules.append(rule)

    def clear_rules(self) -> None:
        """Remove all global rules from the engine."""
        self._global_rules.clear()

    def set_llm_provider(self, provider: Optional[LLMReasoningProvider]) -> None:
        """Set or replace the LLM reasoning provider."""
        self._llm = provider

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def solve(self, problem, *,
              mode: str = "hybrid",
              max_steps: int = 1000,
              max_depth: int = 50,
              timeout_ms: float = 5000.0,
              extra_rules: Optional[List[Dict]] = None,
              extra_facts: Optional[Dict[str, Any]] = None,
              **kwargs) -> ReasoningResult:
        """
        Solve a problem represented as an ATLAS :class:`~prime.atlas.Problem`.

        Args:
            problem:      An :class:`~prime.atlas.Problem` from ATLAS.
            mode:         ``"hybrid"`` (default), ``"rule"``, ``"symbolic"``,
                          or ``"llm_assisted"``.
            max_steps:    Maximum reasoning steps (default 1000).
            max_depth:    Maximum inference depth (default 50).
            timeout_ms:   Timeout in milliseconds (default 5000).
            extra_rules:  Additional rules to apply for this call only.
            extra_facts:  Additional pre-bound facts (name → value).

        Returns:
            A :class:`ReasoningResult`.

        Raises:
            ReasoningError: If the input is structurally invalid.
        """
        try:
            entities     = problem.entities
            goals        = problem.goals
            constraints  = problem.constraints
            expressions  = problem.expressions
            problem_id   = problem.problem_id
            domain       = problem.domain
        except AttributeError as exc:
            raise ReasoningError(
                f"solve() expects an ATLAS Problem, got {type(problem).__name__}: {exc}"
            ) from exc

        # Remap ATLAS goals if target is generic ("find", "calculate", etc.)
        goals = self._remap_goals(goals, entities, expressions)

        return self._solve_context(
            entities     = entities,
            goals        = goals,
            constraints  = constraints,
            expressions  = expressions,
            problem_id   = problem_id,
            domain       = domain,
            rules        = (extra_rules or []) + self._global_rules,
            extra_facts  = extra_facts or {},
            mode         = mode,
            max_steps    = max_steps,
            max_depth    = max_depth,
            timeout_ms   = timeout_ms,
            **kwargs,
        )

    def solve_facts(self, facts: Dict[str, Any], *,
                    rules: Optional[List[Dict]] = None,
                    constraints: Optional[List[Dict]] = None,
                    goals: Optional[List[str]] = None,
                    expressions: Optional[List[Dict]] = None,
                    mode: str = "hybrid",
                    max_steps: int = 1000,
                    max_depth: int = 50,
                    timeout_ms: float = 5000.0,
                    **kwargs) -> ReasoningResult:
        """
        Solve a problem given raw Python facts (without ATLAS).

        This is the low-level API for direct use when ATLAS output is not needed.

        Args:
            facts:        Dict mapping fact name → value (number, bool, or str).
            rules:        List of rule dicts (from :func:`make_rule`).
            constraints:  List of constraint dicts. Each dict needs:
                          ``{"id": str, "left": str, "op": str, "right": str}``.
            goals:        List of fact names to derive.
            expressions:  List of expression dicts. Each dict needs:
                          ``{"id": str, "op": str, "output_id": str, "operand_ids": [...]}``.
            mode:         Reasoning mode.
            max_steps:    Maximum steps.
            max_depth:    Maximum depth.
            timeout_ms:   Timeout in ms.

        Example::

            result = engine.solve_facts(
                facts={"distance": 150, "time": 3},
                expressions=[{"id": "e1", "op": "divide",
                               "output_id": "speed", "operand_ids": ["distance", "time"]}],
                goals=["speed"],
            )
        """
        goal_dicts = []
        for i, g in enumerate(goals or []):
            goal_dicts.append({
                "id": f"g_{i}", "target": g,
                "description": f"Find {g}", "priority": 0, "confidence": 1.0,
            })

        return self._solve_context(
            entities      = [],
            goals         = goal_dicts,
            constraints   = constraints or [],
            expressions   = expressions or [],
            problem_id    = "",
            domain        = "general",
            rules         = (rules or []) + self._global_rules,
            extra_facts   = facts,
            mode          = mode,
            max_steps     = max_steps,
            max_depth     = max_depth,
            timeout_ms    = timeout_ms,
            **kwargs,
        )

    def validate(self, problem) -> Dict[str, Any]:
        """
        Validate that a problem can be reasoned over (structure check only).

        Returns a dict with ``valid`` (bool) and ``issues`` (list of strings).
        """
        issues = []
        try:
            if not hasattr(problem, "entities"):
                issues.append("Problem has no 'entities' attribute")
            if not hasattr(problem, "goals"):
                issues.append("Problem has no 'goals' attribute")
        except Exception as exc:
            issues.append(str(exc))
        return {"valid": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    # Expression evaluation (standalone)
    # ------------------------------------------------------------------

    def eval_expression(self, expr: str,
                        bindings: Optional[Dict[str, float]] = None) -> float:
        """
        Evaluate a safe arithmetic expression string.

        Supports ``+``, ``-``, ``*``, ``/``, ``%``, ``^``, ``abs()``,
        ``min()``, ``max()``, ``sqrt()``.  No ``eval()`` is used.

        Args:
            expr:     Expression string, e.g. ``"150 / 3"`` or ``"x + y * 2"``.
            bindings: Optional dict of variable name → value.

        Returns:
            The numeric result.

        Raises:
            ReasoningError: On parse or evaluation error.

        Example::

            result = engine.eval_expression("150 / 3")           # → 50.0
            result = engine.eval_expression("x + y", {"x": 10, "y": 20})  # → 30.0
        """
        try:
            return _prime_reasoning.eval_expression(expr, bindings or {})
        except _prime_reasoning.ReasoningError as exc:
            raise ReasoningError(str(exc)) from exc

    def check_constraint(self, left: str, op: str, right: str,
                         facts: Dict[str, Any]) -> str:
        """
        Check whether a constraint is satisfied, violated, or unknown.

        Args:
            left:   Fact name to check.
            op:     Operator (``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``).
            right:  Literal value to compare against.
            facts:  Dict of fact name → value.

        Returns:
            One of ``"satisfied"``, ``"violated"``, ``"unknown"``.
        """
        try:
            return _prime_reasoning.check_constraint(left, op, str(right), facts)
        except _prime_reasoning.ReasoningError as exc:
            raise ReasoningError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_context(self, *, entities, goals, constraints, expressions,
                       problem_id, domain, rules, extra_facts,
                       mode, max_steps, max_depth, timeout_ms,
                       max_rule_firings: int = 500,
                       use_rules: bool = True,
                       use_symbolic: bool = True,
                       use_llm: bool = False,
                       detect_circular: bool = True,
                       confidence_threshold: float = 0.0,
                       **_ignored) -> ReasoningResult:

        # Enrich extra_facts with dimension-based aliases for ATLAS operand name resolution.
        # ATLAS expressions use semantic operand IDs ("distance", "time") while
        # entities may have IDs like "e_2". We add aliases by matching quantity dimensions.
        extra_facts = dict(extra_facts)
        DIMENSION_ALIASES = {
            "length":       ["distance", "length", "displacement", "height", "width", "depth"],
            "time":         ["time", "duration", "period", "interval", "elapsed"],
            "mass":         ["mass", "weight"],
            "speed":        ["speed", "velocity"],
            "temperature":  ["temperature"],
            "pressure":     ["pressure"],
            "force":        ["force"],
            "energy":       ["energy", "work"],
            "power":        ["power"],
            "charge":       ["charge"],
            "current":      ["current"],
            "voltage":      ["voltage"],
            "frequency":    ["frequency"],
            "area":         ["area"],
            "volume":       ["volume"],
        }
        for e in entities:
            qty = e.get("quantity")
            if not qty:
                continue
            dim = qty.get("dimension", "")
            val = qty.get("value", 0.0)
            unit = qty.get("unit", "")
            aliases = DIMENSION_ALIASES.get(dim, [])
            # Only add if the alias is not already taken
            for alias in aliases:
                if alias not in extra_facts:
                    extra_facts[alias] = {"value": val, "unit": unit}

        # Optionally consult LLM provider
        all_rules       = list(rules)
        all_expressions = list(expressions)
        if self._llm and mode == "llm_assisted":
            try:
                import json
                proposal = self._llm.propose_reasoning(
                    json.dumps({"entities": entities, "goals": goals}))
                if proposal:
                    # Merge LLM proposals — they go through validation in the engine
                    all_rules       += proposal.get("rules",       [])
                    all_expressions += proposal.get("expressions", [])
            except Exception:
                pass  # LLM unavailable — continue without it

        ctx = {
            "problem_id":     problem_id,
            "problem_domain": domain,
            "entities":       entities,
            "goals":          goals,
            "constraints":    constraints,
            "expressions":    all_expressions,
            "rules":          all_rules,
            "facts":          extra_facts,
            "config": {
                "mode":                  mode,
                "max_steps":             max_steps,
                "max_depth":             max_depth,
                "max_rule_firings":      max_rule_firings,
                "timeout_ms":            timeout_ms,
                "use_rules":             use_rules,
                "use_symbolic":          use_symbolic,
                "use_llm":               use_llm,
                "detect_circular":       detect_circular,
                "confidence_threshold":  confidence_threshold,
            },
        }

        try:
            raw = _prime_reasoning.solve_context(ctx)
        except _prime_reasoning.ReasoningError as exc:
            raise ReasoningError(str(exc)) from exc

        return ReasoningResult(raw)

    def _remap_goals(self, goals, entities, expressions):
        """Remap generic ATLAS goal targets to actual entity/expression IDs."""
        generic = {"find", "calculate", "compute", "determine",
                   "evaluate", "solve", "derive", "get"}

        remapped = []
        for g in goals:
            target = g.get("target", "").lower().strip()
            if target in generic or not target:
                # Try to find the output_id of the first expression
                if expressions:
                    g = dict(g)
                    g["target"] = expressions[0].get("output_id", target)
                elif entities:
                    # Target the last entity (often the unknown)
                    for e in reversed(entities):
                        if not e.get("quantity"):  # unbound
                            g = dict(g)
                            g["target"] = e.get("id", e.get("name", target))
                            break
            remapped.append(g)
        return remapped

    def __repr__(self) -> str:
        return (f"<ReasoningEngine rules={len(self._global_rules)} "
                f"llm={'set' if self._llm else 'none'}>")
