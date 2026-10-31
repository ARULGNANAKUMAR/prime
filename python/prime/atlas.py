"""
prime.atlas — Public ATLAS API for PRIME Phase 4.

ATLAS (Universal Problem Representation Engine) converts heterogeneous problem
inputs into a structured, domain-independent :class:`UniversalProblem`
representation that future PRIME reasoning phases can consume.

ATLAS does NOT solve problems. It represents them.

Typical usage::

    from prime import PRIME

    prime = PRIME()
    atlas = prime.atlas()

    problem = atlas.parse("A car travels 120 km in 3 hours.")

    print(problem.domain)           # "physics"
    print(problem.entities)         # list of entity dicts
    print(problem.goals)            # list of goal dicts
    print(problem.relationships)    # list of relationship dicts

    # Serialization
    json_str = atlas.to_json(problem)
    problem2 = atlas.from_json(json_str)

    # Validation
    result = atlas.validate(problem)
    print(result.valid)             # True / False

    # Structured input
    from prime.atlas import UniversalProblem as UP
    proto = UP(problem_id="my_prob")
    proto.entities.append({
        "id": "e1", "type": "quantity", "name": "distance",
        "quantity": {"value": 120.0, "unit": "km", "dimension": "length"},
        "confidence": 1.0, "attributes": [], "metadata": {}
    })
    problem3 = atlas.parse_structured(proto)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import _prime_atlas


class ATLASError(Exception):
    """Raised when an ATLAS operation fails."""


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Result from :meth:`ATLASEngine.validate`."""
    valid:    bool
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        return (f"<ValidationResult valid={self.valid} "
                f"errors={len(self.errors)} warnings={len(self.warnings)}>")


# ---------------------------------------------------------------------------
# UniversalProblem (Python-level prototype for structured input)
# ---------------------------------------------------------------------------
@dataclass
class UniversalProblemProto:
    """
    A Python-level prototype for :meth:`ATLASEngine.parse_structured`.

    Fill in whatever fields you already know; ATLAS will complete the
    representation (graph, state, etc.).
    """
    problem_id:    str = ""
    original_input:str = ""
    domain:        str = "unknown"
    entities:      List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    constraints:   List[Dict[str, Any]] = field(default_factory=list)
    goals:         List[Dict[str, Any]] = field(default_factory=list)
    operations:    List[Dict[str, Any]] = field(default_factory=list)
    expressions:   List[Dict[str, Any]] = field(default_factory=list)
    assumptions:   List[Dict[str, Any]] = field(default_factory=list)
    evidence:      List[Dict[str, Any]] = field(default_factory=list)
    metadata:      Dict[str, str]       = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Problem — thin Python wrapper around _UniversalProblem
# ---------------------------------------------------------------------------
class Problem:
    """
    A parsed universal problem representation.

    Instances are returned by :meth:`ATLASEngine.parse` and
    :meth:`ATLASEngine.from_json`.  All structural data is accessible via
    attributes that return plain Python dicts / lists.

    ATLAS does NOT solve the problem. This is a representation layer for
    future reasoning phases.
    """

    def __init__(self, c_problem: Any) -> None:
        self._c = c_problem

    # ------------------------------------------------------------------
    # Identity / metadata
    # ------------------------------------------------------------------
    @property
    def problem_id(self) -> str:
        return self._c.problem_id

    @property
    def domain(self) -> str:
        return self._c.domain

    @property
    def domain_confidence(self) -> float:
        return self._c.domain_confidence

    @property
    def original_input(self) -> str:
        return self._c.original_input

    @property
    def schema_version(self) -> str:
        return self._c.schema_version

    # ------------------------------------------------------------------
    # Structural components
    # ------------------------------------------------------------------
    @property
    def entities(self) -> List[Dict[str, Any]]:
        """List of entity dicts (type, name, quantity, confidence, …)."""
        return list(self._c.entities)

    @property
    def relationships(self) -> List[Dict[str, Any]]:
        """List of relationship dicts (source_id, relation, target_id, …)."""
        return list(self._c.relationships)

    @property
    def constraints(self) -> List[Dict[str, Any]]:
        """List of constraint dicts (left, op, right, type, …)."""
        return list(self._c.constraints)

    @property
    def goals(self) -> List[Dict[str, Any]]:
        """List of goal dicts (description, target, priority, …)."""
        return list(self._c.goals)

    @property
    def operations(self) -> List[Dict[str, Any]]:
        """List of operation dicts (type, input_ids, output_ids)."""
        return list(self._c.operations)

    @property
    def expressions(self) -> List[Dict[str, Any]]:
        """List of expression dicts (op, operand_ids, output_id, text)."""
        return list(self._c.expressions)

    @property
    def assumptions(self) -> List[Dict[str, Any]]:
        """List of assumption dicts (description, confidence)."""
        return list(self._c.assumptions)

    @property
    def states(self) -> List[Dict[str, Any]]:
        """List of state dicts (label, entity_ids, constraint_ids)."""
        return list(self._c.states)

    @property
    def graph(self) -> Dict[str, Any]:
        """Graph summary dict (node_count, edge_count)."""
        return dict(self._c.graph)

    # ------------------------------------------------------------------
    # Serialization helper
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return the complete representation as a Python dict."""
        return self._c.to_dict()

    def __repr__(self) -> str:
        return (f"<Problem id={self.problem_id!r} domain={self.domain!r} "
                f"entities={len(self.entities)} goals={len(self.goals)}>")


# ---------------------------------------------------------------------------
# ATLASEngine — main public API
# ---------------------------------------------------------------------------
class ATLASEngine:
    """
    The PRIME ATLAS Engine.

    Converts heterogeneous problem inputs into a domain-independent
    :class:`Problem` (UniversalProblem) representation.

    Obtain via :meth:`prime.PRIME.atlas`::

        prime = PRIME()
        atlas = prime.atlas()

    or construct directly::

        atlas = ATLASEngine()

    **ATLAS does NOT solve problems.** Reasoning belongs to Phase 5.
    """

    def __init__(self,
                 max_input_bytes:   int = 1_048_576,
                 max_entities:      int = 512,
                 max_relationships: int = 1024,
                 max_constraints:   int = 256,
                 max_goals:         int = 64,
                 max_operations:    int = 128,
                 max_graph_nodes:   int = 2048) -> None:
        try:
            self._engine = _prime_atlas._ATLASEngine(
                max_input_bytes   = max_input_bytes,
                max_entities      = max_entities,
                max_relationships = max_relationships,
                max_constraints   = max_constraints,
                max_goals         = max_goals,
                max_operations    = max_operations,
                max_graph_nodes   = max_graph_nodes,
            )
        except _prime_atlas.ATLASError as exc:
            raise ATLASError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> Problem:
        """
        Parse a plain-text problem description into a :class:`Problem`.

        The text is normalized, classified by domain, and then processed
        through the ATLAS pipeline (entity extraction, relationship
        extraction, constraint extraction, goal identification, graph
        construction).

        Args:
            text: Any natural-language or semi-formal problem description.

        Returns:
            A :class:`Problem` with structured representation.

        Raises:
            ATLASError: If the input is empty, too large, or malformed.

        Example::

            problem = atlas.parse("A car travels 120 km in 3 hours.")
            assert problem.domain == "physics"
        """
        if not isinstance(text, str):
            raise ATLASError(f"parse() expects a str, got {type(text).__name__}")
        if not text.strip():
            raise ATLASError("Input text is empty")
        try:
            return Problem(self._engine.parse(text))
        except _prime_atlas.ATLASError as exc:
            raise ATLASError(str(exc)) from exc

    def parse_structured(self, proto: UniversalProblemProto) -> Problem:
        """
        Accept a pre-structured :class:`UniversalProblemProto` and complete
        the representation (graph, initial state).

        This allows callers that have already extracted entities or
        constraints to bypass the text parser.

        Args:
            proto: A :class:`UniversalProblemProto` with any pre-filled fields.

        Returns:
            A completed :class:`Problem`.
        """
        if not isinstance(proto, UniversalProblemProto):
            raise ATLASError("parse_structured() expects an UniversalProblemProto")
        # Build a JSON representation that the C++ engine can round-trip
        import json
        data = {
            "atlas_schema_version": "1.0",
            "problem_id": proto.problem_id or "",
            "original_input": proto.original_input,
            "normalized_input": proto.original_input,
            "domain": proto.domain,
            "domain_confidence": 0.5,
            "entities":      proto.entities,
            "relationships": proto.relationships,
            "constraints":   proto.constraints,
            "goals":         proto.goals,
            "operations":    proto.operations,
            "expressions":   proto.expressions,
            "states":        [],
            "assumptions":   proto.assumptions,
            "evidence":      proto.evidence,
            "logic_clauses": [],
            "graph":         {"nodes": [], "edges": []},
            "metadata":      proto.metadata,
        }
        # Ensure required sub-fields are present in entities
        for e in data["entities"]:
            e.setdefault("id",         "")
            e.setdefault("type",       "concept")
            e.setdefault("name",       "")
            e.setdefault("confidence", 1.0)
            e.setdefault("attributes", [])
            e.setdefault("metadata",   {})
            e.setdefault("quantity",   None)
        for r in data["relationships"]:
            r.setdefault("id",         "")
            r.setdefault("source_id",  "")
            r.setdefault("relation",   "")
            r.setdefault("target_id",  "")
            r.setdefault("confidence", 1.0)
            r.setdefault("metadata",   {})
        for c in data["constraints"]:
            c.setdefault("id",         "")
            c.setdefault("left",       "")
            c.setdefault("op",         "=")
            c.setdefault("right",      "")
            c.setdefault("type",       "domain")
            c.setdefault("confidence", 1.0)
            c.setdefault("metadata",   {})
        for g in data["goals"]:
            g.setdefault("id",            "")
            g.setdefault("description",   "")
            g.setdefault("target",        "")
            g.setdefault("desired_state", "")
            g.setdefault("priority",      0)
            g.setdefault("confidence",    1.0)
            g.setdefault("metadata",      {})

        json_str = json.dumps(data)
        try:
            return Problem(self._engine.from_json(json_str))
        except _prime_atlas.ATLASError as exc:
            raise ATLASError(str(exc)) from exc

    def validate(self, problem: Problem) -> ValidationResult:
        """
        Validate a :class:`Problem` representation.

        Checks required fields, ID uniqueness, confidence ranges, and
        structural integrity.

        Returns:
            A :class:`ValidationResult` with ``valid``, ``errors``, and
            ``warnings``.
        """
        if not isinstance(problem, Problem):
            raise ATLASError("validate() expects a Problem")
        try:
            raw = self._engine.validate(problem._c)
            return ValidationResult(
                valid    = bool(raw["valid"]),
                errors   = list(raw.get("errors",   [])),
                warnings = list(raw.get("warnings", [])),
            )
        except _prime_atlas.ATLASError as exc:
            raise ATLASError(str(exc)) from exc

    def to_json(self, problem: Problem) -> str:
        """
        Serialize a :class:`Problem` to a JSON string.

        The JSON includes the atlas_schema_version field and is safe to
        store or transmit. Use :meth:`from_json` to restore.

        Args:
            problem: A :class:`Problem` to serialize.

        Returns:
            A JSON string.
        """
        if not isinstance(problem, Problem):
            raise ATLASError("to_json() expects a Problem")
        try:
            return self._engine.to_json(problem._c)
        except _prime_atlas.ATLASError as exc:
            raise ATLASError(str(exc)) from exc

    def from_json(self, json_text: str) -> Problem:
        """
        Deserialize a JSON string into a :class:`Problem`.

        Args:
            json_text: A JSON string previously produced by :meth:`to_json`.

        Returns:
            A :class:`Problem`.

        Raises:
            ATLASError: If the JSON is malformed or the schema version is
                unsupported.
        """
        if not isinstance(json_text, str):
            raise ATLASError("from_json() expects a str")
        if not json_text.strip():
            raise ATLASError("JSON input is empty")
        try:
            return Problem(self._engine.from_json(json_text))
        except _prime_atlas.ATLASError as exc:
            raise ATLASError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @staticmethod
    def schema_version() -> str:
        """Return the ATLAS schema version string (e.g. "1.0")."""
        return _prime_atlas.SCHEMA_VERSION

    def __repr__(self) -> str:
        return f"<ATLASEngine schema_version={self.schema_version()!r}>"
