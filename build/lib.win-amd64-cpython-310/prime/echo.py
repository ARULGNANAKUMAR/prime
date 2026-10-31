"""
prime.echo — PRIME Phase 7: ECHO Experience-to-Intelligence Engine.

ECHO = Experience-to-Intelligence Engine

ECHO allows PRIME to learn from its own previous experiences at a structured
knowledge level.  It records completed executions, extracts observations,
detects patterns across multiple experiences, generalises repeated observations
into candidate principles, and exposes a query API so future phases can
retrieve what PRIME has already learned.

ECHO boundary (strict):

  MAY:  record, observe, compare, detect patterns, generalise, create
        knowledge candidates, identify successful/failing strategies,
        detect contradictions, update structured statistics.

  MUST NOT:  update LLM weights, fine-tune models, train neural networks,
             perform gradient descent, modify model parameters, implement
             continual learning, replay-based training, or online gradient
             updates.  Those belong to Phase 8.

Typical usage::

    from prime import PRIME

    prime  = PRIME()
    mosaic = prime.mosaic()
    echo   = prime.echo()

    result  = mosaic.solve("A car travels 150 km in 3 hours.")
    exp     = echo.record(problem_text="A car travels 150 km in 3 hours.",
                          execution_result=result)
    obs     = echo.observe(exp)
    echo.analyze(exp)          # observe → patterns → generalise → validate

    patterns   = echo.get_patterns()
    knowledge  = echo.get_knowledge()
    similar    = echo.find_similar(exp)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums / status codes
# ---------------------------------------------------------------------------

class ExperienceOutcome(str, Enum):
    SUCCESS            = "SUCCESS"
    PARTIAL_SUCCESS    = "PARTIAL_SUCCESS"
    FAILURE            = "FAILURE"
    INVALID            = "INVALID"
    TIMEOUT            = "TIMEOUT"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    UNKNOWN            = "UNKNOWN"


class ExperienceLifecycle(str, Enum):
    CREATED              = "CREATED"
    OBSERVATIONS_EXTRACTED = "OBSERVATIONS_EXTRACTED"
    PATTERNS_ANALYZED    = "PATTERNS_ANALYZED"
    GENERALIZATION_CREATED = "GENERALIZATION_CREATED"
    PRINCIPLE_CANDIDATE  = "PRINCIPLE_CANDIDATE"
    VALIDATED            = "VALIDATED"
    KNOWLEDGE_STORED     = "KNOWLEDGE_STORED"


class EventType(str, Enum):
    TASK_RECEIVED       = "TASK_RECEIVED"
    ATLAS_PARSED        = "ATLAS_PARSED"
    MODULE_SELECTED     = "MODULE_SELECTED"
    REASONING_STARTED   = "REASONING_STARTED"
    REASONING_STEP      = "REASONING_STEP"
    MODULE_EXECUTED     = "MODULE_EXECUTED"
    VALIDATION_STARTED  = "VALIDATION_STARTED"
    VALIDATION_SUCCESS  = "VALIDATION_SUCCESS"
    VALIDATION_FAILURE  = "VALIDATION_FAILURE"
    RESULT_GENERATED    = "RESULT_GENERATED"
    EXPERIENCE_COMPLETED = "EXPERIENCE_COMPLETED"


class ValidationStatus(str, Enum):
    VALID               = "VALID"
    INVALID             = "INVALID"
    PARTIAL             = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTED        = "CONTRADICTED"


class KnowledgeType(str, Enum):
    OBSERVATION    = "OBSERVATION"
    PATTERN        = "PATTERN"
    GENERALIZATION = "GENERALIZATION"
    PRINCIPLE      = "PRINCIPLE"


class PrincipleStatus(str, Enum):
    CANDIDATE       = "CANDIDATE"
    SUPPORTED       = "SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    REJECTED        = "REJECTED"
    UNVALIDATED     = "UNVALIDATED"


class RetentionPolicy(str, Enum):
    KEEP_ALL        = "KEEP_ALL"
    KEEP_RECENT     = "KEEP_RECENT"
    KEEP_SUCCESSFUL = "KEEP_SUCCESSFUL"
    KEEP_HIGH_VALUE = "KEEP_HIGH_VALUE"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass
class ExperienceEvent:
    """A compact, machine-readable event during an experience."""
    event_type:  str
    timestamp:   float = field(default_factory=time.time)
    data:        Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"event_type": self.event_type,
                "timestamp": self.timestamp,
                "data": self.data}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperienceEvent":
        return cls(event_type=d["event_type"],
                   timestamp=d.get("timestamp", 0.0),
                   data=d.get("data", {}))


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """
    A complete, structured record of one PRIME execution.

    Compact by design: large raw traces are not stored; only summaries and
    structural information that ECHO can analyse.
    """
    # Identity
    id:           str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:    float = field(default_factory=time.time)

    # Task context
    task_id:      str = ""
    task_text:    str = ""
    task_domain:  str = "unknown"
    task_category:str = "unknown"   # e.g. "arithmetic", "constraint", "logic"

    # ATLAS structural summary (compact)
    atlas_summary: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"domain": "physics", "entities": ["distance","time"],
    #        "operations": ["division"], "constraints": 0}

    # MOSAIC composition
    modules_used:      List[str] = field(default_factory=list)
    composition_plan:  str = ""     # plan status string
    fallback_used:     bool = False
    selected_reasons:  List[str] = field(default_factory=list)

    # Execution
    result_answer:      str = ""
    result_answer_unit: str = ""
    result_confidence:  float = 0.0
    result_status:      str = ""    # MOSAIC ExecutionResult.status
    reasoning_status:   str = ""

    # Outcome
    outcome:  str = ExperienceOutcome.UNKNOWN.value
    validated:bool = False

    # Events (compact)
    events: List[ExperienceEvent] = field(default_factory=list)

    # Lifecycle
    lifecycle: str = ExperienceLifecycle.CREATED.value

    # Metadata
    tags:     List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed at observation time
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "task_domain": self.task_domain,
            "task_category": self.task_category,
            "atlas_summary": self.atlas_summary,
            "modules_used": self.modules_used,
            "composition_plan": self.composition_plan,
            "fallback_used": self.fallback_used,
            "selected_reasons": self.selected_reasons,
            "result_answer": self.result_answer,
            "result_answer_unit": self.result_answer_unit,
            "result_confidence": self.result_confidence,
            "result_status": self.result_status,
            "reasoning_status": self.reasoning_status,
            "outcome": self.outcome,
            "validated": self.validated,
            "events": [e.to_dict() for e in self.events],
            "lifecycle": self.lifecycle,
            "tags": self.tags,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        obj = cls()
        for k, v in d.items():
            if k == "events":
                obj.events = [ExperienceEvent.from_dict(e) for e in v]
            elif hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        return (f"<Experience id={self.id[:8]}… domain={self.task_domain!r} "
                f"outcome={self.outcome} lifecycle={self.lifecycle}>")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    A structured description of *what happened* in one experience.

    Observations describe facts; they are NOT generalisations.
    """
    id:            str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str = ""
    timestamp:     float = field(default_factory=time.time)

    task_category: str = "unknown"
    task_domain:   str = "unknown"
    modules_used:  List[str] = field(default_factory=list)
    primary_module:str = ""
    outcome:       str = ExperienceOutcome.UNKNOWN.value
    validated:     bool = False
    confidence:    float = 0.0
    fallback_used: bool = False
    reasoning_status: str = ""
    has_arithmetic:   bool = False
    has_constraints:  bool = False
    has_rules:        bool = False
    tags:          List[str] = field(default_factory=list)
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        return (f"<Observation exp={self.experience_id[:8]}… "
                f"domain={self.task_domain!r} outcome={self.outcome}>")


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """
    A detected repeated relationship across multiple experiences.

    Confidence represents *evidence strength*, NOT probability.
    """
    id:                   str = field(default_factory=lambda: str(uuid.uuid4()))
    description:          str = ""
    condition_domain:     str = ""
    condition_category:   str = ""
    condition_module:     str = ""   # primary module observed
    observation_count:    int = 0
    success_count:        int = 0
    failure_count:        int = 0
    confidence:           float = 0.0   # evidence-strength score 0-1
    first_seen:           float = field(default_factory=time.time)
    last_seen:            float = field(default_factory=time.time)
    supporting_experience_ids: List[str] = field(default_factory=list)
    contradicting_experience_ids: List[str] = field(default_factory=list)
    tags:                 List[str] = field(default_factory=list)
    metadata:             Dict[str, Any] = field(default_factory=dict)

    # Versioning
    version:              int = 1
    previous_confidence:  float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def is_contradicted(self) -> bool:
        return self.failure_count > 0 and self.success_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pattern":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        return (f"<Pattern id={self.id[:8]}… "
                f"domain={self.condition_domain!r} "
                f"module={self.condition_module!r} "
                f"obs={self.observation_count} "
                f"conf={self.confidence:.2f}>")


# ---------------------------------------------------------------------------
# Generalization
# ---------------------------------------------------------------------------

@dataclass
class Generalization:
    """
    A broader structured statement derived from repeated observations.

    Generalisations are EVIDENCE-SCOPED: they never claim universal truth.
    """
    id:                 str = field(default_factory=lambda: str(uuid.uuid4()))
    statement:          str = ""
    scope:              str = ""   # e.g. "arithmetic tasks in this session"
    confidence:         float = 0.0
    supporting_pattern_ids: List[str] = field(default_factory=list)
    supporting_experience_ids: List[str] = field(default_factory=list)
    exceptions:         List[str] = field(default_factory=list)
    limitations:        List[str] = field(default_factory=list)
    created_at:         float = field(default_factory=time.time)
    metadata:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Generalization":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        snippet = self.statement[:60] + ("…" if len(self.statement) > 60 else "")
        return f"<Generalization conf={self.confidence:.2f} {snippet!r}>"


# ---------------------------------------------------------------------------
# Knowledge Candidate
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeCandidate:
    """
    A unified experience-derived knowledge item.

    Types: OBSERVATION → PATTERN → GENERALIZATION → PRINCIPLE.
    Confidence is evidence-strength (not statistical probability).
    """
    id:               str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_type:   str = KnowledgeType.OBSERVATION.value
    statement:        str = ""
    conditions:       List[str] = field(default_factory=list)
    source_experience_ids: List[str] = field(default_factory=list)
    supporting_observation_ids: List[str] = field(default_factory=list)
    supporting_pattern_ids: List[str] = field(default_factory=list)
    confidence:       float = 0.0
    validation_status:str = PrincipleStatus.UNVALIDATED.value
    limitations:      List[str] = field(default_factory=list)
    created_at:       float = field(default_factory=time.time)
    updated_at:       float = field(default_factory=time.time)
    version:          int = 1
    change_log:       List[str] = field(default_factory=list)
    metadata:         Dict[str, Any] = field(default_factory=dict)

    # EchoTrace: full provenance chain
    trace: Dict[str, Any] = field(default_factory=dict)

    def update_confidence(self, new_conf: float, reason: str = "") -> None:
        self.previous_confidence = self.confidence
        self.confidence = new_conf
        self.version += 1
        self.updated_at = time.time()
        entry = f"v{self.version}: confidence {self.previous_confidence:.2f}→{new_conf:.2f}"
        if reason:
            entry += f" ({reason})"
        self.change_log.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        if "previous_confidence" not in d:
            d["previous_confidence"] = 0.0
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeCandidate":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        snippet = self.statement[:60] + ("…" if len(self.statement) > 60 else "")
        return (f"<KnowledgeCandidate type={self.knowledge_type} "
                f"conf={self.confidence:.2f} status={self.validation_status} "
                f"{snippet!r}>")


# ---------------------------------------------------------------------------
# EchoTrace
# ---------------------------------------------------------------------------

class EchoTrace:
    """
    Full provenance chain for any derived insight.

    Answers: "Why did ECHO generate this knowledge?"
    The answer always comes from stored evidence — never LLM invention.
    """

    @staticmethod
    def build_for_candidate(
        candidate: KnowledgeCandidate,
        patterns: List[Pattern],
        observations: List[Observation],
        experiences: List[Experience],
    ) -> Dict[str, Any]:
        pat_ids = candidate.supporting_pattern_ids
        obs_ids = candidate.supporting_observation_ids
        exp_ids = candidate.source_experience_ids

        pat_map = {p.id: p.description for p in patterns if p.id in pat_ids}
        obs_map = {o.id: o.outcome for o in observations if o.id in obs_ids}
        exp_map = {e.id: e.task_text[:80] for e in experiences if e.id in exp_ids}

        return {
            "candidate_id": candidate.id,
            "knowledge_type": candidate.knowledge_type,
            "statement": candidate.statement,
            "evidence": {
                "patterns": pat_map,
                "observations": obs_map,
                "experiences": exp_map,
            },
            "confidence": candidate.confidence,
            "validation_status": candidate.validation_status,
            "change_log": candidate.change_log,
            "generated_at": time.time(),
        }


# ---------------------------------------------------------------------------
# Experience Store (SQLite + JSONL fallback)
# ---------------------------------------------------------------------------

class ExperienceStore:
    """
    Lightweight, local, offline experience storage.

    Uses SQLite when available; falls back to JSONL.
    Safe: one malformed record never corrupts the rest.
    """

    def __init__(self, db_path: Optional[str] = None,
                 max_experiences: int = 10_000,
                 retention_policy: str = RetentionPolicy.KEEP_ALL.value) -> None:
        self._max_experiences = max_experiences
        self._retention_policy = retention_policy

        # In-memory store (always available)
        self._experiences:  Dict[str, Experience]         = {}
        self._observations: Dict[str, Observation]        = {}
        self._patterns:     Dict[str, Pattern]            = {}
        self._generalizations: Dict[str, Generalization]  = {}
        self._knowledge:    Dict[str, KnowledgeCandidate] = {}

        # SQLite persistence (optional)
        self._conn: Optional[sqlite3.Connection] = None
        if db_path:
            try:
                self._conn = sqlite3.connect(db_path, check_same_thread=False)
                self._init_db()
            except Exception:
                self._conn = None   # degrade gracefully to in-memory

    # --- DB init -----------------------------------------------------------

    def _init_db(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        for table, col in [
            ("experiences",     "experience_id"),
            ("observations",    "observation_id"),
            ("patterns",        "pattern_id"),
            ("generalizations", "generalization_id"),
            ("knowledge",       "knowledge_id"),
        ]:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                f"({col} TEXT PRIMARY KEY, data TEXT NOT NULL, "
                f"created_at REAL, outcome TEXT, domain TEXT, tags TEXT)"
            )
        self._conn.commit()

    def _upsert(self, table: str, pk_col: str, pk_val: str,
                data: Dict[str, Any], outcome: str = "",
                domain: str = "", tags: List[str] = None) -> None:
        if self._conn is None:
            return
        try:
            tags_str = json.dumps(tags or [])
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} "
                f"({pk_col}, data, created_at, outcome, domain, tags) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                (pk_val, json.dumps(data), time.time(), outcome, domain, tags_str)
            )
            self._conn.commit()
        except Exception:
            pass  # never corrupt in-memory store for a DB failure

    # --- Experiences -------------------------------------------------------

    def add_experience(self, exp: Experience) -> None:
        self._experiences[exp.id] = exp
        self._upsert("experiences", "experience_id", exp.id, exp.to_dict(),
                     outcome=exp.outcome, domain=exp.task_domain, tags=exp.tags)
        self._enforce_retention()

    def get_experience(self, exp_id: str) -> Optional[Experience]:
        """Return one stored experience by ID, or ``None`` when absent."""
        return self._experiences.get(exp_id)

    def get(self, exp_id: str) -> Optional[Experience]:
        """Backward-compatible shorthand for :meth:`get_experience`."""
        return self.get_experience(exp_id)

    def list_experiences(self, outcome: str = "", domain: str = "",
                         tag: str = "", limit: int = 0) -> List[Experience]:
        items = list(self._experiences.values())
        if outcome:
            items = [e for e in items if e.outcome == outcome]
        if domain:
            items = [e for e in items if e.task_domain == domain]
        if tag:
            items = [e for e in items if tag in e.tags]
        items.sort(key=lambda e: e.timestamp, reverse=True)
        return items[:limit] if limit > 0 else items

    def count_experiences(self) -> int:
        return len(self._experiences)

    def delete_experience(self, exp_id: str) -> bool:
        if exp_id in self._experiences:
            del self._experiences[exp_id]
            if self._conn:
                try:
                    self._conn.execute(
                        "DELETE FROM experiences WHERE experience_id=?", (exp_id,))
                    self._conn.commit()
                except Exception:
                    pass
            return True
        return False

    def _enforce_retention(self) -> None:
        if len(self._experiences) <= self._max_experiences:
            return
        excess = len(self._experiences) - self._max_experiences
        policy = self._retention_policy
        items = list(self._experiences.values())
        items.sort(key=lambda e: e.timestamp)   # oldest first

        if policy == RetentionPolicy.KEEP_RECENT.value:
            to_remove = items[:excess]
        elif policy == RetentionPolicy.KEEP_SUCCESSFUL.value:
            failed = [e for e in items if e.outcome == ExperienceOutcome.FAILURE.value]
            to_remove = failed[:excess] if len(failed) >= excess else items[:excess]
        elif policy == RetentionPolicy.KEEP_HIGH_VALUE.value:
            items_scored = sorted(items,
                                  key=lambda e: (e.result_confidence, e.timestamp))
            to_remove = items_scored[:excess]
        else:
            return  # KEEP_ALL

        for exp in to_remove:
            self.delete_experience(exp.id)

    # --- Observations -------------------------------------------------------

    def add_observation(self, obs: Observation) -> None:
        self._observations[obs.id] = obs
        self._upsert("observations", "observation_id", obs.id, obs.to_dict(),
                     outcome=obs.outcome, domain=obs.task_domain, tags=obs.tags)

    def get_observations_for_experience(self, exp_id: str) -> List[Observation]:
        return [o for o in self._observations.values()
                if o.experience_id == exp_id]

    def list_observations(self, domain: str = "", outcome: str = "") -> List[Observation]:
        items = list(self._observations.values())
        if domain:
            items = [o for o in items if o.task_domain == domain]
        if outcome:
            items = [o for o in items if o.outcome == outcome]
        return items

    # --- Patterns -----------------------------------------------------------

    def add_pattern(self, pat: Pattern) -> None:
        self._patterns[pat.id] = pat
        self._upsert("patterns", "pattern_id", pat.id, pat.to_dict(),
                     domain=pat.condition_domain, tags=pat.tags)

    def get_pattern(self, pat_id: str) -> Optional[Pattern]:
        return self._patterns.get(pat_id)

    def list_patterns(self, domain: str = "", category: str = "",
                      module: str = "") -> List[Pattern]:
        items = list(self._patterns.values())
        if domain:
            items = [p for p in items if p.condition_domain == domain]
        if category:
            items = [p for p in items if p.condition_category == category]
        if module:
            items = [p for p in items if p.condition_module == module]
        return items

    # --- Generalizations ----------------------------------------------------

    def add_generalization(self, gen: Generalization) -> None:
        self._generalizations[gen.id] = gen
        self._upsert("generalizations", "generalization_id", gen.id, gen.to_dict())

    def list_generalizations(self) -> List[Generalization]:
        return list(self._generalizations.values())

    # --- Knowledge candidates -----------------------------------------------

    def add_knowledge(self, kc: KnowledgeCandidate) -> None:
        self._knowledge[kc.id] = kc
        self._upsert("knowledge", "knowledge_id", kc.id, kc.to_dict())

    def get_knowledge(self, kc_id: str) -> Optional[KnowledgeCandidate]:
        return self._knowledge.get(kc_id)

    def update_knowledge(self, kc: KnowledgeCandidate) -> None:
        """Update an existing knowledge candidate (versioning)."""
        existing = self._knowledge.get(kc.id)
        if existing is not None:
            # Retain evidence history
            kc.version = existing.version + 1
            kc.change_log = existing.change_log + kc.change_log
        self._knowledge[kc.id] = kc
        self._upsert("knowledge", "knowledge_id", kc.id, kc.to_dict())

    def list_knowledge(self, knowledge_type: str = "",
                       validation_status: str = "") -> List[KnowledgeCandidate]:
        items = list(self._knowledge.values())
        if knowledge_type:
            items = [k for k in items if k.knowledge_type == knowledge_type]
        if validation_status:
            items = [k for k in items if k.validation_status == validation_status]
        return items

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Confidence Estimator
# ---------------------------------------------------------------------------

class ConfidenceEstimator:
    """
    Lightweight evidence-strength estimator.

    Confidence is NOT a calibrated probability; it is a 0-1 score reflecting
    the quantity and consistency of available evidence.
    """

    # Tuning weights
    W_COUNT       = 0.40
    W_CONSISTENCY = 0.35
    W_VALIDATION  = 0.25
    CONTRADICTION_PENALTY_PER = 0.08

    @staticmethod
    def estimate(support_count: int, success_count: int, failure_count: int,
                 validated_count: int = 0, diversity: float = 1.0) -> float:
        """
        Args:
            support_count:  total observations
            success_count:  successful observations
            failure_count:  failed observations
            validated_count: externally validated observations
            diversity:      0-1, how varied the contexts were

        Returns:
            Evidence-strength score in [0, 1].
        """
        if support_count == 0:
            return 0.0

        # Count factor: saturates at ~20 observations
        count_score = min(1.0, math.log1p(support_count) / math.log1p(20))

        # Consistency factor
        total = success_count + failure_count
        if total > 0:
            consistency = success_count / total
        else:
            consistency = 0.5

        # Validation quality
        val_score = min(1.0, validated_count / max(support_count, 1))

        raw = (ConfidenceEstimator.W_COUNT       * count_score
               + ConfidenceEstimator.W_CONSISTENCY * consistency
               + ConfidenceEstimator.W_VALIDATION  * val_score)

        # Contradiction penalty
        contradiction_penalty = (failure_count
                                 * ConfidenceEstimator.CONTRADICTION_PENALTY_PER)

        score = max(0.0, min(1.0, raw - contradiction_penalty))
        return round(score, 4)


# ---------------------------------------------------------------------------
# Observation Extractor
# ---------------------------------------------------------------------------

class ObservationExtractor:
    """
    Converts a raw Experience into one or more structured Observations.

    Observations describe WHAT HAPPENED, not why or what it means.
    """

    def extract(self, exp: Experience) -> List[Observation]:
        obs = Observation(
            experience_id   = exp.id,
            task_category   = exp.task_category,
            task_domain     = exp.task_domain,
            modules_used    = list(exp.modules_used),
            primary_module  = exp.modules_used[0] if exp.modules_used else "",
            outcome         = exp.outcome,
            validated       = exp.validated,
            confidence      = exp.result_confidence,
            fallback_used   = exp.fallback_used,
            reasoning_status= exp.reasoning_status,
            has_arithmetic  = "arithmetic" in exp.task_category.lower()
                              or "arithmetic" in exp.task_domain.lower()
                              or "arithmetic" in exp.atlas_summary.get("domain", "").lower(),
            has_constraints = bool(exp.atlas_summary.get("constraints", 0)),
            has_rules       = "rule" in str(exp.tags).lower()
                              or "logic" in exp.task_domain.lower(),
            tags            = list(exp.tags),
        )
        return [obs]


# ---------------------------------------------------------------------------
# Pattern Detector
# ---------------------------------------------------------------------------

_MIN_OBS_FOR_PATTERN = 2   # minimum observations to form a pattern


class PatternDetector:
    """
    Detects repeated relationships across multiple observations.

    Patterns are described as observed associations, NOT guaranteed rules.
    """

    def detect(self, observations: List[Observation],
               existing_patterns: List[Pattern]) -> List[Pattern]:
        """
        Update or create patterns based on all current observations.

        Returns:
            List of new or updated Pattern objects.
        """
        # Group observations by (domain, category, primary_module)
        groups: Dict[Tuple[str, str, str], List[Observation]] = {}
        for obs in observations:
            key = (obs.task_domain, obs.task_category, obs.primary_module)
            groups.setdefault(key, []).append(obs)

        # Index existing patterns for update
        pat_index: Dict[Tuple[str, str, str], Pattern] = {}
        for pat in existing_patterns:
            key = (pat.condition_domain, pat.condition_category,
                   pat.condition_module)
            pat_index[key] = pat

        updated: List[Pattern] = []
        for key, obs_list in groups.items():
            domain, category, module = key
            if len(obs_list) < _MIN_OBS_FOR_PATTERN:
                continue  # not enough evidence

            successes    = [o for o in obs_list
                            if o.outcome == ExperienceOutcome.SUCCESS.value]
            failures     = [o for o in obs_list
                            if o.outcome == ExperienceOutcome.FAILURE.value]
            validated    = [o for o in obs_list if o.validated]
            exp_ids_succ = [o.experience_id for o in successes]
            exp_ids_fail = [o.experience_id for o in failures]

            conf = ConfidenceEstimator.estimate(
                support_count=len(obs_list),
                success_count=len(successes),
                failure_count=len(failures),
                validated_count=len(validated),
            )

            if key in pat_index:
                pat = pat_index[key]
                pat.previous_confidence  = pat.confidence
                pat.observation_count    = len(obs_list)
                pat.success_count        = len(successes)
                pat.failure_count        = len(failures)
                pat.confidence           = conf
                pat.last_seen            = time.time()
                pat.supporting_experience_ids   = exp_ids_succ
                pat.contradicting_experience_ids = exp_ids_fail
                pat.version             += 1
                pat.description          = self._describe(pat)
            else:
                pat = Pattern(
                    condition_domain   = domain,
                    condition_category = category,
                    condition_module   = module,
                    observation_count  = len(obs_list),
                    success_count      = len(successes),
                    failure_count      = len(failures),
                    confidence         = conf,
                    supporting_experience_ids   = exp_ids_succ,
                    contradicting_experience_ids = exp_ids_fail,
                )
                pat.description = self._describe(pat)

            updated.append(pat)

        return updated

    @staticmethod
    def _describe(pat: Pattern) -> str:
        total = pat.success_count + pat.failure_count
        if pat.is_contradicted:
            return (
                f"In {pat.observation_count} observed {pat.condition_domain!r} "
                f"{pat.condition_category!r} tasks using {pat.condition_module!r}: "
                f"{pat.success_count} successes, {pat.failure_count} failures "
                f"(mixed outcome; confidence {pat.confidence:.2f})."
            )
        return (
            f"Observed association: {pat.condition_domain!r} "
            f"{pat.condition_category!r} tasks with module "
            f"{pat.condition_module!r} succeeded {pat.success_count}/{total} "
            f"times (confidence {pat.confidence:.2f})."
        )


# ---------------------------------------------------------------------------
# Generalization Engine
# ---------------------------------------------------------------------------

_MIN_PATTERN_CONF_FOR_GEN = 0.50
_MIN_OBS_FOR_GEN = 3


class GeneralizationEngine:
    """
    Converts repeated validated patterns into broader structured statements.

    Generalisations are ALWAYS scoped to the observed evidence.
    They NEVER claim universal truth.
    """

    def generalize(self, patterns: List[Pattern],
                   experiences: List[Experience]) -> List[Generalization]:
        """
        Produce generalizations from patterns that have sufficient evidence.
        """
        exp_index = {e.id: e for e in experiences}
        generalizations: List[Generalization] = []

        for pat in patterns:
            if pat.confidence < _MIN_PATTERN_CONF_FOR_GEN:
                continue
            if pat.observation_count < _MIN_OBS_FOR_GEN:
                continue

            scope = (f"{pat.observation_count} observed "
                     f"{pat.condition_domain!r} {pat.condition_category!r} "
                     f"experiences in this session")

            limitations = []
            exceptions  = []
            if pat.is_contradicted:
                limitations.append(
                    f"{pat.failure_count} contradicting observations exist.")
                exceptions.append(
                    "Strategy did not always succeed; conditions may differ.")

            if pat.success_count == pat.observation_count:
                statement = (
                    f"For the {scope}, module {pat.condition_module!r} has "
                    f"consistently produced validated results in "
                    f"{pat.condition_domain!r} {pat.condition_category!r} tasks "
                    f"({pat.success_count}/{pat.observation_count} successes). "
                    f"Evidence strength: {pat.confidence:.2f}."
                )
            else:
                statement = (
                    f"For the {scope}, module {pat.condition_module!r} frequently "
                    f"produced results in {pat.condition_domain!r} "
                    f"{pat.condition_category!r} tasks "
                    f"({pat.success_count}/{pat.observation_count} successes). "
                    f"Contradictions exist ({pat.failure_count} failures). "
                    f"Evidence strength: {pat.confidence:.2f}."
                )

            supporting_exp = [eid for eid in pat.supporting_experience_ids
                              if eid in exp_index]

            gen = Generalization(
                statement      = statement,
                scope          = scope,
                confidence     = pat.confidence * 0.9,  # slightly less than pattern
                supporting_pattern_ids     = [pat.id],
                supporting_experience_ids  = supporting_exp,
                exceptions     = exceptions,
                limitations    = limitations,
            )
            generalizations.append(gen)

        return generalizations


# ---------------------------------------------------------------------------
# Principle Extractor
# ---------------------------------------------------------------------------

_MIN_PATTERN_CONF_FOR_PRINCIPLE = 0.70
_MIN_OBS_FOR_PRINCIPLE = 5


class PrincipleExtractor:
    """
    Derives higher-level structured insights from validated repeated experience.

    Principles are CANDIDATES until sufficient evidence exists.
    They NEVER claim to be established scientific knowledge.
    """

    def extract_principles(
        self,
        patterns: List[Pattern],
        generalizations: List[Generalization],
    ) -> List[KnowledgeCandidate]:
        """
        Produce principle-level KnowledgeCandidates from strong patterns.
        """
        candidates: List[KnowledgeCandidate] = []
        gen_index = {gid: g for g in generalizations
                     for gid in [g.id]}

        for pat in patterns:
            if pat.confidence < _MIN_PATTERN_CONF_FOR_PRINCIPLE:
                continue
            if pat.observation_count < _MIN_OBS_FOR_PRINCIPLE:
                continue

            gen_ids = [g.id for g in generalizations
                       if pat.id in g.supporting_pattern_ids]

            limitations = []
            if pat.is_contradicted:
                limitations.append(
                    f"{pat.failure_count} contradicting observations weaken this principle.")
            limitations.append(
                "This is an evidence-derived candidate; it may not generalise beyond "
                f"the {pat.observation_count} observed cases."
            )

            if not pat.is_contradicted:
                val_status = PrincipleStatus.SUPPORTED.value
            elif pat.success_rate >= 0.8:
                val_status = PrincipleStatus.WEAKLY_SUPPORTED.value
            else:
                val_status = PrincipleStatus.CANDIDATE.value

            statement = (
                f"[Candidate Principle] In {pat.condition_domain!r} "
                f"{pat.condition_category!r} tasks, module "
                f"{pat.condition_module!r} may be a reliable strategy. "
                f"Observed success rate: {pat.success_rate:.0%} "
                f"over {pat.observation_count} cases "
                f"(evidence strength {pat.confidence:.2f}). "
                f"This candidate requires further validation before being "
                f"treated as established knowledge."
            )

            kc = KnowledgeCandidate(
                knowledge_type = KnowledgeType.PRINCIPLE.value,
                statement      = statement,
                conditions     = [
                    f"domain == {pat.condition_domain!r}",
                    f"category == {pat.condition_category!r}",
                    f"primary_module == {pat.condition_module!r}",
                ],
                source_experience_ids = list(pat.supporting_experience_ids),
                supporting_pattern_ids = [pat.id],
                supporting_observation_ids = gen_ids,
                confidence     = pat.confidence,
                validation_status = val_status,
                limitations    = limitations,
                trace          = {
                    "source_pattern": pat.id,
                    "observations": pat.observation_count,
                    "successes": pat.success_count,
                    "failures": pat.failure_count,
                },
            )
            candidates.append(kc)

        return candidates


# ---------------------------------------------------------------------------
# Experience Validator
# ---------------------------------------------------------------------------

class ExperienceValidator:
    """
    Validates experiences and knowledge candidates.

    Reuses outcome information from the execution pipeline; does not re-run
    reasoning.
    """

    def validate_experience(self, exp: Experience) -> str:
        """Returns a ValidationStatus string."""
        if not exp.task_text:
            return ValidationStatus.INVALID.value
        if exp.outcome == ExperienceOutcome.UNKNOWN.value:
            return ValidationStatus.PARTIAL.value
        if exp.outcome in (ExperienceOutcome.FAILURE.value,
                           ExperienceOutcome.INVALID.value):
            return ValidationStatus.INVALID.value
        if exp.validated:
            return ValidationStatus.VALID.value
        return ValidationStatus.PARTIAL.value

    def validate_knowledge_candidate(
        self,
        kc: KnowledgeCandidate,
        store: ExperienceStore,
    ) -> str:
        """Update and return validation status for a knowledge candidate."""
        if len(kc.source_experience_ids) == 0:
            kc.validation_status = ValidationStatus.INSUFFICIENT_EVIDENCE.value
            return kc.validation_status

        # Check for contradictions
        source_exps = [store.get_experience(eid)
                       for eid in kc.source_experience_ids]
        source_exps = [e for e in source_exps if e is not None]

        failures = [e for e in source_exps
                    if e.outcome == ExperienceOutcome.FAILURE.value]
        if failures and kc.confidence < 0.4:
            kc.validation_status = ValidationStatus.CONTRADICTED.value
        elif len(source_exps) < 2:
            kc.validation_status = ValidationStatus.INSUFFICIENT_EVIDENCE.value
        elif kc.confidence >= _MIN_PATTERN_CONF_FOR_PRINCIPLE:
            kc.validation_status = ValidationStatus.VALID.value
        elif kc.confidence >= _MIN_PATTERN_CONF_FOR_GEN:
            kc.validation_status = ValidationStatus.PARTIAL.value
        else:
            kc.validation_status = ValidationStatus.INSUFFICIENT_EVIDENCE.value

        return kc.validation_status


# ---------------------------------------------------------------------------
# Experience Similarity
# ---------------------------------------------------------------------------

class ExperienceSimilarity:
    """
    Lightweight, deterministic structural similarity between experiences.

    Does NOT require embedding models.
    """

    WEIGHTS = {
        "domain":   0.30,
        "category": 0.25,
        "modules":  0.25,
        "tags":     0.10,
        "outcome":  0.10,
    }

    def similarity(self, a: Experience, b: Experience) -> float:
        """Returns a score in [0, 1]."""
        scores: Dict[str, float] = {}

        # Domain
        scores["domain"] = 1.0 if a.task_domain == b.task_domain else 0.0

        # Category
        scores["category"] = 1.0 if a.task_category == b.task_category else 0.0

        # Modules (Jaccard)
        ma, mb = set(a.modules_used), set(b.modules_used)
        union = ma | mb
        scores["modules"] = len(ma & mb) / len(union) if union else 1.0

        # Tags (Jaccard)
        ta, tb = set(a.tags), set(b.tags)
        tunion = ta | tb
        scores["tags"] = len(ta & tb) / len(tunion) if tunion else 1.0

        # Outcome
        scores["outcome"] = 1.0 if a.outcome == b.outcome else 0.0

        total = sum(self.WEIGHTS[k] * scores[k] for k in self.WEIGHTS)
        return round(total, 4)

    def find_similar(self, target: Experience, candidates: List[Experience],
                     top_k: int = 5, min_score: float = 0.3) -> List[Tuple[Experience, float]]:
        scored = [
            (exp, self.similarity(target, exp))
            for exp in candidates
            if exp.id != target.id
        ]
        scored = [(e, s) for e, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Experience Collector
# ---------------------------------------------------------------------------

class ExperienceCollector:
    """
    Captures completed PRIME execution experiences from MOSAIC/ATLAS output.

    Integrates with ExecutionResult and Problem objects without duplicating
    those systems.
    """

    def collect(
        self,
        problem_text: str = "",
        execution_result: Any = None,   # mosaic.ExecutionResult
        atlas_problem:   Any = None,    # atlas.Problem
        mosaic_plan:     Any = None,    # mosaic.CompositionPlan
        extra_tags:      Optional[List[str]] = None,
        extra_metadata:  Optional[Dict[str, Any]] = None,
    ) -> Experience:
        """
        Build an Experience from MOSAIC/ATLAS execution outputs.

        All arguments are optional; the collector degrades gracefully when
        subsystems did not run.
        """
        exp = Experience(
            task_id   = str(uuid.uuid4()),
            task_text = problem_text,
            tags      = list(extra_tags or []),
            metadata  = dict(extra_metadata or {}),
        )

        # --- ATLAS info ---
        if atlas_problem is not None:
            try:
                exp.task_domain   = getattr(atlas_problem, "domain", "unknown") or "unknown"
                exp.atlas_summary = self._summarize_atlas(atlas_problem)
                exp.task_category = self._infer_category(atlas_problem)
                exp.events.append(ExperienceEvent(
                    event_type=EventType.ATLAS_PARSED.value,
                    data={"domain": exp.task_domain}
                ))
            except Exception:
                pass

        # --- MOSAIC plan info ---
        if mosaic_plan is not None:
            try:
                exp.modules_used     = list(getattr(mosaic_plan, "module_ids", []))
                exp.composition_plan = getattr(mosaic_plan, "status", "")
                exp.selected_reasons = list(getattr(mosaic_plan, "selection_reasons", []))
                for mid in exp.modules_used:
                    exp.events.append(ExperienceEvent(
                        event_type=EventType.MODULE_SELECTED.value,
                        data={"module_id": mid}
                    ))
            except Exception:
                pass

        # --- ExecutionResult info ---
        if execution_result is not None:
            try:
                exp.result_answer      = str(getattr(execution_result, "answer", ""))
                exp.result_answer_unit = str(getattr(execution_result, "answer_unit", ""))
                exp.result_confidence  = float(getattr(execution_result, "confidence", 0.0))
                exp.result_status      = str(getattr(execution_result, "status", ""))
                exp.reasoning_status   = str(getattr(execution_result, "reasoning_status", ""))
                exp.fallback_used      = any(
                    getattr(s, "fallback_used", False)
                    for s in getattr(execution_result, "trace_steps", [])
                )

                # Derive modules from trace if not already from plan
                if not exp.modules_used:
                    exp.modules_used = [
                        getattr(s, "module_id", "")
                        for s in getattr(execution_result, "trace_steps", [])
                    ]

                exp.outcome  = self._map_outcome(exp.result_status)
                exp.validated = exp.outcome in (
                    ExperienceOutcome.SUCCESS.value,
                    ExperienceOutcome.PARTIAL_SUCCESS.value,
                )

                exp.events.append(ExperienceEvent(
                    event_type=EventType.RESULT_GENERATED.value,
                    data={"status": exp.result_status,
                          "answer": exp.result_answer[:40]}
                ))
            except Exception:
                pass

        exp.events.append(ExperienceEvent(
            event_type=EventType.EXPERIENCE_COMPLETED.value,
            data={"outcome": exp.outcome}
        ))
        exp.lifecycle = ExperienceLifecycle.CREATED.value
        return exp

    @staticmethod
    def _summarize_atlas(problem: Any) -> Dict[str, Any]:
        """Extract a compact summary from an ATLAS Problem object."""
        summary: Dict[str, Any] = {}
        try:
            summary["domain"] = getattr(problem, "domain", "") or ""
            entities = getattr(problem, "entities", []) or []
            summary["entities"] = [
                getattr(e, "name", str(e)) for e in entities
            ][:10]
            ops = getattr(problem, "operations", []) or []
            summary["operations"] = [
                getattr(o, "type", str(o)) for o in ops
            ][:10]
            constraints = getattr(problem, "constraints", []) or []
            summary["constraints"] = len(constraints)
            goals = getattr(problem, "goals", []) or []
            summary["goals"] = len(goals)
        except Exception:
            pass
        return summary

    @staticmethod
    def _infer_category(problem: Any) -> str:
        """Infer a task category from the ATLAS problem domain."""
        domain = (getattr(problem, "domain", "") or "").lower()
        if domain in ("arithmetic", "mathematics", "math"):
            return "arithmetic"
        if domain in ("physics",):
            # check for typical arithmetic operations
            ops = getattr(problem, "operations", []) or []
            op_types = [getattr(o, "type", str(o)).lower() for o in ops]
            if any(t in ("division", "multiplication", "addition", "subtraction")
                   for t in op_types):
                return "arithmetic"
            return "physics"
        if domain in ("logic", "boolean"):
            return "logic"
        if domain in ("constraint", "scheduling"):
            return "constraint"
        return domain or "unknown"

    @staticmethod
    def _map_outcome(status: str) -> str:
        """Map MOSAIC ExecutionResult.status to ExperienceOutcome."""
        mapping = {
            "SUCCESS":        ExperienceOutcome.SUCCESS.value,
            "PARTIAL_SUCCESS":ExperienceOutcome.PARTIAL_SUCCESS.value,
            "FALLBACK_USED":  ExperienceOutcome.PARTIAL_SUCCESS.value,
            "FAILURE":        ExperienceOutcome.FAILURE.value,
            "MODULE_ERROR":   ExperienceOutcome.FAILURE.value,
            "LIMIT_EXCEEDED": ExperienceOutcome.TIMEOUT.value,
            "INVALID_PLAN":   ExperienceOutcome.INVALID.value,
        }
        return mapping.get(status, ExperienceOutcome.UNKNOWN.value)


# ---------------------------------------------------------------------------
# Experience Query Engine
# ---------------------------------------------------------------------------

class ExperienceQueryEngine:
    """
    Search and retrieve experience-derived knowledge.
    """

    def __init__(self, store: ExperienceStore,
                 similarity: ExperienceSimilarity) -> None:
        self._store      = store
        self._similarity = similarity

    def find_similar(self, target: Experience, top_k: int = 5,
                     min_score: float = 0.3) -> List[Tuple[Experience, float]]:
        all_exps = self._store.list_experiences()
        return self._similarity.find_similar(target, all_exps,
                                             top_k=top_k, min_score=min_score)

    def get_patterns(self, domain: str = "", category: str = "",
                     module: str = "") -> List[Pattern]:
        return self._store.list_patterns(domain=domain, category=category,
                                         module=module)

    def get_principles(self, domain: str = "") -> List[KnowledgeCandidate]:
        kcs = self._store.list_knowledge(
            knowledge_type=KnowledgeType.PRINCIPLE.value)
        if domain:
            kcs = [k for k in kcs if domain.lower() in k.statement.lower()]
        return kcs

    def get_knowledge(self, knowledge_type: str = "",
                      validation_status: str = "") -> List[KnowledgeCandidate]:
        return self._store.list_knowledge(knowledge_type=knowledge_type,
                                          validation_status=validation_status)

    def get_experiences(self, outcome: str = "", domain: str = "",
                        tag: str = "", limit: int = 50) -> List[Experience]:
        return self._store.list_experiences(outcome=outcome, domain=domain,
                                             tag=tag, limit=limit)

    def get_contradictions(self) -> List[Pattern]:
        return [p for p in self._store.list_patterns() if p.is_contradicted]

    def get_recent_experiences(self, n: int = 10) -> List[Experience]:
        return self._store.list_experiences(limit=n)


# ---------------------------------------------------------------------------
# ECHO Engine (main public class)
# ---------------------------------------------------------------------------

class ECHOEngine:
    """
    PRIME Phase 7 — ECHO Experience-to-Intelligence Engine.

    Provides a clean API for recording, analysing, and querying experience-
    derived knowledge.

    Example::

        echo = ECHOEngine()

        exp = echo.record(problem_text="A car travels 150 km in 3 hours.",
                          execution_result=result)
        echo.analyze(exp)

        print(echo.get_patterns())
        print(echo.get_knowledge())
    """

    def __init__(
        self,
        db_path:          Optional[str]  = None,
        max_experiences:  int            = 10_000,
        retention_policy: str            = RetentionPolicy.KEEP_ALL.value,
    ) -> None:
        self._store     = ExperienceStore(db_path=db_path,
                                          max_experiences=max_experiences,
                                          retention_policy=retention_policy)
        self._collector  = ExperienceCollector()
        self._obs_ext    = ObservationExtractor()
        self._pat_det    = PatternDetector()
        self._gen_eng    = GeneralizationEngine()
        self._pri_ext    = PrincipleExtractor()
        self._validator  = ExperienceValidator()
        self._similarity = ExperienceSimilarity()
        self._query      = ExperienceQueryEngine(self._store, self._similarity)
        self._conf_est   = ConfidenceEstimator()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        problem_text:     str       = "",
        execution_result: Any       = None,
        atlas_problem:    Any       = None,
        mosaic_plan:      Any       = None,
        extra_tags:       Optional[List[str]]        = None,
        extra_metadata:   Optional[Dict[str, Any]]   = None,
    ) -> Experience:
        """
        Capture a completed PRIME execution as a structured Experience.

        Returns the created Experience.
        """
        exp = self._collector.collect(
            problem_text     = problem_text,
            execution_result = execution_result,
            atlas_problem    = atlas_problem,
            mosaic_plan      = mosaic_plan,
            extra_tags       = extra_tags,
            extra_metadata   = extra_metadata,
        )
        self._store.add_experience(exp)
        return exp

    def observe(self, experience: Experience) -> List[Observation]:
        """
        Extract structured observations from an Experience.

        Updates the experience lifecycle to OBSERVATIONS_EXTRACTED.
        """
        observations = self._obs_ext.extract(experience)
        for obs in observations:
            self._store.add_observation(obs)

        experience.lifecycle = ExperienceLifecycle.OBSERVATIONS_EXTRACTED.value
        self._store.add_experience(experience)
        return observations

    def detect_patterns(self) -> List[Pattern]:
        """
        Detect or update patterns across all stored observations.

        Returns the current set of patterns.
        """
        all_obs      = self._store.list_observations()
        all_patterns = self._store.list_patterns()
        updated      = self._pat_det.detect(all_obs, all_patterns)
        for pat in updated:
            self._store.add_pattern(pat)
        return updated

    def generalize(self, patterns: Optional[List[Pattern]] = None
                   ) -> List[Generalization]:
        """
        Produce generalisations from current patterns.
        """
        if patterns is None:
            patterns = self._store.list_patterns()
        all_exps = self._store.list_experiences()
        gens     = self._gen_eng.generalize(patterns, all_exps)
        for gen in gens:
            self._store.add_generalization(gen)
        return gens

    def extract_knowledge(self) -> List[KnowledgeCandidate]:
        """
        Extract and validate knowledge candidates (patterns → principles).

        Returns newly created or updated KnowledgeCandidates.
        """
        patterns = self._store.list_patterns()
        gens     = self._store.list_generalizations()
        new_kcs  = self._pri_ext.extract_principles(patterns, gens)

        # Build pattern-level KnowledgeCandidates too
        for pat in patterns:
            if pat.observation_count >= _MIN_OBS_FOR_PATTERN:
                kc = KnowledgeCandidate(
                    knowledge_type = KnowledgeType.PATTERN.value,
                    statement      = pat.description,
                    conditions     = [
                        f"domain == {pat.condition_domain!r}",
                        f"category == {pat.condition_category!r}",
                    ],
                    source_experience_ids  = list(pat.supporting_experience_ids),
                    supporting_pattern_ids = [pat.id],
                    confidence             = pat.confidence,
                    trace                  = {"source_pattern": pat.id},
                )
                self._validator.validate_knowledge_candidate(kc, self._store)
                new_kcs.append(kc)

        for kc in new_kcs:
            self._validator.validate_knowledge_candidate(kc, self._store)
            existing = [k for k in self._store.list_knowledge()
                        if k.statement == kc.statement]
            if existing:
                kc.id = existing[0].id
                self._store.update_knowledge(kc)
            else:
                self._store.add_knowledge(kc)

        return self._store.list_knowledge()

    def analyze(self, experience: Experience) -> Dict[str, Any]:
        """
        Full ECHO pipeline for one experience:
        observe → detect_patterns → generalize → extract_knowledge → validate.

        Returns a summary dict.
        """
        observations = self.observe(experience)
        patterns     = self.detect_patterns()
        experience.lifecycle = ExperienceLifecycle.PATTERNS_ANALYZED.value
        self._store.add_experience(experience)

        gens         = self.generalize(patterns)
        if gens:
            experience.lifecycle = ExperienceLifecycle.GENERALIZATION_CREATED.value
            self._store.add_experience(experience)

        knowledge    = self.extract_knowledge()
        if knowledge:
            experience.lifecycle = ExperienceLifecycle.KNOWLEDGE_STORED.value
            self._store.add_experience(experience)

        val_status   = self._validator.validate_experience(experience)

        return {
            "experience_id":    experience.id,
            "lifecycle":        experience.lifecycle,
            "validation_status":val_status,
            "observations":     len(observations),
            "patterns":         len(patterns),
            "generalizations":  len(gens),
            "knowledge_candidates": len(knowledge),
        }

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def find_similar(self, experience: Experience, top_k: int = 5,
                     min_score: float = 0.3) -> List[Tuple[Experience, float]]:
        """Find experiences structurally similar to *experience*."""
        return self._query.find_similar(experience, top_k=top_k,
                                         min_score=min_score)

    def get_patterns(self, domain: str = "", category: str = "",
                     module: str = "") -> List[Pattern]:
        """Retrieve detected patterns, optionally filtered."""
        return self._query.get_patterns(domain=domain, category=category,
                                         module=module)

    def get_principles(self, domain: str = "") -> List[KnowledgeCandidate]:
        """Retrieve principle-level knowledge candidates."""
        return self._query.get_principles(domain=domain)

    def get_knowledge(self, knowledge_type: str = "",
                      validation_status: str = "") -> List[KnowledgeCandidate]:
        """Retrieve all knowledge candidates, optionally filtered."""
        return self._query.get_knowledge(knowledge_type=knowledge_type,
                                          validation_status=validation_status)

    def get_experiences(self, outcome: str = "", domain: str = "",
                        tag: str = "", limit: int = 50) -> List[Experience]:
        """Retrieve stored experiences, optionally filtered."""
        return self._query.get_experiences(outcome=outcome, domain=domain,
                                            tag=tag, limit=limit)

    def get_contradictions(self) -> List[Pattern]:
        """Return patterns where both successes and failures were observed."""
        return self._query.get_contradictions()

    def get_recent_experiences(self, n: int = 10) -> List[Experience]:
        """Return the *n* most recent experiences."""
        return self._query.get_recent_experiences(n)

    def trace(self, kc_id: str) -> Optional[Dict[str, Any]]:
        """
        Return full provenance for a KnowledgeCandidate.

        Answers: "Why did ECHO generate this knowledge?"
        """
        kc = self._store.get_knowledge(kc_id)
        if kc is None:
            return None
        patterns = [self._store.get_pattern(pid)
                    for pid in kc.supporting_pattern_ids]
        patterns = [p for p in patterns if p is not None]
        obs_ids  = kc.supporting_observation_ids
        obs_list = self._store.list_observations()
        obs_match = [o for o in obs_list if o.id in obs_ids]
        exp_ids  = kc.source_experience_ids
        exp_list = [self._store.get_experience(eid) for eid in exp_ids]
        exp_list = [e for e in exp_list if e is not None]
        return EchoTrace.build_for_candidate(kc, patterns, obs_match, exp_list)

    # ------------------------------------------------------------------
    # Statistics / Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary of ECHO's current state."""
        experiences  = self._store.list_experiences()
        patterns     = self._store.list_patterns()
        gens         = self._store.list_generalizations()
        knowledge    = self._store.list_knowledge()

        outcome_counts: Dict[str, int] = {}
        for e in experiences:
            outcome_counts[e.outcome] = outcome_counts.get(e.outcome, 0) + 1

        return {
            "total_experiences":    len(experiences),
            "total_observations":   len(self._store.list_observations()),
            "total_patterns":       len(patterns),
            "total_generalizations":len(gens),
            "total_knowledge_candidates": len(knowledge),
            "outcome_distribution": outcome_counts,
            "contradicted_patterns": sum(1 for p in patterns if p.is_contradicted),
            "high_confidence_patterns": sum(
                1 for p in patterns if p.confidence >= 0.7),
            "knowledge_by_type": {
                kt.value: sum(1 for k in knowledge if k.knowledge_type == kt.value)
                for kt in KnowledgeType
            },
        }

    def close(self) -> None:
        """Release resources."""
        self._store.close()

    def __repr__(self) -> str:
        s = self.summary()
        return (f"<ECHOEngine experiences={s['total_experiences']} "
                f"patterns={s['total_patterns']} "
                f"knowledge={s['total_knowledge_candidates']}>")


# ---------------------------------------------------------------------------
# ECHOError
# ---------------------------------------------------------------------------

class ECHOError(Exception):
    """Raised when an ECHO operation fails."""
