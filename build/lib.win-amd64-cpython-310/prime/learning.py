"""
prime.learning — PRIME Phase 8: Self-Learning & Continual Learning Engine.

The learning engine consumes structured knowledge produced by ECHO (Phase 7)
and translates it into measurable, controlled, auditable adaptations:

  - Strategy preference scores (per task-context)
  - Knowledge-candidate confidence updates
  - Rule-confidence updates
  - Concept-drift detection

Design principles
─────────────────
• Every update has a Before → After record (LearningTrace).
• No update is committed without passing validation checks.
• Hard constraints (incompatible types, missing modules, resource limits) always
  win; learned preferences are soft bonuses only.
• No neural-framework dependency (no PyTorch / TF / JAX).
  Actual parameter updates are implemented with lightweight Numpy-free arithmetic.
• Rollback to any prior LearningState version is always available.
• Fully offline and local; no network calls.

Phase boundary (strict)
───────────────────────
Phase 8 MAY:  update strategy scores, knowledge confidence, rule confidence,
              detect concept drift, replay experiences, version learning state.
Phase 8 MUST NOT: retrain the full LLM, perform gradient descent on LLM weights,
              implement neural backprop on the base model.
Phase 9 handles: adaptive resource optimisation.
Phase 10 handles: final integration, research validation, paper.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────────────────────

class LearningMode(str, Enum):
    OFF                      = "OFF"
    OBSERVATION_ONLY         = "OBSERVATION_ONLY"
    ADAPTIVE_KNOWLEDGE       = "ADAPTIVE_KNOWLEDGE"
    STRATEGY_LEARNING        = "STRATEGY_LEARNING"
    INCREMENTAL_MODEL_ADAPT  = "INCREMENTAL_MODEL_ADAPT"


class LearningTarget(str, Enum):
    STRATEGY          = "TARGET_STRATEGY"
    RULE              = "TARGET_RULE"
    KNOWLEDGE         = "TARGET_KNOWLEDGE"
    RETRIEVAL         = "TARGET_RETRIEVAL"
    MODULE_PREFERENCE = "TARGET_MODULE_PREFERENCE"
    MODEL_ADAPTER     = "TARGET_MODEL_ADAPTER"
    PARAMETER         = "TARGET_PARAMETER"


class UpdateStatus(str, Enum):
    PENDING   = "PENDING"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    REJECTED  = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class LearningEligibility(str, Enum):
    ELIGIBLE              = "ELIGIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BUDGET_EXHAUSTED      = "BUDGET_EXHAUSTED"
    MODE_DISABLED         = "MODE_DISABLED"
    DUPLICATE_EXPERIENCE  = "DUPLICATE_EXPERIENCE"
    SAFETY_LIMIT          = "SAFETY_LIMIT"
    CONTRADICTED          = "CONTRADICTED"


class DriftStatus(str, Enum):
    STABLE               = "STABLE"
    POSSIBLE_CONCEPT_DRIFT = "POSSIBLE_CONCEPT_DRIFT"
    CONFIRMED_DRIFT      = "CONFIRMED_DRIFT"
    UNKNOWN              = "UNKNOWN"


class ApprovalStatus(str, Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ────────────────────────────────────────────────────────────────────────────
# Reward schedule
# ────────────────────────────────────────────────────────────────────────────

# These are configurable design values, not universal scientific constants.
DEFAULT_REWARDS: Dict[str, float] = {
    "SUCCESS":          +1.0,
    "PARTIAL_SUCCESS":  +0.5,
    "FALLBACK_USED":    +0.3,
    "FAILURE":          -1.0,
    "INVALID":          -1.0,
    "TIMEOUT":          -0.5,
    "UNKNOWN":           0.0,
    "CONSTRAINT_VIOLATION": -0.8,
}


# ────────────────────────────────────────────────────────────────────────────
# Strategy Statistics (per context key)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyStats:
    """
    Running statistics for one (strategy / module, context) pair.

    context_key: compact string identifying the task context
                 e.g. "physics:arithmetic:reasoning"
    """
    strategy_id:    str
    context_key:    str
    attempts:       int   = 0
    successes:      int   = 0
    failures:       int   = 0
    partial:        int   = 0
    total_reward:   float = 0.0
    score:          float = 0.5   # initialised to neutral
    confidence:     float = 0.0
    last_used:      float = field(default_factory=time.time)
    last_updated:   float = field(default_factory=time.time)
    created_at:     float = field(default_factory=time.time)

    # Concept-drift tracking
    recent_rewards: List[float] = field(default_factory=list)  # sliding window
    drift_status:   str = DriftStatus.STABLE.value

    # Exploration tracking
    exploration_uses: int = 0

    def record_reward(self, reward: float,
                      window_size: int = 10) -> None:
        self.attempts      += 1
        self.total_reward  += reward
        self.last_used      = time.time()
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > window_size:
            self.recent_rewards.pop(0)
        if reward > 0:
            if reward >= 1.0:
                self.successes += 1
            else:
                self.partial   += 1
        else:
            self.failures      += 1

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures + self.partial
        return (self.successes + self.partial * 0.5) / total if total > 0 else 0.0

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.attempts if self.attempts > 0 else 0.0

    @property
    def recent_average_reward(self) -> float:
        return sum(self.recent_rewards) / len(self.recent_rewards) \
            if self.recent_rewards else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyStats":
        obj = cls(strategy_id=d["strategy_id"], context_key=d["context_key"])
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        return (f"<StrategyStats strategy={self.strategy_id!r} "
                f"ctx={self.context_key!r} "
                f"score={self.score:.3f} attempts={self.attempts}>")


# ────────────────────────────────────────────────────────────────────────────
# Candidate Update
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateUpdate:
    """
    A proposed learning update that has not yet been committed.

    Every field required to reconstruct Before → After state is stored here
    so the LearningTrace is self-contained.
    """
    id:            str = field(default_factory=lambda: str(uuid.uuid4()))
    target:        str = LearningTarget.STRATEGY.value
    target_id:     str = ""        # e.g. module_id or knowledge_id
    context_key:   str = ""

    old_value:     Any = None
    new_value:     Any = None
    delta:         float = 0.0

    reason:        str = ""
    evidence_ids:  List[str] = field(default_factory=list)  # experience IDs
    pattern_id:    str = ""

    status:        str = UpdateStatus.PENDING.value
    approval:      str = ApprovalStatus.PENDING.value
    validation_errors: List[str] = field(default_factory=list)

    created_at:    float = field(default_factory=time.time)
    committed_at:  float = 0.0
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateUpdate":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def __repr__(self) -> str:
        return (f"<CandidateUpdate target={self.target} id={self.target_id!r} "
                f"delta={self.delta:+.4f} status={self.status}>")


# ────────────────────────────────────────────────────────────────────────────
# Learning Trace Entry
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LearningTraceEntry:
    """
    Full audit record for one committed (or rejected) learning update.

    Answers: What changed? Why? Which experiences caused it?
    Old value? New value? Was the update validated?
    """
    trace_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    update_id:      str = ""
    target:         str = ""
    target_id:      str = ""
    context_key:    str = ""
    old_value:      Any = None
    new_value:      Any = None
    delta:          float = 0.0
    reason:         str = ""
    evidence_ids:   List[str] = field(default_factory=list)
    pattern_id:     str = ""
    status:         str = ""           # COMMITTED or REJECTED
    validation_errors: List[str] = field(default_factory=list)
    learning_version:  str = ""
    timestamp:      float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.2f}"
        return repr(v)

    def explain(self) -> str:
        lines = [
            f"LearningTrace [{self.trace_id[:8]}…]",
            f"  target:   {self.target} / {self.target_id!r}",
            f"  context:  {self.context_key!r}",
            f"  change:   {self._fmt(self.old_value)} → {self._fmt(self.new_value)}"
            f"  (delta={self.delta:+.4f})",
            f"  reason:   {self.reason}",
            f"  evidence: {self.evidence_ids}",
            f"  status:   {self.status}",
            f"  version:  {self.learning_version}",
        ]
        if self.validation_errors:
            lines.append(f"  errors:   {self.validation_errors}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<LearningTraceEntry target={self.target}/{self.target_id!r} "
                f"delta={self.delta:+.4f} {self.status}>")


# ────────────────────────────────────────────────────────────────────────────
# Learning State Version
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LearningStateVersion:
    """Snapshot of key learning state at a point in time (for rollback)."""
    version_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_version: str = ""
    update_id:      str = ""
    update_type:    str = ""
    target:         str = ""
    timestamp:      float = field(default_factory=time.time)
    reason:         str = ""
    # Compact snapshot (strategy scores and knowledge confidences only)
    snapshot:       Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningStateVersion":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ────────────────────────────────────────────────────────────────────────────
# Learned Policy (per-context module preferences exposed to MOSAIC)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LearnedPolicy:
    """
    A compact learned preference table: context_key → {module_id: bonus}.

    MOSAIC adds the bonus to its base capability score.
    Hard constraints always override; bonuses are soft.
    """
    context_key:    str
    preferences:    Dict[str, float] = field(default_factory=dict)
    # module_id → preference bonus in [-MAX_BONUS, +MAX_BONUS]

    MAX_BONUS: float = field(default=0.30, init=False, repr=False)

    def get_bonus(self, module_id: str) -> float:
        return max(-self.MAX_BONUS,
                   min(self.MAX_BONUS,
                       self.preferences.get(module_id, 0.0)))

    def set_bonus(self, module_id: str, bonus: float) -> None:
        self.preferences[module_id] = max(-self.MAX_BONUS,
                                          min(self.MAX_BONUS, bonus))

    def to_dict(self) -> Dict[str, Any]:
        return {"context_key": self.context_key,
                "preferences": self.preferences}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearnedPolicy":
        obj = cls(context_key=d["context_key"])
        obj.preferences = d.get("preferences", {})
        return obj


# ────────────────────────────────────────────────────────────────────────────
# Learning Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LearningConfig:
    """All tunable hyperparameters in one place."""
    mode:                   str   = LearningMode.ADAPTIVE_KNOWLEDGE.value
    learning_rate:          float = 0.08    # score update step size
    min_evidence:           int   = 2       # min supporting experiences
    min_confidence:         float = 0.40    # min ECHO confidence to trigger
    max_update_magnitude:   float = 0.20    # max |delta| per update
    max_score:              float = 1.00
    min_score:              float = 0.00
    max_knowledge_confidence: float = 0.95
    rollback_threshold:     float = 0.25    # if score drops by this much, flag
    concept_drift_threshold: float = 0.30   # |recent_avg - historic_avg| > this
    decay_factor:           float = 0.99    # score decay per time-window
    exploration_rate:       float = 0.05    # fraction of times to explore
    exploration_enabled:    bool  = False   # off by default
    reward_schedule:        Dict[str, float] = field(
                                default_factory=lambda: dict(DEFAULT_REWARDS))
    max_strategy_entries:   int   = 1000
    max_trace_entries:      int   = 5000
    max_versions:           int   = 200
    duplicate_window_secs:  float = 5.0     # experiences within this window
    require_approval:       bool  = False   # SEMI_AUTOMATIC gate
    auto_commit:            bool  = True    # commit after validation passes
    drift_window_size:      int   = 10      # recent window for drift detection

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningConfig":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ────────────────────────────────────────────────────────────────────────────
# Model Adapter Interface
# ────────────────────────────────────────────────────────────────────────────

class ModelAdapter:
    """
    Generic interface for a lightweight trainable component.

    Subclasses implement actual parameter storage and update logic.
    The base class is a no-op adapter (identity).

    Phase 8 ships one concrete subclass: StrategyBiasAdapter —
    a small table of learned scalar biases, one per (context, module) pair,
    updated with a bounded exponential-moving-average rule.
    """

    adapter_type: str = "identity"

    def update(self, context_key: str, target_id: str,
               reward: float, lr: float) -> Dict[str, Any]:
        """Apply one incremental update. Returns {old, new, delta}."""
        return {"old": 0.0, "new": 0.0, "delta": 0.0}

    def get_bias(self, context_key: str, target_id: str) -> float:
        """Return the current learned bias for (context, target)."""
        return 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"adapter_type": self.adapter_type}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        pass


class StrategyBiasAdapter(ModelAdapter):
    """
    Lightweight trainable adapter: one scalar bias per (context, module) pair.

    Update rule (bounded EMA):
        new_bias = clip(old_bias + lr * reward,  -MAX, +MAX)

    This is a real, measurable parameter update:
      Before state → learning update → After state → behavioral difference.

    The bias is exposed to MOSAIC as a soft preference bonus.
    """

    adapter_type: str = "strategy_bias"
    MAX_BIAS: float   = 0.30

    def __init__(self) -> None:
        # biases[(context_key, module_id)] = float
        self._biases: Dict[Tuple[str, str], float] = {}

    def update(self, context_key: str, target_id: str,
               reward: float, lr: float) -> Dict[str, Any]:
        key     = (context_key, target_id)
        old     = self._biases.get(key, 0.0)
        new     = max(-self.MAX_BIAS,
                      min(self.MAX_BIAS, old + lr * reward))
        self._biases[key] = new
        return {"old": old, "new": new, "delta": new - old}

    def get_bias(self, context_key: str, target_id: str) -> float:
        return self._biases.get((context_key, target_id), 0.0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "biases": {f"{k[0]}::{k[1]}": v
                       for k, v in self._biases.items()},
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        raw = d.get("biases", {})
        self._biases = {}
        for encoded_key, val in raw.items():
            parts = encoded_key.split("::", 1)
            if len(parts) == 2:
                self._biases[(parts[0], parts[1])] = float(val)

    def __repr__(self) -> str:
        return f"<StrategyBiasAdapter params={len(self._biases)}>"


# ────────────────────────────────────────────────────────────────────────────
# Stability Controller
# ────────────────────────────────────────────────────────────────────────────

class StabilityController:
    """
    Guards against catastrophic updates, oscillation, and runaway learning.

    All numeric bounds are sourced from LearningConfig.
    """

    def __init__(self, config: LearningConfig) -> None:
        self._cfg = config

    def check_update(self, candidate: CandidateUpdate,
                     current_value: float) -> List[str]:
        """Return a list of error strings; empty list means the update is safe."""
        errors: List[str] = []

        if not isinstance(candidate.new_value, (int, float)):
            # Non-numeric updates (e.g. status strings) bypass numeric checks
            return errors

        # Knowledge and rule confidence updates set an absolute ECHO-derived value.
        # The magnitude check does not apply to them — only bounds do.
        skip_magnitude = candidate.target in (
            LearningTarget.KNOWLEDGE.value,
            LearningTarget.RULE.value,
        )

        if not skip_magnitude:
            delta = abs(candidate.new_value - current_value)
            if delta > self._cfg.max_update_magnitude:
                errors.append(
                    f"Update magnitude {delta:.4f} exceeds limit "
                    f"{self._cfg.max_update_magnitude:.4f}")

        nv = float(candidate.new_value)
        if nv > self._cfg.max_knowledge_confidence:
            errors.append(
                f"New value {nv:.4f} exceeds max_knowledge_confidence "
                f"{self._cfg.max_knowledge_confidence:.4f}")
        if nv < 0.0:
            errors.append(f"New value {nv:.4f} is negative")

        return errors

    def check_drift(self, stats: "StrategyStats") -> str:
        """Return a DriftStatus based on recent vs historic reward."""
        if stats.attempts < self._cfg.drift_window_size:
            return DriftStatus.UNKNOWN.value
        if not stats.recent_rewards:
            return DriftStatus.STABLE.value

        historic_avg = stats.average_reward
        recent_avg   = stats.recent_average_reward
        diff         = abs(recent_avg - historic_avg)

        if diff > self._cfg.concept_drift_threshold * 2:
            return DriftStatus.CONFIRMED_DRIFT.value
        if diff > self._cfg.concept_drift_threshold:
            return DriftStatus.POSSIBLE_CONCEPT_DRIFT.value
        return DriftStatus.STABLE.value


# ────────────────────────────────────────────────────────────────────────────
# Rollback Manager
# ────────────────────────────────────────────────────────────────────────────

class RollbackManager:
    """Manages learning state checkpoints for rollback."""

    def __init__(self) -> None:
        self._versions: Dict[str, LearningStateVersion] = {}
        self._ordered:  List[str] = []    # version_ids newest-last

    def checkpoint(self, snapshot: Dict[str, Any],
                   update_id: str = "", target: str = "",
                   reason: str = "",
                   parent_version: str = "") -> LearningStateVersion:
        ver = LearningStateVersion(
            parent_version = parent_version,
            update_id      = update_id,
            update_type    = target,
            target         = target,
            reason         = reason,
            snapshot       = snapshot,
        )
        self._versions[ver.version_id] = ver
        self._ordered.append(ver.version_id)
        return ver

    def get_version(self, version_id: str) -> Optional[LearningStateVersion]:
        return self._versions.get(version_id)

    def list_versions(self) -> List[LearningStateVersion]:
        return [self._versions[vid] for vid in self._ordered
                if vid in self._versions]

    def latest_version_id(self) -> str:
        return self._ordered[-1] if self._ordered else ""

    def restore(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Return the snapshot for version_id, or None if not found."""
        ver = self._versions.get(version_id)
        return ver.snapshot if ver else None

    def prune(self, max_versions: int) -> None:
        while len(self._ordered) > max_versions:
            oldest = self._ordered.pop(0)
            self._versions.pop(oldest, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "versions": {vid: v.to_dict()
                         for vid, v in self._versions.items()},
            "ordered": self._ordered,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RollbackManager":
        obj = cls()
        obj._ordered = d.get("ordered", [])
        for vid, vd in d.get("versions", {}).items():
            obj._versions[vid] = LearningStateVersion.from_dict(vd)
        return obj


# ────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplayEntry:
    """Lightweight experience replay record (no raw text stored)."""
    experience_id:  str
    context_key:    str
    module_id:      str
    outcome:        str
    reward:         float
    confidence:     float
    timestamp:      float = field(default_factory=time.time)
    signature:      str   = ""   # dedup key

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReplayEntry":
        obj = cls(experience_id=d["experience_id"], context_key=d["context_key"],
                  module_id=d["module_id"], outcome=d["outcome"],
                  reward=d["reward"], confidence=d["confidence"])
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


class ExperienceReplayBuffer:
    """
    Lightweight structured replay buffer.

    Supports: append, sample (recent / high-confidence / failure / diverse),
    dedup via signature, size cap.
    """

    def __init__(self, max_size: int = 2000) -> None:
        self._entries:    List[ReplayEntry] = []
        self._signatures: set  = set()
        self._max_size    = max_size

    def append(self, entry: ReplayEntry) -> bool:
        """Returns False if duplicate."""
        if entry.signature and entry.signature in self._signatures:
            return False
        self._entries.append(entry)
        if entry.signature:
            self._signatures.add(entry.signature)
        if len(self._entries) > self._max_size:
            removed = self._entries.pop(0)
            if removed.signature:
                self._signatures.discard(removed.signature)
        return True

    def sample(self, n: int = 16,
               strategy: str = "recent",
               context_key: str = "",
               outcome: str = "") -> List[ReplayEntry]:
        pool = self._entries
        if context_key:
            pool = [e for e in pool if e.context_key == context_key]
        if outcome:
            pool = [e for e in pool if e.outcome == outcome]
        if not pool:
            return []

        if strategy == "recent":
            pool = sorted(pool, key=lambda e: e.timestamp, reverse=True)
        elif strategy == "high_confidence":
            pool = sorted(pool, key=lambda e: e.confidence, reverse=True)
        elif strategy == "failure":
            pool = [e for e in pool if e.reward < 0]
        elif strategy == "diverse":
            # Return one per unique context_key
            seen: set = set()
            deduped: List[ReplayEntry] = []
            for e in pool:
                if e.context_key not in seen:
                    deduped.append(e)
                    seen.add(e.context_key)
            pool = deduped

        return pool[:n]

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [e.to_dict() for e in self._entries]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperienceReplayBuffer":
        obj = cls()
        for ed in d.get("entries", []):
            obj.append(ReplayEntry.from_dict(ed))
        return obj


# ────────────────────────────────────────────────────────────────────────────
# Learning Metrics
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LearningMetrics:
    total_updates_proposed:  int   = 0
    total_updates_committed: int   = 0
    total_updates_rejected:  int   = 0
    total_rollbacks:         int   = 0
    total_experiences_ingested: int = 0
    total_replay_steps:      int   = 0
    drift_detections:        int   = 0
    avg_delta:               float = 0.0
    _delta_sum:              float = field(default=0.0, repr=False)

    def record_proposal(self) -> None:
        self.total_updates_proposed += 1

    def record_commit(self, delta: float) -> None:
        self.total_updates_committed += 1
        self._delta_sum += abs(delta)
        n = self.total_updates_committed
        self.avg_delta = self._delta_sum / n if n else 0.0

    def record_rejection(self) -> None:
        self.total_updates_rejected += 1

    def record_rollback(self) -> None:
        self.total_rollbacks += 1

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("_delta_sum", None)
        return d


# ────────────────────────────────────────────────────────────────────────────
# Learning State (persistent)
# ────────────────────────────────────────────────────────────────────────────

class LearningState:
    """
    All learned parameters in one serialisable object.

    strategy_stats:       (context_key, strategy_id) → StrategyStats
    knowledge_confidence: knowledge_candidate_id → float
    rule_confidence:      rule_id → float
    policies:             context_key → LearnedPolicy
    adapter:              StrategyBiasAdapter
    """

    def __init__(self) -> None:
        self.version:          str = str(uuid.uuid4())
        self.created_at:       float = time.time()
        self.updated_at:       float = time.time()

        # Core learned parameters
        self._strategy_stats:  Dict[str, StrategyStats] = {}
        # key = f"{context_key}::{strategy_id}"
        self._knowledge_conf:  Dict[str, float] = {}
        self._rule_conf:       Dict[str, float] = {}
        self._policies:        Dict[str, LearnedPolicy] = {}
        self.adapter:          StrategyBiasAdapter = StrategyBiasAdapter()

    # --- Strategy stats ---

    def _stats_key(self, context_key: str, strategy_id: str) -> str:
        return f"{context_key}::{strategy_id}"

    def get_stats(self, context_key: str,
                  strategy_id: str) -> StrategyStats:
        key = self._stats_key(context_key, strategy_id)
        if key not in self._strategy_stats:
            self._strategy_stats[key] = StrategyStats(
                strategy_id=strategy_id,
                context_key=context_key,
            )
        return self._strategy_stats[key]

    def list_stats(self, context_key: str = "") -> List[StrategyStats]:
        items = list(self._strategy_stats.values())
        if context_key:
            items = [s for s in items if s.context_key == context_key]
        return items

    # --- Knowledge confidence ---

    def get_knowledge_confidence(self, kc_id: str,
                                  default: float = 0.5) -> float:
        return self._knowledge_conf.get(kc_id, default)

    def set_knowledge_confidence(self, kc_id: str, conf: float) -> None:
        self._knowledge_conf[kc_id] = max(0.0, min(1.0, conf))

    # --- Rule confidence ---

    def get_rule_confidence(self, rule_id: str,
                             default: float = 0.5) -> float:
        return self._rule_conf.get(rule_id, default)

    def set_rule_confidence(self, rule_id: str, conf: float) -> None:
        self._rule_conf[rule_id] = max(0.0, min(1.0, conf))

    # --- Policies ---

    def get_policy(self, context_key: str) -> LearnedPolicy:
        if context_key not in self._policies:
            self._policies[context_key] = LearnedPolicy(context_key=context_key)
        return self._policies[context_key]

    def list_policies(self) -> List[LearnedPolicy]:
        return list(self._policies.values())

    # --- Snapshot (for rollback) ---

    def snapshot(self) -> Dict[str, Any]:
        """Compact snapshot for rollback — scores and confidences only."""
        return {
            "strategy_scores": {
                k: s.score for k, s in self._strategy_stats.items()
            },
            "knowledge_conf": dict(self._knowledge_conf),
            "rule_conf":      dict(self._rule_conf),
            "adapter":        self.adapter.state_dict(),
        }

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        """Restore scores from a snapshot (does NOT restore full history)."""
        for k, score in snap.get("strategy_scores", {}).items():
            if k in self._strategy_stats:
                self._strategy_stats[k].score = score
        for kc_id, conf in snap.get("knowledge_conf", {}).items():
            self._knowledge_conf[kc_id] = conf
        for rule_id, conf in snap.get("rule_conf", {}).items():
            self._rule_conf[rule_id] = conf
        adapter_data = snap.get("adapter")
        # Always restore adapter — create a fresh one and load snapshot into it
        new_adapter = StrategyBiasAdapter()
        if adapter_data:
            new_adapter.load_state_dict(adapter_data)
        self.adapter = new_adapter
        self.updated_at = time.time()

    # --- Persistence ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version":    self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "strategy_stats": {
                k: s.to_dict() for k, s in self._strategy_stats.items()
            },
            "knowledge_conf": self._knowledge_conf,
            "rule_conf":      self._rule_conf,
            "policies": {
                k: p.to_dict() for k, p in self._policies.items()
            },
            "adapter": self.adapter.state_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningState":
        obj = cls()
        obj.version    = d.get("version", obj.version)
        obj.created_at = d.get("created_at", obj.created_at)
        obj.updated_at = d.get("updated_at", obj.updated_at)
        for k, sd in d.get("strategy_stats", {}).items():
            obj._strategy_stats[k] = StrategyStats.from_dict(sd)
        obj._knowledge_conf = d.get("knowledge_conf", {})
        obj._rule_conf      = d.get("rule_conf", {})
        for k, pd in d.get("policies", {}).items():
            obj._policies[k] = LearnedPolicy.from_dict(pd)
        adapter_data = d.get("adapter")
        if adapter_data:
            obj.adapter.load_state_dict(adapter_data)
        return obj

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "LearningState":
        try:
            text = Path(path).read_text().strip()
            if not text:
                return cls()
            return cls.from_dict(json.loads(text))
        except (json.JSONDecodeError, OSError):
            return cls()


# ────────────────────────────────────────────────────────────────────────────
# Eligibility Checker
# ────────────────────────────────────────────────────────────────────────────

class EligibilityChecker:
    """Determines whether a given experience/pattern justifies learning."""

    def __init__(self, config: LearningConfig) -> None:
        self._cfg = config
        self._recent_sigs: Dict[str, float] = {}   # signature → last_seen

    def check(self, experience: Any,
              pattern: Any = None,
              knowledge: Any = None) -> Tuple[str, str]:
        """
        Returns (LearningEligibility.value, reason_string).
        experience / pattern / knowledge are ECHO objects (duck-typed).
        """
        cfg = self._cfg

        # Mode check
        if cfg.mode == LearningMode.OFF.value:
            return LearningEligibility.MODE_DISABLED.value, "Learning mode is OFF"

        # Experience completeness
        outcome = getattr(experience, "outcome",
                          getattr(experience, "status", "UNKNOWN"))
        if outcome == "UNKNOWN":
            return (LearningEligibility.INSUFFICIENT_EVIDENCE.value,
                    "Experience outcome is UNKNOWN")

        # Evidence threshold via pattern
        if pattern is not None:
            obs_count = getattr(pattern, "observation_count", 0)
            if obs_count < cfg.min_evidence:
                return (LearningEligibility.INSUFFICIENT_EVIDENCE.value,
                        f"Pattern has {obs_count} observations, "
                        f"need {cfg.min_evidence}")
            conf = getattr(pattern, "confidence", 0.0)
            if conf < cfg.min_confidence:
                return (LearningEligibility.INSUFFICIENT_EVIDENCE.value,
                        f"Pattern confidence {conf:.3f} < {cfg.min_confidence:.3f}")

        # Knowledge confidence threshold
        if knowledge is not None:
            conf = getattr(knowledge, "confidence", 0.0)
            if conf < cfg.min_confidence:
                return (LearningEligibility.INSUFFICIENT_EVIDENCE.value,
                        f"Knowledge confidence {conf:.3f} < {cfg.min_confidence:.3f}")

        # Dedup check
        exp_id = getattr(experience, "id", "")
        sig    = self._make_sig(experience)
        now    = time.time()
        if sig in self._recent_sigs:
            age = now - self._recent_sigs[sig]
            if age < cfg.duplicate_window_secs:
                return (LearningEligibility.DUPLICATE_EXPERIENCE.value,
                        f"Duplicate experience (sig={sig[:8]}… age={age:.1f}s)")
        self._recent_sigs[sig] = now
        # Trim old sigs
        self._recent_sigs = {
            s: t for s, t in self._recent_sigs.items()
            if now - t < cfg.duplicate_window_secs * 10
        }

        return LearningEligibility.ELIGIBLE.value, "Eligible for learning"

    @staticmethod
    def _make_sig(experience: Any) -> str:
        """Compact signature for dedup.

        Includes the experience id so two different Experience objects
        with the same structure are NOT treated as duplicates.  The
        duplicate_window_secs guard is intended to prevent the *exact
        same experience object* from being fed twice in rapid succession,
        not to deduplicate structurally similar but distinct experiences.
        """
        exp_id = getattr(experience, "id", "")
        parts = [
            exp_id,
            getattr(experience, "task_domain", ""),
            getattr(experience, "task_category", ""),
            ":".join(sorted(getattr(experience, "modules_used", []) or [])),
            getattr(experience, "outcome", ""),
        ]
        return "|".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Context Key Builder
# ────────────────────────────────────────────────────────────────────────────

def make_context_key(experience: Any) -> str:
    """
    Derive a compact context string from an experience (ECHO object).

    Context format: "<domain>:<category>:<primary_module>"
    """
    domain   = getattr(experience, "task_domain",   "unknown") or "unknown"
    category = getattr(experience, "task_category", "unknown") or "unknown"
    modules  = getattr(experience, "modules_used",  []) or []
    primary  = modules[0] if modules else "unknown"
    return f"{domain}:{category}:{primary}"


# ────────────────────────────────────────────────────────────────────────────
# Feedback Analyzer
# ────────────────────────────────────────────────────────────────────────────

class FeedbackAnalyzer:
    """Converts experience outcome into a scalar reward signal."""

    def __init__(self, config: LearningConfig) -> None:
        self._schedule = config.reward_schedule

    def reward(self, outcome: str) -> float:
        return self._schedule.get(outcome, 0.0)

    def is_positive(self, outcome: str) -> bool:
        return self.reward(outcome) > 0

    def is_negative(self, outcome: str) -> bool:
        return self.reward(outcome) < 0


# ────────────────────────────────────────────────────────────────────────────
# Incremental Update Engine
# ────────────────────────────────────────────────────────────────────────────

class IncrementalUpdateEngine:
    """
    Applies candidate updates to LearningState using bounded, auditable rules.

    Every update records: old_value, new_value, delta, reason, evidence.
    Supports rollback via RollbackManager.
    """

    def __init__(self, config: LearningConfig,
                 state: LearningState,
                 stability: StabilityController,
                 rollback: RollbackManager) -> None:
        self._cfg       = config
        self._state     = state
        self._stability = stability
        self._rollback  = rollback

    def apply(self, candidate: CandidateUpdate) -> LearningTraceEntry:
        """
        Validate and (if safe) commit the candidate update.

        Returns a LearningTraceEntry regardless of outcome.
        """
        trace = LearningTraceEntry(
            update_id    = candidate.id,
            target       = candidate.target,
            target_id    = candidate.target_id,
            context_key  = candidate.context_key,
            old_value    = candidate.old_value,
            new_value    = candidate.new_value,
            delta        = candidate.delta,
            reason       = candidate.reason,
            evidence_ids = list(candidate.evidence_ids),
            pattern_id   = candidate.pattern_id,
        )

        # Checkpoint before mutating
        snap = self._state.snapshot()
        ver  = self._rollback.checkpoint(
            snapshot   = snap,
            update_id  = candidate.id,
            target     = candidate.target,
            reason     = candidate.reason,
            parent_version = self._rollback.latest_version_id(),
        )
        trace.learning_version = ver.version_id

        # Stability check
        current = self._get_current(candidate)
        errors  = self._stability.check_update(candidate, current)
        if errors:
            candidate.status = UpdateStatus.REJECTED.value
            trace.status     = UpdateStatus.REJECTED.value
            trace.validation_errors = errors
            return trace

        # Apply
        self._commit(candidate)
        candidate.status   = UpdateStatus.COMMITTED.value
        candidate.committed_at = time.time()
        trace.status       = UpdateStatus.COMMITTED.value
        trace.new_value    = candidate.new_value
        trace.delta        = candidate.delta

        # Prune old rollback versions
        self._rollback.prune(self._cfg.max_versions)
        return trace

    def _get_current(self, c: CandidateUpdate) -> float:
        if c.target == LearningTarget.STRATEGY.value:
            stats = self._state.get_stats(c.context_key, c.target_id)
            return stats.score
        elif c.target == LearningTarget.KNOWLEDGE.value:
            return self._state.get_knowledge_confidence(c.target_id)
        elif c.target == LearningTarget.RULE.value:
            return self._state.get_rule_confidence(c.target_id)
        elif c.target == LearningTarget.MODULE_PREFERENCE.value:
            pol = self._state.get_policy(c.context_key)
            return pol.get_bonus(c.target_id)
        return float(c.old_value) if isinstance(c.old_value,
                                                  (int, float)) else 0.0

    def _commit(self, c: CandidateUpdate) -> None:
        if c.target == LearningTarget.STRATEGY.value:
            stats = self._state.get_stats(c.context_key, c.target_id)
            stats.score       = float(c.new_value)
            stats.last_updated = time.time()
            stats.confidence  = min(
                stats.confidence + 0.05, 1.0)

        elif c.target == LearningTarget.KNOWLEDGE.value:
            self._state.set_knowledge_confidence(
                c.target_id, float(c.new_value))

        elif c.target == LearningTarget.RULE.value:
            self._state.set_rule_confidence(
                c.target_id, float(c.new_value))

        elif c.target == LearningTarget.MODULE_PREFERENCE.value:
            pol = self._state.get_policy(c.context_key)
            pol.set_bonus(c.target_id, float(c.new_value))

        elif c.target == LearningTarget.MODEL_ADAPTER.value:
            # Adapter update already happened in StrategyLearner;
            # sync policy from adapter
            bias = self._state.adapter.get_bias(
                c.context_key, c.target_id)
            pol = self._state.get_policy(c.context_key)
            pol.set_bonus(c.target_id, bias)

        self._state.updated_at = time.time()


# ────────────────────────────────────────────────────────────────────────────
# Strategy Learner
# ────────────────────────────────────────────────────────────────────────────

class StrategyLearner:
    """
    Core strategy-score updater.

    Maintains per-(context, module) StrategyStats and updates scores using:
        new_score = clip(old_score + lr * reward, 0, 1)

    Also updates the StrategyBiasAdapter for MOSAIC integration.
    """

    def __init__(self, config: LearningConfig,
                 state: LearningState,
                 feedback: FeedbackAnalyzer,
                 update_engine: IncrementalUpdateEngine,
                 replay: ExperienceReplayBuffer) -> None:
        self._cfg     = config
        self._state   = state
        self._fb      = feedback
        self._engine  = update_engine
        self._replay  = replay

    def learn(self, experience: Any) -> List[CandidateUpdate]:
        """
        Produce and apply strategy-score update candidates from one experience.

        Returns list of attempted CandidateUpdate objects.
        """
        if self._cfg.mode not in (
            LearningMode.STRATEGY_LEARNING.value,
            LearningMode.INCREMENTAL_MODEL_ADAPT.value,
            LearningMode.ADAPTIVE_KNOWLEDGE.value,
        ):
            return []

        context_key = make_context_key(experience)
        outcome     = getattr(experience, "outcome", "UNKNOWN")
        reward      = self._fb.reward(outcome)
        modules     = getattr(experience, "modules_used", []) or []
        exp_id      = getattr(experience, "id", "")
        confidence  = getattr(experience, "result_confidence", 0.5)

        candidates: List[CandidateUpdate] = []

        for module_id in modules:
            if not module_id:
                continue

            stats     = self._state.get_stats(context_key, module_id)
            old_score = stats.score
            lr        = self._cfg.learning_rate

            # Update statistics (always, for drift tracking)
            stats.record_reward(reward)

            # Adapter update (actual parameter change in learned parameters)
            self._state.adapter.update(context_key, module_id, reward, lr)

            # Bounded EMA update on score
            raw_new   = old_score + lr * reward
            new_score = max(self._cfg.min_score,
                            min(self._cfg.max_score, raw_new))
            delta     = new_score - old_score

            if abs(delta) < 1e-6:
                continue   # No meaningful change

            c = CandidateUpdate(
                target      = LearningTarget.STRATEGY.value,
                target_id   = module_id,
                context_key = context_key,
                old_value   = old_score,
                new_value   = new_score,
                delta       = delta,
                reason      = (f"outcome={outcome} reward={reward:+.2f} "
                               f"lr={lr:.3f}"),
                evidence_ids = [exp_id] if exp_id else [],
            )
            # Apply through engine so state is updated immediately
            self._engine.apply(c)
            candidates.append(c)

            # Add to replay buffer
            sig = f"{context_key}|{module_id}|{outcome}"
            entry = ReplayEntry(
                experience_id = exp_id,
                context_key   = context_key,
                module_id     = module_id,
                outcome       = outcome,
                reward        = reward,
                confidence    = confidence,
                signature     = sig,
            )
            self._replay.append(entry)

        return candidates

    def replay_update(self, n: int = 8,
                      strategy: str = "recent") -> List[CandidateUpdate]:
        """
        Apply a mini-batch of replayed experiences to prevent catastrophic
        forgetting of less recent patterns.
        """
        entries = self._replay.sample(n=n, strategy=strategy)
        candidates: List[CandidateUpdate] = []
        for entry in entries:
            stats  = self._state.get_stats(entry.context_key, entry.module_id)
            old_s  = stats.score
            new_s  = max(self._cfg.min_score,
                         min(self._cfg.max_score,
                             old_s + self._cfg.learning_rate * 0.5
                             * entry.reward))
            delta  = new_s - old_s
            if abs(delta) < 1e-6:
                continue
            c = CandidateUpdate(
                target      = LearningTarget.STRATEGY.value,
                target_id   = entry.module_id,
                context_key = entry.context_key,
                old_value   = old_s,
                new_value   = new_s,
                delta       = delta,
                reason      = f"replay:{strategy}",
                evidence_ids = [entry.experience_id],
            )
            # Apply so state changes
            self._engine.apply(c)
            candidates.append(c)
        return candidates


# ────────────────────────────────────────────────────────────────────────────
# Knowledge Evaluator
# ────────────────────────────────────────────────────────────────────────────

class KnowledgeEvaluator:
    """
    Updates knowledge-candidate confidence and rule confidence from
    ECHO Pattern / KnowledgeCandidate objects.
    """

    def __init__(self, config: LearningConfig,
                 state: LearningState,
                 update_engine: IncrementalUpdateEngine) -> None:
        self._cfg    = config
        self._state  = state
        self._engine = update_engine

    def evaluate_pattern(self, pattern: Any,
                          experience: Any) -> List[CandidateUpdate]:
        """Propose confidence updates from an ECHO Pattern."""
        if self._cfg.mode == LearningMode.OFF.value:
            return []
        if self._cfg.mode == LearningMode.OBSERVATION_ONLY.value:
            return []

        kc_id  = getattr(pattern, "id", "")
        old_c  = self._state.get_knowledge_confidence(kc_id, 0.5)
        new_c  = getattr(pattern, "confidence", old_c)
        new_c  = max(0.0, min(self._cfg.max_knowledge_confidence, new_c))
        delta  = new_c - old_c

        if abs(delta) < 1e-4:
            return []

        exp_id = getattr(experience, "id", "")
        pat_id = kc_id
        c = CandidateUpdate(
            target       = LearningTarget.KNOWLEDGE.value,
            target_id    = kc_id,
            context_key  = make_context_key(experience),
            old_value    = old_c,
            new_value    = new_c,
            delta        = delta,
            reason       = (f"ECHO pattern confidence={new_c:.3f} "
                            f"obs={getattr(pattern,'observation_count',0)}"),
            evidence_ids = [exp_id] if exp_id else [],
            pattern_id   = pat_id,
        )
        return [c]

    def evaluate_knowledge_candidate(self, kc: Any,
                                      experience: Any) -> List[CandidateUpdate]:
        """Propose confidence updates from an ECHO KnowledgeCandidate."""
        if self._cfg.mode in (LearningMode.OFF.value,
                               LearningMode.OBSERVATION_ONLY.value):
            return []

        kc_id = getattr(kc, "id", "")
        old_c = self._state.get_knowledge_confidence(kc_id, 0.5)
        raw_c = getattr(kc, "confidence", old_c)
        new_c = max(0.0, min(self._cfg.max_knowledge_confidence, raw_c))
        delta = new_c - old_c

        if abs(delta) < 1e-4:
            return []

        exp_id = getattr(experience, "id", "")
        c = CandidateUpdate(
            target       = LearningTarget.KNOWLEDGE.value,
            target_id    = kc_id,
            context_key  = make_context_key(experience),
            old_value    = old_c,
            new_value    = new_c,
            delta        = delta,
            reason       = f"ECHO knowledge candidate confidence={new_c:.3f}",
            evidence_ids = [exp_id] if exp_id else [],
            pattern_id   = getattr(kc, "id", ""),
        )
        return [c]


# ────────────────────────────────────────────────────────────────────────────
# Learned Preference Provider (MOSAIC integration shim)
# ────────────────────────────────────────────────────────────────────────────

class LearnedPreferenceProvider:
    """
    Exposes learned module preference bonuses to MOSAIC.

    MOSAIC calls get_preference_bonus(context_key, module_id) and adds the
    returned value to its base capability score.

    Hard constraints always win; bonuses are soft.
    Bonus is bounded to ±MAX_BONUS (0.30 by default).
    """

    def __init__(self, state: LearningState) -> None:
        self._state = state

    def get_preference_bonus(self, context_key: str,
                              module_id: str) -> float:
        """Return the learned preference bonus for (context, module)."""
        # Adapter bias (actual trained parameter)
        adapter_bias = self._state.adapter.get_bias(context_key, module_id)
        # Policy table bonus
        policy_bonus = self._state.get_policy(context_key).get_bonus(module_id)
        # Combined (clamped)
        combined = max(-0.30, min(0.30, adapter_bias + policy_bonus))
        return combined

    def get_preferred_modules(self, context_key: str,
                               top_k: int = 3,
                               min_bonus: float = 0.0) -> List[str]:
        """Return module_ids with positive bonus, sorted descending."""
        stats = self._state.list_stats(context_key=context_key)
        scored = []
        for s in stats:
            bonus = self.get_preference_bonus(context_key, s.strategy_id)
            if bonus >= min_bonus:
                scored.append((s.strategy_id, bonus))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def get_excluded_modules(self, context_key: str,
                              threshold: float = -0.20) -> List[str]:
        """Return module_ids with strong negative bias."""
        stats = self._state.list_stats(context_key=context_key)
        return [
            s.strategy_id
            for s in stats
            if self.get_preference_bonus(context_key, s.strategy_id) <= threshold
        ]


# ────────────────────────────────────────────────────────────────────────────
# Learning Controller (main orchestrator)
# ────────────────────────────────────────────────────────────────────────────

class LearningController:
    """
    Top-level orchestrator for the self-learning pipeline.

    Flow:
        ECHO experience / pattern / knowledge
            ↓
        EligibilityChecker
            ↓
        StrategyLearner  +  KnowledgeEvaluator
            ↓
        [optional: require_approval gate]
            ↓
        IncrementalUpdateEngine (validate + commit/reject)
            ↓
        LearningTrace  +  LearningMetrics
    """

    def __init__(self, config: Optional[LearningConfig] = None,
                 state_path: Optional[str] = None) -> None:
        self._cfg         = config or LearningConfig()
        self._state       = (LearningState.load(state_path)
                              if state_path and Path(state_path).exists()
                              else LearningState())
        self._state_path  = state_path

        self._stability   = StabilityController(self._cfg)
        self._rollback    = RollbackManager()
        self._update_eng  = IncrementalUpdateEngine(
                                self._cfg, self._state,
                                self._stability, self._rollback)
        self._replay      = ExperienceReplayBuffer()
        self._feedback    = FeedbackAnalyzer(self._cfg)
        self._eligibility = EligibilityChecker(self._cfg)
        self._strategy    = StrategyLearner(
                                self._cfg, self._state,
                                self._feedback, self._update_eng,
                                self._replay)
        self._knowledge   = KnowledgeEvaluator(
                                self._cfg, self._state, self._update_eng)
        self._provider    = LearnedPreferenceProvider(self._state)
        self._metrics     = LearningMetrics()
        self._trace:      List[LearningTraceEntry] = []
        self._pending:    List[CandidateUpdate]    = []   # approval queue

    # ── Public API ─────────────────────────────────────────────────────────

    def enable(self, mode: Optional[str] = None) -> None:
        """Enable learning (optionally set mode)."""
        if mode:
            self._cfg.mode = mode
        elif self._cfg.mode == LearningMode.OFF.value:
            self._cfg.mode = LearningMode.ADAPTIVE_KNOWLEDGE.value

    def disable(self) -> None:
        """Disable learning (sets mode to OFF)."""
        self._cfg.mode = LearningMode.OFF.value

    @property
    def mode(self) -> str:
        return self._cfg.mode

    @property
    def is_enabled(self) -> bool:
        return self._cfg.mode != LearningMode.OFF.value

    def learn_from(self, experience: Any,
                   pattern: Any = None,
                   knowledge: Any = None) -> Dict[str, Any]:
        """
        Full learning pipeline for one experience.

        Args:
            experience: ECHO Experience object (or duck-typed equivalent).
            pattern:    Optional ECHO Pattern object.
            knowledge:  Optional ECHO KnowledgeCandidate object.

        Returns:
            Summary dict: {eligibility, candidates_proposed,
                           committed, rejected, trace_ids}.
        """
        # Eligibility
        elig, reason = self._eligibility.check(experience, pattern, knowledge)
        if elig != LearningEligibility.ELIGIBLE.value:
            return {
                "eligibility": elig,
                "reason":      reason,
                "candidates_proposed": 0,
                "committed":   0,
                "rejected":    0,
                "trace_ids":   [],
            }

        self._metrics.total_experiences_ingested += 1

        # Collect candidate updates
        all_candidates: List[CandidateUpdate] = []

        # Strategy learning
        if self._cfg.mode in (LearningMode.STRATEGY_LEARNING.value,
                               LearningMode.INCREMENTAL_MODEL_ADAPT.value,
                               LearningMode.ADAPTIVE_KNOWLEDGE.value):
            all_candidates.extend(self._strategy.learn(experience))

        # Knowledge / pattern confidence
        if pattern is not None:
            all_candidates.extend(
                self._knowledge.evaluate_pattern(pattern, experience))
        if knowledge is not None:
            all_candidates.extend(
                self._knowledge.evaluate_knowledge_candidate(knowledge,
                                                              experience))

        committed = 0
        rejected  = 0
        trace_ids: List[str] = []

        for c in all_candidates:
            self._metrics.record_proposal()

            # Approval gate
            if self._cfg.require_approval:
                c.approval = ApprovalStatus.PENDING.value
                self._pending.append(c)
                continue

            # Strategy candidates are already applied by StrategyLearner.learn()
            # Knowledge/rule candidates still need applying
            if c.target in (LearningTarget.KNOWLEDGE.value,
                            LearningTarget.RULE.value,
                            LearningTarget.MODULE_PREFERENCE.value,
                            LearningTarget.MODEL_ADAPTER.value):
                trace = self._update_eng.apply(c)
            else:
                # Strategy: already applied; build a synthetic committed trace
                trace = LearningTraceEntry(
                    update_id    = c.id,
                    target       = c.target,
                    target_id    = c.target_id,
                    context_key  = c.context_key,
                    old_value    = c.old_value,
                    new_value    = c.new_value,
                    delta        = c.delta,
                    reason       = c.reason,
                    evidence_ids = list(c.evidence_ids),
                    pattern_id   = c.pattern_id,
                    status       = UpdateStatus.COMMITTED.value,
                    learning_version = self._rollback.latest_version_id(),
                )
                c.status = UpdateStatus.COMMITTED.value

            self._record_trace(trace)
            trace_ids.append(trace.trace_id)

            if trace.status == UpdateStatus.COMMITTED.value:
                committed += 1
                self._metrics.record_commit(trace.delta)
            else:
                rejected += 1
                self._metrics.record_rejection()

        # Drift detection
        context_key = make_context_key(experience)
        for s in self._state.list_stats(context_key=context_key):
            drift = self._stability.check_drift(s)
            if drift != s.drift_status:
                s.drift_status = drift
                if drift != DriftStatus.STABLE.value:
                    self._metrics.drift_detections += 1

        # Prune old rollback versions
        self._rollback.prune(self._cfg.max_versions)

        # Persist if path configured
        if self._state_path:
            self._state.save(self._state_path)

        return {
            "eligibility":         elig,
            "reason":              reason,
            "candidates_proposed": len(all_candidates),
            "committed":           committed,
            "rejected":            rejected,
            "trace_ids":           trace_ids,
        }

    def replay(self, n: int = 8,
               strategy: str = "recent") -> Dict[str, Any]:
        """
        Run a mini-replay batch to guard against catastrophic forgetting.
        """
        if not self.is_enabled:
            return {"replay_steps": 0, "committed": 0}

        candidates = self._strategy.replay_update(n=n, strategy=strategy)
        committed  = 0
        for c in candidates:
            trace = self._update_eng.apply(c)
            self._record_trace(trace)
            if trace.status == UpdateStatus.COMMITTED.value:
                committed += 1
                self._metrics.record_commit(trace.delta)
                self._metrics.total_replay_steps += 1
            else:
                self._metrics.record_rejection()

        return {"replay_steps": len(candidates), "committed": committed}

    # ── Approval API ───────────────────────────────────────────────────────

    def propose_update(self, experience: Any) -> List[CandidateUpdate]:
        """
        Propose candidate updates without committing them.
        Returns the pending candidates (approval required to commit).
        """
        old_require = self._cfg.require_approval
        self._cfg.require_approval = True
        self.learn_from(experience)
        self._cfg.require_approval = old_require
        pending = list(self._pending)
        return pending

    def approve(self, candidate: CandidateUpdate) -> None:
        """Mark a pending candidate as approved."""
        candidate.approval = ApprovalStatus.APPROVED.value

    def reject_candidate(self, candidate: CandidateUpdate) -> None:
        """Mark a pending candidate as rejected (without committing)."""
        candidate.approval = ApprovalStatus.REJECTED.value
        candidate.status   = UpdateStatus.REJECTED.value
        self._pending = [c for c in self._pending if c.id != candidate.id]

    def commit(self, candidate: CandidateUpdate) -> LearningTraceEntry:
        """Commit an approved candidate update."""
        if candidate.approval != ApprovalStatus.APPROVED.value:
            raise LearningError("Candidate not approved; call approve() first.")
        trace = self._update_eng.apply(candidate)
        self._record_trace(trace)
        self._pending = [c for c in self._pending if c.id != candidate.id]
        if trace.status == UpdateStatus.COMMITTED.value:
            self._metrics.record_commit(trace.delta)
        else:
            self._metrics.record_rejection()
        return trace

    # ── Rollback API ───────────────────────────────────────────────────────

    def rollback(self, version_id: str) -> bool:
        """
        Restore learning state to a prior version.
        Returns True on success.
        """
        snap = self._rollback.restore(version_id)
        if snap is None:
            return False
        self._state.restore_snapshot(snap)
        self._metrics.record_rollback()
        if self._state_path:
            self._state.save(self._state_path)
        return True

    def list_versions(self) -> List[LearningStateVersion]:
        """Return all checkpointed learning state versions."""
        return self._rollback.list_versions()

    # ── Query API ──────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Return a human-readable summary of the current learning state."""
        return {
            "mode":           self._cfg.mode,
            "version":        self._state.version,
            "updated_at":     self._state.updated_at,
            "strategy_count": len(self._state.list_stats()),
            "knowledge_count":len(self._state._knowledge_conf),
            "rule_count":     len(self._state._rule_conf),
            "policy_count":   len(self._state.list_policies()),
            "adapter_params": len(self._state.adapter._biases),
            "replay_buffer":  len(self._replay),
            "metrics":        self._metrics.to_dict(),
        }

    def get_strategy_stats(self, context_key: str = "") -> List[StrategyStats]:
        """Return strategy statistics, optionally filtered by context."""
        return self._state.list_stats(context_key=context_key)

    def get_trace(self, n: int = 50) -> List[LearningTraceEntry]:
        """Return the last n learning trace entries."""
        return self._trace[-n:]

    def explain_update(self, trace_id: str) -> Optional[str]:
        """Return a human-readable explanation for one trace entry."""
        for t in self._trace:
            if t.trace_id == trace_id:
                return t.explain()
        return None

    def get_preference_bonus(self, context_key: str,
                              module_id: str) -> float:
        """Return the learned preference bonus for (context, module)."""
        return self._provider.get_preference_bonus(context_key, module_id)

    def get_preferred_modules(self, context_key: str,
                               top_k: int = 3) -> List[str]:
        """Return the top-k preferred module IDs for a context."""
        return self._provider.get_preferred_modules(context_key, top_k)

    def get_mosaic_hints(self, context_key: str) -> Dict[str, Any]:
        """
        Return MOSAIC-compatible hints derived from learned preferences.

        MOSAIC can pass these as kwargs to compose()/solve():
            preferred_modules = [...],  excluded_modules = [...]
        """
        preferred = self._provider.get_preferred_modules(context_key,
                                                          min_bonus=0.05)
        excluded  = self._provider.get_excluded_modules(context_key,
                                                         threshold=-0.15)
        return {
            "preferred_modules": preferred,
            "excluded_modules":  excluded,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return learning metrics."""
        return self._metrics.to_dict()

    def summary(self) -> Dict[str, Any]:
        """High-level summary for display."""
        stats = self._state.list_stats()
        drift_detected = [s for s in stats
                          if s.drift_status != DriftStatus.STABLE.value
                          and s.drift_status != DriftStatus.UNKNOWN.value]
        return {
            "mode":                   self._cfg.mode,
            "total_strategy_entries": len(stats),
            "total_knowledge_entries":len(self._state._knowledge_conf),
            "total_rule_entries":     len(self._state._rule_conf),
            "adapter_parameters":     len(self._state.adapter._biases),
            "replay_buffer_size":     len(self._replay),
            "drift_detected":         len(drift_detected),
            "versions_available":     len(self._rollback.list_versions()),
            "pending_approvals":      len(self._pending),
            "metrics":                self._metrics.to_dict(),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _record_trace(self, trace: LearningTraceEntry) -> None:
        self._trace.append(trace)
        if len(self._trace) > self._cfg.max_trace_entries:
            self._trace.pop(0)


# ────────────────────────────────────────────────────────────────────────────
# LearningError
# ────────────────────────────────────────────────────────────────────────────

class LearningError(Exception):
    """Raised when a Phase 8 learning operation fails."""
