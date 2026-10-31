"""
prime.research — PRIME Phase 10: Research Evaluation Framework.

Provides:
  - SyntheticBenchmark: deterministic tasks with known ground truth
  - BenchmarkSuite: collection of tasks across categories
  - ExperimentRunner: runs baselines / ablations and records results
  - ExperimentResult: machine-readable result record
  - LearningCurveExperiment: measures adaptation over N experiences
  - AblationStudy: systematically disables subsystems

Design principles
─────────────────
• All benchmarks have known ground truth — no fabricated results.
• Every run is reproducible via config + seed.
• Results are JSON-serialisable.
• No scientific claims are made here; results are raw measurements.
• No LLM is required for any benchmark task in the synthetic suite.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .pipeline import PrimeConfig, PrimeResult, SolveStatus, collect_system_info


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark task
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    """A single deterministic benchmark task."""
    task_id:      str
    category:     str          # arithmetic | logic | rule | constraint | retrieval | multistep
    description:  str          # natural-language problem text
    ground_truth: str          # expected answer (string)
    difficulty:   str = "easy" # easy | medium | hard
    tags:         List[str] = field(default_factory=list)
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TaskResult:
    """Result of running one benchmark task."""
    task_id:       str
    category:      str
    status:        str         # from SolveStatus
    answer:        str
    expected:      str
    correct:       bool
    latency_ms:    float
    confidence:    float
    modules_used:  List[str]
    strategy:      str
    error:         str = ""
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Benchmark Suite
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBenchmarkSuite:
    """
    Collection of deterministic tasks with known ground-truth answers.

    Tasks are purely structural — no copyrighted data.
    The ground truth is the exact string the reasoner is expected to produce.
    """

    # RQ1/RQ3 — arithmetic
    ARITHMETIC = [
        BenchmarkTask("arith_01", "arithmetic",
            "A car travels 150 km in 3 hours. What is its average speed in km/h?",
            "50", tags=["speed","division"]),
        BenchmarkTask("arith_02", "arithmetic",
            "A rectangle has width 8 m and height 5 m. What is its area?",
            "40", tags=["area","multiplication"]),
        BenchmarkTask("arith_03", "arithmetic",
            "If x = 12 and y = 4, what is x divided by y?",
            "3", tags=["division"]),
        BenchmarkTask("arith_04", "arithmetic",
            "A train travels at 80 km/h for 2.5 hours. How far does it travel?",
            "200", tags=["distance","multiplication"]),
        BenchmarkTask("arith_05", "arithmetic",
            "What is 15 percent of 200?",
            "30", tags=["percentage"]),
    ]

    # RQ1 — logic / rule reasoning
    LOGIC = [
        BenchmarkTask("logic_01", "logic",
            "All mammals are warm-blooded. A dog is a mammal. Is a dog warm-blooded?",
            "yes", difficulty="easy", tags=["syllogism"]),
        BenchmarkTask("logic_02", "logic",
            "If A implies B, and A is true, what can we conclude about B?",
            "B is true", difficulty="easy", tags=["modus_ponens"]),
        BenchmarkTask("logic_03", "logic",
            "All birds have wings. A penguin is a bird. Does a penguin have wings?",
            "yes", difficulty="easy", tags=["syllogism"]),
    ]

    # RQ1 — constraint satisfaction
    CONSTRAINT = [
        BenchmarkTask("constr_01", "constraint",
            "x must be greater than 5 and less than 10. Is x=7 valid?",
            "valid", tags=["range_check"]),
        BenchmarkTask("constr_02", "constraint",
            "x must be greater than 5 and less than 10. Is x=3 valid?",
            "invalid", tags=["range_check"]),
        BenchmarkTask("constr_03", "constraint",
            "Speed limit is 120 km/h. A car travels at 115 km/h. Is it within the limit?",
            "yes", tags=["limit_check"]),
    ]

    # RQ2/RQ3 — repeated tasks for adaptation measurement
    REPEATED = [
        BenchmarkTask(f"repeat_{i:02d}", "arithmetic",
            f"A car travels {(i+1)*50} km in {i+1} hours. What is its average speed?",
            "50", tags=["speed","repeat"])
        for i in range(10)
    ]

    # RQ4 — resource-constrained variants (structurally identical, different labels)
    RESOURCE_TEST = [
        BenchmarkTask("res_01", "arithmetic",
            "Compute 6 divided by 2.", "3", tags=["simple","resource"]),
        BenchmarkTask("res_02", "arithmetic",
            "Compute 9 divided by 3.", "3", tags=["simple","resource"]),
        BenchmarkTask("res_03", "arithmetic",
            "Compute 12 divided by 4.", "3", tags=["simple","resource"]),
    ]

    @classmethod
    def all_tasks(cls) -> List[BenchmarkTask]:
        return (cls.ARITHMETIC + cls.LOGIC + cls.CONSTRAINT
                + cls.REPEATED + cls.RESOURCE_TEST)

    @classmethod
    def by_category(cls, category: str) -> List[BenchmarkTask]:
        return [t for t in cls.all_tasks() if t.category == category]

    @classmethod
    def core_tasks(cls) -> List[BenchmarkTask]:
        """Minimal fast set for CI / regression tests."""
        return cls.ARITHMETIC[:3] + cls.LOGIC[:2] + cls.CONSTRAINT[:2]


# ─────────────────────────────────────────────────────────────────────────────
# Correctness judge
# ─────────────────────────────────────────────────────────────────────────────

def _judge(answer: str, ground_truth: str) -> bool:
    """
    Determine whether an answer matches the ground truth.
    Case-insensitive; strips whitespace; tries numeric equality for numbers.
    """
    a = answer.strip().lower()
    g = ground_truth.strip().lower()
    if a == g:
        return True
    # Numeric comparison
    try:
        return abs(float(a) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    # Substring check for multi-word answers like "B is true"
    return g in a or a in g


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Machine-readable record for one experiment run."""
    experiment_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name:str = ""
    baseline_name:  str = ""
    timestamp:      str = field(default_factory=lambda:
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    config:         Dict[str, Any] = field(default_factory=dict)
    system_info:    Dict[str, Any] = field(default_factory=dict)
    task_results:   List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate metrics
    n_tasks:        int   = 0
    n_correct:      int   = 0
    n_failed:       int   = 0
    accuracy:       float = 0.0
    mean_latency_ms:float = 0.0
    median_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    mean_confidence:float = 0.0
    notes:          str   = ""

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from task_results."""
        self.n_tasks = len(self.task_results)
        if not self.task_results:
            return
        self.n_correct = sum(1 for r in self.task_results if r.get("correct"))
        self.n_failed  = sum(1 for r in self.task_results
                             if r.get("status") not in
                             (SolveStatus.SUCCESS.value,
                              SolveStatus.PARTIAL_SUCCESS.value))
        self.accuracy  = self.n_correct / self.n_tasks if self.n_tasks else 0.0
        lats = [r["latency_ms"] for r in self.task_results]
        self.mean_latency_ms   = statistics.mean(lats)
        self.median_latency_ms = statistics.median(lats)
        self.std_latency_ms    = statistics.stdev(lats) if len(lats) > 1 else 0.0
        self.min_latency_ms    = min(lats)
        self.max_latency_ms    = max(lats)
        confs = [r.get("confidence", 0.0) for r in self.task_results]
        self.mean_confidence = statistics.mean(confs)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            f"Experiment: {self.experiment_name} / {self.baseline_name}",
            f"  Tasks:      {self.n_tasks}",
            f"  Correct:    {self.n_correct}  ({self.accuracy*100:.1f}%)",
            f"  Failed:     {self.n_failed}",
            f"  Latency:    mean={self.mean_latency_ms:.1f} ms  "
            f"median={self.median_latency_ms:.1f} ms  "
            f"std={self.std_latency_ms:.1f} ms",
            f"  Confidence: {self.mean_confidence:.2f}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Runs a benchmark suite against a PRIME instance and records results.

    The runner is intentionally simple: it calls prime.solve() for each task
    and judges the answer against the ground truth.

    Args:
        prime_factory: Callable[[], PRIME] — creates a fresh PRIME per run.
    """

    def __init__(self, prime_factory: Callable) -> None:
        self._factory = prime_factory

    def run(self, tasks: List[BenchmarkTask],
            config: PrimeConfig,
            experiment_name: str = "",
            baseline_name:   str = "",
            n_repeats:       int = 1,
            ) -> ExperimentResult:
        """
        Run all tasks and return an ExperimentResult.

        Args:
            tasks:           List of BenchmarkTask objects.
            config:          PrimeConfig to use.
            experiment_name: Human label for this experiment.
            baseline_name:   Baseline variant name.
            n_repeats:       How many times to repeat each task (for statistics).
        """
        exp = ExperimentResult(
            experiment_name = experiment_name,
            baseline_name   = baseline_name,
            config          = config.to_dict(),
            system_info     = collect_system_info(),
        )

        prime = self._factory(config)
        try:
            for task in tasks:
                for rep in range(n_repeats):
                    tr = self._run_task(prime, task, rep)
                    exp.task_results.append(tr.to_dict())
        finally:
            try:
                prime.shutdown()
            except Exception:
                pass

        exp.compute_aggregates()
        return exp

    def _run_task(self, prime, task: BenchmarkTask,
                  rep: int) -> TaskResult:
        task_id = f"{task.task_id}_rep{rep}"
        try:
            result: PrimeResult = prime.solve(task.description,
                                               task_id=task_id)
            correct = _judge(result.answer, task.ground_truth)
            return TaskResult(
                task_id      = task.task_id,
                category     = task.category,
                status       = result.status.value,
                answer       = result.answer,
                expected     = task.ground_truth,
                correct      = correct,
                latency_ms   = result.latency_ms,
                confidence   = result.confidence,
                modules_used = result.modules_used,
                strategy     = result.optimizer_strategy,
                error        = result.error,
            )
        except Exception as exc:
            return TaskResult(
                task_id      = task.task_id,
                category     = task.category,
                status       = SolveStatus.INTERNAL_ERROR.value,
                answer       = "",
                expected     = task.ground_truth,
                correct      = False,
                latency_ms   = 0.0,
                confidence   = 0.0,
                modules_used = [],
                strategy     = "",
                error        = str(exc),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Study
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, str] = {
    "full":                "Full PRIME (all subsystems)",
    "mosaic_echo_learning":"MOSAIC + ECHO + Learning, no optimization",
    "mosaic_echo":         "MOSAIC + ECHO, no learning",
    "mosaic_only":         "MOSAIC only, no ECHO or learning",
    "static":              "Static pipeline: ATLAS + Reasoning only",
}


class AblationStudy:
    """
    Systematically compares PRIME baselines on a shared task set.

    Each baseline corresponds to a FeatureFlags preset.
    Results are collected into ExperimentResult objects and can be
    compared or serialised to JSON.

    Research questions addressed:
        RQ1: Can the system work without heavy frameworks?
        RQ3: Does continual adaptation improve repeated tasks?
        RQ4: Does resource-aware optimization reduce cost?
    """

    def __init__(self, prime_factory: Callable) -> None:
        self._runner = ExperimentRunner(prime_factory)

    def run_all(self,
                tasks: Optional[List[BenchmarkTask]] = None,
                baselines: Optional[List[str]] = None,
                n_repeats: int = 1,
                ) -> Dict[str, ExperimentResult]:
        """
        Run all (or selected) ablation baselines.

        Returns a dict: baseline_name → ExperimentResult.
        """
        if tasks is None:
            tasks = SyntheticBenchmarkSuite.core_tasks()
        if baselines is None:
            baselines = list(ABLATION_CONFIGS.keys())

        results: Dict[str, ExperimentResult] = {}
        for baseline in baselines:
            cfg = PrimeConfig.for_ablation(baseline)
            exp = self._runner.run(
                tasks           = tasks,
                config          = cfg,
                experiment_name = "ablation",
                baseline_name   = baseline,
                n_repeats       = n_repeats,
            )
            results[baseline] = exp
        return results

    @staticmethod
    def compare(results: Dict[str, ExperimentResult]) -> str:
        """Return a comparison table of ablation results."""
        lines = ["=" * 72, "Ablation Study Comparison", "=" * 72]
        header = (f"{'Baseline':<26s} {'Acc':>6s} {'Mean lat':>10s} "
                  f"{'Median':>8s} {'Correct':>8s} {'Tasks':>6s}")
        lines.append(header)
        lines.append("-" * 72)
        for name, exp in results.items():
            desc = ABLATION_CONFIGS.get(name, name)
            lines.append(
                f"{name:<26s} {exp.accuracy*100:>5.1f}% "
                f"{exp.mean_latency_ms:>9.1f}ms "
                f"{exp.median_latency_ms:>8.1f}ms "
                f"{exp.n_correct:>7d}/{exp.n_tasks:<5d}"
                f"  [{desc[:28]}]"
            )
        lines.append("=" * 72)
        lines.append("NOTE: All measurements are from this specific hardware/")
        lines.append("      configuration. Results may differ on other systems.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Learning Curve Experiment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LearningCurvePoint:
    """One measurement on the learning curve."""
    n_experiences:  int
    accuracy:       float
    mean_latency_ms:float
    mean_confidence:float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class LearningCurveExperiment:
    """
    Measures how task success evolves as more experiences are accumulated.

    Protocol:
        Phase A: establish baseline on probe tasks.
        Phase B: run training tasks (for experience accumulation).
        Phase C: re-run probe tasks, compare.

    Addresses RQ2/RQ3: does experience-derived adaptation improve performance?
    """

    def __init__(self, prime_factory: Callable) -> None:
        self._factory = prime_factory

    def run(self,
            training_tasks: List[BenchmarkTask],
            probe_tasks:    List[BenchmarkTask],
            checkpoints:    Optional[List[int]] = None,
            config:         Optional[PrimeConfig] = None,
            ) -> List[LearningCurvePoint]:
        """
        Run the experiment and return a list of LearningCurvePoints.

        Args:
            training_tasks: Tasks used to accumulate experience.
            probe_tasks:    Tasks used to measure performance.
            checkpoints:    Experience counts at which to measure (default [0,5,10,20]).
            config:         Config to use (default: full PRIME).
        """
        if checkpoints is None:
            checkpoints = [0, 5, 10, 20]
        if config is None:
            config = PrimeConfig.default()

        prime = self._factory(config)
        curve: List[LearningCurvePoint] = []

        try:
            # Checkpoint 0: baseline before any experience
            if 0 in checkpoints:
                acc, lat, conf = self._measure(prime, probe_tasks)
                curve.append(LearningCurvePoint(0, acc, lat, conf))

            # Accumulate training experience
            exp_count = 0
            for i, task in enumerate(training_tasks):
                try:
                    prime.solve(task.description)
                    exp_count += 1
                except Exception:
                    pass

                if (exp_count in checkpoints and exp_count > 0):
                    acc, lat, conf = self._measure(prime, probe_tasks)
                    curve.append(LearningCurvePoint(exp_count, acc, lat, conf))

            # Final measurement if not already done
            last = curve[-1].n_experiences if curve else -1
            if exp_count not in checkpoints or last != exp_count:
                acc, lat, conf = self._measure(prime, probe_tasks)
                curve.append(LearningCurvePoint(exp_count, acc, lat, conf))

        finally:
            try:
                prime.shutdown()
            except Exception:
                pass

        return curve

    def _measure(self, prime, tasks: List[BenchmarkTask]
                  ) -> Tuple[float, float, float]:
        correct = 0
        latencies: List[float] = []
        confidences: List[float] = []
        for task in tasks:
            try:
                r: PrimeResult = prime.solve(task.description)
                if _judge(r.answer, task.ground_truth):
                    correct += 1
                latencies.append(r.latency_ms)
                confidences.append(r.confidence)
            except Exception:
                latencies.append(0.0)
                confidences.append(0.0)
        n = len(tasks)
        acc  = correct / n if n else 0.0
        lat  = statistics.mean(latencies) if latencies else 0.0
        conf = statistics.mean(confidences) if confidences else 0.0
        return acc, lat, conf


# ─────────────────────────────────────────────────────────────────────────────
# Performance Profiler (research-grade)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentProfile:
    """Timing profile for one PRIME component."""
    component:   str
    n_runs:      int
    mean_ms:     float
    median_ms:   float
    std_ms:      float
    min_ms:      float
    max_ms:      float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        return (f"{self.component:<22s} "
                f"mean={self.mean_ms:>8.3f}ms  "
                f"median={self.median_ms:>8.3f}ms  "
                f"std={self.std_ms:>7.3f}ms")


def profile_component(fn: Callable, n: int = 100,
                       label: str = "") -> ComponentProfile:
    """
    Measure wall-clock latency of a callable over n runs.

    Args:
        fn:    Callable to benchmark (no args).
        n:     Number of timed iterations.
        label: Human-readable component name.

    Returns:
        ComponentProfile with mean/median/std/min/max.
    """
    # Warm-up
    warmup = max(1, n // 10)
    for _ in range(warmup):
        fn()

    timings: List[float] = []
    for _ in range(n):
        t0 = time.monotonic()
        fn()
        timings.append((time.monotonic() - t0) * 1000)

    return ComponentProfile(
        component  = label,
        n_runs     = n,
        mean_ms    = statistics.mean(timings),
        median_ms  = statistics.median(timings),
        std_ms     = statistics.stdev(timings) if len(timings) > 1 else 0.0,
        min_ms     = min(timings),
        max_ms     = max(timings),
    )
