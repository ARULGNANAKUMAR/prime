"""
prime.core — Public PRIME object.

This module provides the single developer-facing class, PRIME. All interaction
with the C++ runtime happens through this class; callers never touch _prime
directly.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import _prime

from .config import RuntimeConfig

if TYPE_CHECKING:
    from .llm import LLMModel
    from .generate import GenerationConfig
    from .echo import ECHOEngine
    from .learning import LearningController
    from .optimizer import AdaptiveOptimizationEngine
    from .pipeline import PrimeConfig, PrimeResult, PrimePipeline


class PrimeError(Exception):
    """Raised when a PRIME operation fails."""


class PRIME:
    """The main PRIME runtime object.

    Typical usage::

        from prime import PRIME

        brain = PRIME()
        print(brain.status())
        brain.shutdown()

    The constructor immediately initializes the C++ runtime. Call
    :meth:`shutdown` when finished to release resources, or use the object
    as a context manager::

        with PRIME() as brain:
            print(brain.status())
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        """Create and initialize a PRIME runtime.

        Args:
            config: Optional :class:`RuntimeConfig`. Defaults to
                ``RuntimeConfig()`` (INFO logging, cpu backend).

        Raises:
            ValueError: If ``config`` contains invalid values.
            PrimeError: If the C++ runtime fails to initialize.
        """
        if config is None:
            config = RuntimeConfig()

        config.validate()

        try:
            self._runtime = _prime._Runtime(
                log_level=config._log_level_int(),
                compute_backend=config.compute_backend,
                max_memory_bytes=config.max_memory_bytes,
            )
            self._runtime.initialize()
        except _prime.PrimeError as exc:
            raise PrimeError(str(exc)) from exc

        self._config = config
        self._alive = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Return a human-readable status string from the C++ runtime.

        Raises:
            PrimeError: If the runtime is not running.
        """
        self._require_alive()
        try:
            return self._runtime.status_string()
        except _prime.PrimeError as exc:
            raise PrimeError(str(exc)) from exc

    def shutdown(self) -> None:
        """Shut down the runtime and release all resources.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if not self._alive:
            return
        try:
            self._runtime.shutdown()
        except _prime.PrimeError as exc:
            raise PrimeError(str(exc)) from exc
        finally:
            self._alive = False

    def is_running(self) -> bool:
        """Return True if the runtime is currently active."""
        if not self._alive:
            return False
        state = self._runtime.get_state()
        return state == _prime.STATE_RUNNING

    @property
    def config(self) -> RuntimeConfig:
        """The configuration used to initialize this runtime."""
        return self._config

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PRIME":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def load_model(self, model_path: str) -> "LLMModel":
        """Load a GGUF model file and return an :class:`~prime.llm.LLMModel`.

        The model file is loaded and validated immediately. No network access
        is performed.

        Args:
            model_path: Path to a ``.gguf`` model file.

        Returns:
            A ready-to-use :class:`~prime.llm.LLMModel`.

        Raises:
            PrimeError: If the runtime is not running.
            FileNotFoundError: If the model file does not exist.
            ``prime.llm.LLMError``: If the model cannot be loaded.
        """
        self._require_alive()
        from .llm import LLMModel
        return LLMModel(model_path)

    def solve(self, task: str,
             config: Optional["PrimeConfig"] = None,
             task_id: Optional[str] = None,
             include_trace: bool = False,
             learning: bool = True,
             optimization: bool = True,
             extra_facts: Optional[dict] = None,
             extra_rules: Optional[list] = None,
             ) -> "PrimeResult":
        """
        Unified high-level solve interface (Phase 10).

        Runs the complete PRIME pipeline:
            ATLAS → MOSAIC → Reasoning → ECHO → Learning → Optimization

        Args:
            task:          Natural-language task description.
            config:        Optional :class:`~prime.pipeline.PrimeConfig`.
                           Defaults to ``PrimeConfig.default()``.
            task_id:       Optional unique task ID; auto-generated if omitted.
            include_trace: If True, attach a full :class:`~prime.pipeline.PrimeTrace`
                           to the returned :class:`~prime.pipeline.PrimeResult`.
            learning:      If False, disables the Phase 8 learning step.
            optimization:  If False, disables the Phase 9 optimizer step.
            extra_facts:   Additional facts injected into reasoning.
            extra_rules:   Additional rules injected into reasoning.

        Returns:
            :class:`~prime.pipeline.PrimeResult` with status, answer, and
            optional trace.

        Example::

            result = prime.solve("A car travels 150 km in 3 hours.")
            print(result.answer)
            print(result.status)
        """
        self._require_alive()
        from .pipeline import PrimeConfig, PrimePipeline

        # Determine whether this is a default (cacheable) solve
        _is_default_config = (config is None and learning and optimization
                              and extra_facts is None and extra_rules is None)

        if config is None:
            config = PrimeConfig.default()

        # Apply per-call overrides
        if not learning:
            config.features.enable_learning = False
        if not optimization:
            config.features.enable_optimization = False

        if _is_default_config:
            # Reuse cached default pipeline; recycle every _PIPELINE_RECYCLE_AFTER solves
            # to bound C++ capsule internal state accumulation (~24 KB/solve).
            _limit = getattr(self, '_pipeline_recycle_after', 500)
            _count = getattr(self, '_pipeline_solve_count', 0)
            if (not hasattr(self, '_default_pipeline')
                    or self._default_pipeline is None
                    or _count >= _limit):
                self._default_pipeline = self._make_pipeline(config)
                _count = 0
            self._pipeline_solve_count = _count + 1
            pipeline = self._default_pipeline
        else:
            pipeline = self._make_pipeline(config)
        return pipeline.solve(
            task         = task,
            task_id      = task_id,
            include_trace= include_trace,
            extra_facts  = extra_facts,
            extra_rules  = extra_rules,
        )

    def pipeline(self, config: Optional["PrimeConfig"] = None) -> "PrimePipeline":
        """
        Return a reusable :class:`~prime.pipeline.PrimePipeline` for
        repeated solve() calls without re-initialising subsystems.

        Args:
            config: Optional :class:`~prime.pipeline.PrimeConfig`.

        Returns:
            A ready-to-use :class:`~prime.pipeline.PrimePipeline`.

        Example::

            pipe   = prime.pipeline()
            result = pipe.solve("A car travels 150 km in 3 hours.")
        """
        self._require_alive()
        from .pipeline import PrimeConfig
        if config is None:
            config = PrimeConfig.default()
        return self._make_pipeline(config)

    def _make_pipeline(self, config: "PrimeConfig") -> "PrimePipeline":
        """Internal: construct a PrimePipeline from config feature flags."""
        from .pipeline import PrimePipeline
        from .learning import LearningConfig

        feats = config.features

        # ATLAS is always available
        atlas_eng = self.atlas()

        # MOSAIC
        mosaic_eng = self.mosaic({
            "max_modules":    config.mosaic_max_modules,
            "allow_fallback": config.mosaic_allow_fallback,
            "allow_llm":      feats.enable_llm,
        }) if feats.enable_mosaic else None

        # ECHO
        echo_eng = self.echo(
            db_path          = config.echo_db_path,
            max_experiences  = config.echo_max_experiences,
            retention_policy = config.echo_retention,
        ) if feats.enable_echo else None

        # Learning
        if feats.enable_learning:
            lc_cfg = LearningConfig(
                mode          = config.learning_mode,
                learning_rate = config.learning_rate,
            )
            learn_ctrl = self.learning(
                config     = lc_cfg,
                state_path = config.learning_state_path,
            )
            learn_ctrl.enable()
        else:
            learn_ctrl = None

        # Optimizer
        if feats.enable_optimization:
            from .optimizer import OptimizerConfig, OptimizationPolicy, OptimizationLevel, ResourceConstraints
            constr = ResourceConstraints(
                max_latency_ms   = config.optimizer_max_latency_ms,
                max_memory_bytes = config.optimizer_max_memory_bytes,
            )
            opt_cfg = OptimizerConfig(
                policy      = OptimizationPolicy(config.optimizer_policy),
                level       = OptimizationLevel(config.optimizer_level),
                constraints = constr,
            )
            opt_eng = self.optimizer(config=opt_cfg)
        else:
            opt_eng = None

        return PrimePipeline(
            atlas_engine       = atlas_eng,
            mosaic_engine      = mosaic_eng,
            echo_engine        = echo_eng,
            learning_controller= learn_ctrl,
            optimizer_engine   = opt_eng,
            config             = config,
        )

    def optimizer(self, policy: str = "BALANCED",
                 level: str = "BALANCED",
                 config=None) -> "AdaptiveOptimizationEngine":
        """Return a new :class:`~prime.optimizer.AdaptiveOptimizationEngine`.

        Phase 9: Adaptive Lightweight Optimization Engine.

        The optimizer measures runtime resources and selects resource-aware
        execution strategies that complement MOSAIC (Phase 6) capability
        matching and Phase 8 learned preferences.

        Args:
            policy: :class:`~prime.optimizer.OptimizationPolicy` name.
                    One of ``"BALANCED"`` (default), ``"LOW_LATENCY"``,
                    ``"LOW_MEMORY"``, ``"LOW_CPU"``, ``"MAX_QUALITY"``,
                    ``"RESOURCE_LIMITED"``, ``"ENERGY_AWARE_PROXY"``.
            level:  :class:`~prime.optimizer.OptimizationLevel` name.
                    One of ``"OFF"``, ``"MINIMAL"``, ``"BALANCED"`` (default),
                    ``"AGGRESSIVE"``.
            config: Optional :class:`~prime.optimizer.OptimizerConfig` or
                    dict of config overrides.

        Returns:
            A ready-to-use :class:`~prime.optimizer.AdaptiveOptimizationEngine`.

        Example::

            optimizer = prime.optimizer()
            mosaic    = prime.mosaic()
            plan, trace = optimizer.optimize_plan(mosaic, problem)
            print(optimizer.explain(trace))
        """
        self._require_alive()
        from .optimizer import (AdaptiveOptimizationEngine, OptimizerConfig,
                                 OptimizationPolicy, OptimizationLevel)
        if isinstance(config, dict):
            cfg = OptimizerConfig.from_dict(config)
        elif config is None:
            cfg = OptimizerConfig(
                policy = OptimizationPolicy(policy),
                level  = OptimizationLevel(level),
            )
        else:
            cfg = config
        return AdaptiveOptimizationEngine(config=cfg)

    def __repr__(self) -> str:
        state = "running" if self._alive else "shutdown"
        return f"<PRIME state={state} backend={self._config.compute_backend}>"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_alive(self) -> None:
        if not self._alive:
            raise PrimeError("PRIME runtime has been shut down")

    def atlas(self, **kwargs):
        """Return a new :class:`~prime.atlas.ATLASEngine` instance.

        ATLAS converts problem descriptions into a structured
        :class:`~prime.atlas.Problem` representation that future PRIME
        reasoning phases can consume.  ATLAS does NOT solve problems.

        Args:
            **kwargs: Optional limit overrides forwarded to
                :class:`~prime.atlas.ATLASEngine` (e.g. ``max_entities=256``).

        Returns:
            A ready-to-use :class:`~prime.atlas.ATLASEngine`.

        Raises:
            PrimeError: If the runtime is not running.

        Example::

            atlas = prime.atlas()
            problem = atlas.parse("A car travels 120 km in 3 hours.")
        """
        self._require_alive()
        from .atlas import ATLASEngine, ATLASError
        try:
            return ATLASEngine(**kwargs)
        except ATLASError as exc:
            raise PrimeError(str(exc)) from exc

    def mosaic(self, config=None):
        """Return a new :class:`~prime.mosaic.MOSAICEngine` instance.

        MOSAIC dynamically selects and composes cognitive modules to solve
        a task. It wraps all previous phases (ATLAS, Memory, Reasoning, LLM)
        as pluggable modules and orchestrates them in a bounded, inspectable
        pipeline.

        Args:
            config: Optional dict of MOSAIC configuration overrides
                    (``max_modules``, ``allow_llm``, ``prefer_lightweight``, …).

        Returns:
            A ready-to-use :class:`~prime.mosaic.MOSAICEngine`.

        Example::

            mosaic  = prime.mosaic()
            result  = mosaic.solve("A car travels 150 km in 3 hours.")
            print(result.answer)   # "50"
        """
        self._require_alive()
        from .mosaic import MOSAICEngine
        return MOSAICEngine(prime_instance=self, config=config or {})

    def reasoner(self, llm_provider=None):
        """Return a new :class:`~prime.reasoning.ReasoningEngine` instance.

        The reasoning engine converts ATLAS :class:`~prime.atlas.Problem`
        representations into :class:`~prime.reasoning.ReasoningResult` objects
        via hybrid symbolic, rule-based, and optional LLM-assisted reasoning.

        Args:
            llm_provider: Optional :class:`~prime.reasoning.LLMReasoningProvider`
                for LLM-assisted mode.

        Returns:
            A ready-to-use :class:`~prime.reasoning.ReasoningEngine`.

        Raises:
            PrimeError: If the runtime is not running.

        Example::

            atlas    = prime.atlas()
            reasoner = prime.reasoner()

            problem = atlas.parse("A car travels 150 km in 3 hours.")
            result  = reasoner.solve(problem)
            print(result.explain())
        """
        self._require_alive()
        from .reasoning import ReasoningEngine
        return ReasoningEngine(llm_provider=llm_provider)

    def memory(self, config=None):
        """Return a new :class:`~prime.memory.MemoryEngine` instance.

        Each call returns a fresh engine. The engine shares no state with the
        runtime itself (no Module registration in Phase 3).

        Args:
            config: Optional :class:`~prime.memory.MemoryConfig`. Defaults to
                ``MemoryConfig()`` (in-memory, LRU eviction).

        Returns:
            A ready-to-use :class:`~prime.memory.MemoryEngine`.

        Raises:
            PrimeError: If the runtime is not running.
        """
        self._require_alive()
        from .memory import MemoryEngine, MemoryConfig
        return MemoryEngine(config or MemoryConfig())

    def learning(self, config=None,
                 state_path: Optional[str] = None) -> "LearningController":
        """Return a new :class:`~prime.learning.LearningController` instance.

        The learning engine consumes ECHO experience knowledge and translates
        it into measurable, controlled strategy-score adaptations.  It does NOT
        modify LLM weights.

        Args:
            config:     Optional :class:`~prime.learning.LearningConfig` or dict
                        of config overrides.
            state_path: Optional path for JSON persistence of learned state.

        Returns:
            A ready-to-use :class:`~prime.learning.LearningController`.

        Example::

            lc = prime.learning()
            lc.enable()
            result = prime.learn_and_solve(\"A car travels 150 km in 3 hours.\")
        """
        self._require_alive()
        from .learning import LearningController, LearningConfig
        if isinstance(config, dict):
            cfg = LearningConfig.from_dict(config)
        elif config is None:
            cfg = LearningConfig()
        else:
            cfg = config
        return LearningController(config=cfg, state_path=state_path)

    def echo(self, db_path: Optional[str] = None,
             max_experiences: int = 10_000,
             retention_policy: str = "KEEP_ALL") -> "ECHOEngine":
        """Return a new :class:`~prime.echo.ECHOEngine` instance.

        ECHO (Experience-to-Intelligence Engine) records completed PRIME
        executions, extracts structured observations, detects repeated
        patterns, generalises observations, and builds a queryable knowledge
        base — all without modifying any model weights.

        Args:
            db_path:          Optional path for SQLite persistence.
                              ``None`` keeps everything in memory.
            max_experiences:  Maximum experiences to retain (default 10 000).
            retention_policy: One of ``"KEEP_ALL"`` (default), ``"KEEP_RECENT"``,
                              ``"KEEP_SUCCESSFUL"``, or ``"KEEP_HIGH_VALUE"``.

        Returns:
            A ready-to-use :class:`~prime.echo.ECHOEngine`.

        Raises:
            PrimeError: If the runtime is not running.

        Example::

            echo   = prime.echo()
            result = prime.mosaic().solve("A car travels 150 km in 3 hours.")
            exp    = echo.record(problem_text="A car travels 150 km in 3 hours.",
                                  execution_result=result)
            echo.analyze(exp)
            print(echo.get_patterns())
            print(echo.get_knowledge())
        """
        self._require_alive()
        from .echo import ECHOEngine
        return ECHOEngine(
            db_path=db_path,
            max_experiences=max_experiences,
            retention_policy=retention_policy,
        )
