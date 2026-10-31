"""PRIME — Phase 10: Full Integration & Research Validation."""

from .core import PRIME, PrimeError
from .config import RuntimeConfig
from .llm import LLMModel, LLMError
from .generate import GenerationConfig
from .memory import MemoryEngine, MemoryConfig, MemoryItem, MemoryError
from .atlas import ATLASEngine, ATLASError, Problem, ValidationResult, UniversalProblemProto
from .reasoning import (
    ReasoningEngine, ReasoningError, ReasoningResult, ReasoningTrace,
    ReasoningStep, GoalResult, ConstraintCheckResult,
    make_rule, make_condition, make_compound_condition, make_conclusion,
    LLMReasoningProvider, MockLLMProvider,
)
from .mosaic import (
    MOSAICEngine, MOSAICError, CompositionPlan, ExecutionResult,
    TaskRequirements, CompositionTraceStep, ModuleMetadata,
    Capability, ResourceCost,
)
from .echo import (
    ECHOEngine, ECHOError,
    Experience, ExperienceOutcome, ExperienceLifecycle, ExperienceEvent,
    Observation, Pattern, Generalization, KnowledgeCandidate,
    KnowledgeType, PrincipleStatus, ValidationStatus,
    RetentionPolicy, EchoTrace,
    ExperienceStore, ExperienceCollector, ObservationExtractor,
    PatternDetector, GeneralizationEngine, PrincipleExtractor,
    ExperienceValidator, ExperienceSimilarity, ConfidenceEstimator,
    ExperienceQueryEngine,
)

from .pipeline import (
    SolveStatus, FeatureFlags, PrimeConfig,
    PhaseTimings, PrimeTrace, PrimeResult,
    PrimePipeline, PipelineError,
    collect_system_info,
)

from .research import (
    BenchmarkTask, TaskResult, SyntheticBenchmarkSuite,
    ExperimentResult, ExperimentRunner,
    AblationStudy, ABLATION_CONFIGS,
    LearningCurveExperiment, LearningCurvePoint,
    ComponentProfile, profile_component,
)

from .optimizer import (
    AdaptiveOptimizationEngine, OptimizerError,
    OptimizationPolicy, OptimizationLevel, ExecutionStrategy,
    OptimizerConfig, ResourceConstraints, CostWeights,
    ResourceMonitor, ResourceSnapshot, CPUInfo, MemoryInfo,
    PerformanceRecord, PerformanceHistory,
    CostModel, OptimizationTrace, CandidateAssessment,
    OptimizationState, OptimizationCache,
    AdaptiveScheduler, ExecutionStrategySelector, ModuleRanker,
    RuntimeProfiler, HysteresisController, ResourceClassifier,
    ThreadSelector, StrategyConfig, STRATEGY_CONFIGS,
    make_optimizer, UNKNOWN,
)

from .learning import (
    LearningController, LearningError,
    LearningMode, LearningTarget, UpdateStatus, LearningEligibility,
    DriftStatus, ApprovalStatus,
    LearningConfig, LearningState, LearningStateVersion,
    LearningMetrics, LearningTraceEntry,
    CandidateUpdate, StrategyStats, LearnedPolicy,
    StrategyBiasAdapter, ModelAdapter,
    ExperienceReplayBuffer, ReplayEntry,
    StabilityController, RollbackManager,
    LearnedPreferenceProvider,
    StrategyLearner, KnowledgeEvaluator, FeedbackAnalyzer,
    EligibilityChecker, IncrementalUpdateEngine,
    make_context_key,
)

__version__ = "1.0.0"
__all__ = [
    "PRIME", "PrimeError", "RuntimeConfig",
    "LLMModel", "LLMError", "GenerationConfig",
    "MemoryEngine", "MemoryConfig", "MemoryItem", "MemoryError",
    "ATLASEngine", "ATLASError", "Problem", "ValidationResult", "UniversalProblemProto",
    "ReasoningEngine", "ReasoningError", "ReasoningResult", "ReasoningTrace",
    "ReasoningStep", "GoalResult", "ConstraintCheckResult",
    "make_rule", "make_condition", "make_compound_condition", "make_conclusion",
    "LLMReasoningProvider", "MockLLMProvider",
    "MOSAICEngine", "MOSAICError", "CompositionPlan", "ExecutionResult",
    "TaskRequirements", "CompositionTraceStep", "ModuleMetadata",
    "Capability", "ResourceCost",
    # Phase 7 — ECHO
    "ECHOEngine", "ECHOError",
    "Experience", "ExperienceOutcome", "ExperienceLifecycle", "ExperienceEvent",
    "Observation", "Pattern", "Generalization", "KnowledgeCandidate",
    "KnowledgeType", "PrincipleStatus", "ValidationStatus",
    "RetentionPolicy", "EchoTrace",
    "ExperienceStore", "ExperienceCollector", "ObservationExtractor",
    "PatternDetector", "GeneralizationEngine", "PrincipleExtractor",
    "ExperienceValidator", "ExperienceSimilarity", "ConfidenceEstimator",
    "ExperienceQueryEngine",
    # Phase 8 — Self-Learning
    "LearningController", "LearningError",
    "LearningMode", "LearningTarget", "UpdateStatus", "LearningEligibility",
    "DriftStatus", "ApprovalStatus",
    "LearningConfig", "LearningState", "LearningStateVersion",
    "LearningMetrics", "LearningTraceEntry",
    "CandidateUpdate", "StrategyStats", "LearnedPolicy",
    "StrategyBiasAdapter", "ModelAdapter",
    "ExperienceReplayBuffer", "ReplayEntry",
    "StabilityController", "RollbackManager",
    "LearnedPreferenceProvider",
    "StrategyLearner", "KnowledgeEvaluator", "FeedbackAnalyzer",
    "EligibilityChecker", "IncrementalUpdateEngine",
    "make_context_key",
    # Phase 9 — Adaptive Optimization
    "AdaptiveOptimizationEngine", "OptimizerError",
    "OptimizationPolicy", "OptimizationLevel", "ExecutionStrategy",
    "OptimizerConfig", "ResourceConstraints", "CostWeights",
    "ResourceMonitor", "ResourceSnapshot", "CPUInfo", "MemoryInfo",
    "PerformanceRecord", "PerformanceHistory",
    "CostModel", "OptimizationTrace", "CandidateAssessment",
    "OptimizationState", "OptimizationCache",
    "AdaptiveScheduler", "ExecutionStrategySelector", "ModuleRanker",
    "RuntimeProfiler", "HysteresisController", "ResourceClassifier",
    "ThreadSelector", "StrategyConfig", "STRATEGY_CONFIGS",
    "make_optimizer", "UNKNOWN",
    # Phase 10 — Integration & Research
    "SolveStatus", "FeatureFlags", "PrimeConfig",
    "PhaseTimings", "PrimeTrace", "PrimeResult",
    "PrimePipeline", "PipelineError", "collect_system_info",
    "BenchmarkTask", "TaskResult", "SyntheticBenchmarkSuite",
    "ExperimentResult", "ExperimentRunner",
    "AblationStudy", "ABLATION_CONFIGS",
    "LearningCurveExperiment", "LearningCurvePoint",
    "ComponentProfile", "profile_component",
]
