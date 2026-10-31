# PRIME Python API Reference

Version 1.0.0

---

## Module `prime`

```python
import prime
prime.__version__   # "1.0.0"
```

---

## Class `PRIME`

Main entry point. Context-manager supported.

```python
from prime import PRIME, RuntimeConfig

prime = PRIME()                        # default config
prime = PRIME(RuntimeConfig(log_level="silent"))
with PRIME() as prime: ...             # auto-shutdown on exit
```

### Methods

#### `prime.solve(task, *, config, task_id, include_trace, learning, optimization, extra_facts, extra_rules) → PrimeResult`

Unified Phase 10 solve pipeline. See full signature in README.

#### `prime.pipeline(config) → PrimePipeline`

Return a reusable `PrimePipeline` (subsystems not re-initialised between calls).

#### `prime.atlas(**kwargs) → ATLASEngine`
#### `prime.mosaic(config) → MOSAICEngine`
#### `prime.reasoner(llm_provider) → ReasoningEngine`
#### `prime.memory(config) → MemoryStore`
#### `prime.echo(db_path, max_experiences, retention_policy) → ECHOEngine`
#### `prime.learning(config, state_path, echo_engine) → LearningController`
#### `prime.optimizer(policy, level, config) → AdaptiveOptimizationEngine`
#### `prime.load_model(model_path) → LLMModel`

#### `prime.shutdown()` / `prime.is_running() → bool`

---

## Class `PrimeConfig`

Unified serialisable configuration.

```python
from prime import PrimeConfig, FeatureFlags

cfg = PrimeConfig()                        # defaults
cfg = PrimeConfig.default()                # alias
cfg = PrimeConfig.minimal()                # ATLAS + Reasoning only
cfg = PrimeConfig.for_ablation("full")     # named baseline
cfg = PrimeConfig.from_json(json_str)

cfg.validate()            # raises ValueError on illegal values
cfg.to_dict()             # → dict
cfg.to_json()             # → str
```

**Key fields** (all have defaults):

| Field | Type | Default |
|---|---|---|
| `log_level` | str | `"warn"` |
| `compute_backend` | str | `"cpu"` |
| `features` | FeatureFlags | `FeatureFlags.full()` |
| `model_path` | str or None | `None` |
| `optimizer_policy` | str | `"BALANCED"` |
| `optimizer_level` | str | `"BALANCED"` |
| `learning_rate` | float | `0.05` |
| `execution_timeout_s` | float or None | `None` |
| `seed` | int or None | `None` |
| `version` | str | `"1.0.0"` |

---

## Class `FeatureFlags`

Toggle individual subsystems.

```python
from prime import FeatureFlags

f = FeatureFlags()                     # all enabled except LLM
f = FeatureFlags.all_off()
f = FeatureFlags.static_baseline()     # ATLAS + Reasoning only
f = FeatureFlags.mosaic_only()
f = FeatureFlags.mosaic_echo()
f = FeatureFlags.mosaic_echo_learning()
f = FeatureFlags.full()                # all on except LLM

f.to_dict()
FeatureFlags.from_dict(d)
```

**Fields**: `enable_memory`, `enable_llm`, `enable_reasoning`,
`enable_mosaic`, `enable_echo`, `enable_learning`, `enable_optimization`

---

## Class `PrimeResult`

Returned by `prime.solve()`.

```python
result.status           # SolveStatus
result.succeeded        # bool
result.task             # str
result.answer           # str
result.answer_unit      # str
result.answer_with_unit # str ("50 km/h")
result.confidence       # float 0–1
result.error            # str
result.latency_ms       # float
result.modules_used     # list[str]
result.optimizer_strategy # str
result.experience_id    # str
result.learning_applied # bool
result.trace_id         # str (UUID)
result.task_id          # str
result.experiment_id    # str
result.trace            # PrimeTrace or None

result.explain()        # → str
result.to_dict()        # → dict (JSON-serialisable)
str(result)             # same as explain()
```

---

## Enum `SolveStatus`

```python
from prime import SolveStatus

SolveStatus.SUCCESS
SolveStatus.PARTIAL_SUCCESS
SolveStatus.INVALID_INPUT
SolveStatus.UNSUPPORTED_TASK
SolveStatus.MODULE_UNAVAILABLE
SolveStatus.MODEL_UNAVAILABLE
SolveStatus.RESOURCE_LIMIT
SolveStatus.TIMEOUT
SolveStatus.VALIDATION_FAILED
SolveStatus.REASONING_FAILED
SolveStatus.COMPOSITION_FAILED
SolveStatus.LEARNING_REJECTED
SolveStatus.OPTIMIZATION_FAILED
SolveStatus.INTERNAL_ERROR

SolveStatus.SUCCESS.succeeded    # True
SolveStatus.TIMEOUT.succeeded    # False
```

---

## Class `PrimeTrace`

Full per-call execution trace (requires `include_trace=True`).

```python
trace.trace_id                 # UUID str
trace.task_id                  # str
trace.atlas_summary            # dict
trace.mosaic_modules           # list[str]
trace.mosaic_selection_reasons # list[str]
trace.reasoning_status         # str
trace.validation_passed        # bool
trace.echo_experience_id       # str
trace.echo_patterns_found      # int
trace.learning_updated         # bool
trace.learning_update_id       # str
trace.optimizer_strategy       # str
trace.optimizer_latency_ms     # float
trace.timings                  # PhaseTimings
trace.config_snapshot          # dict

trace.explain()     # 8-section string from actual data (no LLM)
trace.to_dict()     # → dict
```

---

## Class `PhaseTimings`

Wall-clock per-phase durations (seconds; `None` = phase not run).

```python
t.atlas_s      # float or None
t.mosaic_s     # float or None
t.execution_s  # float or None
t.echo_s       # float or None
t.learning_s   # float or None
t.optimizer_s  # float or None
t.total_s      # float
t.to_dict()    # → dict
```

---

## Class `PrimePipeline`

Reusable pipeline returned by `prime.pipeline()`.

```python
pipe = prime.pipeline(config)
result = pipe.solve(task, task_id=None, include_trace=False,
                    extra_facts=None, extra_rules=None)
```

---

## Research API

### `SyntheticBenchmarkSuite`

```python
from prime import SyntheticBenchmarkSuite

tasks = SyntheticBenchmarkSuite.all_tasks()     # 31 tasks
tasks = SyntheticBenchmarkSuite.core_tasks()    # 7 tasks (CI)
tasks = SyntheticBenchmarkSuite.by_category("arithmetic")

# Task fields
task.task_id       # str
task.category      # str
task.description   # str
task.ground_truth  # str
task.difficulty    # "easy"|"medium"|"hard"
```

### `ExperimentRunner`

```python
from prime import ExperimentRunner, PrimeConfig

runner = ExperimentRunner(factory_fn)
exp    = runner.run(tasks, config, experiment_name="", baseline_name="", n_repeats=1)
```

### `ExperimentResult`

```python
exp.n_tasks             # int
exp.n_correct           # int
exp.accuracy            # float 0–1
exp.mean_latency_ms     # float
exp.median_latency_ms   # float
exp.std_latency_ms      # float
exp.task_results        # list[dict]
exp.config              # dict
exp.system_info         # dict

exp.summary()           # → str
exp.to_json()           # → str
```

### `AblationStudy`

```python
from prime import AblationStudy

study   = AblationStudy(factory_fn)
results = study.run_all(tasks=None, baselines=None, n_repeats=1)
# → dict[baseline_name → ExperimentResult]

print(AblationStudy.compare(results))
```

Supported baseline names: `"static"`, `"mosaic_only"`, `"mosaic_echo"`,
`"mosaic_echo_learning"`, `"full"`.

### `LearningCurveExperiment`

```python
from prime import LearningCurveExperiment

exp   = LearningCurveExperiment(factory_fn)
curve = exp.run(
    training_tasks = [...],
    probe_tasks    = [...],
    checkpoints    = [0, 5, 10, 20],
    config         = PrimeConfig.default(),
)
# → list[LearningCurvePoint]

for pt in curve:
    pt.n_experiences     # int
    pt.accuracy          # float
    pt.mean_latency_ms   # float
    pt.mean_confidence   # float
    pt.to_dict()         # → dict
```

### `profile_component`

```python
from prime import profile_component

prof = profile_component(callable, n=100, label="component_name")

prof.component   # str
prof.n_runs      # int
prof.mean_ms     # float
prof.median_ms   # float
prof.std_ms      # float
prof.min_ms      # float
prof.max_ms      # float
prof.to_dict()   # → dict
str(prof)        # formatted table row
```

### `collect_system_info`

```python
from prime import collect_system_info
info = collect_system_info()
# → dict: os, os_release, machine, python, cpu_count, ram_total_gb, prime_version
```

---

## Phase 9 Optimizer API

```python
from prime import (
    AdaptiveOptimizationEngine, OptimizerConfig,
    OptimizationPolicy, OptimizationLevel, ExecutionStrategy,
    ResourceConstraints, make_optimizer, UNKNOWN,
)

engine = make_optimizer(policy="BALANCED", level="BALANCED")
engine = prime.optimizer(policy="LOW_LATENCY", level="MINIMAL")

trace  = engine.optimize(module_metas=[], task_id="t1",
                          task_signature="sig", learned_hints=None)
plan, trace = engine.optimize_plan(mosaic, problem)

engine.record_result(trace, success=True, latency_ms=5.0)
engine.explain(trace)          # → str from real data
engine.get_state()             # → OptimizationState
engine.get_performance_history() # → PerformanceHistory
engine.summary()               # → dict
engine.invalidate_cache()
engine.set_policy(OptimizationPolicy.LOW_LATENCY)
engine.set_level(OptimizationLevel.MINIMAL)
```

**`ExecutionStrategy`**: `FULL`, `BALANCED`, `LIGHTWEIGHT`, `LOW_MEMORY`,
`LOW_LATENCY`, `CPU_SAVER`

**`OptimizationPolicy`**: `LOW_LATENCY`, `LOW_MEMORY`, `LOW_CPU`, `BALANCED`,
`MAX_QUALITY`, `RESOURCE_LIMITED`, `ENERGY_AWARE_PROXY`

**`OptimizationLevel`**: `OFF`, `MINIMAL`, `BALANCED`, `AGGRESSIVE`
