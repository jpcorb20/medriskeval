"""Evaluation runner for medriskeval.

This module provides the orchestration layer for running evaluations:
- EvaluationRunner: Main pipeline coordinator
- Task: Declarative task configuration
- Caching: Two-level disk cache for generations and judgments
- I/O: JSONL streaming and manifest management

Quick Start:
    >>> from medriskeval.runner import quick_eval, create_psb_task
    >>> 
    >>> # Option 1: Quick evaluation with defaults
    >>> result = quick_eval("psb", target_model, judge_model, max_examples=50)
    >>> 
    >>> # Option 2: Custom task configuration
    >>> task = create_psb_task(max_examples=100)
    >>> runner = EvaluationRunner(target_model, judge_model)
    >>> result = runner.run(task)
    >>> 
    >>> # Access results
    >>> print(result.metrics)
    >>> print(f"Success rate: {result.success_rate:.1%}")

Pipeline Architecture:
    Dataset → TargetPrompt → Generate → JudgePrompt → Judge → Metrics → Artifacts
              (cached)                    (cached)
"""

from medriskeval.runner.cache import (
    DiskCache,
    CacheConfig,
    CacheEntry,
)

from medriskeval.runner.io import (
    RunManifest,
    JSONLWriter,
    JSONLReader,
    ManifestWriter,
    generate_run_id,
    get_git_info,
    create_output_dir,
    find_latest_run,
    load_run_records,
)

from medriskeval.runner.task import (
    Task,
    JudgeConfig,
    create_task,
    create_psb_task,
    create_msb_task,
    create_jbb_task,
    create_xstest_task,
    create_facts_med_task,
    TASK_FACTORIES,
)

from medriskeval.runner.pipeline import (
    EvaluationRunner,
    RunConfig,
    RunResult,
    CLIRunResult,
    run_evaluation,
    quick_eval,
    cli_run_evaluation,
    run_yaml_config,
)


__all__ = [
    # Cache
    "DiskCache",
    "CacheConfig",
    "CacheEntry",
    # I/O
    "RunManifest",
    "JSONLWriter",
    "JSONLReader",
    "ManifestWriter",
    "generate_run_id",
    "get_git_info",
    "create_output_dir",
    "find_latest_run",
    "load_run_records",
    # Task
    "Task",
    "JudgeConfig",
    "create_task",
    "create_psb_task",
    "create_msb_task",
    "create_jbb_task",
    "create_xstest_task",
    "create_facts_med_task",
    "TASK_FACTORIES",
    # Pipeline
    "EvaluationRunner",
    "RunConfig",
    "RunResult",
    "CLIRunResult",
    "run_evaluation",
    "quick_eval",
    "cli_run_evaluation",
    "run_yaml_config",
]
