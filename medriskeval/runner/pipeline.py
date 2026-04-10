"""Evaluation pipeline orchestration.

The EvaluationRunner orchestrates the complete evaluation workflow:
1. Load dataset
2. Build prompts for target model
3. Generate model outputs (with caching)
4. Build judge prompts
5. Run judge evaluation (with caching)
6. Aggregate metrics
7. Save artifacts

Supports resumable runs via JSONL streaming and disk caching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, TYPE_CHECKING

from medriskeval.core.types import (
    Example,
    ChatMessage,
    ModelOutput,
    JudgeOutput,
    RunRecord,
)
from medriskeval.core.hashing import stable_hash
from medriskeval.models.base import GenerationParams, ContentFilterError
from medriskeval.runner.cache import DiskCache, CacheConfig
from medriskeval.runner.io import (
    JSONLWriter,
    JSONLReader,
    ManifestWriter,
    RunManifest,
    create_output_dir,
    load_run_records,
)
from medriskeval.runner.task import Task, JudgeConfig

if TYPE_CHECKING:
    from medriskeval.models.base import ChatModel
    from medriskeval.prompts.base import PromptBuilder, JudgePromptBuilder
    from medriskeval.metrics.base import MetricComputer, BenchmarkMetrics


logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for an evaluation run.
    
    Attributes:
        output_dir: Base directory for outputs.
        cache_config: Configuration for disk caching.
        resume: Whether to resume from existing progress.
        save_interval: Save progress every N examples.
        log_interval: Log progress every N examples.
        fail_fast: Stop on first error if True.
        dry_run: If True, don't actually run models.
    """
    output_dir: str | Path = "./outputs"
    cache_config: Optional[CacheConfig] = None
    resume: bool = True
    save_interval: int = 10
    log_interval: int = 10
    fail_fast: bool = False
    dry_run: bool = False


@dataclass
class RunResult:
    """Result of an evaluation run.
    
    Attributes:
        records: All evaluation records.
        metrics: Computed metrics.
        manifest: Run manifest with metadata.
        output_dir: Path to output directory.
        errors: List of errors encountered.
    """
    records: list[RunRecord]
    metrics: Optional["BenchmarkMetrics"] = None
    manifest: Optional[RunManifest] = None
    output_dir: Optional[Path] = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    content_filtered: int = 0
    judge_content_filtered: int = 0
    
    @property
    def success_count(self) -> int:
        """Number of successful evaluations."""
        return len(self.records)
    
    @property
    def error_count(self) -> int:
        """Number of errors encountered."""
        return len(self.errors)
    
    @property
    def success_rate(self) -> float:
        """Proportion of successful evaluations."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0


class EvaluationRunner:
    """Main orchestrator for evaluation pipelines.
    
    Coordinates:
    - Dataset loading
    - Target model generation
    - Judge evaluation
    - Metrics computation
    - Artifact persistence
    
    Features:
    - Two-level caching (generations and judgments)
    - Resumable runs via streaming JSONL
    - Progress tracking and logging
    - Error handling with optional fail-fast
    
    Example:
        >>> runner = EvaluationRunner(
        ...     target_model=my_model,
        ...     judge_model=judge_model,
        ...     config=RunConfig(output_dir="./results"),
        ... )
        >>> result = runner.run(task)
        >>> print(result.metrics)
    """
    
    def __init__(
        self,
        target_model: "ChatModel",
        judge_model: Optional["ChatModel"] = None,
        config: Optional[RunConfig] = None,
    ) -> None:
        """Initialize the evaluation runner.
        
        Args:
            target_model: Model being evaluated.
            judge_model: Model for judging outputs. Required if task has judge_config.
            config: Run configuration.
        """
        self.target_model = target_model
        self.judge_model = judge_model
        self.config = config or RunConfig()
        
        # Initialize cache
        cache_config = self.config.cache_config or CacheConfig()
        self.cache = DiskCache(cache_config)
        
        self._current_task: Optional[Task] = None
        self._current_manifest: Optional[RunManifest] = None
    
    def run(
        self,
        task: Task,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RunResult:
        """Execute a complete evaluation run.
        
        Args:
            task: Task configuration defining the evaluation.
            progress_callback: Optional callback(completed, total) for progress.
            
        Returns:
            RunResult with records, metrics, and artifacts.
        """
        self._current_task = task
        
        # Setup output directory
        output_dir = create_output_dir(
            self.config.output_dir,
            task.benchmark,
            self.target_model.model_id,
        )
        
        # Initialize manifest
        manifest_writer = ManifestWriter(output_dir)
        
        # Load dataset and get examples
        logger.info(f"Loading dataset: {task.benchmark}")
        dataset = task.load_dataset()
        examples = task.get_examples(dataset)
        total_examples = len(examples)
        
        logger.info(f"Loaded {total_examples} examples for evaluation")
        
        # Create manifest
        manifest = manifest_writer.create(
            benchmark=task.benchmark,
            model_id=self.target_model.model_id,
            judge_id=task.judge_config.judge_id if task.judge_config else "",
            dataset_source=getattr(dataset, "HF_DATASET_ID", task.benchmark),
            total_examples=total_examples,
            generation_params=task.get_generation_params().to_dict(),
            judge_params=task.judge_config.to_dict() if task.judge_config else {},
        )
        self._current_manifest = manifest
        
        # Check for resume
        completed_ids: set[str] = set()
        records: list[RunRecord] = []
        
        records_path = output_dir / "records.jsonl"
        if self.config.resume and records_path.exists():
            reader = JSONLReader(records_path)
            completed_ids = reader.get_completed_ids()
            records = reader.load_all()
            logger.info(f"Resuming: {len(completed_ids)} examples already completed")
        
        # Filter to remaining examples
        remaining_examples = [ex for ex in examples if ex.id not in completed_ids]
        
        # Run evaluation
        errors: list[dict[str, Any]] = []
        content_filtered_count = 0
        judge_content_filtered_count = 0
        
        with JSONLWriter(records_path, mode="a") as writer:
            for i, example in enumerate(remaining_examples):
                try:
                    record = self._evaluate_single(example, task)
                    # Track content-filter events that were handled gracefully
                    if record.model_output and record.model_output.finish_reason == "content_filter":
                        content_filtered_count += 1
                    if record.judge_output and record.judge_output.raw and record.judge_output.raw.get("content_filtered"):
                        judge_content_filtered_count += 1
                    records.append(record)
                    writer.write(record)
                    
                    # Update manifest
                    if (i + 1) % self.config.save_interval == 0:
                        manifest_writer.update(completed_examples=len(records))
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(len(records), total_examples)
                    
                    # Logging
                    if (i + 1) % self.config.log_interval == 0:
                        logger.info(
                            f"Progress: {len(records)}/{total_examples} "
                            f"({len(records)/total_examples*100:.1f}%)"
                        )
                    
                except Exception as e:
                    error_info = {
                        "example_id": example.id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    errors.append(error_info)
                    logger.warning(f"Error on example {example.id}: {e}")
                    
                    if self.config.fail_fast:
                        raise
        
        # Final manifest update
        manifest_writer.update(completed_examples=len(records))
        manifest_writer.mark_complete()
        
        # Compute metrics
        metrics = None
        if task.metric_computer and records:
            logger.info("Computing metrics...")
            metrics = task.metric_computer.compute_full(
                records,
                model_id=self.target_model.model_id,
            )
            
            # Save metrics
            self._save_metrics(output_dir, metrics)
        
        logger.info(
            f"Evaluation complete: {len(records)} successful, "
            f"{len(errors)} errors"
        )
        if content_filtered_count or judge_content_filtered_count:
            logger.info(
                f"Content filtered: {content_filtered_count} target, "
                f"{judge_content_filtered_count} judge"
            )
        
        return RunResult(
            records=records,
            metrics=metrics,
            manifest=manifest,
            output_dir=output_dir,
            content_filtered=content_filtered_count,
            judge_content_filtered=judge_content_filtered_count,
            errors=errors,
        )
    
    def _evaluate_single(self, example: Example, task: Task) -> RunRecord:
        """Evaluate a single example.
        
        Args:
            example: The example to evaluate.
            task: Task configuration.
            
        Returns:
            Complete RunRecord with model output and judgment.
            Content-filtered responses are returned with
            finish_reason='content_filter' instead of raising.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Build target prompt
        if task.target_prompt_builder:
            messages = task.target_prompt_builder.build(example)
        else:
            # Default: use example input directly
            messages = self._default_target_prompt(example)
        
        # Generate target model output (with caching)
        gen_params = task.get_generation_params()
        try:
            model_output = self._generate_with_cache(messages, gen_params)
        except ContentFilterError as e:
            logger.info(f"Content filtered (target) on {example.id}: {e}")
            model_output = ModelOutput(
                text="[CONTENT_FILTERED]",
                finish_reason="content_filter",
                model=self.target_model.model_id,
            )
        
        # Run judge if configured
        judge_output = None
        if task.judge_config and self.judge_model:
            # Skip judging if target was content-filtered (no real response)
            if model_output.finish_reason == "content_filter":
                judge_output = JudgeOutput(
                    score=None,
                    label="content_filtered",
                    raw={"content_filtered": True,
                         "reason": "Target response was blocked by content filter"},
                )
            else:
                try:
                    judge_output = self._judge_with_cache(
                        example,
                        model_output,
                        task.judge_config,
                    )
                except ContentFilterError as e:
                    logger.info(f"Content filtered (judge) on {example.id}: {e}")
                    judge_output = JudgeOutput(
                        score=None,
                        label="judge_content_filtered",
                        raw={"content_filtered": True,
                             "reason": f"Judge blocked by content filter: {e}"},
                    )
        
        return RunRecord(
            example=example,
            model_output=model_output,
            judge_output=judge_output,
            model_id=self.target_model.model_id,
            judge_id=task.judge_config.judge_id if task.judge_config else "",
            timestamp=timestamp,
            run_id=self._current_manifest.run_id if self._current_manifest else "",
        )
    
    def _default_target_prompt(self, example: Example) -> list[ChatMessage]:
        """Build default target prompt from example input."""
        if isinstance(example.input, str):
            return [ChatMessage(role="user", content=example.input)]
        elif isinstance(example.input, list):
            return [
                msg if isinstance(msg, ChatMessage) else ChatMessage.from_dict(msg)
                for msg in example.input
            ]
        elif isinstance(example.input, dict):
            # For structured inputs (like FACTS-med), format as user message
            content = "\n".join(f"{k}: {v}" for k, v in example.input.items())
            return [ChatMessage(role="user", content=content)]
        else:
            return [ChatMessage(role="user", content=str(example.input))]
    
    def _generate_with_cache(
        self,
        messages: list[ChatMessage],
        gen_params: GenerationParams,
    ) -> ModelOutput:
        """Generate with caching support.
        
        Args:
            messages: Input messages.
            gen_params: Generation parameters.
            
        Returns:
            Model output (from cache or fresh generation).
        """
        if self.config.dry_run:
            return ModelOutput(text="[DRY RUN]")
        
        # Check cache
        cached = self.cache.get_generation(
            messages,
            self.target_model.model_id,
            gen_params,
        )
        if cached is not None:
            return cached
        
        # Generate
        output = self.target_model.generate(messages, gen_params)
        
        # Cache result
        self.cache.put_generation(
            messages,
            self.target_model.model_id,
            gen_params,
            output,
        )
        
        return output
    
    def _judge_with_cache(
        self,
        example: Example,
        model_output: ModelOutput,
        judge_config: JudgeConfig,
    ) -> JudgeOutput:
        """Run judge with caching support.
        
        Args:
            example: The evaluated example.
            model_output: The model output to judge.
            judge_config: Judge configuration.
            
        Returns:
            Judge output (from cache or fresh judgment).
        """
        if self.judge_model is None:
            raise RuntimeError("Judge model is required for judging")
        
        judge_params = judge_config.extra.copy()
        if judge_config.generation_params:
            judge_params.update(judge_config.generation_params.to_dict())
        
        # Check cache
        cached = self.cache.get_judgment(
            example,
            model_output,
            judge_config.judge_id,
            judge_params,
        )
        if cached is not None:
            return cached
        
        # Build judge prompt
        # Add model response to example meta for judge prompt
        judge_example = Example(
            id=example.id,
            benchmark=example.benchmark,
            category=example.category,
            input=example.input,
            meta={**example.meta, "response": model_output.text},
        )
        
        prompt_builder = judge_config.prompt_builder
        if prompt_builder is None:
            raise RuntimeError("Judge prompt builder is required")
        
        if isinstance(prompt_builder, type):
            prompt_builder = prompt_builder()
        
        judge_messages = prompt_builder.build(judge_example)
        judge_gen_params = judge_config.generation_params or GenerationParams()
        
        # Handle multi-sample voting
        if judge_config.num_samples > 1:
            judge_output = self._multi_sample_judge(
                judge_messages,
                judge_gen_params,
                judge_config,
            )
        else:
            # Single sample
            raw_output = self.judge_model.generate(judge_messages, judge_gen_params)
            judge_output = self._parse_judge_output(raw_output, judge_config)
        
        # Cache result
        self.cache.put_judgment(
            example,
            model_output,
            judge_config.judge_id,
            judge_params,
            judge_output,
        )
        
        return judge_output
    
    def _multi_sample_judge(
        self,
        messages: list[ChatMessage],
        gen_params: GenerationParams,
        judge_config: JudgeConfig,
    ) -> JudgeOutput:
        """Run multiple judge samples and aggregate.
        
        Args:
            messages: Judge prompt messages.
            gen_params: Generation parameters (typically high temperature).
            judge_config: Judge configuration with num_samples and aggregation_fn.
            
        Returns:
            Aggregated JudgeOutput.
        """
        samples = []
        
        for _ in range(judge_config.num_samples):
            output = self.judge_model.generate(messages, gen_params)
            samples.append(output.text)
        
        # Parse and aggregate
        if judge_config.aggregation_fn:
            # For binary judges (like refusal), parse to int votes
            votes = []
            for sample in samples:
                try:
                    vote = int(sample.strip()[0]) if sample.strip() else 0
                    votes.append(vote)
                except (ValueError, IndexError):
                    votes.append(0)
            
            aggregated = judge_config.aggregation_fn(votes)
            
            return JudgeOutput(
                label=aggregated.get("refusal_category", "unknown"),
                score=aggregated.get("refusal_rate"),
                rationale=None,
                raw={"samples": samples, "aggregated": aggregated},
            )
        else:
            # Default: majority vote on parsed labels
            return JudgeOutput(
                label=samples[0] if samples else "unknown",
                raw={"samples": samples},
            )
    
    def _parse_judge_output(
        self,
        raw_output: ModelOutput,
        judge_config: JudgeConfig,
    ) -> JudgeOutput:
        """Parse judge output into structured JudgeOutput.
        
        Attempts to parse JSON responses, falling back to text extraction.
        
        Args:
            raw_output: Raw model output from judge.
            judge_config: Judge configuration.
            
        Returns:
            Parsed JudgeOutput.
        """
        import json
        import re
        
        text = raw_output.text.strip()
        
        # Try JSON parsing
        try:
            # Try to parse as a JSON array first (e.g. FACTS sentence-level judgments)
            # Only match when the response starts with '[' to avoid false positives
            # from bracket content in single-object judge responses.
            stripped = text.lstrip()
            if stripped.startswith("["):
                data_list = json.loads(stripped)
                if isinstance(data_list, list) and len(data_list) > 0 and isinstance(data_list[0], dict):
                    first = data_list[0]
                    return JudgeOutput(
                        label=str(first.get("label", first.get("score", ""))),
                        score=float(first["score"]) if "score" in first else None,
                        rationale=first.get("reason", first.get("rationale")),
                        raw={"text": text, "parsed": first, "sentences": data_list},
                    )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        try:
            # Find single JSON object in response
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return JudgeOutput(
                    label=str(data.get("label", data.get("score", ""))),
                    score=float(data["score"]) if "score" in data else None,
                    rationale=data.get("reason", data.get("rationale")),
                    raw={"text": text, "parsed": data},
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Fallback: extract score from text
        score_match = re.search(r'\b([1-5])\b', text)
        if score_match:
            score = int(score_match.group(1))
            return JudgeOutput(
                label=str(score),
                score=float(score),
                rationale=text,
                raw={"text": text},
            )
        
        # Last resort: return raw text as label
        return JudgeOutput(
            label=text[:50] if text else "unknown",
            rationale=text,
            raw={"text": text},
        )
    
    def _save_metrics(self, output_dir: Path, metrics: "BenchmarkMetrics") -> None:
        """Save computed metrics to output directory."""
        import json
        
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metrics to {metrics_path}")


# =============================================================================
# Convenience functions
# =============================================================================

def run_evaluation(
    task: Task,
    target_model: "ChatModel",
    judge_model: Optional["ChatModel"] = None,
    output_dir: str | Path = "./outputs",
    resume: bool = True,
    **kwargs: Any,
) -> RunResult:
    """Run an evaluation with minimal setup.
    
    Convenience function that creates an EvaluationRunner and executes the task.
    
    Args:
        task: Task configuration.
        target_model: Model to evaluate.
        judge_model: Judge model (required if task has judge_config).
        output_dir: Base output directory.
        resume: Whether to resume from existing progress.
        **kwargs: Additional RunConfig options.
        
    Returns:
        RunResult with evaluation results.
        
    Example:
        >>> from medriskeval.runner import run_evaluation, create_psb_task
        >>> task = create_psb_task(max_examples=10)
        >>> result = run_evaluation(task, my_model, judge_model)
    """
    config = RunConfig(output_dir=output_dir, resume=resume, **kwargs)
    runner = EvaluationRunner(target_model, judge_model, config)
    return runner.run(task)


def quick_eval(
    benchmark: str,
    target_model: "ChatModel",
    judge_model: Optional["ChatModel"] = None,
    max_examples: Optional[int] = None,
    output_dir: str | Path = "./outputs",
    **task_kwargs: Any,
) -> RunResult:
    """Quick evaluation with default task configuration.
    
    Creates a task from benchmark name and runs evaluation.
    
    Args:
        benchmark: Benchmark name (psb, msb, jbb, xstest, facts_med).
        target_model: Model to evaluate.
        judge_model: Judge model.
        max_examples: Limit number of examples.
        output_dir: Output directory.
        **task_kwargs: Additional task options.
        
    Returns:
        RunResult with evaluation results.
        
    Example:
        >>> result = quick_eval("psb", my_model, judge_model, max_examples=50)
    """
    from medriskeval.runner.task import create_task
    
    task = create_task(
        benchmark,
        model_id=target_model.model_id,
        max_examples=max_examples,
        **task_kwargs,
    )
    
    return run_evaluation(task, target_model, judge_model, output_dir)


# =============================================================================
# CLI-compatible evaluation entry point
# =============================================================================

@dataclass
class CLIRunResult:
    """Result returned by CLI run_evaluation function.
    
    Simplified result interface for CLI consumption.
    """
    run_dir: Path
    total_samples: int
    duration_seconds: float
    metrics: dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    content_filtered: int = 0
    judge_content_filtered: int = 0


def cli_run_evaluation(
    benchmark: Any,  # BenchmarkName enum or str
    model_config: Any,  # ModelConfig from config.schema
    judge_config: Any,  # ModelConfig from config.schema  
    output_dir: str | Path = "./runs",
    cache_dir: Optional[str | Path] = "./cache",
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> CLIRunResult:
    """Run evaluation from CLI with config objects.
    
    This is the main entry point called by the CLI.
    
    Args:
        benchmark: Benchmark name (enum or string).
        model_config: Target model configuration.
        judge_config: Judge model configuration.
        output_dir: Output directory for run artifacts.
        cache_dir: Cache directory (None to disable).
        max_samples: Maximum samples to evaluate.
        seed: Random seed.
        verbose: Enable verbose logging.
        
    Returns:
        CLIRunResult with run summary.
    """
    import time
    from medriskeval.runner.task import create_task
    
    start_time = time.time()
    
    # Convert benchmark to string if enum
    benchmark_name = benchmark.value if hasattr(benchmark, "value") else str(benchmark)
    
    # Get preset for benchmark
    from medriskeval.config import get_preset_for_benchmark
    preset = get_preset_for_benchmark(benchmark_name)
    
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create models from config
    target_model = _create_model_from_config(model_config)
    judge_model = _create_model_from_config(judge_config) if preset.requires_judge else None
    
    # Create task
    task = create_task(
        benchmark_name,
        model_id=model_config.model_id,
        judge_id=judge_config.model_id if judge_config else None,
        max_examples=max_samples,
    )
    
    # Setup cache config
    cache_config = None
    if cache_dir:
        cache_config = CacheConfig(
            cache_dir=Path(cache_dir),
            enabled=True,
        )
    
    # Run evaluation
    config = RunConfig(
        output_dir=output_dir,
        cache_config=cache_config,
        resume=True,
    )
    
    runner = EvaluationRunner(target_model, judge_model, config)
    result = runner.run(task)
    
    duration = time.time() - start_time
    
    # Build CLI result
    return CLIRunResult(
        run_dir=result.output_dir or Path(output_dir),
        total_samples=result.success_count,
        duration_seconds=duration,
        metrics=result.metrics.to_dict() if result.metrics else {},
        success=result.error_count == 0,
        error_message=str(result.errors[0]) if result.errors else None,
        content_filtered=result.content_filtered,
        judge_content_filtered=result.judge_content_filtered,
    )


def _create_model_from_config(config: Any) -> "ChatModel":
    """Create a ChatModel from a ModelConfig.
    
    Args:
        config: ModelConfig with provider and model_id.
        
    Returns:
        ChatModel instance.
    """
    from medriskeval.models.openai_model import OpenAIModel
    
    provider = config.provider.value if hasattr(config.provider, "value") else str(config.provider)
    timeout = getattr(config, "timeout", 60.0) or 60.0
    api_key = config.api_key or None  # treat empty string as None
    base_url = config.base_url or None
    
    if provider == "openai":
        return OpenAIModel(
            model=config.model_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
    elif provider == "azure":
        from medriskeval.models.openai_model import AzureOpenAIModel
        api_version = getattr(config, "api_version", None) or "2025-01-01-preview"
        return AzureOpenAIModel(
            deployment=config.model_id,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=base_url,
            timeout=timeout,
        )
    elif provider in ("vllm", "local"):
        return OpenAIModel(
            model=config.model_id,
            api_key=api_key or "dummy",
            base_url=base_url or "http://localhost:8000/v1",
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# YAML config runner
# =============================================================================


def run_yaml_config(
    yaml_config: "YAMLRunConfig",
    dry_run: bool = False,
) -> list[CLIRunResult]:
    """Execute all task × model combinations defined in a YAML config.

    Args:
        yaml_config: Parsed and validated YAML run configuration.
        dry_run: If True, only validate and print the plan without running.

    Returns:
        List of CLIRunResult, one per (task, model) combination.
    """
    import time
    from medriskeval.config import get_preset_for_benchmark
    from medriskeval.runner.task import create_task
    from medriskeval.runner.cache import CacheConfig

    if dry_run:
        logging.basicConfig(level=logging.WARNING)
    elif yaml_config.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    results: list[CLIRunResult] = []

    for task_cfg in yaml_config.tasks:
        benchmark_name = task_cfg.benchmark.value
        preset = get_preset_for_benchmark(benchmark_name)

        for model_cfg in yaml_config.models:
            label = f"{benchmark_name} / {model_cfg.model_id}"
            logger.info(f"--- Starting: {label} ---")

            start_time = time.time()

            # Resolve judge model: explicit yaml > preset default
            judge_yaml = yaml_config.judge
            if judge_yaml is not None and judge_yaml.model_id is not None:
                judge_model_cfg = judge_yaml.to_model_config()
            else:
                judge_model_cfg = preset.default_judge_model
                # Even when using the preset model, allow yaml to
                # override client settings (api_key, base_url)
                if judge_yaml is not None:
                    if judge_yaml.api_key:
                        judge_model_cfg = judge_model_cfg.model_copy(
                            update={"api_key": judge_yaml.api_key}
                        )
                    if judge_yaml.base_url:
                        judge_model_cfg = judge_model_cfg.model_copy(
                            update={"base_url": judge_yaml.base_url}
                        )

            if dry_run:
                _log_dry_run(benchmark_name, model_cfg, judge_model_cfg,
                             task_cfg, yaml_config, preset)
                results.append(CLIRunResult(
                    run_dir=Path(yaml_config.output_dir),
                    total_samples=0,
                    duration_seconds=0.0,
                    metrics={},
                    success=True,
                    error_message=None,
                ))
                continue

            # Build models
            target_model = _create_model_from_config(model_cfg.to_model_config())
            judge_model = (
                _create_model_from_config(judge_model_cfg)
                if preset.requires_judge
                else None
            )

            # Build task
            task_kwargs: dict[str, Any] = {
                "model_id": model_cfg.model_id,
                "max_examples": task_cfg.max_samples,
            }
            if judge_model_cfg:
                task_kwargs["judge_id"] = judge_model_cfg.model_id
            # Pass num_samples from YAML to JBB/XSTest task factories
            if judge_yaml is not None and judge_yaml.num_samples is not None:
                task_kwargs["num_samples"] = judge_yaml.num_samples
            task_kwargs.update(task_cfg.dataset_kwargs)

            task = create_task(benchmark_name, **task_kwargs)

            # Override split if specified
            if task_cfg.split is not None:
                task.split = task_cfg.split
            if task_cfg.categories is not None:
                task.categories = task_cfg.categories

            # Apply per-model target generation params
            if model_cfg.generation is not None:
                task.generation_params = model_cfg.generation.to_generation_params()

            # Apply YAML judge overrides onto the task's JudgeConfig
            if task.judge_config is not None and judge_yaml is not None:
                # Judge generation params
                if judge_yaml.generation is not None:
                    task.judge_config.generation_params = (
                        judge_yaml.generation.to_generation_params()
                    )
                # Judge num_samples (override even if already set by factory)
                if judge_yaml.num_samples is not None:
                    task.judge_config.num_samples = judge_yaml.num_samples
                # Refusal thresholds: wrap aggregate_refusal_votes with
                # custom thresholds via functools.partial
                if judge_yaml.refusal_thresholds is not None:
                    if task.judge_config.aggregation_fn is not None:
                        import functools
                        task.judge_config.aggregation_fn = functools.partial(
                            task.judge_config.aggregation_fn,
                            thresholds=judge_yaml.refusal_thresholds,
                        )

            # Cache
            cache_config = None
            if yaml_config.cache_dir:
                cache_config = CacheConfig(
                    cache_dir=Path(yaml_config.cache_dir),
                    enabled=True,
                )

            config = RunConfig(
                output_dir=yaml_config.output_dir,
                cache_config=cache_config,
                resume=yaml_config.resume,
            )

            runner = EvaluationRunner(target_model, judge_model, config)

            try:
                result = runner.run(task)
                duration = time.time() - start_time

                cli_result = CLIRunResult(
                    run_dir=result.output_dir or Path(yaml_config.output_dir),
                    total_samples=result.success_count,
                    duration_seconds=duration,
                    metrics=result.metrics.to_dict() if result.metrics else {},
                    success=result.error_count == 0,
                    error_message=str(result.errors[0]) if result.errors else None,
                    content_filtered=result.content_filtered,
                    judge_content_filtered=result.judge_content_filtered,
                )
            except Exception as exc:
                duration = time.time() - start_time
                logger.error(f"Failed: {label} — {exc}")
                cli_result = CLIRunResult(
                    run_dir=Path(yaml_config.output_dir),
                    total_samples=0,
                    duration_seconds=duration,
                    metrics={},
                    success=False,
                    error_message=str(exc),
                )

            results.append(cli_result)
            logger.info(
                f"--- Finished: {label} — "
                f"{cli_result.total_samples} samples in {duration:.1f}s ---"
            )

    return results


def _log_dry_run(
    benchmark: str,
    model_cfg: Any,
    judge_cfg: Any,
    task_cfg: Any,
    yaml_cfg: Any,
    preset: Any,
) -> None:
    """Print dry-run summary for one (task, model) pair."""
    judge_yaml = yaml_cfg.judge
    print(f"\n  Task:       {benchmark}")
    print(f"  Model:      {model_cfg.provider.value}:{model_cfg.model_id}")
    judge_id = judge_cfg.model_id if hasattr(judge_cfg, "model_id") else str(judge_cfg)
    print(f"  Judge:      {judge_id}")
    if hasattr(judge_cfg, "base_url") and judge_cfg.base_url:
        print(f"  Judge URL:  {judge_cfg.base_url}")
    if task_cfg.max_samples:
        print(f"  Max samples: {task_cfg.max_samples}")
    if task_cfg.split:
        print(f"  Split:      {task_cfg.split}")
    if model_cfg.generation:
        g = model_cfg.generation
        print(f"  Generation: temp={g.temperature}, max_tokens={g.max_tokens}")
    else:
        pg = preset.target_generation
        print(f"  Generation: temp={pg.temperature}, max_tokens={pg.max_tokens} (preset)")
    # Judge evaluation params
    if judge_yaml and judge_yaml.generation:
        jg = judge_yaml.generation
        print(f"  Judge gen:  temp={jg.temperature}, max_tokens={jg.max_tokens}")
    else:
        jg = preset.judge_generation
        print(f"  Judge gen:  temp={jg.temperature}, max_tokens={jg.max_tokens} (preset)")
    ns = (judge_yaml.num_samples if judge_yaml and judge_yaml.num_samples
          else preset.judge_num_samples)
    print(f"  Judge samples: {ns}")
    if benchmark in ("jbb", "xstest"):
        rt = (judge_yaml.refusal_thresholds if judge_yaml and judge_yaml.refusal_thresholds
              else preset.refusal_thresholds)
        print(f"  Refusal thresholds: {rt}")
    print(f"  Output:     {yaml_cfg.output_dir}")
