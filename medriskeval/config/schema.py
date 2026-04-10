"""Pydantic configuration schemas for medriskeval.

Defines validated configuration objects for:
- Model configuration
- Generation parameters
- Judge configuration
- Task/benchmark configuration
- Run configuration
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    AZURE = "azure"
    VLLM = "vllm"
    LOCAL = "local"


class BenchmarkName(str, Enum):
    """Supported benchmark names."""
    PSB = "psb"
    MSB = "msb"
    JBB = "jbb"
    XSTEST = "xstest"
    FACTS_MED = "facts_med"


class ModelConfig(BaseModel):
    """Configuration for a model instance.

    Attributes:
        provider: Model provider (openai, vllm, local).
        model_id: Model identifier (e.g., gpt-4, meta-llama/...).
        api_key: Optional API key (defaults to env var).
        base_url: Optional custom API URL.
        timeout: Request timeout in seconds.
    """
    model_config = {"protected_namespaces": ()}
    
    provider: ModelProvider = ModelProvider.OPENAI
    model_id: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: float = 60.0
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_string(cls, model_str: str) -> "ModelConfig":
        """Parse model string like 'openai:gpt-4' or 'vllm:meta-llama/...'

        Args:
            model_str: Model specification string.

        Returns:
            Parsed ModelConfig.
        """
        if ":" in model_str:
            provider_str, model_id = model_str.split(":", 1)
            try:
                provider = ModelProvider(provider_str.lower())
            except ValueError:
                # Default to openai if provider not recognized
                provider = ModelProvider.OPENAI
                model_id = model_str
        else:
            provider = ModelProvider.OPENAI
            model_id = model_str

        return cls(provider=provider, model_id=model_id)

    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.provider.value}:{self.model_id}"


class GenerationConfig(BaseModel):
    """Configuration for text generation.

    Attributes:
        temperature: Sampling temperature (0.0 = deterministic).
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum tokens to generate.
        seed: Random seed for reproducibility.
        stop: Stop sequences.
    """
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=128000)
    seed: Optional[int] = None
    stop: Optional[list[str]] = None

    def to_generation_params(self) -> "GenerationParams":
        """Convert to GenerationParams object."""
        from medriskeval.models.base import GenerationParams

        return GenerationParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=self.stop,
        )


class BaseJudgeConfig(BaseModel):
    """Configuration for judge evaluation.

    Attributes:
        model: Judge model configuration.
        generation: Generation parameters for judge.
        num_samples: Number of samples for voting-based judgments.
        refusal_thresholds: Thresholds for refusal classification (low, high).
    """
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(model_id="gpt-4"))
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    num_samples: int = Field(default=1, ge=1, le=50)
    refusal_thresholds: tuple[float, float] = Field(default=(0.333, 0.667))

    @field_validator("refusal_thresholds")
    @classmethod
    def validate_thresholds(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("Low threshold must be less than high threshold")
        return v


class TaskConfig(BaseModel):
    """Configuration for an evaluation task.

    Attributes:
        benchmark: Benchmark name.
        split: Dataset split to evaluate.
        categories: Optional category filter.
        max_examples: Optional limit on examples.
        dataset_kwargs: Additional dataset arguments.
    """
    benchmark: BenchmarkName
    split: str = "test"
    categories: Optional[list[str]] = None
    max_examples: Optional[int] = Field(default=None, ge=1)
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Configuration for an evaluation run.

    Attributes:
        task: Task configuration.
        target_model: Model being evaluated.
        target_generation: Generation params for target model.
        judge: Judge configuration.
        output_dir: Base output directory.
        cache_enabled: Whether to use disk caching.
        resume: Whether to resume from existing progress.
        log_level: Logging level.
    """
    task: TaskConfig
    target_model: ModelConfig
    target_generation: GenerationConfig = Field(default_factory=GenerationConfig)
    judge: Optional[BaseJudgeConfig] = None
    output_dir: str = "./runs"
    cache_enabled: bool = True
    resume: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_cli_args(
        cls,
        task: str,
        model: str,
        judge: Optional[str] = None,
        output_dir: str = "./runs",
        max_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> "RunConfig":
        """Create RunConfig from CLI arguments.

        Args:
            task: Benchmark name (psb, msb, jbb, xstest, facts_med).
            model: Target model string (e.g., openai:gpt-4).
            judge: Judge model string (optional).
            output_dir: Output directory.
            max_examples: Limit on examples.
            **kwargs: Additional options.

        Returns:
            Configured RunConfig.
        """
        from medriskeval.config.presets import get_preset_for_benchmark

        # Parse benchmark
        benchmark = BenchmarkName(task.lower())

        # Get preset for this benchmark
        preset = get_preset_for_benchmark(benchmark)

        # Parse target model
        target_model = ModelConfig.from_string(model)

        # Configure judge
        judge_config = None
        if judge:
            judge_model = ModelConfig.from_string(judge)
            judge_config = BaseJudgeConfig(
                model=judge_model,
                generation=preset.judge_generation,
                num_samples=preset.judge_num_samples,
                refusal_thresholds=preset.refusal_thresholds,
            )
        elif preset.requires_judge:
            # Use preset's default judge
            judge_config = BaseJudgeConfig(
                model=preset.default_judge_model,
                generation=preset.judge_generation,
                num_samples=preset.judge_num_samples,
                refusal_thresholds=preset.refusal_thresholds,
            )

        return cls(
            task=TaskConfig(
                benchmark=benchmark,
                max_examples=max_examples,
                **kwargs.get("task_kwargs", {}),
            ),
            target_model=target_model,
            target_generation=preset.target_generation,
            judge=judge_config,
            output_dir=output_dir,
            cache_enabled=kwargs.get("cache_enabled", True),
            resume=kwargs.get("resume", True),
            log_level=kwargs.get("log_level", "INFO"),
        )


class SummaryConfig(BaseModel):
    """Configuration for summary generation.

    Attributes:
        run_dir: Path to run directory.
        output_format: Output format(s).
        include_categories: Include category breakdown.
        precision: Decimal precision for metrics.
    """
    run_dir: str
    output_format: list[Literal["table", "json", "csv"]] = Field(
        default_factory=lambda: ["table", "csv"]
    )
    include_categories: bool = True
    precision: int = Field(default=3, ge=1, le=6)
