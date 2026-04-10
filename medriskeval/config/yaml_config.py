"""YAML configuration loader for medriskeval.

Supports launching evaluations from YAML config files with:
- Multiple tasks (benchmarks)
- Multiple model configurations with per-model client/generation settings
- Judge model with api_key, base_url, generation, num_samples, refusal_thresholds

Example YAML:

    tasks:
      - benchmark: psb
        max_samples: 50
      - benchmark: jbb
        split: harmful

    models:
      - provider: openai
        model_id: gpt-4.1-mini
        api_key: ${OPENAI_API_KEY}
        base_url: https://api.openai.com/v1
        generation:
          temperature: 0.0
          max_tokens: 512
      - provider: vllm
        model_id: meta-llama/Llama-3-8B
        base_url: http://localhost:8000/v1
        generation:
          max_tokens: 128

    judge:
      provider: openai
      model_id: gpt-4
      api_key: ${JUDGE_API_KEY}
      base_url: https://custom-endpoint.openai.azure.com/
      generation:
        temperature: 1.0
        max_tokens: 8
      num_samples: 10
      refusal_thresholds: [0.333, 0.667]

    output_dir: ./runs
    cache_dir: ./cache
    verbose: false
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from medriskeval.config.schema import (
    BenchmarkName,
    GenerationConfig,
    ModelConfig,
    ModelProvider,
)


# ---------------------------------------------------------------------------
# YAML Pydantic schemas
# ---------------------------------------------------------------------------


class YAMLModelConfig(BaseModel):
    """Model configuration from YAML.

    Extends ModelConfig with an inline generation block so each model
    can carry its own generation overrides.
    """

    model_config = {"protected_namespaces": ()}

    provider: ModelProvider = ModelProvider.OPENAI
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: float = 60.0
    generation: Optional[GenerationConfig] = None

    def to_model_config(self) -> ModelConfig:
        """Convert to the canonical ModelConfig used by the runner."""
        return ModelConfig(
            provider=self.provider,
            model_id=self.model_id,
            api_key=self.api_key,
            base_url=self.base_url,
            api_version=self.api_version,
            timeout=self.timeout,
        )


class YAMLJudgeConfig(BaseModel):
    """Judge model and evaluation configuration from YAML.

    Combines client settings (provider, api_key, base_url) with
    judge-specific evaluation parameters (generation, num_samples,
    refusal_thresholds).
    """

    model_config = {"protected_namespaces": ()}

    provider: ModelProvider = ModelProvider.OPENAI
    model_id: Optional[str] = None  # None → use preset default per benchmark
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: float = 60.0
    generation: Optional[GenerationConfig] = None
    num_samples: Optional[int] = Field(default=None, ge=1)
    refusal_thresholds: Optional[tuple[float, float]] = None

    def to_model_config(self) -> ModelConfig:
        """Convert to the canonical ModelConfig used by the runner."""
        return ModelConfig(
            provider=self.provider,
            model_id=self.model_id or "gpt-4",
            api_key=self.api_key,
            base_url=self.base_url,
            api_version=self.api_version,
            timeout=self.timeout,
        )


class YAMLTaskConfig(BaseModel):
    """Single task entry from YAML."""

    benchmark: BenchmarkName
    split: Optional[str] = None
    categories: Optional[list[str]] = None
    max_samples: Optional[int] = Field(default=None, ge=1)
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)


class YAMLRunConfig(BaseModel):
    """Top-level YAML run configuration.

    Attributes:
        tasks: One or more benchmark tasks to run.
        models: One or more target model configurations.
        judge: Optional judge override (model + evaluation params).
        output_dir: Base output directory.
        cache_dir: Cache directory (null to disable).
        seed: Global random seed.
        verbose: Enable verbose logging.
        resume: Resume from existing progress.
    """

    tasks: list[YAMLTaskConfig]
    models: list[YAMLModelConfig]
    judge: Optional[YAMLJudgeConfig] = None
    output_dir: str = "./runs"
    cache_dir: Optional[str] = "./cache"
    seed: Optional[int] = None
    verbose: bool = False
    resume: bool = True

    @field_validator("tasks", mode="before")
    @classmethod
    def _coerce_tasks(cls, v: Any) -> Any:
        """Accept a single task dict as well as a list."""
        if isinstance(v, dict):
            return [v]
        return v

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models(cls, v: Any) -> Any:
        """Accept a single model dict as well as a list."""
        if isinstance(v, dict):
            return [v]
        return v


# ---------------------------------------------------------------------------
# Env-var interpolation
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:default}`` in strings."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            name = m.group(1)
            default = m.group(2)
            env_val = os.environ.get(name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            raise ValueError(
                f"Environment variable '{name}' is not set and no default provided"
            )
        return _ENV_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_yaml_config(path: str | Path) -> YAMLRunConfig:
    """Load and validate a YAML configuration file.

    Environment variables in the form ``${VAR}`` or ``${VAR:default}``
    are interpolated before validation.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated YAMLRunConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is invalid or missing required fields.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config support. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    raw = _interpolate_env(raw)
    return YAMLRunConfig.model_validate(raw)
