"""Configuration module for medriskeval.

Provides pydantic schemas and paper-aligned presets for benchmark evaluation.
"""

from medriskeval.config.schema import (
    BenchmarkName,
    GenerationConfig,
    BaseJudgeConfig,
    ModelConfig,
    ModelProvider,
    RunConfig,
    SummaryConfig,
    TaskConfig,
)
from medriskeval.config.presets import (
    BenchmarkPreset,
    FACTS_MED_PRESET,
    JBB_PRESET,
    MSB_PRESET,
    PRESETS,
    PSB_PRESET,
    XSTEST_PRESET,
    customize_preset,
    get_preset_for_benchmark,
    list_presets,
)

from medriskeval.config.yaml_config import (
    YAMLRunConfig,
    YAMLModelConfig,
    YAMLJudgeConfig,
    YAMLTaskConfig,
    load_yaml_config,
)

__all__ = [
    # Schema classes
    "BenchmarkName",
    "GenerationConfig",
    "BaseJudgeConfig",
    "ModelConfig",
    "ModelProvider",
    "RunConfig",
    "SummaryConfig",
    "TaskConfig",
    # YAML config
    "YAMLRunConfig",
    "YAMLModelConfig",
    "YAMLJudgeConfig",
    "YAMLTaskConfig",
    "load_yaml_config",
    # Preset classes
    "BenchmarkPreset",
    # Preset instances
    "FACTS_MED_PRESET",
    "JBB_PRESET",
    "MSB_PRESET",
    "PRESETS",
    "PSB_PRESET",
    "XSTEST_PRESET",
    # Functions
    "customize_preset",
    "get_preset_for_benchmark",
    "list_presets",
]
