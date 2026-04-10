"""Core module for medriskeval.

This module provides the foundational types, registries, and utilities
used throughout the evaluation framework.
"""

from medriskeval.core.types import (
    Role,
    ChatMessage,
    Example,
    UsageStats,
    ModelOutput,
    JudgeOutput,
    RunRecord,
    ExampleList,
    MessageList,
    RunRecordList,
)

from medriskeval.core.registry import (
    RegistryError,
    DuplicateRegistrationError,
    NotFoundError,
    BaseRegistry,
    ClassRegistry,
    FunctionRegistry,
    DatasetRegistry,
    JudgeRegistry,
    ModelRegistry,
    TaskRegistry,
    MetricRegistry,
    PromptRegistry,
    list_all_registries,
)

from medriskeval.core.hashing import (
    stable_hash,
    hash_example,
    hash_messages,
    hash_prompt,
    hash_judgment,
    hash_run_config,
    combine_hashes,
    short_hash,
)

__all__ = [
    # Types
    "Role",
    "ChatMessage",
    "Example",
    "UsageStats",
    "ModelOutput",
    "JudgeOutput",
    "RunRecord",
    "ExampleList",
    "MessageList",
    "RunRecordList",
    # Registry classes
    "RegistryError",
    "DuplicateRegistrationError",
    "NotFoundError",
    "BaseRegistry",
    "ClassRegistry",
    "FunctionRegistry",
    # Global registries
    "DatasetRegistry",
    "JudgeRegistry",
    "ModelRegistry",
    "TaskRegistry",
    "MetricRegistry",
    "PromptRegistry",
    "list_all_registries",
    # Hashing utilities
    "stable_hash",
    "hash_example",
    "hash_messages",
    "hash_prompt",
    "hash_judgment",
    "hash_run_config",
    "combine_hashes",
    "short_hash",
]
