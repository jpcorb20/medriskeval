"""Model interfaces for medriskeval.

This module provides a unified interface for interacting with various
LLM providers including OpenAI, Azure OpenAI, and vLLM.

Example usage:

    from medriskeval.models import OpenAIModel, GenerationParams
    from medriskeval.core.types import ChatMessage, Role

    # Initialize model
    model = OpenAIModel(model="gpt-4")

    # Create conversation
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="Hello!"),
    ]

    # Generate response
    params = GenerationParams(temperature=0.0, max_tokens=100, seed=42)
    output = model.generate(messages, params)
    
    print(output.text)
    print(f"Tokens used: {output.usage.total_tokens}")
"""

from medriskeval.models.base import (
    ChatModel,
    ContentFilterError,
    ContextLengthError,
    GenerationParams,
    ModelError,
    ModelOutput,
    RateLimitError,
    UsageStats,
)
from medriskeval.models.batching import (
    BatchConfig,
    BatchResult,
    batch_generate_async,
    batch_generate_sync,
    chunk_list,
    estimate_messages_tokens,
    estimate_tokens,
)
from medriskeval.models.retry import (
    DEFAULT_RETRY_CONFIG,
    RateLimiter,
    RetryConfig,
    retry_async,
    retry_sync,
)

# Lazy imports for provider-specific models to avoid import errors
# when dependencies are not installed


def _get_openai_model():
    from medriskeval.models.openai_model import OpenAIModel
    return OpenAIModel


def _get_azure_openai_model():
    from medriskeval.models.openai_model import AzureOpenAIModel
    return AzureOpenAIModel


def _get_vllm_model():
    from medriskeval.models.vllm_model import VLLMModel
    return VLLMModel


def _get_vllm_server_model():
    from medriskeval.models.vllm_model import VLLMServerModel
    return VLLMServerModel


# Expose classes via property-like access for backwards compatibility
class _LazyModule:
    """Lazy module loader to avoid import errors for optional dependencies."""

    @property
    def OpenAIModel(self):
        return _get_openai_model()

    @property
    def AzureOpenAIModel(self):
        return _get_azure_openai_model()

    @property
    def VLLMModel(self):
        return _get_vllm_model()

    @property
    def VLLMServerModel(self):
        return _get_vllm_server_model()


_lazy = _LazyModule()

# For direct imports, we use __getattr__ for lazy loading
def __getattr__(name: str):
    if name == "OpenAIModel":
        return _get_openai_model()
    elif name == "AzureOpenAIModel":
        return _get_azure_openai_model()
    elif name == "VLLMModel":
        return _get_vllm_model()
    elif name == "VLLMServerModel":
        return _get_vllm_server_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base types
    "ChatModel",
    "GenerationParams",
    "ModelOutput",
    "UsageStats",
    # Exceptions
    "ModelError",
    "RateLimitError",
    "ContentFilterError",
    "ContextLengthError",
    # Retry utilities
    "RetryConfig",
    "RateLimiter",
    "retry_sync",
    "retry_async",
    "DEFAULT_RETRY_CONFIG",
    # Batching utilities
    "BatchConfig",
    "BatchResult",
    "batch_generate_sync",
    "batch_generate_async",
    "chunk_list",
    "estimate_tokens",
    "estimate_messages_tokens",
    # Provider implementations (lazy loaded)
    "OpenAIModel",
    "AzureOpenAIModel",
    "VLLMModel",
    "VLLMServerModel",
]
