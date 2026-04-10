"""Base classes for chat model interface.

This module defines the abstract ChatModel interface and supporting types
for unified model access across different providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from medriskeval.core.types import ChatMessage, ModelOutput, UsageStats


@dataclass
class GenerationParams:
    """Parameters for text generation.

    Attributes:
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).
        top_p: Nucleus sampling probability threshold.
        max_tokens: Maximum tokens to generate.
        seed: Random seed for reproducibility (when supported by provider).
        stop: Optional stop sequences.
        presence_penalty: Penalty for token presence (OpenAI-style).
        frequency_penalty: Penalty for token frequency (OpenAI-style).
    """
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: Optional[int] = None
    stop: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        if self.stop is not None:
            d["stop"] = self.stop
        if self.presence_penalty != 0.0:
            d["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            d["frequency_penalty"] = self.frequency_penalty
        return d


class ChatModel(ABC):
    """Abstract base class for chat model providers.

    This interface provides a unified way to interact with different
    LLM providers (OpenAI, vLLM, etc.) for both generation and judge calls.

    Implementations should handle:
    - Message format conversion to provider-specific format
    - Parameter mapping to provider API
    - Error handling and retries (via retry module)
    - Batch processing optimization (via batching module)
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier (e.g., 'gpt-4', 'meta-llama/...')."""
        ...

    @property
    def supports_seed(self) -> bool:
        """Whether this model supports seeded generation for reproducibility."""
        return False

    @property
    def supports_batching(self) -> bool:
        """Whether this model supports native batch API."""
        return False

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        gen_params: Optional[GenerationParams] = None,
    ) -> ModelOutput:
        """Generate a response for a single conversation.
        
        Args:
            messages: Sequence of chat messages forming the conversation.
            gen_params: Generation parameters. Uses defaults if not provided.
            
        Returns:
            ModelOutput containing the generated text and metadata.
            
        Raises:
            ModelError: If generation fails after retries.
        """
        ...

    def generate_batch(
        self,
        messages_list: Sequence[Sequence[ChatMessage]],
        gen_params: Optional[GenerationParams] = None,
    ) -> list[ModelOutput]:
        """Generate responses for multiple conversations.
        
        Default implementation calls generate() sequentially.
        Providers with native batch support should override this.
        
        Args:
            messages_list: List of conversations, each a sequence of messages.
            gen_params: Generation parameters applied to all requests.
            
        Returns:
            List of ModelOutput objects, one per input conversation.
        """
        return [self.generate(msgs, gen_params) for msgs in messages_list]

    def _format_messages(
        self, messages: Sequence[ChatMessage]
    ) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to provider-specific format.
        
        Default implementation returns OpenAI-compatible format.
        Override for providers with different message formats.
        """
        return [msg.to_dict() for msg in messages]


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class RateLimitError(ModelError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


class ContentFilterError(ModelError):
    """Raised when content is blocked by safety filters."""
    pass


class ContextLengthError(ModelError):
    """Raised when input exceeds model's context length."""
    pass
