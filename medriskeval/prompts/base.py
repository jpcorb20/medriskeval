"""Base prompt builder interface for medriskeval.

This module defines the abstract PromptBuilder interface that all prompt
formatters must implement. It provides a consistent API for converting
evaluation examples into model-ready chat messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from medriskeval.core.types import ChatMessage, Example


@dataclass
class PromptConfig:
    """Configuration for prompt builders.
    
    Attributes:
        include_system_prompt: Whether to include system instructions.
        json_output: Whether to request JSON-formatted output.
        strict_json: Whether to enforce strict JSON schema validation.
        max_context_tokens: Optional limit on context length (for truncation).
    """
    include_system_prompt: bool = True
    json_output: bool = True
    strict_json: bool = False
    max_context_tokens: Optional[int] = None


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.
    
    Prompt builders convert Examples into chat messages suitable for
    model inference. They encapsulate:
    - System prompt construction
    - User message formatting
    - Output format specification
    
    Subclasses implement:
    - name: identifier for the prompt type
    - build(): convert Example to chat messages
    
    Example usage:
        builder = PSBMSBJudgePromptBuilder()
        messages = builder.build(example)
        output = model.generate(messages)
    """
    
    # Subclasses should define these
    name: str = ""
    description: str = ""
    
    def __init__(self, config: Optional[PromptConfig] = None) -> None:
        """Initialize the prompt builder.
        
        Args:
            config: Optional configuration for prompt construction.
        """
        self.config = config or PromptConfig()
    
    @abstractmethod
    def build(self, example: Example) -> list[ChatMessage]:
        """Build chat messages from an evaluation example.
        
        Args:
            example: The evaluation example to format.
            
        Returns:
            List of ChatMessage objects forming the prompt.
        """
        pass
    
    def build_batch(self, examples: list[Example]) -> list[list[ChatMessage]]:
        """Build chat messages for multiple examples.
        
        Default implementation calls build() for each example.
        Override for optimized batch processing if needed.
        
        Args:
            examples: List of evaluation examples.
            
        Returns:
            List of message lists, one per example.
        """
        return [self.build(ex) for ex in examples]


class JudgePromptBuilder(PromptBuilder):
    """Base class for LLM-as-judge prompt builders.
    
    Judge prompt builders produce prompts that instruct an LLM to
    evaluate model outputs according to specific criteria. They include:
    - System instruction defining the judge role
    - Rubric or evaluation criteria
    - Output format contract (usually JSON)
    
    Attributes:
        system_instruction: The system prompt defining judge behavior.
        rubric: The evaluation rubric or scoring guidelines.
        output_format: Description of expected output format.
    """
    
    @property
    @abstractmethod
    def system_instruction(self) -> str:
        """Return the system instruction for the judge."""
        ...
    
    @property
    @abstractmethod
    def rubric(self) -> str:
        """Return the evaluation rubric."""
        ...
    
    @property
    @abstractmethod
    def output_format(self) -> str:
        """Return the expected output format specification."""
        ...
    
    def get_system_message(self) -> ChatMessage:
        """Construct the system message for the judge.
        
        Returns:
            ChatMessage with role 'system' containing instructions.
        """
        return ChatMessage(
            role="system",
            content=self.system_instruction,
        )


class TargetPromptBuilder(PromptBuilder):
    """Base class for target model prompt builders.
    
    Target prompt builders format inputs for the model being evaluated.
    They transform benchmark examples into the appropriate input format
    for generation (without judgment instructions).
    """
    
    @abstractmethod
    def build(self, example: Example) -> list[ChatMessage]:
        """Build prompt for the target model.
        
        Args:
            example: The benchmark example.
            
        Returns:
            Chat messages to send to the model being evaluated.
        """
        pass


# Type alias for prompt building functions
PromptBuilderFn = type[PromptBuilder]
