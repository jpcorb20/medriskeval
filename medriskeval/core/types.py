"""Core data types for medriskeval.

This module defines the fundamental data structures used throughout the evaluation framework.
All types are designed to be JSON-serializable for caching and logging purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum


class Role(str, Enum):
    """Standard chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """Normalized chat message format.

    Attributes:
        role: The role of the message sender (system, user, assistant, tool).
        content: The text content of the message.
        name: Optional name identifier for the sender (useful for multi-agent scenarios).
    """
    role: Role | str
    content: str
    name: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize role to string for serialization compatibility
        if isinstance(self.role, Role):
            self.role = self.role.value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
        )


@dataclass
class Example:
    """A single evaluation example from a benchmark dataset.

    Attributes:
        id: Unique identifier for this example.
        benchmark: Name of the benchmark this example belongs to.
        category: Category or subcategory within the benchmark.
        input: The input data (prompt, question, or structured input).
        meta: Additional metadata associated with this example.
    """
    id: str
    benchmark: str
    category: str
    input: str | list[ChatMessage] | dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        input_data = self.input
        if isinstance(self.input, list):
            input_data = [
                m.to_dict() if isinstance(m, ChatMessage) else m 
                for m in self.input
            ]
        return {
            "id": self.id,
            "benchmark": self.benchmark,
            "category": self.category,
            "input": input_data,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Example:
        """Create from dictionary."""
        input_data = data["input"]
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], dict):
            # Check if it looks like chat messages
            if "role" in input_data[0] and "content" in input_data[0]:
                input_data = [ChatMessage.from_dict(m) for m in input_data]
        return cls(
            id=data["id"],
            benchmark=data["benchmark"],
            category=data["category"],
            input=input_data,
            meta=data.get("meta", {}),
        )


@dataclass
class UsageStats:
    """Token usage statistics from model inference.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used (prompt + completion).
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageStats:
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class ModelOutput:
    """Output from a model generation call.

    Attributes:
        text: The generated text content.
        finish_reason: Why generation stopped (e.g., 'stop', 'length', 'content_filter').
        usage: Token usage statistics.
        raw: Raw provider response payload for auditability.
        model: The model identifier that produced this output.
    """
    text: str
    finish_reason: Optional[str] = None
    messages: Optional[list[ChatMessage]] = None
    usage: Optional[UsageStats] = None
    raw: Optional[dict[str, Any]] = None
    model: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {"text": self.text, "finish_reason": self.finish_reason}
        if self.messages is not None:
            d["messages"] = [m.to_dict() for m in self.messages]
        if self.usage is not None:
            d["usage"] = self.usage.to_dict()
        if self.raw is not None:
            d["raw"] = self.raw
        if self.model is not None:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelOutput:
        """Create from dictionary."""
        messages = None
        if "messages" in data and data["messages"] is not None:
            messages = [ChatMessage.from_dict(m) for m in data["messages"]]
        usage = None
        if "usage" in data and data["usage"] is not None:
            usage = UsageStats.from_dict(data["usage"])
        return cls(
            text=data["text"],
            messages=messages,
            usage=usage,
            raw=data.get("raw"),
        )


@dataclass
class JudgeOutput:
    """Output from a judge evaluation.

    Attributes:
        label: Categorical judgment (e.g., "safe", "unsafe", "pass", "fail").
        score: Optional numeric score (e.g., 0-1 scale, 1-5 rating).
        rationale: Optional explanation for the judgment.
        raw: Optional raw response from the judge for debugging.
    """
    label: str
    score: Optional[float] = None
    rationale: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {"label": self.label}
        if self.score is not None:
            d["score"] = self.score
        if self.rationale is not None:
            d["rationale"] = self.rationale
        if self.raw is not None:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeOutput:
        """Create from dictionary."""
        return cls(
            label=data["label"],
            score=data.get("score"),
            rationale=data.get("rationale"),
            raw=data.get("raw"),
        )


@dataclass
class RunRecord:
    """A complete record of a single evaluation run.

    This is the primary unit of evaluation output, containing all information
    needed to reproduce and analyze the evaluation of a single example.

    Attributes:
        example: The input example that was evaluated.
        model_output: The output from the model being evaluated.
        judge_output: The judgment of the model's output.
        model_id: Identifier of the model being evaluated.
        judge_id: Identifier of the judge used.
        timestamp: ISO format timestamp of when the evaluation was run.
        run_id: Unique identifier for this evaluation run.
        meta: Additional metadata about the run.
    """
    example: Example
    model_output: ModelOutput
    judge_output: Optional[JudgeOutput] = None
    model_id: str = ""
    judge_id: str = ""
    timestamp: str = ""
    run_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = {
            "example": self.example.to_dict(),
            "model_output": self.model_output.to_dict(),
            "model_id": self.model_id,
            "judge_id": self.judge_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "meta": self.meta,
        }
        if self.judge_output is not None:
            d["judge_output"] = self.judge_output.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Create from dictionary."""
        judge_output = None
        if "judge_output" in data and data["judge_output"] is not None:
            judge_output = JudgeOutput.from_dict(data["judge_output"])
        return cls(
            example=Example.from_dict(data["example"]),
            model_output=ModelOutput.from_dict(data["model_output"]),
            judge_output=judge_output,
            model_id=data.get("model_id", ""),
            judge_id=data.get("judge_id", ""),
            timestamp=data.get("timestamp", ""),
            run_id=data.get("run_id", ""),
            meta=data.get("meta", {}),
        )


# Type aliases for common patterns
ExampleList = list[Example]
MessageList = list[ChatMessage]
RunRecordList = list[RunRecord]
