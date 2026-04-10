"""Stable hashing utilities for caching.

This module provides deterministic hashing functions for caching model outputs
and evaluation results. The hashes are stable across Python sessions and
suitable for use as cache keys.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from medriskeval.core.types import Example, ChatMessage


def _normalize_value(obj: Any) -> Any:
    """Recursively normalize an object for consistent JSON serialization.
    
    Handles:
    - Dataclasses with to_dict() method
    - Dictionaries (sorted by key)
    - Lists and tuples
    - Sets (converted to sorted lists)
    - Basic types (str, int, float, bool, None)
    """
    if obj is None:
        return None
    
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if hasattr(obj, "to_dict"):
        return _normalize_value(obj.to_dict())
    
    if isinstance(obj, dict):
        return {k: _normalize_value(v) for k, v in sorted(obj.items())}
    
    if isinstance(obj, (list, tuple)):
        return [_normalize_value(item) for item in obj]
    
    if isinstance(obj, set):
        return [_normalize_value(item) for item in sorted(obj, key=str)]
    
    # Fallback: convert to string
    return str(obj)


def _to_canonical_json(obj: Any) -> str:
    """Convert an object to a canonical JSON string for hashing.
    
    The output is deterministic: same input always produces same output,
    regardless of dict ordering or object type variations.
    """
    normalized = _normalize_value(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_hash(obj: Any, algorithm: str = "sha256", length: int | None = 16) -> str:
    """Compute a stable hash of any JSON-serializable object.
    
    Args:
        obj: Any object that can be normalized to JSON (including dataclasses
             with to_dict() methods).
        algorithm: Hash algorithm to use (sha256, sha1, md5, etc.).
        length: Number of hex characters to return. None for full hash.
        
    Returns:
        Hexadecimal hash string, truncated to `length` characters if specified.
        
    Example:
        >>> stable_hash({"b": 2, "a": 1})  # Same as stable_hash({"a": 1, "b": 2})
        'e3b0c44298fc1c14'
    """
    canonical = _to_canonical_json(obj)
    h = hashlib.new(algorithm)
    h.update(canonical.encode("utf-8"))
    digest = h.hexdigest()
    
    if length is not None:
        return digest[:length]
    return digest


def hash_example(example: "Example") -> str:
    """Compute a stable hash for an Example.
    
    The hash includes: id, benchmark, category, and input.
    Meta is excluded to allow metadata changes without cache invalidation.
    """
    hash_data = {
        "id": example.id,
        "benchmark": example.benchmark,
        "category": example.category,
        "input": example.input,
    }
    return stable_hash(hash_data)


def hash_messages(messages: list["ChatMessage"]) -> str:
    """Compute a stable hash for a list of chat messages."""
    return stable_hash([m.to_dict() if hasattr(m, "to_dict") else m for m in messages])


def hash_prompt(
    prompt: str | list["ChatMessage"],
    model_id: str,
    **kwargs: Any,
) -> str:
    """Compute a cache key for a model prompt.
    
    This is the primary function for generating cache keys for model outputs.
    
    Args:
        prompt: The prompt string or list of chat messages.
        model_id: Identifier of the model being called.
        **kwargs: Additional parameters that affect the output (e.g., temperature,
                  max_tokens). Only include parameters that change the output.
                  
    Returns:
        A stable hash string suitable for use as a cache key.
        
    Example:
        >>> cache_key = hash_prompt(
        ...     "What is the diagnosis?",
        ...     model_id="gpt-4",
        ...     temperature=0.0,
        ...     max_tokens=100,
        ... )
    """
    if isinstance(prompt, str):
        prompt_data: Any = prompt
    else:
        prompt_data = [m.to_dict() if hasattr(m, "to_dict") else m for m in prompt]
    
    hash_data = {
        "prompt": prompt_data,
        "model_id": model_id,
        "params": kwargs,
    }
    return stable_hash(hash_data)


def hash_judgment(
    example_hash: str,
    model_output_hash: str,
    judge_id: str,
    **kwargs: Any,
) -> str:
    """Compute a cache key for a judge evaluation.
    
    Args:
        example_hash: Hash of the example being judged.
        model_output_hash: Hash of the model output being judged.
        judge_id: Identifier of the judge.
        **kwargs: Additional judge parameters.
        
    Returns:
        A stable hash string for caching judge outputs.
    """
    hash_data = {
        "example_hash": example_hash,
        "model_output_hash": model_output_hash,
        "judge_id": judge_id,
        "params": kwargs,
    }
    return stable_hash(hash_data)


def hash_run_config(
    benchmark: str,
    model_id: str,
    judge_id: str | None = None,
    **kwargs: Any,
) -> str:
    """Compute a hash for a run configuration.
    
    Useful for naming output directories or files based on configuration.
    
    Args:
        benchmark: Name of the benchmark.
        model_id: Identifier of the model.
        judge_id: Identifier of the judge (optional).
        **kwargs: Additional configuration parameters.
        
    Returns:
        A short hash string representing this configuration.
    """
    hash_data = {
        "benchmark": benchmark,
        "model_id": model_id,
        "judge_id": judge_id,
        "config": kwargs,
    }
    return stable_hash(hash_data, length=8)


# =============================================================================
# Utility functions for working with hashes
# =============================================================================

def combine_hashes(*hashes: str) -> str:
    """Combine multiple hashes into a single hash.
    
    Useful for creating composite cache keys.
    """
    return stable_hash(list(hashes))


def short_hash(obj: Any, length: int = 8) -> str:
    """Compute a short hash for display purposes.
    
    Not suitable for caching due to higher collision probability,
    but useful for logging and display.
    """
    return stable_hash(obj, length=length)
