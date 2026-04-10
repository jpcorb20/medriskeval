"""Disk cache for model generations and judge outputs.

This module provides two-level caching:
1. Target model generations (keyed by prompt + model + params)
2. Judge outputs (keyed by example + model output + judge + params)

Cache files are stored as JSON with stable hashes as filenames.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypeVar

from medriskeval.core.hashing import (
    stable_hash,
    hash_prompt,
    hash_judgment,
    hash_example,
)
from medriskeval.core.types import (
    Example,
    ModelOutput,
    JudgeOutput,
    ChatMessage,
)
from medriskeval.models.base import GenerationParams


T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for the disk cache.
    
    Attributes:
        cache_dir: Root directory for cache storage.
        enabled: Whether caching is enabled.
        generation_subdir: Subdirectory for generation cache.
        judge_subdir: Subdirectory for judge output cache.
        max_age_days: Maximum age of cache entries (None = no expiry).
    """
    cache_dir: str | Path = field(default_factory=lambda: _default_cache_dir())
    enabled: bool = True
    generation_subdir: str = "generations"
    judge_subdir: str = "judgments"
    max_age_days: Optional[int] = None
    
    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)


def _default_cache_dir() -> Path:
    """Get the default cache directory."""
    cache_root = os.environ.get("MEDRISKEVAL_CACHE_DIR")
    if cache_root:
        return Path(cache_root) / "runner_cache"
    
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "medriskeval" / "runner_cache"
    
    return Path.home() / ".cache" / "medriskeval" / "runner_cache"


@dataclass
class CacheEntry:
    """A single cache entry with metadata.
    
    Attributes:
        key: The cache key (hash).
        data: The cached data.
        created_at: ISO timestamp of when the entry was created.
        model_id: Model that produced this output (for generations).
        meta: Additional metadata.
    """
    key: str
    data: dict[str, Any]
    created_at: str
    model_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "key": self.key,
            "data": self.data,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "meta": self.meta,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            key=d["key"],
            data=d["data"],
            created_at=d.get("created_at", ""),
            model_id=d.get("model_id", ""),
            meta=d.get("meta", {}),
        )


class DiskCache:
    """Two-level disk cache for generations and judgments.
    
    The cache uses stable hashes as filenames to enable:
    - Deterministic cache keys across runs
    - Separate caching of expensive operations
    - Easy cache inspection and cleanup
    
    Directory structure:
        cache_dir/
            generations/
                <hash>.json
            judgments/
                <hash>.json
    
    Example:
        >>> cache = DiskCache(CacheConfig(cache_dir="./cache"))
        >>> cache.put_generation(messages, model_id, params, output)
        >>> cached = cache.get_generation(messages, model_id, params)
    """
    
    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the disk cache.
        
        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._ensure_dirs()
    
    def _ensure_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        if not self.config.enabled:
            return
        
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.judge_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def generation_dir(self) -> Path:
        """Path to generation cache directory."""
        return self.config.cache_dir / self.config.generation_subdir
    
    @property
    def judge_dir(self) -> Path:
        """Path to judge output cache directory."""
        return self.config.cache_dir / self.config.judge_subdir
    
    # =========================================================================
    # Generation Cache
    # =========================================================================
    
    def _generation_key(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model_id: str,
        gen_params: GenerationParams | dict[str, Any] | None = None,
    ) -> str:
        """Compute cache key for a generation request."""
        params_dict = {}
        if gen_params is not None:
            if isinstance(gen_params, GenerationParams):
                params_dict = gen_params.to_dict()
            else:
                params_dict = gen_params
        
        # Normalize messages
        msg_list = [
            m.to_dict() if hasattr(m, "to_dict") else m
            for m in messages
        ]
        
        return hash_prompt(msg_list, model_id, **params_dict)
    
    def get_generation(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model_id: str,
        gen_params: GenerationParams | dict[str, Any] | None = None,
    ) -> ModelOutput | None:
        """Look up a cached generation.
        
        Args:
            messages: The input messages.
            model_id: The model identifier.
            gen_params: Generation parameters.
            
        Returns:
            Cached ModelOutput or None if not found.
        """
        if not self.config.enabled:
            return None
        
        key = self._generation_key(messages, model_id, gen_params)
        entry = self._load_entry(self.generation_dir / f"{key}.json")
        
        if entry is None:
            return None
        
        # Check expiry
        if self._is_expired(entry.created_at):
            self._delete_entry(self.generation_dir / f"{key}.json")
            return None
        
        return ModelOutput.from_dict(entry.data)
    
    def put_generation(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model_id: str,
        gen_params: GenerationParams | dict[str, Any] | None,
        output: ModelOutput,
    ) -> str:
        """Store a generation in the cache.
        
        Args:
            messages: The input messages.
            model_id: The model identifier.
            gen_params: Generation parameters.
            output: The model output to cache.
            
        Returns:
            The cache key.
        """
        if not self.config.enabled:
            return ""
        
        key = self._generation_key(messages, model_id, gen_params)
        entry = CacheEntry(
            key=key,
            data=output.to_dict(),
            created_at=datetime.utcnow().isoformat(),
            model_id=model_id,
        )
        
        self._save_entry(self.generation_dir / f"{key}.json", entry)
        return key
    
    def has_generation(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model_id: str,
        gen_params: GenerationParams | dict[str, Any] | None = None,
    ) -> bool:
        """Check if a generation is cached."""
        return self.get_generation(messages, model_id, gen_params) is not None
    
    # =========================================================================
    # Judge Cache
    # =========================================================================
    
    def _judge_key(
        self,
        example: Example,
        model_output: ModelOutput,
        judge_id: str,
        judge_params: dict[str, Any] | None = None,
    ) -> str:
        """Compute cache key for a judge request."""
        example_hash = hash_example(example)
        output_hash = stable_hash(model_output.to_dict())
        
        return hash_judgment(
            example_hash,
            output_hash,
            judge_id,
            **(judge_params or {}),
        )
    
    def get_judgment(
        self,
        example: Example,
        model_output: ModelOutput,
        judge_id: str,
        judge_params: dict[str, Any] | None = None,
    ) -> JudgeOutput | None:
        """Look up a cached judgment.
        
        Args:
            example: The evaluated example.
            model_output: The model output that was judged.
            judge_id: The judge model identifier.
            judge_params: Judge-specific parameters.
            
        Returns:
            Cached JudgeOutput or None if not found.
        """
        if not self.config.enabled:
            return None
        
        key = self._judge_key(example, model_output, judge_id, judge_params)
        entry = self._load_entry(self.judge_dir / f"{key}.json")
        
        if entry is None:
            return None
        
        if self._is_expired(entry.created_at):
            self._delete_entry(self.judge_dir / f"{key}.json")
            return None
        
        return JudgeOutput.from_dict(entry.data)
    
    def put_judgment(
        self,
        example: Example,
        model_output: ModelOutput,
        judge_id: str,
        judge_params: dict[str, Any] | None,
        output: JudgeOutput,
    ) -> str:
        """Store a judgment in the cache.
        
        Args:
            example: The evaluated example.
            model_output: The model output that was judged.
            judge_id: The judge model identifier.
            judge_params: Judge-specific parameters.
            output: The judge output to cache.
            
        Returns:
            The cache key.
        """
        if not self.config.enabled:
            return ""
        
        key = self._judge_key(example, model_output, judge_id, judge_params)
        entry = CacheEntry(
            key=key,
            data=output.to_dict(),
            created_at=datetime.utcnow().isoformat(),
            model_id=judge_id,
            meta={"judge_params": judge_params},
        )
        
        self._save_entry(self.judge_dir / f"{key}.json", entry)
        return key
    
    def has_judgment(
        self,
        example: Example,
        model_output: ModelOutput,
        judge_id: str,
        judge_params: dict[str, Any] | None = None,
    ) -> bool:
        """Check if a judgment is cached."""
        return self.get_judgment(example, model_output, judge_id, judge_params) is not None
    
    # =========================================================================
    # Internal Utilities
    # =========================================================================
    
    def _load_entry(self, path: Path) -> CacheEntry | None:
        """Load a cache entry from disk."""
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CacheEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None
    
    def _save_entry(self, path: Path, entry: CacheEntry) -> None:
        """Save a cache entry to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _delete_entry(self, path: Path) -> None:
        """Delete a cache entry."""
        if path.exists():
            path.unlink()
    
    def _is_expired(self, created_at: str) -> bool:
        """Check if a cache entry is expired."""
        if self.config.max_age_days is None:
            return False
        
        try:
            created = datetime.fromisoformat(created_at)
            age = datetime.utcnow() - created
            return age.days > self.config.max_age_days
        except (ValueError, TypeError):
            return False
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def clear_generations(self) -> int:
        """Clear all cached generations.
        
        Returns:
            Number of entries deleted.
        """
        return self._clear_dir(self.generation_dir)
    
    def clear_judgments(self) -> int:
        """Clear all cached judgments.
        
        Returns:
            Number of entries deleted.
        """
        return self._clear_dir(self.judge_dir)
    
    def clear_all(self) -> int:
        """Clear all cached data.
        
        Returns:
            Total number of entries deleted.
        """
        return self.clear_generations() + self.clear_judgments()
    
    def _clear_dir(self, directory: Path) -> int:
        """Clear all JSON files in a directory."""
        if not directory.exists():
            return 0
        
        count = 0
        for f in directory.glob("*.json"):
            f.unlink()
            count += 1
        return count
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache size and entry counts.
        """
        gen_count = len(list(self.generation_dir.glob("*.json"))) if self.generation_dir.exists() else 0
        judge_count = len(list(self.judge_dir.glob("*.json"))) if self.judge_dir.exists() else 0
        
        gen_size = sum(f.stat().st_size for f in self.generation_dir.glob("*.json")) if self.generation_dir.exists() else 0
        judge_size = sum(f.stat().st_size for f in self.judge_dir.glob("*.json")) if self.judge_dir.exists() else 0
        
        return {
            "enabled": self.config.enabled,
            "cache_dir": str(self.config.cache_dir),
            "generations": {"count": gen_count, "size_bytes": gen_size},
            "judgments": {"count": judge_count, "size_bytes": judge_size},
            "total_entries": gen_count + judge_count,
            "total_size_bytes": gen_size + judge_size,
        }
