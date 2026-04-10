"""Data I/O utilities for medriskeval.

This module provides utilities for loading data from various sources:
- Hugging Face datasets
- Local JSONL files
- CSV files
- Caching utilities
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

# Optional imports with graceful fallback
try:
    from datasets import load_dataset as hf_load_dataset, Dataset as HFDataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    HFDataset = None  # type: ignore


def get_cache_dir(subdir: str | None = None) -> Path:
    """Get the cache directory for medriskeval.
    
    Uses the following priority:
    1. MEDRISKEVAL_CACHE_DIR environment variable
    2. HF_HOME/medriskeval if HF_HOME is set
    3. ~/.cache/medriskeval
    
    Args:
        subdir: Optional subdirectory within the cache.
        
    Returns:
        Path to the cache directory (created if needed).
    """
    cache_root = os.environ.get("MEDRISKEVAL_CACHE_DIR")
    
    if cache_root is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_root = os.path.join(hf_home, "medriskeval")
        else:
            cache_root = os.path.join(os.path.expanduser("~"), ".cache", "medriskeval")
    
    cache_path = Path(cache_root)
    if subdir:
        cache_path = cache_path / subdir
    
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_hf_dataset(
    path: str,
    name: str | None = None,
    split: str | None = None,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a dataset from Hugging Face Hub.
    
    Args:
        path: Dataset identifier on HF Hub (e.g., "microsoft/PatientSafetyBench").
        name: Optional dataset configuration name.
        split: Optional split to load (e.g., "train", "test").
        cache_dir: Optional cache directory override.
        **kwargs: Additional arguments passed to datasets.load_dataset().
        
    Returns:
        A Hugging Face Dataset or DatasetDict object.
        
    Raises:
        ImportError: If the datasets library is not installed.
        DatasetLoadError: If the dataset cannot be loaded.
    """
    if not HAS_DATASETS:
        raise ImportError(
            "The 'datasets' library is required for Hugging Face datasets. "
            "Install it with: pip install datasets"
        )
    
    if cache_dir is None:
        cache_dir = str(get_cache_dir("hf_datasets"))
    
    try:
        # trust_remote_code is deprecated in newer versions of datasets
        kwargs.pop("trust_remote_code", None)
        return hf_load_dataset(
            path,
            name=name,
            split=split,
            cache_dir=cache_dir,
            **kwargs,
        )
    except Exception as e:
        from medriskeval.datasets.base import DatasetLoadError
        raise DatasetLoadError(f"Failed to load HF dataset '{path}': {e}") from e


def load_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load data from a JSONL file.
    
    Args:
        path: Path to the JSONL file.
        max_samples: Optional maximum number of samples to load.
        
    Returns:
        List of dictionaries, one per line in the file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    path = Path(path)
    samples = []
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    return samples


def save_jsonl(
    data: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    """Save data to a JSONL file.
    
    Args:
        data: Sequence of dictionaries to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def iter_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate over a JSONL file without loading it all into memory.
    
    Args:
        path: Path to the JSONL file.
        max_samples: Optional maximum number of samples to yield.
        
    Yields:
        Dictionaries, one per line in the file.
    """
    path = Path(path)
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def load_csv(
    path: str | Path,
    **kwargs: Any,
) -> Any:
    """Load data from a CSV file using Hugging Face datasets.
    
    Args:
        path: Path to the CSV file.
        **kwargs: Additional arguments passed to load_dataset().
        
    Returns:
        A Hugging Face Dataset object.
    """
    if not HAS_DATASETS:
        raise ImportError(
            "The 'datasets' library is required for CSV loading. "
            "Install it with: pip install datasets"
        )
    
    return hf_load_dataset("csv", data_files=str(path), **kwargs)


def load_csv_simple(
    path: str | Path,
    delimiter: str = ",",
) -> list[dict[str, Any]]:
    """Load CSV without external dependencies.
    
    Args:
        path: Path to the CSV file.
        delimiter: Field delimiter character.
        
    Returns:
        List of dictionaries with column headers as keys.
    """
    import csv
    
    path = Path(path)
    samples = []
    
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            samples.append(dict(row))
    
    return samples


class DataCache:
    """Simple file-based cache for processed datasets.
    
    Caches processed Example lists to avoid re-processing on each load.
    """
    
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the cache.
        
        Args:
            cache_dir: Directory for cache files. Uses default if None.
        """
        if cache_dir is None:
            self.cache_dir = get_cache_dir("processed")
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        # Sanitize the key for use as a filename
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.jsonl"
    
    def exists(self, key: str) -> bool:
        """Check if a cached entry exists."""
        return self._get_cache_path(key).exists()
    
    def get(self, key: str) -> list[dict[str, Any]] | None:
        """Retrieve cached data.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached data as list of dicts, or None if not cached.
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            return load_jsonl(cache_path)
        return None
    
    def set(self, key: str, data: Sequence[dict[str, Any]]) -> None:
        """Store data in the cache.
        
        Args:
            key: Cache key.
            data: Data to cache.
        """
        save_jsonl(data, self._get_cache_path(key))
    
    def clear(self, key: str | None = None) -> None:
        """Clear cached data.
        
        Args:
            key: Specific key to clear, or None to clear all.
        """
        if key is not None:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.jsonl"):
                cache_file.unlink()
