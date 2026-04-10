"""I/O utilities for evaluation runner.

Provides:
- JSONL writer for streaming run records
- Manifest writer for run metadata
- Utilities for loading/resuming runs
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, TextIO

from medriskeval.core.types import RunRecord, Example
from medriskeval.core.hashing import hash_run_config, stable_hash


@dataclass
class RunManifest:
    """Manifest containing metadata about an evaluation run.
    
    Attributes:
        run_id: Unique identifier for this run.
        benchmark: Name of the benchmark being evaluated.
        model_id: Identifier of the model being evaluated.
        judge_id: Identifier of the judge model (if applicable).
        dataset_source: Source of the dataset (HF path, local path, etc.).
        dataset_version: Version or hash of the dataset.
        generation_params: Parameters used for model generation.
        judge_params: Parameters used for judge evaluation.
        started_at: ISO timestamp when the run started.
        completed_at: ISO timestamp when the run completed (None if incomplete).
        git_commit: Git commit hash of the medriskeval codebase.
        git_dirty: Whether there were uncommitted changes.
        total_examples: Total number of examples in the evaluation.
        completed_examples: Number of examples completed.
        output_dir: Directory where outputs are stored.
        extra: Additional metadata.
    """
    run_id: str
    benchmark: str
    model_id: str
    judge_id: str = ""
    dataset_source: str = ""
    dataset_version: str = ""
    generation_params: dict[str, Any] = field(default_factory=dict)
    judge_params: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    total_examples: int = 0
    completed_examples: int = 0
    output_dir: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "model_id": self.model_id,
            "judge_id": self.judge_id,
            "dataset_source": self.dataset_source,
            "dataset_version": self.dataset_version,
            "generation_params": self.generation_params,
            "judge_params": self.judge_params,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "total_examples": self.total_examples,
            "completed_examples": self.completed_examples,
            "output_dir": self.output_dir,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            benchmark=data.get("benchmark", ""),
            model_id=data.get("model_id", ""),
            judge_id=data.get("judge_id", ""),
            dataset_source=data.get("dataset_source", ""),
            dataset_version=data.get("dataset_version", ""),
            generation_params=data.get("generation_params", {}),
            judge_params=data.get("judge_params", {}),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            git_commit=data.get("git_commit", ""),
            git_dirty=data.get("git_dirty", False),
            total_examples=data.get("total_examples", 0),
            completed_examples=data.get("completed_examples", 0),
            output_dir=data.get("output_dir", ""),
            extra=data.get("extra", {}),
        )


def get_git_info() -> dict[str, Any]:
    """Get current git commit information.
    
    Returns:
        Dictionary with 'commit' and 'dirty' fields.
    """
    info = {"commit": "", "dirty": False}
    
    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:12]
        
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())
            
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return info


def generate_run_id(
    benchmark: str,
    model_id: str,
    judge_id: str = "",
    timestamp: str | None = None,
) -> str:
    """Generate a unique run identifier.
    
    Format: {benchmark}_{model_short}_{timestamp}_{hash}
    
    Args:
        benchmark: Benchmark name.
        model_id: Model identifier.
        judge_id: Judge identifier (optional).
        timestamp: Override timestamp (for testing).
        
    Returns:
        Unique run identifier string.
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Short model name (last component of path, truncated)
    model_short = model_id.split("/")[-1][:20]
    
    # Hash for uniqueness
    config_hash = hash_run_config(benchmark, model_id, judge_id)[:6]
    
    return f"{benchmark}_{model_short}_{timestamp}_{config_hash}"


class JSONLWriter:
    """Streaming JSONL writer for run records.
    
    Writes records one at a time to enable resumable runs and
    reduce memory usage for large evaluations.
    
    Example:
        >>> with JSONLWriter("output/records.jsonl") as writer:
        ...     for record in records:
        ...         writer.write(record)
    """
    
    def __init__(
        self,
        path: str | Path,
        mode: str = "a",
        flush_interval: int = 1,
    ) -> None:
        """Initialize the JSONL writer.
        
        Args:
            path: Path to the output file.
            mode: File mode ('w' for overwrite, 'a' for append).
            flush_interval: Flush to disk every N writes.
        """
        self.path = Path(path)
        self.mode = mode
        self.flush_interval = flush_interval
        self._file: Optional[TextIO] = None
        self._write_count = 0
    
    def __enter__(self) -> JSONLWriter:
        self.open()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
    
    def open(self) -> None:
        """Open the file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, self.mode, encoding="utf-8")
    
    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def write(self, record: RunRecord | dict[str, Any]) -> None:
        """Write a single record to the file.
        
        Args:
            record: RunRecord or dictionary to write.
        """
        if self._file is None:
            raise RuntimeError("Writer is not open. Use 'with' statement or call open().")
        
        if isinstance(record, RunRecord):
            data = record.to_dict()
        else:
            data = record
        
        self._file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._write_count += 1
        
        if self._write_count % self.flush_interval == 0:
            self._file.flush()
    
    def write_many(self, records: Sequence[RunRecord | dict[str, Any]]) -> None:
        """Write multiple records."""
        for record in records:
            self.write(record)
    
    @property
    def count(self) -> int:
        """Number of records written."""
        return self._write_count


class JSONLReader:
    """Reader for JSONL run record files.
    
    Supports streaming reads for memory efficiency.
    
    Example:
        >>> reader = JSONLReader("output/records.jsonl")
        >>> for record in reader.iter_records():
        ...     process(record)
    """
    
    def __init__(self, path: str | Path) -> None:
        """Initialize the reader.
        
        Args:
            path: Path to the JSONL file.
        """
        self.path = Path(path)
    
    def iter_lines(self) -> Iterator[dict[str, Any]]:
        """Iterate over raw JSON lines."""
        if not self.path.exists():
            return
        
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    
    def iter_records(self) -> Iterator[RunRecord]:
        """Iterate over RunRecord objects."""
        for data in self.iter_lines():
            try:
                yield RunRecord.from_dict(data)
            except (KeyError, TypeError):
                continue
    
    def load_all(self) -> list[RunRecord]:
        """Load all records into memory."""
        return list(self.iter_records())
    
    def count(self) -> int:
        """Count number of records in file."""
        return sum(1 for _ in self.iter_lines())
    
    def get_completed_ids(self) -> set[str]:
        """Get set of example IDs that have been completed.
        
        Useful for resuming interrupted runs.
        """
        ids = set()
        for data in self.iter_lines():
            example_data = data.get("example", {})
            example_id = example_data.get("id")
            if example_id:
                ids.add(example_id)
        return ids


class ManifestWriter:
    """Writer for run manifest files.
    
    Creates and updates manifest.json files that track run metadata.
    """
    
    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the manifest writer.
        
        Args:
            output_dir: Directory to write the manifest to.
        """
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
    
    def create(
        self,
        benchmark: str,
        model_id: str,
        judge_id: str = "",
        dataset_source: str = "",
        total_examples: int = 0,
        generation_params: dict[str, Any] | None = None,
        judge_params: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RunManifest:
        """Create a new manifest for a run.
        
        Args:
            benchmark: Name of the benchmark.
            model_id: Model identifier.
            judge_id: Judge identifier.
            dataset_source: Dataset source path/URL.
            total_examples: Total number of examples.
            generation_params: Generation parameters.
            judge_params: Judge parameters.
            extra: Additional metadata.
            
        Returns:
            The created RunManifest.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        git_info = get_git_info()
        run_id = generate_run_id(benchmark, model_id, judge_id)
        
        manifest = RunManifest(
            run_id=run_id,
            benchmark=benchmark,
            model_id=model_id,
            judge_id=judge_id,
            dataset_source=dataset_source,
            dataset_version="",  # Could compute hash of dataset
            generation_params=generation_params or {},
            judge_params=judge_params or {},
            started_at=datetime.utcnow().isoformat(),
            completed_at="",
            git_commit=git_info["commit"],
            git_dirty=git_info["dirty"],
            total_examples=total_examples,
            completed_examples=0,
            output_dir=str(self.output_dir),
            extra=extra or {},
        )
        
        self._save(manifest)
        return manifest
    
    def update(
        self,
        completed_examples: int | None = None,
        completed_at: str | None = None,
        **extra_updates: Any,
    ) -> RunManifest | None:
        """Update an existing manifest.
        
        Args:
            completed_examples: Update completed count.
            completed_at: Mark as completed with timestamp.
            **extra_updates: Additional fields to update in extra.
            
        Returns:
            Updated manifest or None if not found.
        """
        manifest = self.load()
        if manifest is None:
            return None
        
        if completed_examples is not None:
            manifest.completed_examples = completed_examples
        
        if completed_at is not None:
            manifest.completed_at = completed_at
        
        if extra_updates:
            manifest.extra.update(extra_updates)
        
        self._save(manifest)
        return manifest
    
    def mark_complete(self) -> RunManifest | None:
        """Mark the run as complete."""
        return self.update(completed_at=datetime.utcnow().isoformat())
    
    def load(self) -> RunManifest | None:
        """Load existing manifest."""
        if not self.manifest_path.exists():
            return None
        
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RunManifest.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None
    
    def _save(self, manifest: RunManifest) -> None:
        """Save manifest to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)


def create_output_dir(
    base_dir: str | Path,
    benchmark: str,
    model_id: str,
    timestamp: str | None = None,
) -> Path:
    """Create a structured output directory for a run.
    
    Creates directory structure:
        base_dir/
            benchmark/
                model_short_timestamp/
                    manifest.json
                    records.jsonl
                    metrics.json
    
    Args:
        base_dir: Base output directory.
        benchmark: Benchmark name.
        model_id: Model identifier.
        timestamp: Override timestamp.
        
    Returns:
        Path to the created run directory.
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    model_short = model_id.split("/")[-1][:30]
    run_dir = Path(base_dir) / benchmark / f"{model_short}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def find_latest_run(base_dir: str | Path, benchmark: str) -> Path | None:
    """Find the most recent run directory for a benchmark.
    
    Args:
        base_dir: Base output directory.
        benchmark: Benchmark name.
        
    Returns:
        Path to the latest run directory, or None if not found.
    """
    benchmark_dir = Path(base_dir) / benchmark
    if not benchmark_dir.exists():
        return None
    
    run_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Sort by modification time
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs[0]


def load_run_records(output_dir: str | Path) -> list[RunRecord]:
    """Load all run records from an output directory.
    
    Args:
        output_dir: Path to run output directory.
        
    Returns:
        List of RunRecord objects.
    """
    records_path = Path(output_dir) / "records.jsonl"
    if not records_path.exists():
        return []
    
    reader = JSONLReader(records_path)
    return reader.load_all()
