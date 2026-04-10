"""Base classes for metrics computation in medriskeval.

This module provides abstract base classes and common utilities for computing
benchmark-level and category-level metrics from evaluation run records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, Sequence
from collections import defaultdict

from medriskeval.core.types import RunRecord, JudgeOutput


T = TypeVar("T")


@dataclass
class MetricResult:
    """Container for computed metric results.
    
    Attributes:
        name: Name of the metric.
        value: Primary metric value (can be float, dict, etc.).
        count: Number of samples used to compute the metric.
        breakdown: Optional breakdown by category or other dimension.
        meta: Additional metadata about the computation.
    """
    name: str
    value: Any
    count: int
    breakdown: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "count": self.count,
            "breakdown": self.breakdown,
            "meta": self.meta,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            count=data.get("count", 0),
            breakdown=data.get("breakdown", {}),
            meta=data.get("meta", {}),
        )


@dataclass
class BenchmarkMetrics:
    """Collection of metrics for a benchmark evaluation.
    
    Attributes:
        benchmark: Name of the benchmark (psb, msb, jbb, xstest, facts_med).
        model_id: Identifier of the evaluated model.
        total_count: Total number of examples evaluated.
        metrics: Dictionary of metric name to MetricResult.
        category_metrics: Metrics broken down by category.
    """
    benchmark: str
    model_id: str
    total_count: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    category_metrics: dict[str, dict[str, MetricResult]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "benchmark": self.benchmark,
            "model_id": self.model_id,
            "total_count": self.total_count,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "category_metrics": {
                cat: {k: v.to_dict() for k, v in metrics.items()}
                for cat, metrics in self.category_metrics.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkMetrics:
        """Create from dictionary."""
        metrics = {
            k: MetricResult.from_dict(v) 
            for k, v in data.get("metrics", {}).items()
        }
        category_metrics = {
            cat: {k: MetricResult.from_dict(v) for k, v in cat_metrics.items()}
            for cat, cat_metrics in data.get("category_metrics", {}).items()
        }
        return cls(
            benchmark=data["benchmark"],
            model_id=data.get("model_id", ""),
            total_count=data.get("total_count", 0),
            metrics=metrics,
            category_metrics=category_metrics,
        )
    
    def add_metric(self, result: MetricResult) -> None:
        """Add a benchmark-level metric."""
        self.metrics[result.name] = result
    
    def add_category_metric(self, category: str, result: MetricResult) -> None:
        """Add a category-level metric."""
        if category not in self.category_metrics:
            self.category_metrics[category] = {}
        self.category_metrics[category][result.name] = result


class MetricComputer(ABC):
    """Abstract base class for metric computation.
    
    Subclasses implement specific metrics for different benchmarks
    (safety scores, refusal rates, groundedness proportions, etc.).
    """
    
    name: str = ""
    description: str = ""
    supported_benchmarks: list[str] = []
    
    @abstractmethod
    def compute(
        self, 
        records: Sequence[RunRecord],
        **kwargs,
    ) -> MetricResult:
        """Compute metrics from run records.
        
        Args:
            records: Sequence of evaluation run records.
            **kwargs: Additional computation options.
            
        Returns:
            MetricResult containing computed metrics.
        """
        pass
    
    def compute_by_category(
        self,
        records: Sequence[RunRecord],
        **kwargs,
    ) -> dict[str, MetricResult]:
        """Compute metrics broken down by category.
        
        Args:
            records: Sequence of evaluation run records.
            **kwargs: Additional computation options.
            
        Returns:
            Dictionary mapping category names to MetricResults.
        """
        # Group records by category
        by_category: dict[str, list[RunRecord]] = defaultdict(list)
        for record in records:
            category = record.example.category
            by_category[category].append(record)
        
        # Compute metrics for each category
        results = {}
        for category, cat_records in sorted(by_category.items()):
            results[category] = self.compute(cat_records, **kwargs)
        
        return results
    
    def compute_full(
        self,
        records: Sequence[RunRecord],
        model_id: str = "",
        **kwargs,
    ) -> BenchmarkMetrics:
        """Compute both benchmark-level and category-level metrics.
        
        Args:
            records: Sequence of evaluation run records.
            model_id: Identifier of the evaluated model.
            **kwargs: Additional computation options.
            
        Returns:
            BenchmarkMetrics with both overall and per-category results.
        """
        if not records:
            benchmark = self.supported_benchmarks[0] if self.supported_benchmarks else "unknown"
            return BenchmarkMetrics(
                benchmark=benchmark,
                model_id=model_id,
                total_count=0,
            )
        
        benchmark = records[0].example.benchmark
        
        # Compute overall metrics
        overall = self.compute(records, **kwargs)
        
        # Compute per-category metrics
        category_results = self.compute_by_category(records, **kwargs)
        
        result = BenchmarkMetrics(
            benchmark=benchmark,
            model_id=model_id,
            total_count=len(records),
        )
        result.add_metric(overall)
        
        for category, metric in category_results.items():
            result.add_category_metric(category, metric)
        
        return result


def group_by_category(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    """Group run records by their example category.
    
    Args:
        records: Sequence of run records to group.
        
    Returns:
        Dictionary mapping category names to lists of records.
    """
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.example.category].append(record)
    return dict(grouped)


def group_by_benchmark(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    """Group run records by their benchmark name.
    
    Args:
        records: Sequence of run records to group.
        
    Returns:
        Dictionary mapping benchmark names to lists of records.
    """
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.example.benchmark].append(record)
    return dict(grouped)


def extract_score(record: RunRecord) -> float | None:
    """Extract numeric score from a run record's judge output.
    
    Args:
        record: The run record to extract score from.
        
    Returns:
        The numeric score, or None if not available.
    """
    if record.judge_output is None:
        return None
    return record.judge_output.score


def extract_label(record: RunRecord) -> str | None:
    """Extract label from a run record's judge output.
    
    Args:
        record: The run record to extract label from.
        
    Returns:
        The label string, or None if not available.
    """
    if record.judge_output is None:
        return None
    return record.judge_output.label


def count_labels(records: Sequence[RunRecord]) -> dict[str, int]:
    """Count occurrences of each label in run records.
    
    Args:
        records: Sequence of run records.
        
    Returns:
        Dictionary mapping label strings to counts.
    """
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        label = extract_label(record)
        if label is not None:
            counts[label] += 1
    return dict(counts)


def count_scores(records: Sequence[RunRecord]) -> dict[int, int]:
    """Count occurrences of each integer score in run records.
    
    Args:
        records: Sequence of run records.
        
    Returns:
        Dictionary mapping integer scores to counts.
    """
    counts: dict[int, int] = defaultdict(int)
    for record in records:
        score = extract_score(record)
        if score is not None:
            counts[int(score)] += 1
    return dict(counts)
