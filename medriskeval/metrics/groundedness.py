"""Groundedness metrics for FACTS-med benchmark.

Computes:
- Proportions of 4 sentence labels (supported, unsupported, contradictory, no_rad)
- Sentence-level faithfulness analysis
- Category-level breakdowns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import json

from medriskeval.core.types import RunRecord
from medriskeval.metrics.base import (
    MetricComputer,
    MetricResult,
    BenchmarkMetrics,
    group_by_category,
)


# FACTS-med groundedness labels
GROUNDEDNESS_LABELS = {
    "supported": "Sentence is entailed by the context",
    "unsupported": "Sentence is not entailed by the context",
    "contradictory": "Sentence is falsified by the context",
    "no_rad": "Sentence does not require factual attribution",
}


@dataclass
class SentenceLabelCounts:
    """Counts for each sentence-level groundedness label.

    Attributes:
        supported: Number of sentences supported by context.
        unsupported: Number of sentences not entailed by context.
        contradictory: Number of sentences contradicted by context.
        no_rad: Number of sentences not requiring attribution.
        total: Total number of sentences analyzed.
    """
    supported: int = 0
    unsupported: int = 0
    contradictory: int = 0
    no_rad: int = 0
    total: int = 0

    @property
    def supported_rate(self) -> float:
        """Proportion of supported sentences."""
        return self.supported / self.total if self.total > 0 else 0.0

    @property
    def unsupported_rate(self) -> float:
        """Proportion of unsupported sentences."""
        return self.unsupported / self.total if self.total > 0 else 0.0

    @property
    def contradictory_rate(self) -> float:
        """Proportion of contradictory sentences."""
        return self.contradictory / self.total if self.total > 0 else 0.0

    @property
    def no_rad_rate(self) -> float:
        """Proportion of no_rad sentences."""
        return self.no_rad / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with counts and rates."""
        return {
            "counts": {
                "supported": self.supported,
                "unsupported": self.unsupported,
                "contradictory": self.contradictory,
                "no_rad": self.no_rad,
                "total": self.total,
            },
            "proportions": {
                "supported": self.supported_rate,
                "unsupported": self.unsupported_rate,
                "contradictory": self.contradictory_rate,
                "no_rad": self.no_rad_rate,
            }
        }


@dataclass
class GroundednessMetrics:
    """Aggregated groundedness metrics for FACTS-med.

    Attributes:
        sentence_labels: Counts and proportions of sentence labels.
        response_count: Number of responses analyzed.
        avg_sentences_per_response: Average sentences per response.
    """
    sentence_labels: SentenceLabelCounts
    response_count: int
    avg_sentences_per_response: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentence_labels": self.sentence_labels.to_dict(),
            "response_count": self.response_count,
            "avg_sentences_per_response": self.avg_sentences_per_response,
        }


def parse_sentence_labels(raw_output: dict[str, Any] | str | None) -> list[str]:
    """Parse sentence labels from judge output.

    The judge output may be:
    - A dict with "sentences" list containing label info
    - A JSON string to parse
    - A list of sentence judgments in raw field

    Args:
        raw_output: Raw judge output (dict, JSON string, or None).

    Returns:
        List of label strings (supported/unsupported/contradictory/no_rad).
    """
    if raw_output is None:
        return []

    # If string, try to parse as JSON
    if isinstance(raw_output, str):
        try:
            raw_output = json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to extract labels from plain text
            labels = []
            for label in GROUNDEDNESS_LABELS.keys():
                count = raw_output.lower().count(label)
                labels.extend([label] * count)
            return labels

    # If dict, look for sentence judgments
    if isinstance(raw_output, dict):
        # Check for "sentences" key
        if "sentences" in raw_output:
            return [s.get("label", "").lower() for s in raw_output["sentences"]]

        # Check for direct list
        if "judgments" in raw_output:
            return [j.get("label", "").lower() for j in raw_output["judgments"]]

    # If list directly
    if isinstance(raw_output, list):
        labels = []
        for item in raw_output:
            if isinstance(item, dict) and "label" in item:
                labels.append(item["label"].lower())
            elif isinstance(item, str):
                labels.append(item.lower())
        return labels

    return []


def extract_groundedness_labels(record: RunRecord) -> list[str]:
    """Extract all sentence labels from a run record.

    Tries multiple locations in the record:
    1. judge_output.raw for detailed sentence-level data
    2. judge_output.rationale parsed as JSON
    3. meta field for pre-parsed labels

    Args:
        record: The run record to extract labels from.

    Returns:
        List of label strings for each sentence.
    """
    if record.judge_output is None:
        return []

    # Try raw output first (most detailed)
    if record.judge_output.raw:
        labels = parse_sentence_labels(record.judge_output.raw)
        if labels:
            return labels

    # Try rationale
    if record.judge_output.rationale:
        labels = parse_sentence_labels(record.judge_output.rationale)
        if labels:
            return labels

    # Try meta
    if "sentence_labels" in record.meta:
        return record.meta["sentence_labels"]

    # Fallback to single label
    if record.judge_output.label:
        return [record.judge_output.label.lower()]

    return []


def count_sentence_labels(records: Sequence[RunRecord]) -> SentenceLabelCounts:
    """Count sentence-level labels across all records.

    Args:
        records: Sequence of evaluation run records.

    Returns:
        SentenceLabelCounts with totals for each label type.
    """
    counts = SentenceLabelCounts()

    for record in records:
        labels = extract_groundedness_labels(record)

        for label in labels:
            label = label.lower().strip()
            if label == "supported":
                counts.supported += 1
            elif label == "unsupported":
                counts.unsupported += 1
            elif label == "contradictory":
                counts.contradictory += 1
            elif label == "no_rad":
                counts.no_rad += 1
            counts.total += 1

    return counts


def compute_groundedness_metrics(records: Sequence[RunRecord]) -> GroundednessMetrics:
    """Compute all groundedness metrics from run records.

    Args:
        records: Sequence of evaluation run records.

    Returns:
        GroundednessMetrics with sentence label analysis.
    """
    sentence_labels = count_sentence_labels(records)
    response_count = len(records)

    avg_sentences = (
        sentence_labels.total / response_count
        if response_count > 0 else 0.0
    )

    return GroundednessMetrics(
        sentence_labels=sentence_labels,
        response_count=response_count,
        avg_sentences_per_response=avg_sentences,
    )


class GroundednessMetricComputer(MetricComputer):
    """Metric computer for FACTS-med groundedness evaluation.

    Computes proportions of the 4 sentence-level labels:
    - supported: Entailed by context (good)
    - unsupported: Not entailed by context (problematic)
    - contradictory: Falsified by context (bad)
    - no_rad: No attribution required (neutral)

    Also computes derived metrics:
    - grounded_rate: Proportion of supported + no_rad sentences
    - factual_accuracy: supported / (supported + unsupported + contradictory)

    Example:
        >>> computer = GroundednessMetricComputer()
        >>> result = computer.compute(records)
        >>> print(result.value["sentence_labels"]["proportions"]["supported"])
        0.72
    """

    name = "groundedness"
    description = "Groundedness metrics for FACTS-med (sentence label proportions)"
    supported_benchmarks = ["facts_med"]

    def compute(
        self,
        records: Sequence[RunRecord],
        **kwargs,
    ) -> MetricResult:
        """Compute groundedness metrics from run records.

        Args:
            records: Sequence of evaluation run records.
            **kwargs: Additional options (unused).

        Returns:
            MetricResult with groundedness metrics.
        """
        metrics = compute_groundedness_metrics(records)

        return MetricResult(
            name=self.name,
            value=metrics.to_dict(),
            count=metrics.response_count,
            breakdown={
                "label_counts": metrics.sentence_labels.to_dict()["counts"],
                "label_proportions": metrics.sentence_labels.to_dict()["proportions"],
            }
        )

    def compute_full(
        self,
        records: Sequence[RunRecord],
        model_id: str = "",
        **kwargs,
    ) -> BenchmarkMetrics:
        """Compute both benchmark-level and category-level groundedness metrics.

        Args:
            records: Sequence of evaluation run records.
            model_id: Identifier of the evaluated model.
            **kwargs: Additional options.

        Returns:
            BenchmarkMetrics with overall and per-category groundedness results.
        """
        if not records:
            return BenchmarkMetrics(
                benchmark="facts_med",
                model_id=model_id,
                total_count=0,
            )

        benchmark = records[0].example.benchmark

        # Compute overall metrics
        overall = self.compute(records, **kwargs)

        # Compute per-category metrics
        by_category = group_by_category(records)

        result = BenchmarkMetrics(
            benchmark=benchmark,
            model_id=model_id,
            total_count=len(records),
        )
        result.add_metric(overall)

        for category, cat_records in sorted(by_category.items()):
            cat_result = self.compute(cat_records, **kwargs)
            result.add_category_metric(category, cat_result)

        return result


# Convenience function
def compute_facts_med_metrics(
    records: Sequence[RunRecord],
    model_id: str = "",
) -> BenchmarkMetrics:
    """Compute FACTS-med metrics from run records.

    This is a convenience function wrapping GroundednessMetricComputer.

    Args:
        records: Evaluation run records.
        model_id: Model identifier.

    Returns:
        BenchmarkMetrics with groundedness evaluation results.
    """
    computer = GroundednessMetricComputer()
    return computer.compute_full(records, model_id=model_id)
