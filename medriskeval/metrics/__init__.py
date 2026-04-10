"""Metrics computation for medriskeval.

This module provides metric computers and utilities for computing
benchmark-level and category-level evaluation metrics.

Available Metrics:
- SafetyMetricComputer: PSB/MSB 1-5 score histogram, safe-rate
- RefusalMetricComputer: XSTest/JBB compliance/partial/refusal rates
- GroundednessMetricComputer: FACTS-med sentence label proportions

Usage:
    from medriskeval.metrics import SafetyMetricComputer, compute_psb_msb_metrics

    # Using computer class
    computer = SafetyMetricComputer()
    result = computer.compute_full(records, model_id="gpt-4")

    # Using convenience function
    metrics = compute_psb_msb_metrics(records, model_id="gpt-4")
"""

from medriskeval.metrics.base import (
    MetricComputer,
    MetricResult,
    BenchmarkMetrics,
    group_by_category,
    group_by_benchmark,
    extract_score,
    extract_label,
    count_labels,
    count_scores,
)

from medriskeval.metrics.safety import (
    SafetyMetricComputer,
    SafetyMetrics,
    ScoreHistogram,
    compute_score_histogram,
    compute_safe_rate,
    compute_unsafe_rate,
    compute_mean_score,
    compute_safety_metrics,
    compute_psb_msb_metrics,
    SAFE_SCORES,
    UNSAFE_SCORES,
    SCORE_LABELS,
)

from medriskeval.metrics.refusal import (
    RefusalMetricComputer,
    RefusalDistribution,
    RefusalMetricsBySplit,
    compute_refusal_distribution,
    compute_refusal_by_split,
    compute_jbb_metrics,
    compute_xstest_metrics,
    compute_attack_success_rate,
    compute_over_refusal_rate,
    normalize_label,
    REFUSAL_LABELS,
)

from medriskeval.metrics.groundedness import (
    GroundednessMetricComputer,
    GroundednessMetrics,
    SentenceLabelCounts,
    count_sentence_labels,
    compute_groundedness_metrics,
    compute_facts_med_metrics,
    parse_sentence_labels,
    extract_groundedness_labels,
    GROUNDEDNESS_LABELS,
)


__all__ = [
    # Base
    "MetricComputer",
    "MetricResult",
    "BenchmarkMetrics",
    "group_by_category",
    "group_by_benchmark",
    "extract_score",
    "extract_label",
    "count_labels",
    "count_scores",
    # Safety (PSB/MSB)
    "SafetyMetricComputer",
    "SafetyMetrics",
    "ScoreHistogram",
    "compute_score_histogram",
    "compute_safe_rate",
    "compute_unsafe_rate",
    "compute_mean_score",
    "compute_safety_metrics",
    "compute_psb_msb_metrics",
    "SAFE_SCORES",
    "UNSAFE_SCORES",
    "SCORE_LABELS",
    # Refusal (XSTest/JBB)
    "RefusalMetricComputer",
    "RefusalDistribution",
    "RefusalMetricsBySplit",
    "compute_refusal_distribution",
    "compute_refusal_by_split",
    "compute_jbb_metrics",
    "compute_xstest_metrics",
    "compute_attack_success_rate",
    "compute_over_refusal_rate",
    "normalize_label",
    "REFUSAL_LABELS",
    # Groundedness (FACTS-med)
    "GroundednessMetricComputer",
    "GroundednessMetrics",
    "SentenceLabelCounts",
    "count_sentence_labels",
    "compute_groundedness_metrics",
    "compute_facts_med_metrics",
    "parse_sentence_labels",
    "extract_groundedness_labels",
    "GROUNDEDNESS_LABELS"
]
