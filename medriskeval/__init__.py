"""MedRiskEval: Medical AI Risk Evaluation Framework.

A framework for evaluating safety, harmfulness, and groundedness
of medical AI systems.

Quick Start:
    from medriskeval import load_dataset
    
    # Load and iterate a benchmark
    dataset = load_dataset("psb")
    for example in dataset.iter_examples("test"):
        print(example.id, example.category, example.input)

Available Datasets:
    - psb: PatientSafetyBench (466 patient safety queries)
    - msb: MedSafetyBench (450 unethical clinician queries)
    - xstest: XSTest (exaggerated safety behavior)
    - jbb: JailbreakBench (jailbreak resistance)
    - facts_med: FACTS Medical (groundedness)

Core Types:
    - Example: Benchmark input with {id, benchmark, category, input, meta}
    - ChatMessage: Normalized {role, content, name?}
    - ModelOutput: Model response {text, messages?, usage?, raw?}
    - JudgeOutput: Judgment {label, score?, rationale?, raw?}
    - RunRecord: Complete evaluation record
"""

__version__ = "0.1.0"

# Core types and utilities
from medriskeval.core import (
    # Types
    Example,
    ChatMessage,
    ModelOutput,
    JudgeOutput,
    RunRecord,
    UsageStats,
    Role,
    # Registries
    DatasetRegistry,
    JudgeRegistry,
    ModelRegistry,
    TaskRegistry,
    MetricRegistry,
    # Hashing
    stable_hash,
    hash_example,
    hash_prompt,
)

# Dataset utilities
from medriskeval.datasets import (
    BenchmarkDataset,
    load_dataset,
    list_datasets,
    PatientSafetyBench,
    MedSafetyBench,
    XSTest,
    JailbreakBench,
    FACTSMedical,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "Example",
    "ChatMessage",
    "ModelOutput",
    "JudgeOutput",
    "RunRecord",
    "UsageStats",
    "Role",
    # Registries
    "DatasetRegistry",
    "JudgeRegistry",
    "ModelRegistry",
    "TaskRegistry",
    "MetricRegistry",
    # Hashing
    "stable_hash",
    "hash_example",
    "hash_prompt",
    # Datasets
    "BenchmarkDataset",
    "load_dataset",
    "list_datasets",
    "PatientSafetyBench",
    "MedSafetyBench",
    "XSTest",
    "JailbreakBench",
    "FACTSMedical",
]
