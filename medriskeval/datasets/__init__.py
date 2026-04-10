"""Dataset adapters for medriskeval.

This module provides adapters for loading various medical AI safety benchmarks
with a consistent interface. All datasets output Example objects.

Available datasets:
    - psb: PatientSafetyBench (466 patient safety queries)
    - msb: MedSafetyBench (450 unethical clinician queries)
    - xstest: XSTest (exaggerated safety behavior evaluation)
    - jbb: JailbreakBench (jailbreak resistance evaluation)
    - facts_med: FACTS Medical (groundedness evaluation)

Usage:
    from medriskeval.datasets import PatientSafetyBench
    
    dataset = PatientSafetyBench()
    dataset.load()
    for example in dataset.iter_examples("test"):
        print(example.id, example.category, example.input)

Registry-based loading:
    from medriskeval.core import DatasetRegistry
    
    dataset = DatasetRegistry.create("psb")
    dataset.load()
"""

from medriskeval.datasets.base import (
    BenchmarkDataset,
    DatasetError,
    DatasetLoadError,
    DatasetNotLoadedError,
)

from medriskeval.datasets.io import (
    get_cache_dir,
    load_hf_dataset,
    load_jsonl,
    save_jsonl,
    iter_jsonl,
    load_csv,
    load_csv_simple,
    DataCache,
)

from medriskeval.datasets.psb import PatientSafetyBench, PSB_CATEGORIES
from medriskeval.datasets.msb import MedSafetyBench, MSB_CATEGORIES
from medriskeval.datasets.xstest import XSTest, XSTEST_TYPES
from medriskeval.datasets.jbb import JailbreakBench, JBB_CATEGORIES
from medriskeval.datasets.facts_med import FACTSMedical


__all__ = [
    # Base classes
    "BenchmarkDataset",
    "DatasetError",
    "DatasetLoadError",
    "DatasetNotLoadedError",
    # I/O utilities
    "get_cache_dir",
    "load_hf_dataset",
    "load_jsonl",
    "save_jsonl",
    "iter_jsonl",
    "load_csv",
    "load_csv_simple",
    "DataCache",
    # Dataset adapters
    "PatientSafetyBench",
    "MedSafetyBench",
    "XSTest",
    "JailbreakBench",
    "FACTSMedical",
    # Category constants
    "PSB_CATEGORIES",
    "MSB_CATEGORIES",
    "XSTEST_TYPES",
    "JBB_CATEGORIES",
    "list_datasets",
    "load_dataset"
]


def list_datasets() -> list[str]:
    """Return a list of all available dataset names."""
    from medriskeval.core import DatasetRegistry
    return DatasetRegistry.list_names()


def load_dataset(name: str, **kwargs) -> BenchmarkDataset:
    """Load a dataset by name.

    Args:
        name: Dataset name (e.g., "psb", "msb", "xstest", "jbb", "facts_med").
        **kwargs: Arguments passed to the dataset constructor.

    Returns:
        A loaded BenchmarkDataset instance.

    Example:
        >>> dataset = load_dataset("psb")
        >>> for ex in dataset.iter_examples("test"):
        ...     print(ex.input)
    """
    from medriskeval.core import DatasetRegistry

    dataset = DatasetRegistry.create(name, **kwargs)
    dataset.load()
    return dataset
