"""PatientSafetyBench (PSB) dataset adapter.

PatientSafetyBench contains 466 short patient queries across 5 risk categories,
designed to evaluate medical AI safety for patient-facing interactions.

Source: microsoft/PatientSafetyBench on Hugging Face Hub
"""

from __future__ import annotations

from typing import Iterator, Sequence

from medriskeval.core.types import Example
from medriskeval.core.registry import DatasetRegistry
from medriskeval.datasets.base import BenchmarkDataset
from medriskeval.datasets.io import load_hf_dataset


# PSB risk categories
PSB_CATEGORIES = [
    "emergency_triage",
    "medication_safety", 
    "diagnosis_advice",
    "treatment_recommendation",
    "mental_health_crisis",
]


@DatasetRegistry.register("psb")
class PatientSafetyBench(BenchmarkDataset):
    """PatientSafetyBench dataset adapter.
    
    466 short patient queries across 5 risk categories designed to evaluate
    how medical AI systems handle potentially dangerous patient queries.
    
    Categories:
        - emergency_triage: Emergency situations requiring immediate care
        - medication_safety: Drug interactions, dosing, contraindications
        - diagnosis_advice: Self-diagnosis requests
        - treatment_recommendation: Treatment suggestions without context
        - mental_health_crisis: Mental health emergencies
    
    Example usage:
        >>> dataset = PatientSafetyBench()
        >>> dataset.load()
        >>> for example in dataset.iter_examples("test"):
        ...     print(example.category, example.input)
    """
    
    name = "psb"
    description = "PatientSafetyBench: 466 patient safety queries across 5 risk categories"
    
    HF_DATASET_ID = "microsoft/PatientSafetyBench"
    
    def __init__(
        self,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the PSB dataset adapter.
        
        Args:
            cache_dir: Optional cache directory for HF datasets.
            **kwargs: Additional configuration.
        """
        super().__init__(cache_dir=cache_dir, **kwargs)
        self._data = None
    
    def load(self) -> None:
        """Load the PatientSafetyBench dataset from Hugging Face."""
        self._data = load_hf_dataset(
            self.HF_DATASET_ID,
            split="train",  # PSB only has a train split
            cache_dir=self.cache_dir,
        )
        self._loaded = True
    
    def splits(self) -> Sequence[str]:
        """Return available splits.
        
        Note: PSB only has a single split which we expose as "test"
        since it's intended for evaluation, not training.
        """
        return ["test"]
    
    def iter_examples(self, split: str = "test") -> Iterator[Example]:
        """Iterate over PSB examples.
        
        Args:
            split: The split to iterate (only "test" is valid).
            
        Yields:
            Example objects with PSB data.
            
        Raises:
            ValueError: If split is not "test".
            DatasetNotLoadedError: If load() hasn't been called.
        """
        self._ensure_loaded()
        self._validate_split(split)
        
        for idx, item in enumerate(self._data):
            # Extract category from the data
            # PSB format: {'content': str, 'category': str, ...}
            category = item.get("category", "unknown")
            content = item.get("content", "")
            
            # Build metadata from remaining fields
            meta = {
                k: v for k, v in item.items() 
                if k not in ("content", "category")
            }
            
            yield Example(
                id=f"psb_{idx}",
                benchmark=self.name,
                category=category,
                input=content,
                meta=meta,
            )
    
    def iter_by_category(
        self, 
        category: str,
        split: str = "test",
    ) -> Iterator[Example]:
        """Iterate over examples in a specific category.
        
        Args:
            category: The risk category to filter by.
            split: The split to iterate.
            
        Yields:
            Example objects matching the category.
        """
        for example in self.iter_examples(split):
            if example.category == category:
                yield example
    
    def get_categories(self) -> list[str]:
        """Return the list of risk categories in PSB."""
        return list(PSB_CATEGORIES)
    
    def category_counts(self, split: str = "test") -> dict[str, int]:
        """Count examples per category.
        
        Args:
            split: The split to count.
            
        Returns:
            Dictionary mapping category names to counts.
        """
        self._ensure_loaded()
        counts: dict[str, int] = {}
        for example in self.iter_examples(split):
            counts[example.category] = counts.get(example.category, 0) + 1
        return counts
