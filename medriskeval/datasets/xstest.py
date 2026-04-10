"""XSTest dataset adapter.

XSTest is a test suite for identifying exaggerated safety behaviors
in language models. It contains both safe and unsafe prompts designed
to test whether models can distinguish between genuinely harmful content
and benign content that might superficially appear harmful.

Source: walledai/XSTest on Hugging Face Hub
"""

from __future__ import annotations

from typing import Iterator, Sequence

from medriskeval.core.types import Example
from medriskeval.core.registry import DatasetRegistry
from medriskeval.datasets.base import BenchmarkDataset
from medriskeval.datasets.io import load_hf_dataset


# XSTest type categories
XSTEST_TYPES = [
    "homonyms",
    "figurative_language",
    "safe_targets",
    "safe_contexts",
    "definitions",
    "real_discrimination",
    "historical_events",
    "public_figures",
    "fictional_characters",
    "privacy",
]


@DatasetRegistry.register("xstest")
class XSTest(BenchmarkDataset):
    """XSTest dataset adapter.
    
    XSTest evaluates exaggerated safety behaviors ("over-refusal") where
    models refuse to answer benign queries that superficially resemble
    harmful ones, while also testing responses to genuinely harmful content.
    
    Example usage:
        >>> dataset = XSTest()
        >>> dataset.load()
        >>> for example in dataset.iter_examples("test"):
        ...     print(example.meta["label"], example.input)
    """
    
    name = "xstest"
    description = "XSTest: Evaluating exaggerated safety behaviors in LLMs"
    
    HF_DATASET_ID = "walledai/XSTest"
    
    def __init__(
        self,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the XSTest dataset adapter.
        
        Args:
            cache_dir: Optional cache directory for HF datasets.
            **kwargs: Additional configuration.
        """
        super().__init__(cache_dir=cache_dir, **kwargs)
        self._data = None
    
    def load(self) -> None:
        """Load the XSTest dataset from Hugging Face."""
        self._data = load_hf_dataset(
            self.HF_DATASET_ID,
            cache_dir=self.cache_dir,
        )
        self._loaded = True
    
    def splits(self) -> Sequence[str]:
        """Return available splits."""
        if self._data is None:
            return ["test"]
        return list(self._data.keys())
    
    def iter_examples(self, split: str = "test") -> Iterator[Example]:
        """Iterate over XSTest examples.
        
        Args:
            split: The split to iterate.
            
        Yields:
            Example objects with XSTest data.
            
        Note:
            The meta field contains:
            - label: "safe" or "unsafe" indicating ground truth
            - type: The specific test type (e.g., "homonyms", "figurative_language")
            - note: Additional context about the example
        """
        self._ensure_loaded()
        self._validate_split(split)
        
        for idx, item in enumerate(self._data[split]):
            # XSTest format: {'prompt': str, 'label': str, 'type': str, ...}
            prompt = item.get("prompt", "")
            label = item.get("label", "unknown")
            test_type = item.get("type", "unknown")
            
            # Use type as category for aggregation
            category = test_type
            
            # Build metadata
            meta = {
                "label": label,  # "safe" or "unsafe"
                "type": test_type,
            }
            # Add any additional fields
            for k, v in item.items():
                if k not in ("prompt", "label", "type"):
                    meta[k] = v
            
            yield Example(
                id=f"xstest_{split}_{idx}",
                benchmark=self.name,
                category=category,
                input=prompt,
                meta=meta,
            )
    
    def iter_safe_examples(self, split: str = "test") -> Iterator[Example]:
        """Iterate over safe (should not refuse) examples.
        
        Args:
            split: The split to iterate.
            
        Yields:
            Example objects labeled as "safe".
        """
        for example in self.iter_examples(split):
            if example.meta.get("label") == "safe":
                yield example
    
    def iter_unsafe_examples(self, split: str = "test") -> Iterator[Example]:
        """Iterate over unsafe (should refuse) examples.
        
        Args:
            split: The split to iterate.
            
        Yields:
            Example objects labeled as "unsafe".
        """
        for example in self.iter_examples(split):
            if example.meta.get("label") == "unsafe":
                yield example
    
    def iter_by_type(self, test_type: str, split: str = "test") -> Iterator[Example]:
        """Iterate over examples of a specific test type.
        
        Args:
            test_type: The test type to filter by.
            split: The split to iterate.
            
        Yields:
            Example objects matching the test type.
        """
        for example in self.iter_examples(split):
            if example.category == test_type:
                yield example
    
    def get_types(self) -> list[str]:
        """Return the list of test types in XSTest."""
        return list(XSTEST_TYPES)
    
    def label_counts(self, split: str = "test") -> dict[str, int]:
        """Count examples per label (safe/unsafe).
        
        Args:
            split: The split to count.
            
        Returns:
            Dictionary mapping labels to counts.
        """
        self._ensure_loaded()
        counts: dict[str, int] = {}
        for example in self.iter_examples(split):
            label = example.meta.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def type_counts(self, split: str = "test") -> dict[str, int]:
        """Count examples per test type.
        
        Args:
            split: The split to count.
            
        Returns:
            Dictionary mapping types to counts.
        """
        self._ensure_loaded()
        counts: dict[str, int] = {}
        for example in self.iter_examples(split):
            counts[example.category] = counts.get(example.category, 0) + 1
        return counts
