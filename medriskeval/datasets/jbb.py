"""JailbreakBench (JBB) dataset adapter.

JailbreakBench provides a standardized benchmark for evaluating LLM
resistance to jailbreak attacks. It contains both harmful and benign
behaviors to test model safety boundaries.

Source: JailbreakBench/JBB-Behaviors on Hugging Face Hub
"""

from __future__ import annotations

from typing import Iterator, Literal, Sequence

from medriskeval.core.types import Example
from medriskeval.core.registry import DatasetRegistry
from medriskeval.datasets.base import BenchmarkDataset
from medriskeval.datasets.io import load_hf_dataset


# JBB behavior categories
JBB_CATEGORIES = [
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making",
]


@DatasetRegistry.register("jbb")
class JailbreakBench(BenchmarkDataset):
    """JailbreakBench dataset adapter.
    
    JBB-Behaviors provides harmful and benign goals to test whether
    models will comply with potentially dangerous requests.
    
    Splits:
        - harmful: Requests that models should refuse
        - benign: Similar-sounding but safe requests
    
    Example usage:
        >>> dataset = JailbreakBench()
        >>> dataset.load()
        >>> for example in dataset.iter_examples("harmful"):
        ...     print(example.category, example.input)
    """
    
    name = "jbb"
    description = "JailbreakBench: Evaluating LLM jailbreak resistance"
    
    HF_DATASET_ID = "JailbreakBench/JBB-Behaviors"
    HF_CONFIG = "behaviors"
    
    def __init__(
        self,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the JBB dataset adapter.
        
        Args:
            cache_dir: Optional cache directory for HF datasets.
            **kwargs: Additional configuration.
        """
        super().__init__(cache_dir=cache_dir, **kwargs)
        self._data = None
    
    def load(self) -> None:
        """Load the JailbreakBench dataset from Hugging Face."""
        self._data = load_hf_dataset(
            self.HF_DATASET_ID,
            name=self.HF_CONFIG,
            cache_dir=self.cache_dir,
        )
        self._loaded = True
    
    def splits(self) -> Sequence[str]:
        """Return available splits.
        
        Returns:
            ["harmful", "benign"] - the two behavior types.
        """
        if self._data is None:
            return ["harmful", "benign"]
        return list(self._data.keys())
    
    def iter_examples(self, split: str = "harmful") -> Iterator[Example]:
        """Iterate over JBB examples.
        
        Args:
            split: The split to iterate ("harmful" or "benign").
            
        Yields:
            Example objects with JBB data.
            
        Note:
            The input is the "Goal" field from JBB, representing
            what the user is trying to get the model to do.
        """
        self._ensure_loaded()
        self._validate_split(split)
        
        for idx, item in enumerate(self._data[split]):
            # JBB format: {'Goal': str, 'Target': str, 'Category': str, ...}
            goal = item.get("Goal", "")
            category = item.get("Category", "unknown")
            target = item.get("Target", "")
            
            # Build metadata
            meta = {
                "split_type": split,  # "harmful" or "benign"
                "target": target,  # Expected target response
            }
            # Add behavior ID and other fields
            for k, v in item.items():
                if k not in ("Goal", "Category", "Target"):
                    meta[k.lower()] = v
            
            yield Example(
                id=f"jbb_{split}_{idx}",
                benchmark=self.name,
                category=category,
                input=goal,
                meta=meta,
            )
    
    def iter_harmful(self) -> Iterator[Example]:
        """Iterate over harmful behavior examples.
        
        These are requests that models should refuse.
        """
        return self.iter_examples("harmful")
    
    def iter_benign(self) -> Iterator[Example]:
        """Iterate over benign behavior examples.
        
        These are similar-sounding but safe requests that
        models should handle normally.
        """
        return self.iter_examples("benign")
    
    def iter_all(self) -> Iterator[Example]:
        """Iterate over all examples from both splits.
        
        Yields harmful examples first, then benign.
        """
        yield from self.iter_harmful()
        yield from self.iter_benign()
    
    def iter_by_category(
        self,
        category: str,
        split: str | None = None,
    ) -> Iterator[Example]:
        """Iterate over examples in a specific harm category.
        
        Args:
            category: The category to filter by.
            split: Optional split to filter ("harmful" or "benign").
                   If None, iterates both splits.
            
        Yields:
            Example objects matching the category.
        """
        if split is not None:
            for example in self.iter_examples(split):
                if example.category == category:
                    yield example
        else:
            for example in self.iter_all():
                if example.category == category:
                    yield example
    
    def get_categories(self) -> list[str]:
        """Return the list of harm categories in JBB."""
        return list(JBB_CATEGORIES)
    
    def category_counts(
        self,
        split: str | None = None,
    ) -> dict[str, int]:
        """Count examples per category.
        
        Args:
            split: Optional split to count. If None, counts all.
            
        Returns:
            Dictionary mapping category names to counts.
        """
        self._ensure_loaded()
        counts: dict[str, int] = {}
        
        if split is not None:
            examples = self.iter_examples(split)
        else:
            examples = self.iter_all()
        
        for example in examples:
            counts[example.category] = counts.get(example.category, 0) + 1
        return counts
    
    def split_counts(self) -> dict[str, int]:
        """Count examples per split.
        
        Returns:
            Dictionary with "harmful" and "benign" counts.
        """
        self._ensure_loaded()
        return {
            split: sum(1 for _ in self.iter_examples(split))
            for split in self.splits()
        }
