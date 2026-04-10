"""MedSafetyBench (MSB) dataset adapter.

MedSafetyBench contains 450 samples across 9 AMA ethical categories,
representing ill-intentioned clinician queries designed to test
whether medical AI will comply with unethical requests.

Source: AI4LIFE-GROUP/med-safety-bench on GitHub
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

from medriskeval.core.types import Example
from medriskeval.core.registry import DatasetRegistry
from medriskeval.datasets.base import BenchmarkDataset, DatasetLoadError
from medriskeval.datasets.io import load_hf_dataset, get_cache_dir


# MSB AMA ethical categories (1-9)
MSB_CATEGORIES = {
    "category_1": "Competence, Compassion, and Respect for Human Dignity",
    "category_2": "Professionalism and Honesty",
    "category_3": "Law and Responsibility to Society",
    "category_4": "Patient Rights and Confidentiality",
    "category_5": "Continued Study and Information Sharing",
    "category_6": "Physician's Freedom of Choice",
    "category_7": "Community and Public Health",
    "category_8": "Responsibility to Patient",
    "category_9": "Medical Care for All",
}


@DatasetRegistry.register("msb")
class MedSafetyBench(BenchmarkDataset):
    """MedSafetyBench dataset adapter.
    
    450 samples across 9 AMA ethical categories representing queries
    from an ill-intentioned clinician perspective.
    
    Categories (based on AMA Code of Medical Ethics):
        1. Competence, Compassion, and Respect for Human Dignity
        2. Professionalism and Honesty
        3. Law and Responsibility to Society
        4. Patient Rights and Confidentiality
        5. Continued Study and Information Sharing
        6. Physician's Freedom of Choice
        7. Community and Public Health
        8. Responsibility to Patient
        9. Medical Care for All
    
    Example usage:
        >>> dataset = MedSafetyBench()
        >>> dataset.load()
        >>> for example in dataset.iter_examples("test"):
        ...     print(example.category, example.input)
    """
    
    name = "msb"
    description = "MedSafetyBench: 450 unethical clinician queries across 9 AMA categories"
    
    # GitHub repo for the dataset
    GITHUB_REPO = "https://github.com/AI4LIFE-GROUP/med-safety-bench.git"
    DATA_PATH = "datasets/test/gpt4"
    
    def __init__(
        self,
        cache_dir: str | None = None,
        local_path: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the MSB dataset adapter.
        
        Args:
            cache_dir: Optional cache directory.
            local_path: Optional path to local clone of the repository.
            **kwargs: Additional configuration.
        """
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.local_path = local_path
        self._data: dict[str, list[dict]] = {}
    
    def load(self) -> None:
        """Load the MedSafetyBench dataset.
        
        Attempts to load from local path if provided, otherwise downloads
        CSV files from the GitHub repository.
        """
        repo_path = self._get_repo_path()
        data_dir = Path(repo_path) / self.DATA_PATH
        
        if not data_dir.exists():
            raise DatasetLoadError(
                f"MedSafetyBench data directory not found: {data_dir}. "
                f"Clone the repository first: git clone {self.GITHUB_REPO}"
            )
        
        # Load each category CSV
        for category_num in range(1, 10):
            category_name = f"category_{category_num}"
            csv_path = data_dir / f"med_safety_demonstrations_{category_name}.csv"
            
            if not csv_path.exists():
                raise DatasetLoadError(f"Missing MSB file: {csv_path}")
            
            try:
                dataset = load_hf_dataset(
                    "csv",
                    data_files=str(csv_path),
                    split="train",
                    cache_dir=self.cache_dir,
                )
                self._data[category_name] = list(dataset)
            except Exception as e:
                raise DatasetLoadError(
                    f"Failed to load MSB category {category_num}: {e}"
                ) from e
        
        self._loaded = True
    
    def _get_repo_path(self) -> str:
        """Get the path to the med-safety-bench repository."""
        if self.local_path:
            return self.local_path
        
        # Check common locations
        candidates = [
            Path.cwd() / "med-safety-bench",
            get_cache_dir("repos") / "med-safety-bench",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # Return default expected location
        return str(candidates[0])
    
    def splits(self) -> Sequence[str]:
        """Return available splits.
        
        Note: MSB only provides test data.
        """
        return ["test"]
    
    def iter_examples(self, split: str = "test") -> Iterator[Example]:
        """Iterate over MSB examples.
        
        Args:
            split: The split to iterate (only "test" is valid).
            
        Yields:
            Example objects with MSB data.
        """
        self._ensure_loaded()
        self._validate_split(split)
        
        global_idx = 0
        for category_name, samples in sorted(self._data.items()):
            for item in samples:
                # MSB format: {'harmful_medical_request': str, ...}
                content = item.get("harmful_medical_request", "")
                
                # Get the item ID if present, otherwise use index
                item_id = item.get("Unnamed: 0", item.get("id", global_idx))
                
                # Build metadata
                meta = {
                    "original_id": item_id,
                    "category_description": MSB_CATEGORIES.get(category_name, ""),
                }
                # Add any additional fields
                for k, v in item.items():
                    if k not in ("harmful_medical_request", "Unnamed: 0", "id"):
                        meta[k] = v
                
                yield Example(
                    id=f"msb_{category_name}_{item_id}",
                    benchmark=self.name,
                    category=category_name,
                    input=content,
                    meta=meta,
                )
                global_idx += 1
    
    def iter_by_category(
        self,
        category: str | int,
        split: str = "test",
    ) -> Iterator[Example]:
        """Iterate over examples in a specific category.
        
        Args:
            category: Category name (e.g., "category_1") or number (1-9).
            split: The split to iterate.
            
        Yields:
            Example objects matching the category.
        """
        if isinstance(category, int):
            category = f"category_{category}"
        
        for example in self.iter_examples(split):
            if example.category == category:
                yield example
    
    def get_categories(self) -> list[str]:
        """Return the list of category names."""
        return [f"category_{i}" for i in range(1, 10)]
    
    def get_category_descriptions(self) -> dict[str, str]:
        """Return category names with their AMA descriptions."""
        return dict(MSB_CATEGORIES)
    
    def category_counts(self, split: str = "test") -> dict[str, int]:
        """Count examples per category.
        
        Args:
            split: The split to count.
            
        Returns:
            Dictionary mapping category names to counts.
        """
        self._ensure_loaded()
        return {cat: len(samples) for cat, samples in self._data.items()}
