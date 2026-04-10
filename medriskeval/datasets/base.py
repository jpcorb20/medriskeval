"""Base dataset interface for medriskeval.

This module defines the abstract base class that all benchmark dataset
adapters must implement, ensuring a consistent interface for data loading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Sequence

from medriskeval.core.types import Example


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark dataset adapters.

    All dataset implementations must subclass this and implement:
    - name: class attribute with the benchmark name
    - load(): initialize/download the dataset
    - splits(): return available data splits
    - iter_examples(split): yield Example objects

    Example usage:
        dataset = PatientSafetyBench()
        dataset.load()
        for example in dataset.iter_examples("test"):
            print(example.id, example.category)
    """

    # Subclasses must define these
    name: str = ""
    description: str = ""

    def __init__(self, cache_dir: str | None = None, **kwargs) -> None:
        """Initialize the dataset adapter.

        Args:
            cache_dir: Optional directory for caching downloaded data.
            **kwargs: Additional configuration options.
        """
        self.cache_dir = cache_dir
        self.config = kwargs
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load or download the dataset.

        This method should:
        - Download data if not cached
        - Parse/preprocess as needed
        - Set self._loaded = True when complete

        Raises:
            DatasetLoadError: If the dataset cannot be loaded.
        """
        pass

    @abstractmethod
    def splits(self) -> Sequence[str]:
        """Return the available data splits.

        Returns:
            A sequence of split names (e.g., ["train", "test", "validation"]).
        """
        pass

    @abstractmethod
    def iter_examples(self, split: str) -> Iterator[Example]:
        """Iterate over examples in a given split.

        Args:
            split: The name of the split to iterate over.

        Yields:
            Example objects with consistent schema.

        Raises:
            ValueError: If the split name is invalid.
            DatasetNotLoadedError: If load() hasn't been called.
        """
        pass

    def get_examples(self, split: str) -> list[Example]:
        """Return all examples in a split as a list.

        This is a convenience method; prefer iter_examples() for large datasets.

        Args:
            split: The name of the split.

        Returns:
            List of all Example objects in the split.
        """
        return list(self.iter_examples(split))

    def __len__(self) -> int:
        """Return total number of examples across all splits."""
        if not self._loaded:
            return 0
        return sum(
            sum(1 for _ in self.iter_examples(split))
            for split in self.splits()
        )

    def _ensure_loaded(self) -> None:
        """Raise an error if the dataset hasn't been loaded."""
        if not self._loaded:
            raise DatasetNotLoadedError(
                f"{self.name}: Dataset not loaded. Call load() first."
            )

    def _validate_split(self, split: str) -> None:
        """Raise an error if the split is invalid."""
        valid_splits = self.splits()
        if split not in valid_splits:
            raise ValueError(
                f"{self.name}: Invalid split '{split}'. "
                f"Available splits: {list(valid_splits)}"
            )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"


class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass


class DatasetLoadError(DatasetError):
    """Raised when a dataset cannot be loaded."""
    pass


class DatasetNotLoadedError(DatasetError):
    """Raised when operations are attempted on an unloaded dataset."""
    pass
