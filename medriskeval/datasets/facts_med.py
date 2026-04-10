"""FACTS Medical subset dataset adapter.

FACTS (Faithful Assessment of Clinical Text Summarization) provides
examples for evaluating groundedness/faithfulness of model outputs
against provided context documents.

The medical subset contains 219 samples under 5000 tokens, filtered
from the larger FACTS benchmark.

Source: Local CSV file (FACTS_examples.csv filtered by domain="Medical")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

from medriskeval.core.types import Example
from medriskeval.core.registry import DatasetRegistry
from medriskeval.datasets.base import BenchmarkDataset, DatasetLoadError
from medriskeval.datasets.io import load_hf_dataset, load_csv_simple, get_cache_dir


@DatasetRegistry.register("facts_med")
class FACTSMedical(BenchmarkDataset):
    """FACTS Medical subset dataset adapter.
    
    Evaluates groundedness/faithfulness of model outputs against
    provided context documents. Each example includes:
    - system_instruction: Instructions for the model
    - context_document: The reference document to ground responses in
    - user_request: The user's query
    
    Example.input is a structured dict with these three fields to
    enable proper prompt construction for groundedness evaluation.
    
    Example usage:
        >>> dataset = FACTSMedical()
        >>> dataset.load()
        >>> for example in dataset.iter_examples("test"):
        ...     instruction = example.input["instruction"]
        ...     context = example.input["context_document"]
        ...     query = example.input["query"]
    """
    
    name = "facts_med"
    description = "FACTS Medical: 219 groundedness evaluation samples"
    
    # Maximum token length to include
    MAX_TOKEN_LENGTH = 5000
    
    def __init__(
        self,
        cache_dir: str | None = None,
        csv_path: str | None = None,
        max_token_length: int = MAX_TOKEN_LENGTH,
        **kwargs,
    ) -> None:
        """Initialize the FACTS Medical dataset adapter.
        
        Args:
            cache_dir: Optional cache directory.
            csv_path: Path to FACTS_examples.csv file.
            max_token_length: Maximum token length to include (default 5000).
            **kwargs: Additional configuration.
        """
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.csv_path = csv_path
        self.max_token_length = max_token_length
        self._data: list[dict[str, Any]] = []
    
    def load(self) -> None:
        """Load the FACTS Medical dataset.
        
        Loads from CSV and filters to Medical domain only.
        """
        csv_path = self._find_csv_path()
        
        try:
            # Try loading with HF datasets first
            dataset = load_hf_dataset(
                "csv",
                data_files=str(csv_path),
                split="train",
                cache_dir=self.cache_dir,
            )
            # Filter to Medical domain
            dataset = dataset.filter(lambda x: x["domain"] == "Medical")
            self._data = list(dataset)
        except Exception:
            # Fallback to simple CSV loading
            all_data = load_csv_simple(csv_path)
            self._data = [
                row for row in all_data
                if row.get("domain") == "Medical"
            ]
        
        if not self._data:
            raise DatasetLoadError(
                f"No Medical domain samples found in {csv_path}"
            )
        
        self._loaded = True
    
    def _find_csv_path(self) -> Path:
        """Find the FACTS CSV file."""
        if self.csv_path:
            path = Path(self.csv_path)
            if path.exists():
                return path
            raise DatasetLoadError(f"CSV file not found: {self.csv_path}")
        
        # Search common locations
        candidates = [
            Path.cwd() / "FACTS_examples.csv",
            get_cache_dir("facts") / "FACTS_examples.csv",
            Path.home() / "FACTS_examples.csv",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        raise DatasetLoadError(
            "FACTS_examples.csv not found. Please provide csv_path parameter or "
            f"place the file in one of: {[str(c) for c in candidates]}"
        )
    
    def splits(self) -> Sequence[str]:
        """Return available splits.
        
        Note: FACTS provides a single set exposed as "test".
        """
        return ["test"]
    
    def iter_examples(self, split: str = "test") -> Iterator[Example]:
        """Iterate over FACTS Medical examples.
        
        Args:
            split: The split to iterate (only "test" is valid).
            
        Yields:
            Example objects with structured input containing:
            - instruction: System instruction
            - context_document: Reference document
            - query: User request
            
        Note:
            Only yields examples under max_token_length.
        """
        self._ensure_loaded()
        self._validate_split(split)
        
        for idx, item in enumerate(self._data):
            # Get the key fields
            instruction = item.get("system_instruction", "")
            context = item.get("context_document", "")
            query = item.get("user_request", "")
            domain = item.get("domain", "Medical")
            
            # Build structured input for groundedness evaluation
            structured_input = {
                "instruction": instruction,
                "context_document": context,
                "query": query,
            }
            
            # Build metadata
            meta: dict[str, Any] = {
                "domain": domain,
            }
            # Add other fields except the main ones
            for k, v in item.items():
                if k not in ("system_instruction", "context_document", 
                             "user_request", "domain"):
                    meta[k] = v
            
            yield Example(
                id=f"facts_med_{idx}",
                benchmark=self.name,
                category=domain,  # Could be extended for sub-categories
                input=structured_input,
                meta=meta,
            )
    
    def iter_with_token_filter(
        self,
        max_tokens: int | None = None,
        split: str = "test",
    ) -> Iterator[Example]:
        """Iterate with explicit token length filtering.
        
        Args:
            max_tokens: Maximum token length (uses instance default if None).
            split: The split to iterate.
            
        Yields:
            Example objects under the token limit.
            
        Note:
            Token counting is approximate based on character length / 4.
        """
        max_tokens = max_tokens or self.max_token_length
        
        for example in self.iter_examples(split):
            # Approximate token count
            if isinstance(example.input, dict):
                total_chars = sum(
                    len(str(v)) for v in example.input.values()
                )
            else:
                total_chars = len(str(example.input))
            
            approx_tokens = total_chars // 4
            
            if approx_tokens <= max_tokens:
                yield example
    
    def format_prompt(
        self,
        example: Example,
        template: str | None = None,
    ) -> str:
        """Format an example into a prompt string.
        
        Args:
            example: The example to format.
            template: Optional custom template. Uses default if None.
            
        Returns:
            Formatted prompt string ready for model input.
        """
        if template is None:
            template = (
                "---CONTEXT/REFERENCE---\n{context_document}\n\n"
                "---USER REQUEST---\n{query}"
            )
        
        if isinstance(example.input, dict):
            return template.format(**example.input)
        else:
            return str(example.input)
    
    def get_system_instruction(self, example: Example) -> str:
        """Extract the system instruction from an example.
        
        Args:
            example: The example to extract from.
            
        Returns:
            The system instruction string.
        """
        if isinstance(example.input, dict):
            return example.input.get("instruction", "")
        return ""
