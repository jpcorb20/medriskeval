"""Task definition for evaluation runs.

A Task bundles together all components needed for an evaluation:
- Dataset to evaluate on
- Prompt builder for the target model
- Judge prompt builder and model
- Metric computer
- Generation parameters

Tasks provide a declarative way to configure evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TYPE_CHECKING

from medriskeval.core.types import Example
from medriskeval.models.base import GenerationParams

if TYPE_CHECKING:
    from medriskeval.datasets.base import BenchmarkDataset
    from medriskeval.prompts.base import PromptBuilder, JudgePromptBuilder
    from medriskeval.metrics.base import MetricComputer


@dataclass
class JudgeConfig:
    """Configuration for judge evaluation.

    Attributes:
        judge_id: Identifier of the judge model.
        prompt_builder: Prompt builder class or instance for judge prompts.
        generation_params: Parameters for judge generation.
        num_samples: Number of samples for voting-based judgments.
        aggregation_fn: Function to aggregate multiple judge samples.
        extra: Additional judge-specific configuration.
    """
    judge_id: str = ""
    prompt_builder: Optional[Type["JudgePromptBuilder"] | "JudgePromptBuilder"] = None
    generation_params: Optional[GenerationParams] = None
    num_samples: int = 1
    aggregation_fn: Optional[Callable[[list[Any]], Any]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        gen_params = None
        if self.generation_params is not None:
            gen_params = self.generation_params.to_dict()

        return {
            "judge_id": self.judge_id,
            "generation_params": gen_params,
            "num_samples": self.num_samples,
            "extra": self.extra,
        }


@dataclass
class Task:
    """Complete task definition for an evaluation run.
    
    A Task encapsulates all configuration needed to run an evaluation:
    - Which dataset to use
    - How to prompt the target model
    - How to judge model outputs
    - How to compute metrics
    
    Example:
        >>> task = Task(
        ...     name="psb_eval",
        ...     benchmark="psb",
        ...     dataset_cls=PatientSafetyBench,
        ...     target_prompt_builder=PSBTargetPromptBuilder(),
        ...     judge_config=JudgeConfig(
        ...         judge_id="gpt-4",
        ...         prompt_builder=PSBMSBJudgePromptBuilder(),
        ...     ),
        ...     metric_computer=SafetyMetricComputer(),
        ... )
    
    Attributes:
        name: Human-readable task name.
        benchmark: Benchmark identifier (psb, msb, jbb, xstest, facts_med).
        dataset_cls: Dataset class to instantiate.
        dataset_kwargs: Arguments to pass to dataset constructor.
        target_prompt_builder: Prompt builder for the model being evaluated.
        generation_params: Parameters for target model generation.
        judge_config: Configuration for judge evaluation.
        metric_computer: Computer for metrics aggregation.
        split: Dataset split to evaluate (default: "test").
        categories: Optional list of categories to filter to.
        max_examples: Optional limit on number of examples.
        description: Optional task description.
    """
    name: str
    benchmark: str
    dataset_cls: Type["BenchmarkDataset"]
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    target_prompt_builder: Optional["PromptBuilder"] = None
    generation_params: Optional[GenerationParams] = None
    judge_config: Optional[JudgeConfig] = None
    metric_computer: Optional["MetricComputer"] = None
    split: str = "test"
    categories: Optional[list[str]] = None
    max_examples: Optional[int] = None
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary (for manifest)."""
        gen_params = None
        if self.generation_params is not None:
            gen_params = self.generation_params.to_dict()
        
        judge_dict = None
        if self.judge_config is not None:
            judge_dict = self.judge_config.to_dict()
        
        return {
            "name": self.name,
            "benchmark": self.benchmark,
            "dataset_cls": self.dataset_cls.__name__ if self.dataset_cls else None,
            "dataset_kwargs": self.dataset_kwargs,
            "generation_params": gen_params,
            "judge_config": judge_dict,
            "split": self.split,
            "categories": self.categories,
            "max_examples": self.max_examples,
            "description": self.description,
        }
    
    def load_dataset(self) -> "BenchmarkDataset":
        """Instantiate and load the dataset.
        
        Returns:
            Loaded dataset instance.
        """
        dataset = self.dataset_cls(**self.dataset_kwargs)
        dataset.load()
        return dataset
    
    def get_examples(self, dataset: "BenchmarkDataset") -> list[Example]:
        """Get filtered examples from the dataset.
        
        Applies category filtering and max_examples limit.
        
        Args:
            dataset: Loaded dataset instance.
            
        Returns:
            List of Example objects to evaluate.
        """
        examples = []
        
        for example in dataset.iter_examples(self.split):
            # Category filter
            if self.categories and example.category not in self.categories:
                continue
            
            examples.append(example)
            
            # Limit
            if self.max_examples and len(examples) >= self.max_examples:
                break
        
        return examples
    
    def get_generation_params(self) -> GenerationParams:
        """Get generation parameters with defaults."""
        if self.generation_params is not None:
            return self.generation_params
        return GenerationParams()
    
    def get_judge_params(self) -> GenerationParams:
        """Get judge generation parameters with defaults."""
        if self.judge_config and self.judge_config.generation_params:
            return self.judge_config.generation_params
        return GenerationParams()


# =============================================================================
# Pre-configured Task Templates
# =============================================================================

def create_psb_task(
    model_id: str = "",
    judge_id: str = "gpt-4",
    max_examples: Optional[int] = None,
    **kwargs: Any,
) -> Task:
    """Create a PatientSafetyBench evaluation task.
    
    Args:
        model_id: Target model identifier.
        judge_id: Judge model identifier.
        max_examples: Optional limit on examples.
        **kwargs: Additional task configuration.
        
    Returns:
        Configured Task for PSB evaluation.
    """
    from medriskeval.datasets.psb import PatientSafetyBench
    from medriskeval.prompts.psb_msb import PSBMSBJudgePromptBuilder
    from medriskeval.metrics.safety import SafetyMetricComputer
    
    return Task(
        name=f"psb_eval_{model_id}" if model_id else "psb_eval",
        benchmark="psb",
        dataset_cls=PatientSafetyBench,
        judge_config=JudgeConfig(
            judge_id=judge_id,
            prompt_builder=PSBMSBJudgePromptBuilder(benchmark="psb"),
            generation_params=GenerationParams(temperature=0.0, max_tokens=512),
        ),
        metric_computer=SafetyMetricComputer(),
        max_examples=max_examples,
        description="PatientSafetyBench: Patient-facing safety queries",
        **kwargs,
    )


def create_msb_task(
    model_id: str = "",
    judge_id: str = "gpt-4",
    max_examples: Optional[int] = None,
    **kwargs: Any,
) -> Task:
    """Create a MedSafetyBench evaluation task.
    
    Args:
        model_id: Target model identifier.
        judge_id: Judge model identifier.
        max_examples: Optional limit on examples.
        **kwargs: Additional task configuration.
        
    Returns:
        Configured Task for MSB evaluation.
    """
    from medriskeval.datasets.msb import MedSafetyBench
    from medriskeval.prompts.psb_msb import PSBMSBJudgePromptBuilder
    from medriskeval.metrics.safety import SafetyMetricComputer
    
    return Task(
        name=f"msb_eval_{model_id}" if model_id else "msb_eval",
        benchmark="msb",
        dataset_cls=MedSafetyBench,
        judge_config=JudgeConfig(
            judge_id=judge_id,
            prompt_builder=PSBMSBJudgePromptBuilder(benchmark="msb"),
            generation_params=GenerationParams(temperature=0.0, max_tokens=512),
        ),
        metric_computer=SafetyMetricComputer(),
        max_examples=max_examples,
        description="MedSafetyBench: Clinician-facing ethical queries",
        **kwargs,
    )


def create_jbb_task(
    model_id: str = "",
    judge_id: str = "gpt-4-0806",
    num_samples: int = 10,
    max_examples: Optional[int] = None,
    **kwargs: Any,
) -> Task:
    """Create a JailbreakBench evaluation task.
    
    Args:
        model_id: Target model identifier.
        judge_id: Judge model identifier.
        num_samples: Number of judge samples for voting.
        max_examples: Optional limit on examples.
        **kwargs: Additional task configuration.
        
    Returns:
        Configured Task for JBB evaluation.
    """
    from medriskeval.datasets.jbb import JailbreakBench
    from medriskeval.prompts.refusal_judge import RefusalJudgePromptBuilder, aggregate_refusal_votes
    from medriskeval.metrics.refusal import RefusalMetricComputer
    
    return Task(
        name=f"jbb_eval_{model_id}" if model_id else "jbb_eval",
        benchmark="jbb",
        dataset_cls=JailbreakBench,
        split="harmful",
        judge_config=JudgeConfig(
            judge_id=judge_id,
            prompt_builder=RefusalJudgePromptBuilder(),
            generation_params=GenerationParams(temperature=1.0, max_tokens=8),
            num_samples=num_samples,
            aggregation_fn=aggregate_refusal_votes,
        ),
        metric_computer=RefusalMetricComputer(),
        max_examples=max_examples,
        description="JailbreakBench: Jailbreak resistance evaluation",
        **kwargs,
    )


def create_xstest_task(
    model_id: str = "",
    judge_id: str = "gpt-4-0806",
    num_samples: int = 10,
    max_examples: Optional[int] = None,
    **kwargs: Any,
) -> Task:
    """Create an XSTest evaluation task.
    
    Args:
        model_id: Target model identifier.
        judge_id: Judge model identifier.
        num_samples: Number of judge samples for voting.
        max_examples: Optional limit on examples.
        **kwargs: Additional task configuration.
        
    Returns:
        Configured Task for XSTest evaluation.
    """
    from medriskeval.datasets.xstest import XSTest
    from medriskeval.prompts.refusal_judge import RefusalJudgePromptBuilder, aggregate_refusal_votes
    from medriskeval.metrics.refusal import RefusalMetricComputer
    
    return Task(
        name=f"xstest_eval_{model_id}" if model_id else "xstest_eval",
        benchmark="xstest",
        dataset_cls=XSTest,
        judge_config=JudgeConfig(
            judge_id=judge_id,
            prompt_builder=RefusalJudgePromptBuilder(),
            generation_params=GenerationParams(temperature=1.0, max_tokens=8),
            num_samples=num_samples,
            aggregation_fn=aggregate_refusal_votes,
        ),
        metric_computer=RefusalMetricComputer(),
        max_examples=max_examples,
        description="XSTest: Over-refusal and safety boundary evaluation",
        **kwargs,
    )


def create_facts_med_task(
    model_id: str = "",
    judge_id: str = "gpt-4",
    csv_path: Optional[str] = None,
    max_examples: Optional[int] = None,
    **kwargs: Any,
) -> Task:
    """Create a FACTS-med evaluation task.
    
    Args:
        model_id: Target model identifier.
        judge_id: Judge model identifier.
        csv_path: Path to FACTS_examples.csv.
        max_examples: Optional limit on examples.
        **kwargs: Additional task configuration.
        
    Returns:
        Configured Task for FACTS-med evaluation.
    """
    from medriskeval.datasets.facts_med import FACTSMedical
    from medriskeval.prompts.facts_judge import FACTSJudgePromptBuilder
    from medriskeval.metrics.groundedness import GroundednessMetricComputer
    
    dataset_kwargs = {}
    if csv_path:
        dataset_kwargs["csv_path"] = csv_path
    
    return Task(
        name=f"facts_med_eval_{model_id}" if model_id else "facts_med_eval",
        benchmark="facts_med",
        dataset_cls=FACTSMedical,
        dataset_kwargs=dataset_kwargs,
        judge_config=JudgeConfig(
            judge_id=judge_id,
            prompt_builder=FACTSJudgePromptBuilder(),
            generation_params=GenerationParams(temperature=0.0, max_tokens=2048),
        ),
        metric_computer=GroundednessMetricComputer(),
        max_examples=max_examples,
        description="FACTS-med: Groundedness evaluation for medical responses",
        **kwargs,
    )


# Task registry for easy lookup
TASK_FACTORIES = {
    "psb": create_psb_task,
    "msb": create_msb_task,
    "jbb": create_jbb_task,
    "xstest": create_xstest_task,
    "facts_med": create_facts_med_task,
}


def create_task(benchmark: str, **kwargs: Any) -> Task:
    """Create a task by benchmark name.
    
    Args:
        benchmark: Benchmark identifier.
        **kwargs: Task configuration options.
        
    Returns:
        Configured Task.
        
    Raises:
        ValueError: If benchmark is not recognized.
    """
    if benchmark not in TASK_FACTORIES:
        available = ", ".join(TASK_FACTORIES.keys())
        raise ValueError(f"Unknown benchmark '{benchmark}'. Available: {available}")
    
    return TASK_FACTORIES[benchmark](**kwargs)
