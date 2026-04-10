"""CLI run command for medriskeval.

Usage:
    medriskeval run --task psb --model openai:gpt-4.1-mini --judge openai:gpt-4-0806
    medriskeval run --task jbb --model vllm:mistral-7b --output-dir ./my_runs
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from medriskeval.config import (
    BenchmarkName,
    ModelConfig,
    get_preset_for_benchmark,
)

app = typer.Typer(help="Run benchmark evaluations")


def parse_model_string(model_str: str) -> ModelConfig:
    """Parse 'provider:model_id' format into ModelConfig."""
    try:
        return ModelConfig.from_string(model_str)
    except ValueError as e:
        typer.echo(f"Error parsing model string '{model_str}': {e}", err=True)
        raise typer.Exit(1)


@app.command()
def run_benchmark(
    task: Annotated[
        str,
        typer.Option(
            "--task", "-t",
            help="Benchmark task: psb, msb, jbb, xstest, facts_med",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="Target model in provider:model_id format (e.g., openai:gpt-4.1-mini)",
        ),
    ],
    judge: Annotated[
        Optional[str],
        typer.Option(
            "--judge", "-j",
            help="Judge model (default from preset). Format: provider:model_id",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o",
            help="Output directory (default: ./runs)",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Cache directory (default: ./cache)",
        ),
    ] = None,
    max_samples: Annotated[
        Optional[int],
        typer.Option(
            "--max-samples", "-n",
            help="Maximum samples to evaluate (for quick testing)",
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Disable caching",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Verbose output",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show configuration without running",
        ),
    ] = False,
) -> None:
    """Run a benchmark evaluation.

    Examples:
        medriskeval run --task psb --model openai:gpt-4.1-mini
        medriskeval run --task jbb --model openai:gpt-4 --judge openai:gpt-4-0806
        medriskeval run --task xstest --model vllm:mistral-7b -n 100
    """
    # Validate benchmark name
    try:
        benchmark = BenchmarkName(task.lower())
    except ValueError:
        available = ", ".join(b.value for b in BenchmarkName)
        typer.echo(f"Error: Unknown benchmark '{task}'. Available: {available}", err=True)
        raise typer.Exit(1)

    # Parse model configurations
    model_config = parse_model_string(model)

    # Get preset and determine judge
    preset = get_preset_for_benchmark(benchmark)
    if judge:
        judge_config = parse_model_string(judge)
    else:
        judge_config = preset.default_judge_model
        if verbose:
            typer.echo(f"Using default judge from preset: {judge_config.model_id}")

    # Dry run: show configuration and exit
    if dry_run:
        typer.echo("\n=== Dry Run Configuration ===")
        typer.echo(f"Benchmark: {benchmark.value}")
        typer.echo(f"Target Model: {model_config.provider.value}:{model_config.model_id}")
        typer.echo(f"Judge Model: {judge_config.provider.value}:{judge_config.model_id}")
        typer.echo(f"Output Directory: {output_dir or './runs'}")
        typer.echo(f"Cache Directory: {cache_dir or './cache'}")
        typer.echo(f"Max Samples: {max_samples or 'all'}")
        typer.echo(f"Seed: {seed or 'none'}")
        typer.echo(f"Cache Enabled: {not no_cache}")
        typer.echo("\nPreset Settings:")
        typer.echo(f"  Judge Samples: {preset.judge_num_samples}")
        typer.echo(f"  Target Temperature: {preset.target_generation.temperature}")
        typer.echo(f"  Judge Temperature: {preset.judge_generation.temperature}")
        if benchmark in (BenchmarkName.JBB, BenchmarkName.XSTEST):
            typer.echo(f"  Refusal Thresholds: {preset.refusal_thresholds}")
        return

    # Import runner here to avoid circular imports
    from medriskeval.runner import cli_run_evaluation

    typer.echo(f"\n🏃 Starting evaluation: {benchmark.value}")
    typer.echo(f"   Target: {model_config.model_id}")
    typer.echo(f"   Judge: {judge_config.model_id}")

    try:
        # Run the evaluation
        result = cli_run_evaluation(
            benchmark=benchmark,
            model_config=model_config,
            judge_config=judge_config,
            output_dir=str(output_dir) if output_dir else "./runs",
            cache_dir=str(cache_dir) if cache_dir and not no_cache else None,
            max_samples=max_samples,
            seed=seed,
            verbose=verbose,
        )

        typer.echo(f"\n✅ Evaluation complete!")
        typer.echo(f"   Run directory: {result.run_dir}")
        typer.echo(f"   Total samples: {result.total_samples}")
        typer.echo(f"   Duration: {result.duration_seconds:.1f}s")

        # Print summary metrics
        if result.metrics:
            typer.echo("\n📊 Summary Metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    typer.echo(f"   {key}: {value:.3f}")
                else:
                    typer.echo(f"   {key}: {value}")

    except Exception as e:
        typer.echo(f"\n❌ Evaluation failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command("list-tasks")
def list_tasks() -> None:
    """List available benchmark tasks with descriptions."""
    from medriskeval.config import list_presets

    typer.echo("\n📋 Available Benchmark Tasks:\n")

    for preset_info in list_presets():
        typer.echo(f"  {preset_info['benchmark']:12} - {preset_info['description']}")
        typer.echo(f"{'':14} Judge: {preset_info['judge_model']} ({preset_info['judge_samples']} samples)")
        typer.echo()


@app.command("validate")
def validate_config(
    task: Annotated[
        str,
        typer.Option("--task", "-t", help="Benchmark task"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Target model"),
    ],
    judge: Annotated[
        Optional[str],
        typer.Option("--judge", "-j", help="Judge model"),
    ] = None,
) -> None:
    """Validate configuration without running.

    Checks that model strings parse correctly and benchmark is valid.
    """
    errors = []

    # Validate benchmark
    try:
        benchmark = BenchmarkName(task.lower())
        typer.echo(f"✅ Benchmark: {benchmark.value}")
    except ValueError:
        errors.append(f"Unknown benchmark: {task}")
        typer.echo(f"❌ Benchmark: {task}")

    # Validate target model
    try:
        model_config = ModelConfig.from_string(model)
        typer.echo(f"✅ Target Model: {model_config.provider.value}:{model_config.model_id}")
    except ValueError as e:
        errors.append(f"Invalid model string: {e}")
        typer.echo(f"❌ Target Model: {model}")

    # Validate judge model
    if judge:
        try:
            judge_config = ModelConfig.from_string(judge)
            typer.echo(f"✅ Judge Model: {judge_config.provider.value}:{judge_config.model_id}")
        except ValueError as e:
            errors.append(f"Invalid judge string: {e}")
            typer.echo(f"❌ Judge Model: {judge}")
    else:
        typer.echo("ℹ️  Judge Model: (using preset default)")

    if errors:
        typer.echo(f"\n❌ Validation failed with {len(errors)} error(s)")
        raise typer.Exit(1)
    else:
        typer.echo("\n✅ Configuration is valid")


if __name__ == "__main__":
    app()
