"""Main CLI entry point for medriskeval.

Usage:
    medriskeval run --task psb --model openai:gpt-4.1-mini --judge openai:gpt-4-0806
    medriskeval summarize runs/psb_20240115_123456
    medriskeval list-runs ./runs
    medriskeval compare runs/run1 runs/run2
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from medriskeval.cli.run import run_benchmark
from medriskeval.cli.summarize import app as summarize_app

# Create main app
app = typer.Typer(
    name="medriskeval",
    help="Medical Risk Evaluation Framework",
    no_args_is_help=True,
    add_completion=False,
)

# Register run command directly (avoids double "run run" nesting)
app.command("run")(run_benchmark)
app.add_typer(summarize_app, name="summarize", help="Summarize evaluation results")


@app.command("run-config")
def run_config(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to YAML configuration file"),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show execution plan without running"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Run evaluations from a YAML configuration file.

    Executes every (task, model) combination declared in the config.

    Examples:
        medriskeval run-config eval.yaml
        medriskeval run-config eval.yaml --dry-run
    """
    from medriskeval.config import load_yaml_config
    from medriskeval.runner import run_yaml_config

    try:
        yaml_cfg = load_yaml_config(config_path)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        typer.echo(f"Error loading config: {exc}", err=True)
        raise typer.Exit(1)

    # CLI flags can override the YAML values
    if verbose:
        yaml_cfg.verbose = True

    n_combos = len(yaml_cfg.tasks) * len(yaml_cfg.models)
    task_names = ", ".join(t.benchmark.value for t in yaml_cfg.tasks)
    model_names = ", ".join(m.model_id for m in yaml_cfg.models)

    typer.echo(f"\nConfig: {config_path}")
    typer.echo(f"  Tasks:  {task_names}")
    typer.echo(f"  Models: {model_names}")
    typer.echo(f"  Runs:   {n_combos} combination(s)")

    if dry_run:
        typer.echo("\n=== Dry Run Plan ===")

    results = run_yaml_config(yaml_cfg, dry_run=dry_run)

    if dry_run:
        return

    # Summary table
    typer.echo(f"\n{'='*60}")
    typer.echo("Run Summary")
    typer.echo(f"{'='*60}")
    total_filtered = 0
    total_judge_filtered = 0
    for r in results:
        status = "OK" if r.success else "FAIL"
        typer.echo(
            f"  [{status}] {r.run_dir.name:40s}  "
            f"samples={r.total_samples:>5}  "
            f"time={r.duration_seconds:>6.1f}s"
        )
        if r.content_filtered or r.judge_content_filtered:
            typer.echo(
                f"         Content filtered: "
                f"{r.content_filtered} target, "
                f"{r.judge_content_filtered} judge"
            )
        if r.error_message:
            typer.echo(f"         Error: {r.error_message}")
        total_filtered += r.content_filtered
        total_judge_filtered += r.judge_content_filtered

    if total_filtered or total_judge_filtered:
        typer.echo(f"\nContent Filter Summary:")
        typer.echo(f"  Target responses blocked: {total_filtered}")
        typer.echo(f"  Judge  responses blocked: {total_judge_filtered}")
        typer.echo(
            "  Note: Filtered target prompts are recorded with "
            "finish_reason='content_filter'."
        )
        typer.echo(
            "  To reduce filtering, disable content filters on your "
            "Azure OpenAI deployment."
        )

    failures = sum(1 for r in results if not r.success)
    if failures:
        typer.echo(f"\n{failures}/{len(results)} run(s) failed.")
        raise typer.Exit(1)
    else:
        typer.echo(f"\nAll {len(results)} run(s) succeeded.")


@app.command()
def version() -> None:
    """Show medriskeval version."""
    typer.echo("medriskeval v0.1.0")


@app.command("list-tasks")
def list_tasks() -> None:
    """List available benchmark tasks."""
    from medriskeval.config import list_presets

    typer.echo("\n📋 Available Benchmark Tasks:\n")

    for preset_info in list_presets():
        typer.echo(f"  {preset_info['benchmark']:12} - {preset_info['description']}")
        typer.echo(f"{'':14} Default Judge: {preset_info['judge_model']} ({preset_info['judge_samples']} samples)")
        typer.echo()


@app.command("info")
def info() -> None:
    """Show information about medriskeval."""
    typer.echo("""
╔══════════════════════════════════════════════════════════════════╗
║                    medriskeval v0.1.0                            ║
║           Medical Risk Evaluation Framework                      ║
╚══════════════════════════════════════════════════════════════════╝

Supported Benchmarks:
  • PSB (PatientSafetyBench)  - Patient safety evaluation
  • MSB (MedSafetyBench)      - Medical safety evaluation
  • JBB (JailbreakBench)      - Jailbreak resistance
  • XSTest                    - Over-refusal detection
  • FACTS-med                 - Groundedness evaluation

Quick Start:
  medriskeval run --task psb --model openai:gpt-4.1-mini
  medriskeval summarize runs/<run_id>

Documentation:
  https://github.com/jpcorb20/medriskeval
""")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
