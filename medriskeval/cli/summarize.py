"""CLI summarize command for medriskeval.

Usage:
    medriskeval summarize runs/psb_20240115_123456
    medriskeval summarize runs/psb_20240115_123456 --format csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(help="Summarize and export evaluation results")


def find_manifest(run_dir: Path) -> Optional[Path]:
    """Find manifest.json in a run directory."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    return None


def find_metrics(run_dir: Path) -> Optional[Path]:
    """Find metrics.json in a run directory."""
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return metrics_path
    return None


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.command("summarize")
def summarize_run(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to run directory (e.g., runs/psb_20240115_123456)",
        ),
    ],
    output_format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format: table, json, csv, markdown",
        ),
    ] = "table",
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output file path (default: stdout or auto-named)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Include additional details",
        ),
    ] = False,
) -> None:
    """Summarize results from a completed evaluation run.
    
    Examples:
        medriskeval summarize runs/psb_20240115_123456
        medriskeval summarize runs/jbb_run --format csv -o results.csv
        medriskeval summarize runs/xstest_run --format markdown
    """
    # Validate run directory
    if not run_dir.exists():
        typer.echo(f"Error: Run directory not found: {run_dir}", err=True)
        raise typer.Exit(1)
    
    if not run_dir.is_dir():
        typer.echo(f"Error: Not a directory: {run_dir}", err=True)
        raise typer.Exit(1)
    
    # Load manifest
    manifest_path = find_manifest(run_dir)
    if not manifest_path:
        typer.echo(f"Error: No manifest.json found in {run_dir}", err=True)
        raise typer.Exit(1)
    
    manifest = load_json(manifest_path)
    
    # Load metrics
    metrics_path = find_metrics(run_dir)
    metrics = load_json(metrics_path) if metrics_path else {}
    
    # Format and output
    if output_format == "table":
        _print_table_summary(manifest, metrics, verbose)
    elif output_format == "json":
        _output_json(manifest, metrics, output_file)
    elif output_format == "csv":
        _output_csv(manifest, metrics, output_file)
    elif output_format == "markdown":
        _output_markdown(manifest, metrics, output_file, verbose)
    else:
        typer.echo(f"Error: Unknown format '{output_format}'. Use: table, json, csv, markdown", err=True)
        raise typer.Exit(1)


def _extract_flat_metrics(metrics: dict) -> list[tuple[str, str]]:
    """Extract human-readable flat metrics from the nested metrics.json structure.

    Returns a list of (label, formatted_value) tuples.
    """
    rows: list[tuple[str, str]] = []
    benchmark = metrics.get("benchmark", "")

    # Top-level info
    if "model_id" in metrics:
        rows.append(("Model", metrics["model_id"]))
    if "total_count" in metrics:
        rows.append(("Total Samples", str(metrics["total_count"])))

    # Extract from metrics.safety (PSB/MSB)
    safety = (metrics.get("metrics") or {}).get("safety", {})
    if safety:
        val = safety.get("value", {})
        safe_rate = val.get("safe_rate")
        if safe_rate is not None:
            rows.append(("Safe Rate", f"{safe_rate:.1%}"))
        unsafe_rate = val.get("unsafe_rate")
        if unsafe_rate is not None:
            rows.append(("Unsafe Rate", f"{unsafe_rate:.1%}"))
        mean_score = val.get("mean_score")
        if mean_score is not None:
            rows.append(("Mean Score", f"{mean_score:.2f}"))

        # Score distribution
        histogram = val.get("histogram", {})
        counts = histogram.get("counts", {})
        labels = safety.get("meta", {}).get("score_labels", {})
        if counts:
            rows.append(("", ""))
            rows.append(("**Score Distribution**", ""))
            for score in sorted(counts.keys(), key=int):
                label = labels.get(score, "")
                label_str = f" ({label})" if label else ""
                rows.append((f"  Score {score}{label_str}", str(counts[score])))

    # Extract from metrics.refusal (JBB/XSTest)
    refusal = (metrics.get("metrics") or {}).get("refusal", {})
    if refusal:
        val = refusal.get("value", {})
        for key in ("refusal_rate", "compliance_rate", "partial_rate"):
            v = val.get(key)
            if v is not None:
                rows.append((key.replace("_", " ").title(), f"{v:.1%}"))

        label_counts = val.get("label_counts", {})
        if label_counts:
            rows.append(("", ""))
            rows.append(("**Label Distribution**", ""))
            for label, count in label_counts.items():
                rows.append((f"  {label}", str(count)))

    # Extract from metrics.groundedness (FACTS-med)
    groundedness = (metrics.get("metrics") or {}).get("groundedness", {})
    if groundedness:
        val = groundedness.get("value", {})
        overall = val.get("overall_groundedness_score")
        if overall is not None:
            rows.append(("Overall Groundedness", f"{overall:.1%}"))

        proportions = val.get("sentence_label_proportions", {})
        if proportions:
            rows.append(("", ""))
            rows.append(("**Sentence Labels**", ""))
            for label, prop in proportions.items():
                rows.append((f"  {label}", f"{prop:.1%}"))

    # Category breakdown
    category_metrics = metrics.get("category_metrics", {})
    if category_metrics:
        rows.append(("", ""))
        rows.append(("**Per-Category Results**", ""))
        for cat_name, cat_data in category_metrics.items():
            cat_safety = cat_data.get("safety", {})
            cat_val = cat_safety.get("value", {}) if isinstance(cat_safety, dict) else {}
            cat_safe_rate = cat_val.get("safe_rate")
            cat_mean = cat_val.get("mean_score")
            if cat_safe_rate is not None:
                rows.append((f"  {cat_name} safe_rate", f"{cat_safe_rate:.1%}"))
            if cat_mean is not None:
                rows.append((f"  {cat_name} mean_score", f"{cat_mean:.2f}"))

    return rows


def _print_table_summary(manifest: dict, metrics: dict, verbose: bool) -> None:
    """Print a formatted table summary to stdout."""
    typer.echo("\n" + "=" * 60)
    typer.echo("EVALUATION SUMMARY")
    typer.echo("=" * 60)

    # Run info
    typer.echo(f"\n[Run Information]")
    typer.echo(f"   Run ID:     {manifest.get('run_id', 'N/A')}")
    typer.echo(f"   Benchmark:  {manifest.get('benchmark', 'N/A')}")
    typer.echo(f"   Started:    {manifest.get('started_at', manifest.get('start_time', 'N/A'))}")
    typer.echo(f"   Completed:  {manifest.get('completed_at', manifest.get('end_time', 'N/A'))}")

    # Model info
    typer.echo(f"\n[Model Configuration]")
    typer.echo(f"   Target:     {manifest.get('model_id', manifest.get('model_config', {}).get('model_id', 'N/A'))}")
    typer.echo(f"   Judge:      {manifest.get('judge_id', manifest.get('judge_config', {}).get('model_id', 'N/A'))}")

    # Sample counts
    typer.echo(f"\n[Samples]")
    typer.echo(f"   Total:      {manifest.get('total_examples', manifest.get('total_samples', 'N/A'))}")
    typer.echo(f"   Completed:  {manifest.get('completed_examples', manifest.get('generated_count', 'N/A'))}")

    # Metrics
    if metrics:
        typer.echo(f"\n[Metrics]")
        rows = _extract_flat_metrics(metrics)
        for label, value in rows:
            if label.startswith("**"):
                typer.echo(f"\n   {label.strip('*')}:")
            elif label == "":
                continue
            elif label.startswith("  "):
                typer.echo(f"      {label.strip()}: {value}")
            else:
                typer.echo(f"   {label}: {value}")

    if verbose:
        _print_verbose_info(manifest)

    typer.echo("\n" + "=" * 60)


def _print_safety_metrics(metrics: dict) -> None:
    """Print safety metrics (PSB/MSB)."""
    histogram = metrics.get("score_histogram", {})
    if histogram:
        typer.echo("   Score Distribution:")
        for score in ["1", "2", "3", "4", "5"]:
            count = histogram.get(score, 0)
            bar = "█" * min(count, 40)
            typer.echo(f"      {score}: {count:4d} {bar}")

    safe_rate = metrics.get("safe_rate")
    if safe_rate is not None:
        typer.echo(f"   Safe Rate (score ≤ 2): {safe_rate:.1%}")

    mean_score = metrics.get("mean_score")
    if mean_score is not None:
        typer.echo(f"   Mean Score: {mean_score:.2f}")


def _print_refusal_metrics(metrics: dict) -> None:
    """Print refusal metrics (JBB/XSTest)."""
    compliance_rate = metrics.get("compliance_rate")
    if compliance_rate is not None:
        typer.echo(f"   Compliance Rate: {compliance_rate:.1%}")

    refusal_rate = metrics.get("refusal_rate")
    if refusal_rate is not None:
        typer.echo(f"   Refusal Rate: {refusal_rate:.1%}")

    partial_rate = metrics.get("partial_rate")
    if partial_rate is not None:
        typer.echo(f"   Partial Rate: {partial_rate:.1%}")

    label_counts = metrics.get("label_counts", {})
    if label_counts:
        typer.echo("   Label Distribution:")
        for label, count in label_counts.items():
            typer.echo(f"      {label}: {count}")


def _print_groundedness_metrics(metrics: dict) -> None:
    """Print groundedness metrics (FACTS-med)."""
    sentence_labels = metrics.get("sentence_label_proportions", {})
    if sentence_labels:
        typer.echo("   Sentence Label Proportions:")
        for label, prop in sentence_labels.items():
            bar = "█" * int(prop * 40)
            typer.echo(f"      {label:20}: {prop:.1%} {bar}")

    overall_score = metrics.get("overall_groundedness_score")
    if overall_score is not None:
        typer.echo(f"   Overall Groundedness: {overall_score:.1%}")


def _format_value(value) -> str:
    """Format a value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_verbose_info(manifest: dict) -> None:
    """Print verbose information."""
    typer.echo(f"\n[Additional Details]")

    git_info = manifest.get("git_info", {})
    if git_info:
        typer.echo(f"   Git Commit: {git_info.get('commit', 'N/A')}")
        typer.echo(f"   Git Branch: {git_info.get('branch', 'N/A')}")

    gen_config = manifest.get("generation_config", {})
    if gen_config:
        typer.echo(f"   Temperature: {gen_config.get('temperature', 'N/A')}")
        typer.echo(f"   Max Tokens: {gen_config.get('max_tokens', 'N/A')}")


def _output_json(manifest: dict, metrics: dict, output_file: Optional[Path]) -> None:
    """Output summary as JSON."""
    summary = {
        "manifest": manifest,
        "metrics": metrics,
    }

    output = json.dumps(summary, indent=2)

    if output_file:
        output_file.write_text(output, encoding="utf-8")
        typer.echo(f"JSON written to {output_file}")
    else:
        typer.echo(output)


def _output_csv(
    manifest: dict,
    metrics: dict,
    output_file: Optional[Path]
) -> None:
    """Output summary as CSV."""
    import csv
    from io import StringIO

    # Flatten metrics for CSV
    rows = []
    row = {
        "run_id": manifest.get("run_id"),
        "benchmark": manifest.get("benchmark"),
        "model": manifest.get("model_id", manifest.get("model_config", {}).get("model_id")),
        "judge": manifest.get("judge_id", manifest.get("judge_config", {}).get("model_id")),
        "total_samples": manifest.get("total_examples", manifest.get("total_samples")),
        "status": "completed" if manifest.get("completed_at") else manifest.get("status"),
    }

    # Add flattened metrics
    for label, value in _extract_flat_metrics(metrics):
        if label and not label.startswith("**"):
            row[label.strip()] = value

    rows.append(row)

    # Write CSV
    if output_file:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        typer.echo(f"CSV written to {output_file}")
    else:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        typer.echo(output.getvalue())


def _output_markdown(
    manifest: dict,
    metrics: dict,
    output_file: Optional[Path],
    verbose: bool,
) -> None:
    """Output summary as Markdown."""
    lines = []

    lines.append(f"# Evaluation Summary: {manifest.get('run_id', 'N/A')}")
    lines.append("")
    lines.append("## Run Information")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Benchmark | {manifest.get('benchmark', 'N/A')} |")
    lines.append(f"| Target Model | {manifest.get('model_id', manifest.get('model_config', {}).get('model_id', 'N/A'))} |")
    lines.append(f"| Judge Model | {manifest.get('judge_id', manifest.get('judge_config', {}).get('model_id', 'N/A'))} |")
    lines.append(f"| Total Samples | {manifest.get('total_examples', manifest.get('total_samples', 'N/A'))} |")
    lines.append(f"| Status | {'completed' if manifest.get('completed_at') else manifest.get('status', 'N/A')} |")
    lines.append("")

    if metrics:
        rows = _extract_flat_metrics(metrics)
        current_section = None
        for label, value in rows:
            if label.startswith("**"):
                section_name = label.strip("*")
                if current_section != section_name:
                    lines.append("")
                    lines.append(f"### {section_name}")
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")
                    current_section = section_name
            elif label == "":
                continue
            elif label.startswith("  "):
                lines.append(f"| {label.strip()} | {value} |")
            else:
                if current_section is None:
                    lines.append("## Metrics")
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")
                    current_section = "metrics"
                lines.append(f"| {label} | {value} |")
        lines.append("")

    output = "\n".join(lines)

    if output_file:
        output_file.write_text(output, encoding="utf-8")
        typer.echo(f"Markdown written to {output_file}")
    else:
        typer.echo(output)


@app.command("compare")
def compare_runs(
    run_dirs: Annotated[
        list[Path],
        typer.Argument(
            help="Paths to run directories to compare",
        ),
    ],
    output_format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format: table, csv, markdown",
        ),
    ] = "table",
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output file path",
        ),
    ] = None,
) -> None:
    """Compare metrics across multiple evaluation runs.

    Example:
        medriskeval compare runs/run1 runs/run2 runs/run3 --format markdown
    """
    if len(run_dirs) < 2:
        typer.echo("Error: Need at least 2 run directories to compare", err=True)
        raise typer.Exit(1)

    # Load all runs
    runs = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            typer.echo(f"Warning: Skipping non-existent directory: {run_dir}", err=True)
            continue

        manifest_path = find_manifest(run_dir)
        if not manifest_path:
            typer.echo(f"Warning: No manifest in {run_dir}", err=True)
            continue

        manifest = load_json(manifest_path)
        metrics_path = find_metrics(run_dir)
        metrics = load_json(metrics_path) if metrics_path else {}

        runs.append({
            "run_dir": str(run_dir),
            "manifest": manifest,
            "metrics": metrics,
        })

    if len(runs) < 2:
        typer.echo("Error: Need at least 2 valid runs to compare", err=True)
        raise typer.Exit(1)

    if output_format == "table":
        _print_comparison_table(runs)
    elif output_format == "csv":
        _output_comparison_csv(runs, output_file)
    elif output_format == "markdown":
        _output_comparison_markdown(runs, output_file)
    else:
        typer.echo(f"Error: Unknown format '{output_format}'", err=True)
        raise typer.Exit(1)


def _print_comparison_table(runs: list[dict]) -> None:
    """Print comparison table."""
    typer.echo("\n" + "=" * 80)
    typer.echo("RUN COMPARISON")
    typer.echo("=" * 80)

    # Header
    headers = ["Metric"] + [r["manifest"].get("run_id", "N/A")[:20] for r in runs]
    typer.echo("\n" + " | ".join(f"{h:20}" for h in headers))
    typer.echo("-" * (22 * len(headers)))

    # Collect all unique metrics
    all_metrics = set()
    for run in runs:
        for key, value in run["metrics"].items():
            if isinstance(value, dict):
                for k in value.keys():
                    all_metrics.add(f"{key}.{k}")
            else:
                all_metrics.add(key)

    # Print rows
    for metric in sorted(all_metrics):
        row = [f"{metric:20}"]
        for run in runs:
            value = _get_nested_metric(run["metrics"], metric)
            row.append(f"{_format_value(value):20}")
        typer.echo(" | ".join(row))

    typer.echo("=" * 80)


def _get_nested_metric(metrics: dict, key: str):
    """Get a potentially nested metric value."""
    if "." in key:
        parts = key.split(".", 1)
        if parts[0] in metrics and isinstance(metrics[parts[0]], dict):
            return metrics[parts[0]].get(parts[1], "N/A")
    return metrics.get(key, "N/A")


def _output_comparison_csv(runs: list[dict], output_file: Optional[Path]) -> None:
    """Output comparison as CSV."""
    import csv
    from io import StringIO

    # Collect all metrics
    all_metrics = set()
    for run in runs:
        for key, value in run["metrics"].items():
            if isinstance(value, dict):
                for k in value.keys():
                    all_metrics.add(f"{key}.{k}")
            else:
                all_metrics.add(key)

    # Build rows
    fieldnames = ["run_id", "benchmark", "model"] + sorted(all_metrics)
    rows = []

    for run in runs:
        row = {
            "run_id": run["manifest"].get("run_id"),
            "benchmark": run["manifest"].get("benchmark"),
            "model": run["manifest"].get("model_config", {}).get("model_id"),
        }
        for metric in all_metrics:
            row[metric] = _get_nested_metric(run["metrics"], metric)
        rows.append(row)

    if output_file:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        typer.echo(f"Comparison CSV written to {output_file}")
    else:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        typer.echo(output.getvalue())


def _output_comparison_markdown(runs: list[dict], output_file: Optional[Path]) -> None:
    """Output comparison as Markdown."""
    lines = []

    lines.append("# Run Comparison")
    lines.append("")

    # Collect all metrics
    all_metrics = set()
    for run in runs:
        for key, value in run["metrics"].items():
            if isinstance(value, dict):
                for k in value.keys():
                    all_metrics.add(f"{key}.{k}")
            else:
                all_metrics.add(key)

    # Header
    run_ids = [r["manifest"].get("run_id", "N/A") for r in runs]
    lines.append("| Metric | " + " | ".join(run_ids) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(runs)) + "|")

    # Rows
    for metric in sorted(all_metrics):
        values = [_format_value(_get_nested_metric(r["metrics"], metric)) for r in runs]
        lines.append(f"| {metric} | " + " | ".join(values) + " |")

    output = "\n".join(lines)

    if output_file:
        output_file.write_text(output, encoding="utf-8")
        typer.echo(f"Comparison Markdown written to {output_file}")
    else:
        typer.echo(output)


def _add_run(runs: list, run_dir: Path, manifest_path: Path, benchmark_filter: Optional[str]) -> None:
    """Add a run to the list if it matches the filter."""
    manifest = load_json(manifest_path)
    run_benchmark = manifest.get("benchmark", "").lower()

    if benchmark_filter and run_benchmark != benchmark_filter.lower():
        return

    runs.append({
        "path": str(run_dir),
        "run_id": manifest.get("run_id", "N/A"),
        "benchmark": run_benchmark,
        "model": manifest.get("model_id", manifest.get("model_config", {}).get("model_id", "N/A")),
        "status": "completed" if manifest.get("completed_at") or manifest.get("end_time") else manifest.get("status", "N/A"),
        "samples": manifest.get("total_examples", manifest.get("total_samples", "N/A")),
    })


@app.command("list-runs")
def list_runs(
    runs_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to runs directory",
        ),
    ] = Path("runs"),
    benchmark: Annotated[
        Optional[str],
        typer.Option(
            "--benchmark", "-b",
            help="Filter by benchmark",
        ),
    ] = None,
) -> None:
    """List all completed runs in a directory.

    Example:
        medriskeval list-runs ./runs
        medriskeval list-runs ./runs --benchmark psb
    """
    if not runs_dir.exists():
        typer.echo(f"Error: Runs directory not found: {runs_dir}", err=True)
        raise typer.Exit(1)

    runs = []
    # Search up to 2 levels deep: runs/<run>/ and runs/<benchmark>/<run>/
    candidates = sorted(runs_dir.iterdir())
    for subdir in candidates:
        if not subdir.is_dir():
            continue

        manifest_path = find_manifest(subdir)
        if manifest_path:
            # Direct run directory: runs/<run>/
            _add_run(runs, subdir, manifest_path, benchmark)
        else:
            # Benchmark subdirectory: runs/<benchmark>/<run>/
            for nested in sorted(subdir.iterdir()):
                if not nested.is_dir():
                    continue
                nested_manifest = find_manifest(nested)
                if nested_manifest:
                    _add_run(runs, nested, nested_manifest, benchmark)

    if not runs:
        typer.echo("No runs found.")
        return

    typer.echo(f"\nFound {len(runs)} run(s):\n")
    typer.echo(f"{'Run ID':<25} {'Benchmark':<12} {'Model':<20} {'Samples':<10} {'Status':<10}")
    typer.echo("-" * 80)

    for run in runs:
        typer.echo(
            f"{run['run_id']:<25} "
            f"{run['benchmark']:<12} "
            f"{run['model']:<20} "
            f"{str(run['samples']):<10} "
            f"{run['status']:<10}"
        )


if __name__ == "__main__":
    app()
