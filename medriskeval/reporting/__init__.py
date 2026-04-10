"""Reporting and output generation for medriskeval.

This module provides utilities for exporting evaluation results
to various formats (JSON, CSV) and rendering summary tables.

Usage:
    from medriskeval.reporting import (
        export_metrics_to_json,
        export_metrics_to_csv,
        render_summary_table,
        save_report,
    )
    
    # Export to JSON
    json_str = export_metrics_to_json(metrics, "output/results.json")
    
    # Export to CSV
    csv_str = export_metrics_to_csv(metrics, "output/results.csv")
    
    # Render Markdown table
    table = render_summary_table(metrics)
    print(table)
    
    # Save in multiple formats
    outputs = save_report(metrics, "output/", formats=["json", "csv", "md"])
"""

from medriskeval.reporting.tables import (
    # Configuration
    TableConfig,
    # JSON export
    export_metrics_to_json,
    load_metrics_from_json,
    # CSV export
    export_metrics_to_csv,
    export_category_metrics_to_csv,
    # Summary tables
    render_safety_summary_table,
    render_refusal_summary_table,
    render_groundedness_summary_table,
    render_summary_table,
    render_multi_benchmark_summary,
    # File I/O
    save_report,
)


__all__ = [
    "TableConfig",
    "export_metrics_to_json",
    "load_metrics_from_json",
    "export_metrics_to_csv",
    "export_category_metrics_to_csv",
    "render_safety_summary_table",
    "render_refusal_summary_table",
    "render_groundedness_summary_table",
    "render_summary_table",
    "render_multi_benchmark_summary",
    "save_report",
]
